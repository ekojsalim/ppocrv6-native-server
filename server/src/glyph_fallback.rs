//! HTTP policy and bounded CPU work; native model predictions stay independent.
use anyhow::{bail, Context, Result};
use image::DynamicImage;
use ppocrv6_native_server::glyph_matcher::*;
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

const MAX_ATTEMPTS: usize = 32;
const WORK_BUDGET: Duration = Duration::from_millis(100);
const MAX_DICTIONARY_BYTES: u64 = 128 * 1024 * 1024;

pub(crate) struct Fallback {
    dict: Dictionary,
    descriptors: Vec<[u8; 64]>,
    blocks: VerificationIndex,
}

pub(crate) fn info(enabled: bool) -> Value {
    json!({"enabled":enabled, "source":"cpu_template_v2", "trigger":"empty_text",
        "max_attempts_per_request":MAX_ATTEMPTS, "work_budget_ms":WORK_BUDGET.as_millis(),
        "max_concurrent_requests":1, "min_similarity":MIN_SIMILARITY,
        "min_different_character_margin":MIN_MARGIN,
        "score_contract":"accepted=1; model=null; similarity is not probability",
        "coverage":"U+4E00..U+9FFF; template-dependent; no line-like glyphs"})
}

impl Fallback {
    pub(crate) fn load_optional(path: &Path) -> Result<Option<Self>> {
        let metadata = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).context("inspect glyph dictionary"),
        };
        if metadata.len() > MAX_DICTIONARY_BYTES {
            bail!("glyph dictionary exceeds 128 MiB");
        }
        let mut dict: Dictionary =
            serde_json::from_reader(std::io::BufReader::new(fs::File::open(path)?))
                .context("read glyph dictionary")?;
        validate_dictionary(&dict)?;
        if dict.templates.len() > 100_000 {
            bail!("too many glyph templates");
        }
        for t in &mut dict.templates {
            // Keep dictionary recovery compatible with every current CJK policy.
            if !t
                .character
                .chars()
                .all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            {
                bail!("glyph dictionary contains a character outside the supported Han block");
            }
            t.shape.norm = energy(&t.shape);
        }
        let descriptors = dict
            .templates
            .iter()
            .map(|t| t.shape.descriptor())
            .collect();
        let blocks = VerificationIndex::new(&dict);
        Ok(Some(Self {
            dict,
            descriptors,
            blocks,
        }))
    }

    pub(crate) fn apply<E>(
        &self,
        response: &mut Value,
        images: &[String],
        available: bool,
        accepted_score: bool,
        mut decode: impl FnMut(&str) -> std::result::Result<DynamicImage, E>,
    ) -> std::result::Result<(), E> {
        let started = Instant::now();
        let mut attempted = 0;
        let mut applied = 0;
        let mut skipped = 0;
        if let Some(predictions) = response
            .get_mut("predictions")
            .and_then(Value::as_array_mut)
        {
            for (prediction, image) in predictions.iter_mut().zip(images) {
                if prediction.get("text").and_then(Value::as_str) != Some("") {
                    continue;
                }
                let skip = if !available {
                    Some("busy")
                } else if attempted >= MAX_ATTEMPTS {
                    Some("request_limit")
                } else if started.elapsed() >= WORK_BUDGET {
                    Some("work_budget")
                } else {
                    None
                };
                if let Some(reason) = skip {
                    prediction["cpu_fallback"] = json!({"applied":false,"skipped":reason});
                    skipped += 1;
                    continue;
                }
                attempted += 1;
                let mut shape = match normalize(decode(image)?) {
                    Ok(s) => s,
                    Err(reason) => {
                        prediction["cpu_fallback"] = json!({"applied":false,"rejection":reason});
                        continue;
                    }
                };
                shape.norm = energy(&shape);
                let (candidates, _) =
                    rank_profiled(&shape, &self.dict, &self.descriptors, false, &self.blocks);
                let accept = accepted(&candidates);
                let detail = json!({"applied":accept, "source":"cpu_template_v2",
                    "similarity":candidates.first().map(|c|c.similarity),
                    "similarity_type":"aligned_ink_cosine_minus_aspect_penalty_not_probability",
                    "required_margin":MIN_MARGIN,"acceptance_margin_verified":accept});
                if accept {
                    let original = prediction.clone();
                    *prediction = json!({"text":candidates[0].character,
                        "score":if accepted_score {Some(1.0)} else {None},
                        "score_type":if accepted_score {"binary_acceptance"} else {"unavailable"},
                        "class_ids":[],"per_char_scores":[],
                        "original_model":original,"cpu_fallback":detail});
                    applied += 1;
                } else {
                    prediction["cpu_fallback"] = detail;
                }
            }
        }
        response["cpu_fallback"] = json!({"attempted":attempted,"applied":applied,
            "skipped":skipped,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0});
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    fn image() -> DynamicImage {
        DynamicImage::ImageLuma8(GrayImage::from_fn(48, 48, |x, y| {
            Luma([
                if ((x == 12 || x == 35) && (10..38).contains(&y))
                    || ((y == 10 || y == 37) && (12..36).contains(&x))
                {
                    0
                } else {
                    255
                },
            ])
        }))
    }
    fn fixture() -> Fallback {
        let mut a = normalize(image()).unwrap();
        a.norm = energy(&a);
        let mut b = a.clone();
        b.pixels.rotate_left(12);
        b.rows.rotate_left(12);
        b.norm = energy(&b);
        let dict = Dictionary {
            version: 2,
            provenance: Value::Null,
            templates: vec![
                Template {
                    character: "口".into(),
                    source: "fixture".into(),
                    shape: a,
                },
                Template {
                    character: "土".into(),
                    source: "fixture".into(),
                    shape: b,
                },
            ],
        };
        let descriptors = dict
            .templates
            .iter()
            .map(|t| t.shape.descriptor())
            .collect();
        let blocks = VerificationIndex::new(&dict);
        Fallback {
            dict,
            descriptors,
            blocks,
        }
    }
    #[test]
    fn response_preserves_model_and_does_not_invent_probability_or_ctc_ids() {
        let f = fixture();
        let original = json!({"text":"","score":0.0,"class_ids":[],"timesteps":[0,0]});
        let nonempty = json!({"text":"未","score":0.001});
        for accepted_score in [false, true] {
            let mut r = json!({"predictions":[original.clone(),nonempty.clone()]});
            f.apply(
                &mut r,
                &["a".into(), "b".into()],
                true,
                accepted_score,
                |_| Ok::<_, ()>(image()),
            )
            .unwrap();
            let p = &r["predictions"][0];
            assert_eq!(p["text"], "口");
            assert_eq!(p["original_model"], original);
            assert_eq!(p["class_ids"], json!([]));
            assert_eq!(
                p["score"],
                if accepted_score {
                    json!(1.0)
                } else {
                    Value::Null
                }
            );
            assert_eq!(r["predictions"][1], nonempty);
        }
    }
    #[test]
    fn busy_and_request_limit_abstain_without_decoding_excess_images() {
        let f = fixture();
        let images = vec!["x".into(); 40];
        for available in [false, true] {
            let mut r = json!({"predictions":vec![json!({"text":"","score":0.0});40]});
            let mut decoded = 0;
            f.apply(&mut r, &images, available, false, |_| {
                decoded += 1;
                Ok::<_, ()>(DynamicImage::new_rgb8(8, 8))
            })
            .unwrap();
            assert_eq!(decoded, if available { 32 } else { 0 });
            assert!(r["predictions"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["text"] == ""));
            assert_eq!(r["cpu_fallback"]["skipped"], if available { 8 } else { 40 });
        }
    }
    #[test]
    fn absent_dictionary_is_optional_but_invalid_dictionary_is_an_error() {
        let root =
            std::env::temp_dir().join(format!("glyph-dictionary-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        assert!(Fallback::load_optional(&root.join("missing.json"))
            .unwrap()
            .is_none());
        fs::write(root.join("invalid.json"), b"{}").unwrap();
        assert!(Fallback::load_optional(&root.join("invalid.json")).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
