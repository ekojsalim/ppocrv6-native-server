//! Offline evaluation and asset generation; uses the serving matcher.
mod gpu_export;
use anyhow::{bail, Context, Result};
use ppocrv6_native_server::glyph_matcher::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};
#[derive(Deserialize)]
struct TemplateInput {
    character: String,
    source: String,
    path: String,
}
#[derive(Deserialize)]
struct BuildInput {
    provenance: Value,
    templates: Vec<TemplateInput>,
}
#[derive(Deserialize)]
struct Query {
    id: String,
    path: String,
    group: String,
    // Null means it is unsafe to infer a character (negative or damaged crop).
    expected: Option<String>,
    // Null means recognition was not run. Evaluate forced-empty behavior only.
    model: Option<Value>,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("export-gpu") if args.len()==5 => gpu_export::export(&args[2],&args[3],&args[4])?,
        Some("build") if args.len() == 4 => {
            let input: BuildInput = read_json(&args[2])?;
            let mut templates = Vec::new();
            let mut skipped = BTreeMap::<String, usize>::new();
            for t in input.templates {
                let image = image::open(&t.path).with_context(|| t.path.clone())?;
                match normalize(image) {
                    Ok(shape) => templates.push(Template { character: t.character, source: t.source, shape }),
                    Err(reason) => *skipped.entry(reason.into()).or_default() += 1,
                }
            }
            let dict = Dictionary { version: 2, provenance: input.provenance, templates };
            validate_dictionary(&dict)?;
            fs::write(&args[3], serde_json::to_vec(&dict)?)?;
            println!("{}",json!({"templates":dict.templates.len(),"skipped":skipped,"asset_bytes":fs::metadata(&args[3])?.len()}));
        }
        Some("evaluate") if args.len() == 5 || (args.len() == 6 && args[5] == "--exhaustive") => {
            let started = Instant::now();
            let mut dict: Dictionary = read_json(&args[2])?;
            validate_dictionary(&dict)?;
            for template in &mut dict.templates { template.shape.norm=energy(&template.shape); }
            let descriptors: Vec<_> = dict.templates.iter().map(|t| t.shape.descriptor()).collect();
            let blocks=VerificationIndex::new(&dict);
            let load_ms = started.elapsed().as_secs_f64()*1000.0;
            let queries: Vec<Query> = read_json(&args[3])?;
            let mut results = Vec::new();
            for q in queries {
                let started = Instant::now();
                let shape = normalize(image::open(Path::new(&q.path)).with_context(|| q.path.clone())?);
                let normalize_ms = started.elapsed().as_secs_f64()*1000.0;
                let started = Instant::now();
                let (candidates, profile, rejection) = match shape {
                    Ok(mut s) => { s.norm=energy(&s); let (c,p)=rank_profiled(&s, &dict, &descriptors, args.len() == 6, &blocks); (c,p,None) },
                    Err(reason) => (vec![],MatchStats::default(), Some(reason)),
                };
                let match_ms = started.elapsed().as_secs_f64()*1000.0;
                let accept = accepted(&candidates);
                let proposed = if accept { Some(candidates[0].character.as_str()) } else { None };
                let (applied, final_text) = replay(q.model.as_ref(), &candidates)?;
                results.push(json!({"id":q.id,"group":q.group,"expected":q.expected,"original_model":q.model,
                    "text":final_text,"applied":applied,"proposed_if_empty":proposed,
                    "provenance":if applied {"cpu_template_v2"} else {"unchanged"},
                    "similarity_type":"aligned_ink_cosine_minus_aspect_penalty_not_probability",
                    "profile":profile,"candidates":candidates,"acceptance_margin_verified":accept,"rejection":rejection,"decode_normalize_ms":normalize_ms,"match_ms":match_ms}));
            }
            let output = json!({"schema_version":2,"dictionary_provenance":dict.provenance,
                "templates":dict.templates.len(),"load_ms":load_ms,
                "template_struct_bytes":std::mem::size_of::<Template>() * dict.templates.len(),
                "descriptor_bytes":descriptors.len()*64,"bound_bytes":blocks.bytes(),
                "acceptance_competitors":"full_dictionary_verified_before_acceptance",
                "shortlist":if args.len()==6 {dict.templates.len()} else {SHORTLIST},
                "min_similarity":MIN_SIMILARITY,"min_different_character_margin":MIN_MARGIN,
                "results":results});
            fs::write(&args[4], serde_json::to_vec_pretty(&output)?)?;
        }
        _ => bail!("usage: cpu_glyph_fallback build TEMPLATES.json DICTIONARY.json | evaluate DICTIONARY.json QUERIES.json OUTPUT.json [--exhaustive] | export-gpu DICTIONARY.json QUERIES.json DIRECTORY"),
    }
    Ok(())
}
