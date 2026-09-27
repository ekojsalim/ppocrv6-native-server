//! CPU glyph template matching with conservative global competitor verification.
use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, DynamicImage, GrayImage, Luma};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::{collections::BTreeMap, fs, time::Instant};

pub const SIDE: usize = 32;
pub const SHORTLIST: usize = 256;
// Exploratory operating point, not calibrated production confidence.
pub const MIN_SIMILARITY: f32 = 0.93;
pub const MIN_MARGIN: f32 = 0.04;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shape {
    pub rows: [u32; SIDE],
    pub aspect: f32,
    #[serde(with = "pixel_bytes")]
    pub pixels: Vec<u8>,
    #[serde(skip)]
    pub norm: f32,
}

mod pixel_bytes {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(pixels: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&STANDARD.encode(pixels))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        STANDARD
            .decode(String::deserialize(d)?)
            .map_err(serde::de::Error::custom)
    }
}
impl Shape {
    pub fn descriptor(&self) -> [u8; 64] {
        let mut d = [0; 64];
        for y in 0..8 {
            for x in 0..8 {
                d[y * 8 + x] = (0..4)
                    .map(|dy| ((self.rows[y * 4 + dy] >> (x * 4)) & 15).count_ones() as u8)
                    .sum();
            }
        }
        d
    }
}

// Normalize the original crop, not the padded native NCHW tensor. Preserve
// aspect ratio; deliberately abstain on rules, low contrast and clipped ink.
pub fn normalize(image: DynamicImage) -> std::result::Result<Shape, &'static str> {
    let rgba = image.to_rgba8();
    let (w, h) = rgba.dimensions();
    if w < 5 || h < 5 || u64::from(w) * u64::from(h) > 1_048_576 {
        return Err("dimensions");
    }
    let gray = GrayImage::from_fn(w, h, |x, y| {
        let p = rgba.get_pixel(x, y);
        let rgb = (u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 3;
        Luma([((rgb * u32::from(p[3]) + 255 * (255 - u32::from(p[3])) + 127) / 255) as u8])
    });
    let mut border = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                border.push(gray.get_pixel(x, y)[0]);
            }
        }
    }
    border.sort_unstable();
    let bg = border[border.len() / 2];
    let light_background = bg >= 128;
    let foreground = |p: u8| {
        if light_background {
            bg.saturating_sub(p)
        } else {
            p.saturating_sub(bg)
        }
    };
    let max_contrast = gray.pixels().map(|p| foreground(p[0])).max().unwrap_or(0);
    if max_contrast < 64 {
        return Err("blank_or_low_contrast");
    }
    let threshold = (max_contrast / 3).max(32);
    let (mut x0, mut y0, mut x1, mut y1, mut count) = (w, h, 0, 0, 0u32);
    for (x, y, p) in gray.enumerate_pixels() {
        if foreground(p[0]) >= threshold {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            count += 1;
        }
    }
    if count < 8 {
        return Err("too_little_ink");
    }
    if x0 == 0 || y0 == 0 || x1 == w - 1 || y1 == h - 1 {
        return Err("border_ink");
    }
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let aspect = bw as f32 / bh as f32;
    let density = count as f32 / (bw * bh) as f32;
    if !(0.25..=4.0).contains(&aspect) || bh * 5 < h || bw * 7 < w || bw < 3 || bh < 3 {
        return Err("rule_or_small_mark");
    }
    if !(0.04..=0.80).contains(&density) {
        return Err("ink_density");
    }
    // Keep antialiased stroke coverage through resizing. Thresholding before
    // shrinking loses thin strokes and makes rasterization differences dominant.
    let crop = GrayImage::from_fn(bw, bh, |x, y| {
        Luma([
            ((u32::from(foreground(gray.get_pixel(x + x0, y + y0)[0])) * 255)
                / u32::from(max_contrast))
            .min(255) as u8,
        ])
    });
    let scale = 28.0 / bw.max(bh) as f32;
    let nw = (bw as f32 * scale).round().max(1.0) as u32;
    let nh = (bh as f32 * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(&crop, nw, nh, FilterType::Triangle);
    let (ox, oy) = ((32 - nw) / 2, (32 - nh) / 2);
    let mut rows = [0; SIDE];
    let mut pixels = vec![0; SIDE * SIDE];
    for (x, y, p) in resized.enumerate_pixels() {
        pixels[(y + oy) as usize * SIDE + (x + ox) as usize] = p[0];
        if p[0] >= 128 {
            rows[(y + oy) as usize] |= 1 << (x + ox);
        }
    }
    if rows.iter().all(|r| *r == 0) {
        return Err("empty_after_resize");
    }
    Ok(Shape {
        rows,
        aspect,
        pixels,
        norm: 0.0,
    })
}

#[derive(Serialize, Deserialize)]
pub struct Template {
    pub character: String,
    pub source: String,
    pub shape: Shape,
}
#[derive(Serialize, Deserialize)]
pub struct Dictionary {
    pub version: u32,
    pub provenance: Value,
    pub templates: Vec<Template>,
}
#[derive(Debug, Serialize)]
pub struct Candidate {
    pub character: String,
    pub source: String,
    pub similarity: f32,
}

pub fn energy(shape: &Shape) -> f32 {
    if shape.norm > 0.0 {
        return shape.norm;
    }
    shape
        .pixels
        .iter()
        .map(|p| f32::from(*p).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn aspect_penalty(a: &Shape, b: &Shape) -> f32 {
    0.1 * (a.aspect / b.aspect).ln().abs().min(1.0)
}

pub fn similarity(a: &Shape, b: &Shape) -> f32 {
    similarity_mask(a, b, 0x1ff)
}

pub fn similarity_mask(a: &Shape, b: &Shape, mask: u16) -> f32 {
    let denominator = energy(a) * energy(b);
    let mut best = 0.0f32;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if mask & (1 << ((dy + 1) * 3 + dx + 1)) == 0 {
                continue;
            }
            let mut dot = 0u32;
            let (x0, x1) = ((-dx).max(0) as usize, (32 - dx).min(32) as usize);
            let (y0, y1) = ((-dy).max(0) as usize, (32 - dy).min(32) as usize);
            for y in y0..y1 {
                let by = (y as i32 + dy) as usize;
                let bx = (x0 as i32 + dx) as usize;
                let left = &a.pixels[y * SIDE + x0..y * SIDE + x1];
                let right = &b.pixels[by * SIDE + bx..by * SIDE + bx + (x1 - x0)];
                dot += left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| u32::from(*a) * u32::from(*b))
                    .sum::<u32>();
            }
            best = best.max(dot as f32 / denominator);
        }
    }
    (best - aspect_penalty(a, b)).max(0.0)
}

// Quantizing upward preserves the Cauchy-Schwarz upper bound. Q is small
// enough for signed SSE2 multiply-add; normalized blocks keep the TOTAL dot
// product below 2^31 (including upward rounding), so lane accumulation is safe.
pub const Q: f32 = 16384.0;
pub struct VerificationIndex {
    pub coarse: Vec<[u16; 64]>,
    pub fine: Vec<[u16; 256]>,
}
impl VerificationIndex {
    pub fn new(dict: &Dictionary) -> Self {
        Self {
            coarse: dict
                .templates
                .iter()
                .map(|t| quantized_blocks(&t.shape, 0, 0, 4))
                .collect(),
            fine: dict
                .templates
                .iter()
                .map(|t| quantized_blocks(&t.shape, 0, 0, 2))
                .collect(),
        }
    }
    pub fn bytes(&self) -> usize {
        self.coarse.len() * 128 + self.fine.len() * 512
    }
}
pub fn quantized_blocks<const N: usize>(shape: &Shape, dx: i32, dy: i32, cell: usize) -> [u16; N] {
    assert_eq!(N, (SIDE / cell) * (SIDE / cell));
    let mut blocks = [0u32; N];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let (sx, sy) = (x as i32 + dx, y as i32 + dy);
            if (0..32).contains(&sx) && (0..32).contains(&sy) {
                let p = u32::from(shape.pixels[sy as usize * SIDE + sx as usize]);
                blocks[(y / cell) * (SIDE / cell) + x / cell] += p * p;
            }
        }
    }
    let norm = energy(shape);
    blocks.map(|v| ((v as f32).sqrt() / norm * Q).ceil() as u16)
}

#[inline]
pub fn quantized_dot<const N: usize>(a: &[u16; N], b: &[u16; N]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        assert_eq!(N % 8, 0);
        // SSE2 is guaranteed on x86_64. Arrays contain N initialized u16s;
        // callers use N=64 or 256 and normalized, upward-quantized energies.
        unsafe {
            let mut sum = _mm_setzero_si128();
            for i in (0..N).step_by(8) {
                let left = _mm_loadu_si128(a.as_ptr().add(i).cast());
                let right = _mm_loadu_si128(b.as_ptr().add(i).cast());
                sum = _mm_add_epi32(sum, _mm_madd_epi16(left, right));
            }
            sum = _mm_add_epi32(sum, _mm_srli_si128::<8>(sum));
            sum = _mm_add_epi32(sum, _mm_srli_si128::<4>(sum));
            _mm_cvtsi128_si32(sum) as u32
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.iter()
            .zip(b)
            .map(|(a, b)| u32::from(*a) * u32::from(*b))
            .sum()
    }
}

#[derive(Default, Serialize)]
pub struct MatchStats {
    pub retrieval_ms: f64,
    pub shortlist_ms: f64,
    pub verification_ms: f64,
    pub verification_pixel_comparisons: usize,
    pub verification_pixel_alignments: usize,
}

#[cfg(test)]
pub fn rank(
    shape: &Shape,
    dict: &Dictionary,
    descriptors: &[[u8; 64]],
    exhaustive: bool,
    blocks: &VerificationIndex,
) -> Vec<Candidate> {
    rank_profiled(shape, dict, descriptors, exhaustive, blocks).0
}

pub fn rank_profiled(
    shape: &Shape,
    dict: &Dictionary,
    descriptors: &[[u8; 64]],
    exhaustive: bool,
    blocks: &VerificationIndex,
) -> (Vec<Candidate>, MatchStats) {
    let mut stats = MatchStats::default();
    let started = Instant::now();
    let d = shape.descriptor();
    let mut distances: Vec<(u32, usize)> = descriptors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let distance = d
                .iter()
                .zip(t)
                .map(|(a, b)| {
                    let v = i32::from(*a) - i32::from(*b);
                    (v * v) as u32
                })
                .sum();
            (distance, i)
        })
        .collect();
    if !exhaustive && distances.len() > SHORTLIST {
        distances.select_nth_unstable(SHORTLIST);
        distances.truncate(SHORTLIST);
    }
    stats.retrieval_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut by_character: BTreeMap<&str, Candidate> = BTreeMap::new();
    for (_, i) in distances {
        let t = &dict.templates[i];
        let score = similarity(shape, &t.shape);
        let entry = by_character
            .entry(&t.character)
            .or_insert_with(|| Candidate {
                character: t.character.clone(),
                source: t.source.clone(),
                similarity: -1.0,
            });
        if score > entry.similarity {
            entry.similarity = score;
            entry.source.clone_from(&t.source);
        }
    }
    let mut ranked: Vec<_> = by_character.into_values().collect();
    ranked.sort_by(|a, b| {
        b.similarity
            .total_cmp(&a.similarity)
            .then_with(|| a.character.cmp(&b.character))
    });
    stats.shortlist_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    // A shortlist may miss a competing character. Before emitting a proposal,
    // verify every template capable of violating the acceptance margin. The
    // bound avoids expensive pixel comparisons for provably weaker templates.
    if !exhaustive && accepted(&ranked) {
        let floor = ranked[0].similarity - MIN_MARGIN;
        let offsets: Vec<_> = (-1..=1)
            .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
            .collect();
        let coarse: Vec<[u16; 64]> = offsets
            .iter()
            .map(|&(dx, dy)| quantized_blocks(shape, dx, dy, 4))
            .collect();
        let fine: Vec<[u16; 256]> = offsets
            .iter()
            .map(|&(dx, dy)| quantized_blocks(shape, dx, dy, 2))
            .collect();
        let mut verified: BTreeMap<String, Candidate> = ranked
            .into_iter()
            .map(|c| (c.character.clone(), c))
            .collect();
        for (i, t) in dict.templates.iter().enumerate() {
            // Round the comparison threshold DOWN, and retain numerical slack.
            let required = ((floor + aspect_penalty(shape, &t.shape) - 0.00001) as f64
                * f64::from(Q * Q))
            .max(0.0)
            .floor() as u32;
            let mut mask = 0u16;
            for j in 0..9 {
                if quantized_dot(&coarse[j], &blocks.coarse[i]) >= required
                    && quantized_dot(&fine[j], &blocks.fine[i]) >= required
                {
                    // The bound shifts the QUERY; pixel comparison shifts the
                    // TEMPLATE, so their alignment signs are opposite.
                    mask |= 1 << (8 - j);
                }
            }
            if mask == 0 {
                continue;
            }
            stats.verification_pixel_comparisons += 1;
            stats.verification_pixel_alignments += mask.count_ones() as usize;
            let score = similarity_mask(shape, &t.shape, mask);
            if score < floor {
                continue;
            }
            let c = verified
                .entry(t.character.clone())
                .or_insert_with(|| Candidate {
                    character: t.character.clone(),
                    source: t.source.clone(),
                    similarity: -1.0,
                });
            if score > c.similarity {
                c.similarity = score;
                c.source.clone_from(&t.source);
            }
        }
        ranked = verified.into_values().collect();
        ranked.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.character.cmp(&b.character))
        });
    }
    stats.verification_ms = started.elapsed().as_secs_f64() * 1000.0;
    ranked.truncate(3);
    (ranked, stats)
}

pub fn accepted(candidates: &[Candidate]) -> bool {
    candidates.len() >= 2
        && candidates[0].similarity >= MIN_SIMILARITY
        && candidates[0].similarity - candidates[1].similarity >= MIN_MARGIN
}

// A null model is an offline matching-only query, never an actual recovery.
pub fn replay(
    original: Option<&Value>,
    candidates: &[Candidate],
) -> Result<(bool, Option<String>)> {
    let original_text = match original {
        Some(model) => Some(
            model
                .get("text")
                .and_then(Value::as_str)
                .context("model.text must be a string")?,
        ),
        None => None,
    };
    let applied = original_text == Some("") && accepted(candidates);
    Ok((
        applied,
        if applied {
            Some(candidates[0].character.clone())
        } else {
            original_text.map(str::to_owned)
        },
    ))
}

pub fn validate_dictionary(dict: &Dictionary) -> Result<()> {
    if dict.version != 2 || dict.templates.is_empty() {
        bail!("invalid dictionary version or empty dictionary");
    }
    for t in &dict.templates {
        if t.character.chars().count() != 1
            || !t.shape.aspect.is_finite()
            || t.shape.aspect <= 0.0
            || t.shape.rows.iter().all(|r| *r == 0)
            || t.shape.pixels.len() != SIDE * SIDE
            || energy(&t.shape) == 0.0
        {
            bail!("invalid template");
        }
    }
    Ok(())
}

pub fn read_json<T: serde::de::DeserializeOwned>(p: &str) -> Result<T> {
    serde_json::from_slice(&fs::read(p).with_context(|| p.to_owned())?)
        .with_context(|| p.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn glyph() -> DynamicImage {
        DynamicImage::ImageLuma8(GrayImage::from_fn(48, 48, |x, y| {
            Luma([
                if (x == 12 || x == 35) && (10..38).contains(&y)
                    || (y == 10 || y == 37) && (12..36).contains(&x)
                {
                    0
                } else {
                    255
                },
            ])
        }))
    }
    #[test]
    fn polarity_and_transparent_background() {
        let s = normalize(glyph()).unwrap();
        let mut inverted = glyph();
        inverted.invert();
        assert_eq!(s.rows, normalize(inverted).unwrap().rows);
        let gray = glyph().to_luma8();
        let rgba = image::RgbaImage::from_fn(48, 48, |x, y| {
            image::Rgba([0, 0, 0, 255 - gray.get_pixel(x, y)[0]])
        });
        assert_eq!(
            s.rows,
            normalize(DynamicImage::ImageRgba8(rgba)).unwrap().rows
        );
    }
    #[test]
    fn rejects_blank_rule_and_clipping() {
        assert!(normalize(DynamicImage::ImageLuma8(GrayImage::from_pixel(
            48,
            48,
            Luma([255])
        )))
        .is_err());
        let rule = GrayImage::from_fn(48, 48, |x, y| {
            Luma([if (5..43).contains(&x) && y == 24 {
                0
            } else {
                255
            }])
        });
        assert!(normalize(DynamicImage::ImageLuma8(rule)).is_err());
        let clipped = GrayImage::from_fn(48, 48, |x, y| {
            Luma([if x < 20 && (5..40).contains(&y) {
                0
            } else {
                255
            }])
        });
        assert!(normalize(DynamicImage::ImageLuma8(clipped)).is_err());
    }
    #[test]
    fn quantized_bounds_and_simd_cover_each_alignment() {
        fn check<const N: usize>(a: &Shape, b: &Shape, cell: usize) {
            let template = quantized_blocks::<N>(b, 0, 0, cell);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let query = quantized_blocks::<N>(a, dx, dy, cell);
                    let scalar: u64 = query
                        .iter()
                        .zip(template)
                        .map(|(a, b)| u64::from(*a) * u64::from(b))
                        .sum();
                    assert!(scalar < i32::MAX as u64);
                    let dot = quantized_dot(&query, &template);
                    assert_eq!(u64::from(dot), scalar);
                    let opposite = (1 - dy) * 3 + (1 - dx);
                    let exact = similarity_mask(a, b, 1 << opposite);
                    assert!(dot as f32 / (Q * Q) + 0.00001 >= exact);
                }
            }
        }
        let mut seed = 7u32;
        let mut random_shape = || {
            let mut pixels = vec![0; SIDE * SIDE];
            for p in &mut pixels {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *p = (seed & 255) as u8;
            }
            Shape {
                rows: [1; SIDE],
                aspect: 1.0,
                pixels,
                norm: 0.0,
            }
        };
        for _ in 0..32 {
            let a = random_shape();
            let b = random_shape();
            check::<64>(&a, &b, 4);
            check::<256>(&a, &b, 2);
            check::<256>(&a, &a, 2);
        }
        // Concentrated energy stresses signed SIMD lanes and rounding. Pixel
        // offsets cross block boundaries and catch reversed shift masks.
        for &(x, y) in &[(0, 0), (3, 3), (15, 16), (31, 31)] {
            let mut a = Shape {
                rows: [1; SIDE],
                aspect: 1.0,
                pixels: vec![0; SIDE * SIDE],
                norm: 0.0,
            };
            a.pixels[y * SIDE + x] = 255;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (bx, by) = (x as i32 + dx, y as i32 + dy);
                    if !(0..32).contains(&bx) || !(0..32).contains(&by) {
                        continue;
                    }
                    let mut b = a.clone();
                    b.pixels.fill(0);
                    b.pixels[by as usize * SIDE + bx as usize] = 255;
                    check::<64>(&a, &b, 4);
                    check::<256>(&a, &b, 2);
                }
            }
        }
    }

    #[test]
    fn global_verification_finds_rival_outside_shortlist() {
        let shape = normalize(glyph()).unwrap();
        let mut other = shape.clone();
        other.pixels.rotate_left(12);
        other.rows.rotate_left(12);
        let mut templates = Vec::new();
        for _ in 0..SHORTLIST - 1 {
            templates.push(Template {
                character: "口".into(),
                source: "a".into(),
                shape: shape.clone(),
            });
        }
        templates.push(Template {
            character: "土".into(),
            source: "b".into(),
            shape: other,
        });
        templates.push(Template {
            character: "囗".into(),
            source: "hidden".into(),
            shape: shape.clone(),
        });
        let dict = Dictionary {
            version: 2,
            provenance: Value::Null,
            templates,
        };
        // Simulate a retrieval miss independently of the pixel comparison.
        let mut descriptors = vec![shape.descriptor(); dict.templates.len()];
        *descriptors.last_mut().unwrap() = [255; 64];
        let blocks = VerificationIndex::new(&dict);
        let ranked = rank(&shape, &dict, &descriptors, false, &blocks);
        assert_eq!(ranked[0].similarity, ranked[1].similarity);
        assert!(!accepted(&ranked));
        let full = rank(&shape, &dict, &descriptors, true, &blocks);
        assert_eq!(ranked[0].character, full[0].character);
        assert_eq!(ranked[1].character, full[1].character);
    }

    #[test]
    fn only_empty_model_results_can_be_replaced() {
        let candidates = vec![
            Candidate {
                character: "末".into(),
                source: "a".into(),
                similarity: 0.99,
            },
            Candidate {
                character: "未".into(),
                source: "b".into(),
                similarity: 0.80,
            },
        ];
        let original = json!({"text":"未", "score":0.001});
        assert_eq!(
            replay(Some(&original), &candidates).unwrap(),
            (false, Some("未".into()))
        );
        assert_eq!(
            replay(Some(&json!({"text":"", "score":0.0})), &candidates).unwrap(),
            (true, Some("末".into()))
        );
        assert_eq!(replay(None, &candidates).unwrap(), (false, None));
        assert!(replay(Some(&json!({"score":1.0})), &candidates).is_err());
    }

    #[test]
    fn rival_is_a_different_character_and_collision_abstains() {
        let shape = normalize(glyph()).unwrap();
        let dict = Dictionary {
            version: 2,
            provenance: Value::Null,
            templates: vec![
                Template {
                    character: "口".into(),
                    source: "a".into(),
                    shape: shape.clone(),
                },
                Template {
                    character: "口".into(),
                    source: "b".into(),
                    shape: shape.clone(),
                },
                Template {
                    character: "囗".into(),
                    source: "c".into(),
                    shape: shape.clone(),
                },
            ],
        };
        let descriptors = dict
            .templates
            .iter()
            .map(|t| t.shape.descriptor())
            .collect::<Vec<_>>();
        let ranked = rank(
            &shape,
            &dict,
            &descriptors,
            false,
            &VerificationIndex::new(&dict),
        );
        assert_eq!(ranked.len(), 2);
        assert_ne!(ranked[0].character, ranked[1].character);
        assert!(!accepted(&ranked));
        assert!(!accepted(&ranked[..1]));
    }
}
