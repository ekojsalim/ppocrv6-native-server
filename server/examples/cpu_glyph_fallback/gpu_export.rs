//! Private experiment interchange; not a server or model-bundle format.
use super::*;
use std::io::{BufWriter, Write};

pub(super) fn export(dict_path: &str, queries_path: &str, output: &str) -> Result<()> {
    let dict: Dictionary = read_json(dict_path)?;
    validate_dictionary(&dict)?;
    let queries: Vec<Query> = read_json(queries_path)?;
    let out = Path::new(output);
    fs::create_dir_all(out)?;
    let file = |name: &str| -> Result<BufWriter<fs::File>> {
        Ok(BufWriter::new(fs::File::create(out.join(name))?))
    };
    let (mut pixels, mut desc, mut coarse, mut fine, mut norms, mut aspects, mut labels) = (
        file("pixels.bin")?,
        file("desc.bin")?,
        file("coarse.bin")?,
        file("fine.bin")?,
        file("norms.bin")?,
        file("aspects.bin")?,
        file("labels.bin")?,
    );
    for t in &dict.templates {
        pixels.write_all(&t.shape.pixels)?;
        desc.write_all(&t.shape.descriptor())?;
        for v in quantized_blocks::<64>(&t.shape, 0, 0, 4) {
            coarse.write_all(&v.to_le_bytes())?;
        }
        for v in quantized_blocks::<256>(&t.shape, 0, 0, 2) {
            fine.write_all(&v.to_le_bytes())?;
        }
        norms.write_all(&energy(&t.shape).to_le_bytes())?;
        aspects.write_all(&t.shape.aspect.to_le_bytes())?;
        labels.write_all(&(t.character.chars().next().unwrap() as u32).to_le_bytes())?;
    }
    // Propagate buffered write errors instead of relying on Drop.
    for f in [
        &mut pixels,
        &mut desc,
        &mut coarse,
        &mut fine,
        &mut norms,
        &mut aspects,
        &mut labels,
    ] {
        f.flush()?;
    }
    let mut packed = file("queries.bin")?;
    let mut meta = Vec::new();
    for q in queries {
        let started = Instant::now();
        let image = image::open(&q.path)?;
        let result = normalize(image);
        let decode_normalize_ms = started.elapsed().as_secs_f64() * 1000.0;
        let valid = result.is_ok();
        let mut shape = result.unwrap_or(Shape {
            rows: [0; SIDE],
            aspect: 1.0,
            pixels: vec![0; SIDE * SIDE],
            norm: 1.0,
        });
        let prepared = Instant::now();
        shape.norm = energy(&shape);
        let descriptor = shape.descriptor();
        let coarse: [[u16; 64]; 9] = std::array::from_fn(|i| {
            quantized_blocks(&shape, i as i32 % 3 - 1, i as i32 / 3 - 1, 4)
        });
        let fine: [[u16; 256]; 9] = std::array::from_fn(|i| {
            quantized_blocks(&shape, i as i32 % 3 - 1, i as i32 / 3 - 1, 2)
        });
        let prepare_ms = prepared.elapsed().as_secs_f64() * 1000.0;
        packed.write_all(&u32::from(valid).to_le_bytes())?;
        packed.write_all(&shape.norm.to_le_bytes())?;
        packed.write_all(&shape.aspect.to_le_bytes())?;
        packed.write_all(&shape.pixels)?;
        packed.write_all(&descriptor)?;
        for row in coarse {
            for v in row {
                packed.write_all(&v.to_le_bytes())?;
            }
        }
        for row in fine {
            for v in row {
                packed.write_all(&v.to_le_bytes())?;
            }
        }
        meta.push(json!({"id":q.id,"expected":q.expected,"valid":valid,"original_model":q.model,"decode_normalize_ms":decode_normalize_ms,"prepare_ms":prepare_ms}));
    }
    packed.flush()?;
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(
            &json!({"version":1,"templates":dict.templates.len(),"queries":meta,"record_bytes":6860}),
        )?,
    )?;
    println!(
        "{}",
        json!({"templates":dict.templates.len(),"queries":meta.len(),"directory":out})
    );
    Ok(())
}
