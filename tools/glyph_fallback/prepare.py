#!/usr/bin/env python3
"""Generate private, reproducible CPU glyph experiment assets (offline only).

Requires Pillow and fonttools. Never copies fonts or fixtures into source paths.
Noto TTC inputs use face 2 (SC); a different collection needs an explicit edit.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
from fontTools.ttLib import TTFont


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font_asset(path, index, weight, size):
    face = TTFont(path, fontNumber=index, lazy=True)
    supported = set(face.getBestCmap())
    face.close()
    font = ImageFont.truetype(str(path), size, index=index)
    if weight is not None:
        font.set_variation_by_axes([weight])
    return font, supported


def render(font, char, side=64):
    image = Image.new("L", (side, side), 255)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    draw.text(((side - right + left) // 2 - left, (side - bottom + top) // 2 - top),
              char, fill=0, font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sans", "serif", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("heldout", "fixtures", "predictions", "vocabulary"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--templates-only", action="store_true",
                        help="Build dictionary inputs without evaluation fixtures.")
    parser.add_argument("--bar-fixture", type=Path)
    args = parser.parse_args()
    if not args.templates_only and not all((args.heldout, args.fixtures, args.predictions, args.vocabulary)):
        parser.error("evaluation requires --heldout, --fixtures, --predictions and --vocabulary")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    template_dir, query_dir = out / "templates", out / "queries"
    template_dir.mkdir(exist_ok=True)
    query_dir.mkdir(exist_ok=True)
    provenance = {"pillow": PIL.__version__, "fonts": [],
                  "character_range": "U+4E00..U+9FFF", "seed": 20260927,
                  "normalization": "Rust cpu_glyph_fallback v2",
                  "note": "Font asset hashes identify data, not font redistribution permission."}
    templates = []
    for name, path, weight in [("noto_sans_sc_100", args.sans, 100), ("noto_serif_sc_400", args.serif, 400), ("noto_sans_sc_400", args.sans, 400)]:
        font, supported = font_asset(path, 2, weight, 48)
        provenance["fonts"].append({"source": name, "path": str(path.resolve()),
                                     "sha256": digest(path), "index": 2, "weight": weight,
                                     "size": 48, "role": "dictionary"})
        for code in range(0x4E00, 0xA000):
            if code not in supported:
                continue
            path = template_dir / f"{name}_{code:04X}.png"
            render(font, chr(code)).save(path)
            templates.append({"character": chr(code), "source": name, "path": str(path)})
        print(f"prepared {name}", flush=True)
    (out / "templates.json").write_text(json.dumps({"provenance": provenance, "templates": templates}, ensure_ascii=False))
    if args.templates_only:
        return
    files = sorted(args.fixtures.glob("U+*.png"))
    response = json.loads(args.predictions.read_text())
    predictions = response["predictions"]
    if len(files) != len(predictions):
        raise ValueError("Fixture ordering/count must match saved sorted HTTP request")
    vocabulary = set(args.vocabulary.read_text().splitlines())
    queries = []
    for path, prediction in zip(files, predictions):
        char = chr(int(path.stem[2:], 16))
        queries.append({"id": path.stem, "path": str(path.resolve()), "group": "historical_synthetic",
                        "expected": char, "model": prediction})
    font, supported = font_asset(args.heldout, 0, None, 39)
    provenance["heldout_font"] = {"path": str(args.heldout.resolve()), "sha256": digest(args.heldout),
                                  "size": 39, "canvas": 57, "index": 0}

    def add(image, name, group, expected=None):
        path = query_dir / (name + ".png")
        image.save(path)
        queries.append({"id": name, "path": str(path), "group": group, "expected": expected, "model": None})

    codes = sorted(set(range(0x4E00, 0x5200)) | set(map(ord, "未末土士口囗日曰己已巳人入八距")))
    for code in codes:
        if code in supported:
            add(render(font, chr(code), 57), f"droid_{code:04X}", "heldout_font", chr(code))
    for char in "未末土士口囗日曰己已巳人入八":
        base = render(font, char, 57)
        add(ImageOps.invert(base), f"inverse_{ord(char):04X}", "heldout_polarity", char)
        # Missing ink is an ambiguity stress test, not a claim of illegibility.
        damaged = base.copy()
        box = ImageOps.invert(base).getbbox()
        y = (box[1] + box[3]) // 2
        ImageDraw.Draw(damaged).rectangle((box[0], y-1, box[2], y+1), fill=255)
        add(damaged, f"missing_{ord(char):04X}", "damaged")
        clipped = base.crop((box[0]+2, 0, 57, 57))
        add(clipped, f"clipped_{ord(char):04X}", "clipped")
    for i, char in enumerate(".,:;!?-_=+|/\\()[]{}<>…·。、「」一丨丿丶〇"):
        add(render(font, char, 57), f"punctuation_{i}", "punctuation_and_rules")
    for value in (0, 64, 127, 192, 255):
        add(Image.new("L", (48, 48), value), f"blank_{value}", "blank")
    rng = random.Random(20260927)
    for i in range(100):
        im = Image.new("L", (48,48), 255)
        draw = ImageDraw.Draw(im)
        for _ in range(10 + i * 3):
            draw.point((rng.randrange(2,46),rng.randrange(2,46)), fill=rng.randrange(128))
        add(im, f"noise_{i}", "noise")
    if args.bar_fixture:
        # This particular historical fixture was visually inspected: a bar,
        # despite a filename claiming U+8DDD. Do not generalize filename truth.
        add(Image.open(args.bar_fixture), "historical_mislabeled_bar", "historical_negative")
    (out / "queries.json").write_text(json.dumps(queries, ensure_ascii=False))
    (out / "evaluation_provenance.json").write_text(json.dumps({
        "generation": provenance, "prediction_sha256": digest(args.predictions),
        "prediction_character_policy": response.get("character_policy"),
        "prediction_score_mode": response.get("score_mode"),
        "vocabulary_sha256": digest(args.vocabulary),
        "fixtures": [{"name": p.name, "sha256": digest(p)} for p in files],
        "empty_failures": [{"id": q["id"], "expected": q["expected"],
                            "in_vocabulary": q["expected"] in vocabulary} for q in queries
                           if q["model"] is not None and not q["model"]["text"]],
    }, ensure_ascii=False, indent=2))
    print(f"{len(templates)} templates; {len(queries)} queries", flush=True)


if __name__ == "__main__":
    main()
