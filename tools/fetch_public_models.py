#!/usr/bin/env python3
"""Download the pinned PP-OCRv6 medium sources with Python's standard library."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

MODELS = {
    "det": ("PaddlePaddle/PP-OCRv6_medium_det_onnx", "61323801669c338b7891481ec7bac61ce31b576a"),
    "rec": ("PaddlePaddle/PP-OCRv6_medium_rec_onnx", "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/source/source"))
    root = p.parse_args().out_dir
    metadata = {"models": {}}
    for kind, (repo, revision) in MODELS.items():
        target = root / kind
        target.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("README.md", "inference.yml", "inference.json", "inference.onnx"):
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            tmp = target / (name + ".part")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
                tmp.replace(target / name)
            finally:
                tmp.unlink(missing_ok=True)
            with (target / name).open("rb") as f:
                files[name] = hashlib.file_digest(f, "sha256").hexdigest()
        metadata["models"][kind] = {"repo": repo, "revision": revision, "files_sha256": files}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

if __name__ == "__main__":
    main()
