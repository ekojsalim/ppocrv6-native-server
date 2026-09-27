#!/usr/bin/env bash
# Public builder entry point. CUDA_ARCH selects the deployment GPU (default 120).
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# < 1 || $# > 2 )); then
  echo 'usage: tools/build_models.sh OUTPUT_DIRECTORY [PREPARED_SOURCE_DIRECTORY]' >&2
  exit 2
fi
python=${PYTHON:-python3}
source=${2:-artifacts/source}
if (( $# == 1 )); then
  "$python" tools/prepare_model_artifacts.py --root "$source"
fi
bash tools/compiled_models/fetch_cutlass.sh
PYTHON="$python" bash tools/compiled_models/package.sh "$source" "$1"
mkdir -p "$1/licenses"
cp LICENSES/* "$1/licenses/"
# Source cards are separate to avoid overwriting a same-named upstream file.
cp "$source/source/det/README.md" "$1/licenses/detector-model-card.md"
cp "$source/source/rec/README.md" "$1/licenses/recognizer-model-card.md"
"$python" - "$1" "$source" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);p=root/'package.json';meta=json.loads(p.read_text())
source=Path(sys.argv[2]);provenance={}
source_meta=source/'source/metadata.json'
if source_meta.exists():
    data=json.loads(source_meta.read_text())
    for kind, model in data.get('models',{}).items():
        provenance[kind]={k:v for k,v in model.items() if k in ('repo','revision','sha','license')}
for kind in ('det','rec'):
    provenance.setdefault(kind,{})['onnx_sha256']=hashlib.sha256((source/'source'/kind/'inference.onnx').read_bytes()).hexdigest()
f=root/'source-metadata.json';f.write_text(json.dumps(provenance,indent=2)+'\n')
meta['files_sha256']['source-metadata.json']=hashlib.sha256(f.read_bytes()).hexdigest()

for f in (root/'licenses').iterdir():
    meta['files_sha256'][str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
p.write_text(json.dumps(meta,indent=2)+'\n')
PY
