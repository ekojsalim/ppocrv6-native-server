#!/usr/bin/env bash
# Prepare the supported deployment catalogs. ONNX is used only at build time.
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: package.sh SOURCE_MODEL_DIRECTORY OUTPUT_DIRECTORY' >&2;exit 2;fi
model=$(realpath "$1");out=$(realpath -m "$2");python=${PYTHON:-python3}
if [[ -e "$out/package.json" ]]; then echo "Use a fresh output directory; this package is already complete" >&2;exit 2;fi
mkdir -p "$out/recognition" "$out/detection" "$out/classifier"
: > "$out/recognition/models.tsv"
for spec in '1 80' '8 80' '32 80' '128 80' '8 384' '1 3200' '8 3200'; do
 read -r batch width <<< "$spec";name=b${batch}w${width};d="$out/recognition/$name";mkdir -p "$d"
 "$python" tools/compiled_models/export.py --model "$model/derived/rec-hidden.onnx" --out "$d" --batch "$batch" --width "$width" --kind recognition > "$d/export.log"
 bash tools/compiled_models/build.sh recognition "$d" > "$d/build.log" 2>&1
 printf '%s/model.so\t%s/weights.f32\n' "$name" "$name" >> "$out/recognition/models.tsv"
 echo "Built recognition $name"
done
: > "$out/detection/models.tsv"
for spec in '256 256' '640 640' '1280 992' '1280 1280'; do
 read -r height width <<< "$spec";name=h${height}w${width};d="$out/detection/$name";mkdir -p "$d"
 "$python" tools/compiled_models/export.py --model "$model/source/det/inference.onnx" --out "$d" --height "$height" --width "$width" --kind detection > "$d/export.log"
 bash tools/compiled_models/build.sh detection "$d" > "$d/build.log" 2>&1
 printf '%s/model.so\t%s/weights.f32\n' "$name" "$name" >> "$out/detection/models.tsv"
 echo "Built detection $name"
done
# Shapes share weights. Keep one verified copy per network in the deployment.
for kind in recognition detection; do
 first=1
 while read -r library weights; do
  if [[ $first == 1 ]];then cp "$out/$kind/$weights" "$out/$kind/weights.f32";first=0;fi
  cmp "$out/$kind/$weights" "$out/$kind/weights.f32"
 done < "$out/$kind/models.tsv"
 while read -r library weights; do
  rm "$out/$kind/$weights"
  printf '%s\tweights.f32\n' "$library"
  (cd "$out/$kind/${library%/*}";sha256sum model.so ../weights.f32 > SHA256SUMS)
 done < "$out/$kind/models.tsv" > "$out/$kind/models.tsv.new"
 mv "$out/$kind/models.tsv.new" "$out/$kind/models.tsv"
done
cp "$model"/classifier/{weight.fp16.bin,bias.fp16.bin,characters.txt,metadata.json} "$out/classifier/"
"$python" - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);files={}
# Keep deployment metadata portable, including when reusing older prepared assets.
metadata_path=root/'classifier/metadata.json'
metadata=json.loads(metadata_path.read_text())
metadata={k:v for k,v in metadata.items() if k in ('character_count','weight_shape','bias_shape','weight_dtype','bias_dtype')}
metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
for path in sorted(root.rglob('*')):
 if path.is_file() and path.name not in ('package.json','build.log','export.log','model.inc'):
  files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'package.json').write_text(json.dumps({'format_version':1,'model_abi_version':2,'files_sha256':files},indent=2)+'\n')
PY
