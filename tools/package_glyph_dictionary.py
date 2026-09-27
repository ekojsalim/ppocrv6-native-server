#!/usr/bin/env python3
"""Install a v2 dictionary and its font notices into an existing model bundle."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dictionary', type=Path)
    p.add_argument('model_dir', type=Path)
    p.add_argument('--font-license', type=Path, action='append', required=True)
    args = p.parse_args()
    if args.dictionary.stat().st_size > 128 * 1024 * 1024:
        p.error('dictionary exceeds 128 MiB')
    data = json.loads(args.dictionary.read_text())
    templates = data.get('templates', [])
    if data.get('version') != 2 or not 1 <= len(templates) <= 100_000:
        p.error('expected a nonempty v2 dictionary with at most 100,000 templates')
    for t in templates:
        c, s = t['character'], t['shape']
        pixels = base64.b64decode(s['pixels'], validate=True)
        if (len(c) != 1 or not 0x4e00 <= ord(c) <= 0x9fff
                or len(pixels) != 1024 or not any(pixels)
                or len(s['rows']) != 32 or not any(s['rows'])
                or not math.isfinite(s['aspect']) or s['aspect'] <= 0):
            p.error('invalid template')
    # Remove development-machine font paths; preserve identity and rendering data.
    provenance = data.get('provenance', {})
    data['provenance'] = {k: provenance[k] for k in
        ('character_range', 'normalization', 'pillow', 'seed') if k in provenance}
    data['provenance']['fonts'] = [
        {k: f[k] for k in ('source', 'sha256', 'index', 'weight', 'size', 'role') if k in f}
        for f in provenance.get('fonts', [])]
    notices = [f.read_bytes() for f in args.font_license]
    manifest_path = args.model_dir / 'package.json'
    manifest = json.loads(manifest_path.read_text())
    out = args.model_dir / 'glyph-fallback'
    if out.exists():
        p.error('glyph-fallback already exists; use a fresh destination bundle')
    out.mkdir()
    (out / 'dictionary.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    for i, notice in enumerate(notices):
        (out / f'font-license-{i + 1}.txt').write_bytes(notice)
    for path in out.iterdir():
        manifest['files_sha256'][str(path.relative_to(args.model_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    print(f'Installed {len(templates)} templates in {out}')


if __name__ == '__main__':
    main()
