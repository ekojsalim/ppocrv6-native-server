#!/usr/bin/env python3
"""Replay one saved diagnostic case against a running OCR server."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('case', type=Path)
p.add_argument('--url', default='http://127.0.0.1:8184')
p.add_argument('--output', type=Path, help='Write response JSON to this file instead of stdout.')
a = p.parse_args()
if a.case.stat().st_size > 16 * 1024 * 1024:
    p.error('case exceeds 16 MiB')
case = json.loads(a.case.read_text())
if case.get('schema_version') != 1 or not isinstance(case.get('image'), str):
    p.error('expected a replayable schema-v1 case')
endpoint = {'glyph': 'glyphs', 'line': 'lines', 'page': 'ocr'}.get(case.get('kind'))
if endpoint is None:
    p.error('unknown case kind')
body = dict(case['request'], image=case['image'])
request = Request(a.url.rstrip('/') + '/v1/' + endpoint + '/recognize',
                  data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
with urlopen(request, timeout=120) as response:
    result = json.dumps(json.load(response), ensure_ascii=False, indent=2) + '\n'
if a.output:
    a.output.write_text(result)
else:
    print(result, end='')
