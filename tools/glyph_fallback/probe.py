#!/usr/bin/env python3
"""Explicitly query an existing test server; never starts/stops a service.

`pack QUERIES.json REQUESTS.json` reads local crops; `run URL REQUESTS.json
OUTPUT.json` needs only the packed file and Python's standard library.
"""
import base64
import json
import sys
import urllib.request
from pathlib import Path


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "pack":
        queries = json.loads(Path(sys.argv[2]).read_text())
        packed = [{"id": q["id"], "image": base64.b64encode(Path(q["path"]).read_bytes()).decode()} for q in queries]
        Path(sys.argv[3]).write_text(json.dumps(packed))
    elif len(sys.argv) == 5 and sys.argv[1] == "run":
        queries = json.loads(Path(sys.argv[3]).read_text())
        results = {}
        for policy in ("all", "cjk_focus", "cjk_focus_fallback"):
            rows, responses = [], []
            for start in range(0, len(queries), 512):
                chunk = queries[start:start+512]
                payload = {"images": [q["image"] for q in chunk], "batch_size": 128,
                           "character_policy": policy, "score_mode": "model", "return_timesteps": True}
                request = urllib.request.Request(sys.argv[2].rstrip("/") + "/v1/glyphs/recognize",
                    json.dumps(payload).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.load(response)
                if len(data["predictions"]) != len(chunk):
                    raise ValueError("Prediction count mismatch")
                rows.extend({"id":q["id"], "prediction":p} for q,p in zip(chunk, data["predictions"]))
                responses.append({k:v for k,v in data.items() if k != "predictions"})
            results[policy] = {"predictions": rows, "responses": responses}
            print(policy, len(rows), "empty", sum(not r["prediction"]["text"] for r in rows), flush=True)
        Path(sys.argv[4]).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    elif len(sys.argv) == 6 and sys.argv[1] == "attach":
        queries = json.loads(Path(sys.argv[2]).read_text())
        snapshot = json.loads(Path(sys.argv[3]).read_text())[sys.argv[4]]
        by_id = {r["id"]:r["prediction"] for r in snapshot["predictions"]}
        for query in queries:
            query["model"] = by_id[query["id"]]
        Path(sys.argv[5]).write_text(json.dumps(queries, ensure_ascii=False))
    else:
        raise SystemExit(__doc__ + "\nattach QUERIES.json SNAPSHOTS.json POLICY OUTPUT.json")


if __name__ == "__main__":
    main()
