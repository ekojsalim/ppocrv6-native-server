#!/usr/bin/env python3
"""Select a deterministic shortlist audit, or compare against exhaustive output."""
import json
import statistics
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ("select", "compare"):
        raise SystemExit("audit.py select RESULTS QUERIES OUTPUT | compare RESULTS EXHAUSTIVE OUTPUT")
    rows = json.loads(Path(sys.argv[2]).read_text())["results"]
    if sys.argv[1] == "select":
        queries = {q["id"]: q for q in json.loads(Path(sys.argv[3]).read_text())}
        result = []
        for row in rows:
            near = bool(row["candidates"] and row["candidates"][0]["similarity"] >= .8)
            if ((row["original_model"] is not None and not row["original_model"]["text"])
                or row["group"] == "heldout_polarity"
                or (near and row["group"] != "historical_synthetic")
                or row["proposed_if_empty"] is not None):
                result.append(queries[row["id"]])
    else:
        exhaustive = json.loads(Path(sys.argv[3]).read_text())["results"]
        by_id = {row["id"]: row for row in rows}
        differences = []
        winner_differences, decision_differences = 0, 0
        for row in exhaustive:
            short = by_id[row["id"]]
            a = [c["character"] for c in short["candidates"][:2]]
            b = [c["character"] for c in row["candidates"][:2]]
            winner_differences += a[:1] != b[:1]
            decision_differences += short["proposed_if_empty"] != row["proposed_if_empty"]
            if a != b:
                differences.append({"id":row["id"], "expected":row["expected"],
                    "shortlist":short["candidates"][:2], "exhaustive":row["candidates"][:2]})
        result = {"count":len(exhaustive), "top_two_differ":len(differences),
                  "winner_differ":winner_differences, "decision_differ":decision_differences,
                  "match_median_ms":statistics.median(r["match_ms"] for r in exhaustive),
                  "differences":differences}
    Path(sys.argv[4]).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
