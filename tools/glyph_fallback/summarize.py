#!/usr/bin/env python3
"""Summarize actual empty recovery separately from simulated empty inputs."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))] if values else None


def summarize(document):
    groups = defaultdict(list)
    for row in document["results"]:
        groups[row["group"]].append(row)
    result = {"templates": document["templates"], "load_ms": document["load_ms"],
              "thresholds": [document["min_similarity"], document["min_different_character_margin"]],
              "groups": {}}
    for group, rows in groups.items():
        proposals = [r for r in rows if r["proposed_if_empty"] is not None]
        measured = [r for r in rows if r["rejection"] is None]
        result["groups"][group] = {
            "count": len(rows), "proposed_if_empty": len(proposals),
            "correct_proposals": sum(r["proposed_if_empty"] == r["expected"] for r in proposals),
            "wrong_or_unsafe_proposals": sum(r["proposed_if_empty"] != r["expected"] for r in proposals),
            "abstained": len(rows)-len(proposals),
            "normalization_rejected": sum(r["rejection"] is not None for r in rows),
            "match_median_ms": statistics.median(r["match_ms"] for r in measured) if measured else None,
            "match_p95_ms": percentile([r["match_ms"] for r in measured], .95),
            "decode_normalize_median_ms": statistics.median(r["decode_normalize_ms"] for r in rows),
        }
    actual = [r for r in document["results"] if r["original_model"] is not None]
    if actual:
        applied = [r for r in actual if r["applied"]]
        correct_before = [r for r in actual if r["original_model"]["text"] == r["expected"]]
        result["actual_model_replay"] = {
            "count": len(actual), "labelled_count": sum(r["expected"] is not None for r in actual),
            "negative_nonempty_before": sum(r["expected"] is None and bool(r["original_model"]["text"]) for r in actual),
            "negative_nonempty_after": sum(r["expected"] is None and bool(r["text"]) for r in actual),
            "empty": sum(not r["original_model"]["text"] for r in actual),
            "applied": len(applied), "fallback_frequency": len(applied)/len(actual),
            "recovered": sum(r["text"] == r["expected"] for r in applied),
            "wrong_fills": sum(r["text"] != r["expected"] for r in applied),
            "correct_before": len(correct_before),
            "correct_after": sum(r["text"] == r["expected"] for r in actual),
            "newly_wrong_previously_correct": sum(r["text"] != r["expected"] for r in correct_before),
            "nonempty_changed": sum(r["text"] != r["original_model"]["text"] for r in actual if r["original_model"]["text"]),
            "empty_details": [r for r in actual if not r["original_model"]["text"]],
        }
    # Sensitivity only: these are NOT independently calibrated operating points.
    result["threshold_sensitivity_note"] = "Post-hoc recorded-candidate analysis; changed thresholds are not reverified against the full dictionary."
    result["threshold_sensitivity"] = []
    for minimum in (.80, .85, .90, .93, .96):
        for margin in (.02, .04, .08):
            counts = {}
            for group, rows in groups.items():
                selected = [r for r in rows if len(r["candidates"]) >= 2
                            and r["candidates"][0]["similarity"] >= minimum
                            and r["candidates"][0]["similarity"] - r["candidates"][1]["similarity"] >= margin]
                counts[group] = {"accepted": len(selected), "wrong_or_unsafe": sum(
                    r["candidates"][0]["character"] != r["expected"] for r in selected)}
            empty_selected = [r for r in actual if not r["original_model"]["text"]
                              and len(r["candidates"]) >= 2
                              and r["candidates"][0]["similarity"] >= minimum
                              and r["candidates"][0]["similarity"]-r["candidates"][1]["similarity"] >= margin]
            result["threshold_sensitivity"].append({"minimum": minimum, "margin": margin, "groups": counts,
                "actual_empty_recovered": sum(r["candidates"][0]["character"] == r["expected"] for r in empty_selected),
                "actual_empty_wrong_or_unsafe": sum(r["candidates"][0]["character"] != r["expected"] for r in empty_selected)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for key, value in summary.items():
        if key not in ("threshold_sensitivity", "actual_model_replay"):
            print(key, json.dumps(value, ensure_ascii=False))
    actual = summary.get("actual_model_replay", {})
    print("actual_model_replay", json.dumps({k:v for k,v in actual.items() if k != "empty_details"}))
    for row in actual.get("empty_details", []):
        print(row["id"], row["expected"], "=>", row["text"], row["rejection"], row["candidates"][:2])


if __name__ == "__main__":
    main()
