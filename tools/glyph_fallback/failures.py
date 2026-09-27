#!/usr/bin/env python3
"""Mine frozen PP-OCR failure cohorts without changing the empty-only gate."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def cohort(row):
    model = row['original_model']
    if model is None:
        return 'unmeasured'
    text = model['text']
    if row['expected'] is None:
        return 'negative_nonempty' if text else 'negative_empty'
    if not text:
        return 'glyph_empty'
    return 'glyph_correct' if text == row['expected'] else 'glyph_wrong_nonempty'


def metrics(rows):
    proposed = [r for r in rows if r['proposed_if_empty'] is not None]
    return {
        'inputs': len(rows),
        'unique_labels': len({r['expected'] for r in rows if r['expected'] is not None}),
        'correct_dictionary_proposals': sum(r['proposed_if_empty'] == r['expected'] for r in proposed),
        'wrong_or_unsafe_dictionary_proposals': sum(r['proposed_if_empty'] != r['expected'] for r in proposed),
        'dictionary_abstentions': len(rows) - len(proposed),
        'normalization_rejected': sum(r['rejection'] is not None for r in rows),
        'actual_correct_recoveries': sum(r['applied'] and r['text'] == r['expected'] for r in rows),
        'actual_wrong_fills': sum(r['applied'] and r['text'] != r['expected'] for r in rows),
    }


def mine(queries, rows):
    by_id = {q['id']: q for q in queries}
    if len(by_id) != len(queries) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate query/result IDs')
    if set(by_id) != {r['id'] for r in rows}:
        raise ValueError('query/result IDs differ')
    cohorts = defaultdict(list)
    for row in rows:
        q = by_id[row['id']]
        if (q['expected'], q['group'], q.get('model')) != (
                row['expected'], row['group'], row['original_model']):
            raise ValueError(f"query/result snapshot mismatch: {row['id']}")
        cohorts[cohort(row)].append(row)
    report = {'note': 'Frozen evaluation cohorts, not training templates. Nonempty proposals are diagnostics only.',
              'cohorts': {}}
    manifests = {}
    for name, members in sorted(cohorts.items()):
        groups = defaultdict(list)
        for row in members:
            groups[row['group']].append(row)
        report['cohorts'][name] = {**metrics(members), 'groups': {
            group: metrics(items) for group, items in sorted(groups.items())}}
        manifests[name] = [by_id[r['id']] for r in members]
    failures = cohorts['glyph_empty'] + cohorts['glyph_wrong_nonempty']
    manifests['glyph_failures'] = [by_id[r['id']] for r in failures]
    report['glyph_failures'] = metrics(failures)
    report['empty_unrecovered_by_group'] = dict(Counter(
        r['group'] for r in cohorts['glyph_empty'] if not (r['applied'] and r['text'] == r['expected'])))
    return report, manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report, manifests = mine(json.loads(args.queries.read_text()),
                             json.loads(args.results.read_text())['results'])
    report['sources'] = {name: {'path': str(path.resolve()),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in [('queries', args.queries), ('results', args.results)]}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, queries in manifests.items():
        (args.output / f'{name}.json').write_text(json.dumps(queries, ensure_ascii=False, indent=2))
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({name: {k: v for k, v in stats.items() if k != 'groups'}
                      for name, stats in report['cohorts'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
