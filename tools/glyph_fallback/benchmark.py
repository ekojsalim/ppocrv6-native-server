#!/usr/bin/env python3
"""Paired sequential matcher benchmarks with decision parity checks.

Use an empty-only query manifest for realistic gate workload; loading is reported
separately. The binaries must use the same dictionary format and score semantics.
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path


def latency(rows):
    values=sorted(r['match_ms'] for r in rows)
    if not values:
        return {'median_ms':None,'p95_ms':None,'max_ms':None,'total_ms':0}
    return {'median_ms':statistics.median(values),
            'p95_ms':values[int(.95*(len(values)-1))], 'max_ms':max(values),
            'total_ms':sum(values)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','dictionary','queries','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats < 1:p.error('--repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    runs={'before':[],'after':[]}
    for repeat in range(args.repeats):
        # Alternate order to reduce systematic temperature/frequency bias.
        for name in (('before','after') if repeat%2==0 else ('after','before')):
            target=args.output/f'{name}-{repeat}.json'
            subprocess.run([str(getattr(args,name).resolve()),'evaluate',str(args.dictionary),
                            str(args.queries),str(target)],check=True)
            document=json.loads(target.read_text())
            runs[name].append(document)
        before={r['id']:r for r in runs['before'][-1]['results']}
        after={r['id']:r for r in runs['after'][-1]['results']}
        assert before.keys()==after.keys()
        for ident,row in before.items():
            for key in ('text','applied','proposed_if_empty','original_model'):
                assert row[key]==after[ident][key],(ident,key)
            if row['proposed_if_empty'] is not None:
                assert row['candidates'][0]==after[ident]['candidates'][0],ident
    summary={}
    for name,documents in runs.items():
        rows=[r for d in documents for r in d['results']]
        summary[name]={'runs':args.repeats,'count_per_run':len(documents[0]['results']),
            'all':latency(rows), 'accepted':latency([r for r in rows if r['applied']]),
            'production':latency([r for r in rows if r['id']=='production_attachment'])
                if any(r['id']=='production_attachment' for r in rows) else None,
            'median_load_ms':statistics.median(d['load_ms'] for d in documents)}
    summary['parity']='All replay decisions, original predictions and accepted winners/scores identical.'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
