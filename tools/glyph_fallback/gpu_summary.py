#!/usr/bin/env python3
"""Check GPU proposal parity and summarize warmed host wall time and VRAM."""
import argparse
import json
import statistics
from pathlib import Path


def latency(rows):
    values=sorted(r['wall_ms'] for r in rows)
    if not values:return None
    return {'count':len(values),'median_ms':statistics.median(values),
            'p95_ms':values[int(.95*(len(values)-1))],'max_ms':max(values)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('results','manifest','cpu-results','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    data=[json.loads(line) for line in args.results.read_text().splitlines()]
    memory=data[0];assert memory['event']=='memory'
    rows=data[1:];meta=json.loads(args.manifest.read_text())['queries']
    cpu={r['id']:r for r in json.loads(args.cpu_results.read_text())['results']}
    errors=[];deltas=[];recoveries={};wrong={}
    seen=set()
    for row in rows:
        assert row['event']=='query'
        key=(row['round'],row['index']);assert key not in seen;seen.add(key)
        q=meta[row['index']];ref=cpu[q['id']]
        proposal=chr(row['codepoint']) if row['accepted'] else None
        if proposal!=ref['proposed_if_empty']:errors.append({'id':q['id'],'gpu':proposal,'cpu':ref['proposed_if_empty']})
        if proposal and ref['proposed_if_empty']:
            deltas.append(abs(row['score']-ref['candidates'][0]['similarity']))
        if proposal and q['original_model'] is not None and q['original_model']['text']=='':
            target=recoveries if proposal==q['expected'] else wrong
            target[row['round']]=target.get(row['round'],0)+1
    rounds={r['round'] for r in rows}
    assert len(rows)==len(rounds)*len(meta)
    production=[r for r in rows if meta[r['index']]['id']=='production_attachment']
    prep=[q for q in meta if q['id']=='production_attachment']
    summary={'memory':memory,'allocated_mib':memory['allocated_bytes']/2**20,
             'observed_allocation_delta_mib':(memory['free_before']-memory['free_after'])/2**20,
             'rounds':len(rounds),'queries_per_round':len(meta),'parity_errors':errors,
             'max_accepted_score_delta':max(deltas,default=0),
             'recovered_per_round':recoveries,'wrong_fills_per_round':wrong,
             'valid_queries':latency([r for r in rows if r['valid']]),
             'accepted':latency([r for r in rows if r['accepted']]),
             'production':latency(production),'production_cpu_preparation':prep}
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='production_cpu_preparation'},indent=2))
    if errors or summary['max_accepted_score_delta']>1e-6:raise SystemExit('GPU/CPU parity failed')


if __name__=='__main__':main()
