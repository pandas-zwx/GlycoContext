"""Summarize every prespecified final seed without seed selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CELLS=('HeLa','H9','macro')
METRICS=('AUROC','AP','MCC','local_AUROC','Brier','logloss')


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/final_evaluation'))
    a=p.parse_args();model=a.root/'model'/'full'/'final';out=a.root/'summary';out.mkdir(exist_ok=True)
    rows=[];raw=[]
    for seed in range(5):
        run=model/f'seed{seed}';training=json.loads((run/'training.json').read_text())
        if training['status'] != 'COMPLETE':raise ValueError(f'seed{seed} is not complete')
        metrics=json.loads((run/'metrics.json').read_text());raw.append(metrics)
        for cell in CELLS:
            source=metrics['macro'] if cell == 'macro' else metrics['cells'][cell]['metrics']
            for metric in METRICS:
                rows.append(dict(seed=seed,cell=cell,metric=metric,value=float(source[metric])))
    aggregate=[]
    for cell in CELLS:
        for metric in METRICS:
            values=np.array([r['value'] for r in rows if r['cell']==cell and r['metric']==metric])
            aggregate.append(dict(cell=cell,metric=metric,n_seeds=5,mean=float(values.mean()),
                population_sd=float(values.std()),values_by_seed=[float(x) for x in values]))
    freeze=a.root/'freeze'/'manifest.json'
    payload=dict(status='COMPLETE',model='full',partition='final',seeds=list(range(5)),
        seed_selection=False,aggregate=aggregate,per_seed=rows,freeze_manifest_sha256=sha256(freeze),
        interpretation='Seed SD measures optimization variability on one internal RNA-group-isolated test set; it is not cross-study uncertainty.',
        external_validation=False)
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (out/'per_seed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with (out/'aggregate.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=aggregate[0].keys());writer.writeheader();writer.writerows(aggregate)
    lookup={(r['cell'],r['metric']):r for r in aggregate}
    lines=['# Frozen full-model final evaluation','',
        'All five prespecified seeds are included; no seed was selected using final performance.','',
        '| Cell | AUROC | AP | MCC | local AUROC | Brier | log loss |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cell in CELLS:
        vals=[]
        for metric in METRICS:
            r=lookup[cell,metric];vals.append(f"{r['mean']:.4f} ± {r['population_sd']:.4f}")
        lines.append('| '+cell+' | '+' | '.join(vals)+' |')
    lines += ['', 'This is an internal test from the same data source, isolated by RNA groups; it is not independent experimental or cross-study validation.',
        'Seed SD describes optimization variability, not biological-replicate or cross-study uncertainty.','']
    (out/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
