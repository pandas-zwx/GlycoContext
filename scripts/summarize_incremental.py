"""Summarize the prespecified development-only incremental component path."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


PATH = (
    ('inc_cnn_seq', 'sequence CNN'),
    ('inc_cnn_structure', '+ CNN RNAfold channels'),
    ('inc_rnafm', '+ RNA-FM'),
    ('inc_gnn_backbone', '+ backbone GNN'),
    ('no_cell', '+ GNN pair messages'),
    ('full', '+ cell conditioning'),
)
METRICS = ('AUROC', 'AP', 'MCC', 'local_AUROC')
CELLS = ('HeLa', 'H9', 'macro')


def metric_path(root, arm, fold, seed):
    if arm in {'no_cell'}:
        base = root/'ablations'/arm
    elif arm == 'full':
        base = root/'model'/arm
    else:
        base = root/'incremental'/arm
    return base/f'dev{fold}'/f'seed{seed}'/'metrics.json'


def load(root):
    values={}
    for arm,_ in PATH:
        for fold in range(3):
            for seed in range(5):
                file=metric_path(root,arm,fold,seed)
                if not file.exists():
                    raise FileNotFoundError(file)
                data=json.loads(file.read_text())
                for cell in CELLS:
                    source=data['macro'] if cell == 'macro' else data['cells'][cell]['metrics']
                    for metric in METRICS:
                        values[arm,fold,seed,cell,metric]=float(source[metric])
    return values


def hierarchical_ci(delta, rng, draws):
    # Outer RNA-group folds are the independent units; seeds are resampled within folds.
    folds=rng.integers(0,3,(draws,3))
    seeds=rng.integers(0,5,(draws,3,5))
    samples=delta[folds[:,:,None],seeds].mean((1,2))
    return np.quantile(samples,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/development'))
    p.add_argument('--output',type=Path,default=Path('runs/development/incremental_analysis'))
    p.add_argument('--bootstrap-draws',type=int,default=100000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    values=load(a.root);rng=np.random.default_rng(20261103)
    summary=[];paired=[]
    for arm,label in PATH:
        for cell in CELLS:
            for metric in METRICS:
                x=np.array([[values[arm,f,s,cell,metric] for s in range(5)] for f in range(3)])
                summary.append(dict(arm=arm,label=label,cell=cell,metric=metric,n=15,
                    mean=float(x.mean()),sd=float(x.std()),fold_means=[float(v) for v in x.mean(1)]))
    for (previous,_),(arm,label) in zip(PATH,PATH[1:]):
        for cell in CELLS:
            for metric in METRICS:
                delta=np.array([[values[arm,f,s,cell,metric]-values[previous,f,s,cell,metric]
                    for s in range(5)] for f in range(3)])
                low,high=hierarchical_ci(delta,rng,a.bootstrap_draws)
                paired.append(dict(previous=previous,arm=arm,addition=label,cell=cell,metric=metric,n=15,
                    mean_delta=float(delta.mean()),fold_mean_deltas=[float(v) for v in delta.mean(1)],
                    hierarchical_bootstrap_ci95=[float(low),float(high)]))
    payload=dict(path=[dict(arm=x,label=y) for x,y in PATH],summary=summary,paired_steps=paired,
        bootstrap=dict(draws=a.bootstrap_draws,seed=20261103,
            design='resample 3 outer RNA-group folds, then 5 paired seeds within each selected fold'),
        interpretation='Adjacent differences are conditional on the prespecified order; folds, not 15 runs, are independent units.',
        final_used=False)
    (a.output/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    for name,rows in [('summary.csv',summary),('paired_steps.csv',paired)]:
        with (a.output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    lookup={(r['arm'],r['cell'],r['metric']):r for r in summary}
    delta={(r['arm'],r['cell'],r['metric']):r for r in paired}
    lines=['# Incremental component analysis','',
        'Development only: 3 RNA-group-isolated folds x 5 seeds. Final was not used.','',
        '## Macro results','',
        '| Step | AUROC | AP | MCC | local AUROC | ΔAUROC | ΔAP | ΔMCC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in PATH:
        means=[lookup[arm,'macro',m]['mean'] for m in METRICS]
        ds=[delta[arm,'macro',m]['mean_delta'] if (arm,'macro',m) in delta else None for m in METRICS[:3]]
        fmt=lambda v: '—' if v is None else f'{v:+.4f}'
        lines.append(f"| {label} | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} | {means[3]:.4f} | {fmt(ds[0])} | {fmt(ds[1])} | {fmt(ds[2])} |")
    lines += ['', 'Adjacent deltas are paired by fold and seed and are conditional on this fixed order.',
        'Confidence intervals are stored in `paired_steps.csv` and use hierarchical bootstrap over folds and seeds.','']
    (a.output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
