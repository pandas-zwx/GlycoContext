"""Summarize frozen full, baselines and every final ablation without selection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS=(
    ('full','Full','model/full/final'),
    ('rf','RF','baselines/rf/final'),
    ('rnafm_lgbm','RNA-FM LightGBM','baselines/rnafm_lgbm/final'),
    ('no_cnn','no CNN','ablations/no_cnn/final'),
    ('no_gnn','no GNN','ablations/no_gnn/final'),
    ('no_rnafm','no RNA-FM','ablations/no_rnafm/final'),
    ('no_rnafold','no RNAfold','ablations/no_rnafold/final'),
    ('no_cell','no cell','ablations/no_cell/final'),
    ('no_cnn_structure','no CNN structure','ablations/no_cnn_structure/final'),
    ('no_gnn_pair','no GNN pair','ablations/no_gnn_pair/final'),
)
CORE_METHODS=('full','rf','rnafm_lgbm','no_cnn','no_gnn','no_rnafm','no_rnafold','no_cell')
CELLS=('HeLa','H9','macro')
METRICS=('AUROC','AP','MCC','local_AUROC','Brier','logloss')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/final_evaluation'))
    a=p.parse_args();out=a.root/'comparison';out.mkdir(exist_ok=True)
    raw={};per_seed=[]
    for method,label,relative in METHODS:
        for seed in range(5):
            run=a.root/relative/f'seed{seed}';training=json.loads((run/'training.json').read_text())
            if training['status'] != 'COMPLETE':raise ValueError(f'{method} seed{seed} incomplete')
            data=json.loads((run/'metrics.json').read_text())
            for cell in CELLS:
                source=data['macro'] if cell=='macro' else data['cells'][cell]['metrics']
                for metric in METRICS:
                    value=float(source[metric]);raw[method,seed,cell,metric]=value
                    per_seed.append(dict(method=method,label=label,seed=seed,cell=cell,metric=metric,value=value))
    aggregate=[];paired=[]
    for method,label,_ in METHODS:
        for cell in CELLS:
            for metric in METRICS:
                values=np.array([raw[method,s,cell,metric] for s in range(5)])
                aggregate.append(dict(method=method,label=label,cell=cell,metric=metric,n_seeds=5,
                    mean=float(values.mean()),population_sd=float(values.std())))
                if method!='full':
                    delta=values-np.array([raw['full',s,cell,metric] for s in range(5)])
                    paired.append(dict(method=method,label=label,cell=cell,metric=metric,n_pairs=5,
                        mean_delta_vs_full=float(delta.mean()),population_sd_delta=float(delta.std()),
                        deltas_by_seed=[float(x) for x in delta]))
    payload=dict(status='COMPLETE',methods=[dict(method=x,label=y,path=z) for x,y,z in METHODS],
        seeds=list(range(5)),aggregate=aggregate,paired_vs_full=paired,
        disclosure='All methods retain the five prespecified seeds. Epoch selection uses S, thresholds use C, and evaluation-partition metrics are used only for reporting.',
        scope='Internal same-source RNA-group-isolated test; not independent experimental or cross-study validation.')
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    for name,rows in [('aggregate.csv',aggregate),('paired_vs_full.csv',paired),('per_seed.csv',per_seed)]:
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    lookup={(r['method'],r['cell'],r['metric']):r for r in aggregate}
    delta={(r['method'],r['cell'],r['metric']):r for r in paired}
    lines=['# Frozen final model comparison','',
        'All five prespecified seeds are included for every method; no seed was selected using final performance.','',
        '## Macro results','',
        '| Method | AUROC | AP | MCC | local AUROC | ΔAUROC vs Full | ΔAP | ΔMCC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,label,_ in METHODS:
        vals=[lookup[method,'macro',m] for m in METRICS[:4]]
        fmt=lambda r:f"{r['mean']:.4f} ± {r['population_sd']:.4f}"
        ds=[None if method=='full' else delta[method,'macro',m]['mean_delta_vs_full'] for m in METRICS[:3]]
        dformat=lambda v:'—' if v is None else f'{v:+.4f}'
        lines.append(f"| {label} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(vals[3])} | {dformat(ds[0])} | {dformat(ds[1])} | {dformat(ds[2])} |")
    for cell in ('HeLa','H9'):
        lines += ['',f'## {cell}','',
            '| Method | AUROC | AP | MCC | local AUROC |','|---|---:|---:|---:|---:|']
        for method,label,_ in METHODS:
            vals=[lookup[method,cell,m] for m in METRICS[:4]]
            lines.append('| '+label+' | '+' | '.join(fmt(v) for v in vals)+' |')
    lines += ['',payload['disclosure'],'',payload['scope'],
        'Seed SD measures optimization variability on this single test set, not cross-study uncertainty.','']
    (out/'report.md').write_text('\n'.join(lines))
    core_rows=[r for r in aggregate if r['method'] in CORE_METHODS]
    with (out/'core_aggregate.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=core_rows[0].keys());writer.writeheader();writer.writerows(core_rows)
    core=['# Core-module final comparison','',
        'Main reporting subset: Full, two competitive baselines, and module-level ablations only.','',
        '`no CNN structure` and `no GNN pair` are retained in the complete archive but excluded from this main table as fine-grained pathway ablations.']
    for cell in CELLS:
        core += ['',f'## {cell}','',
            '| Method | AUROC | AP | MCC | local AUROC |','|---|---:|---:|---:|---:|']
        for method,label,_ in METHODS:
            if method not in CORE_METHODS:continue
            vals=[lookup[method,cell,m] for m in METRICS[:4]]
            core.append('| '+label+' | '+' | '.join(fmt(v) for v in vals)+' |')
    core += ['',payload['disclosure'],'',payload['scope'],
        'Seed SD measures optimization variability on this single test set, not cross-study uncertainty.','']
    (out/'core_report.md').write_text('\n'.join(core))


if __name__=='__main__':main()
