"""Evaluate saved predictions. No threshold fitting or model selection."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset
from glycocontext.io import write_json,sha256
from glycocontext.metrics import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records,_,_=read_dataset(a.data);lookup={r['candidate_id']:r for r in records}
    with a.predictions.open() as f:pred=list(csv.DictReader(f))
    if not pred or len({r['candidate_id'] for r in pred}) != len(pred):raise ValueError('Unique nonempty predictions required')
    r=[lookup[v['candidate_id']] for v in pred]
    if any('y' not in row for row in r):raise ValueError('Evaluation requires supplied labels')
    y=np.array([v['y'] for v in r]);p=np.array([float(v['probability']) for v in pred]);t=np.array([float(v['threshold']) for v in pred])
    cells=np.array([v['cell'] for v in r]);parents=np.array([v['sequence_sha256'] for v in r]);bases=np.array([v['center_base'] for v in r])
    result={}
    for c,name in enumerate(('HeLa','H9')):
        take=cells==c
        if take.any():result[name]=dict(n=int(take.sum()),positive=int(y[take].sum()),metrics=evaluate(y[take],p[take],t[take],parents[take],bases[take]))
    keys=['AUROC','AP','MCC','local_AUROC','Brier','logloss']
    macro={k:float(np.mean([v['metrics'][k] for v in result.values()])) if all(v['metrics'][k] is not None for v in result.values()) else None for k in keys}
    if a.output.exists():raise FileExistsError('Preserve earlier evaluation')
    write_json(a.output,dict(cells=result,macro=macro,predictions_sha256=sha256(a.predictions),thresholds_reestimated=False,
        interpretation='Metrics evaluate supplied binary labels; AP is average precision, not trapezoidal PR-AUC.'))
    print(result,macro)


if __name__=='__main__':main()
