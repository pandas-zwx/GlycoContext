"""Score specified RNA candidates using fixed saved weights and C thresholds."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset,FeatureBank,roles
from glycocontext.model import GlycoContext
from glycocontext.io import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--features',type=Path,default=Path('features'));p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',type=Path);p.add_argument('--role',choices=['E','S','C','test'],default='test')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();records,_,_=read_dataset(a.data);bank=FeatureBank(a.features,records)
    take=roles(records,read_json(a.split))[a.role] if a.split else np.arange(len(records))
    if not len(take):raise ValueError('Empty requested role')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=GlycoContext(state.get('ablation','full')).to(a.device).eval();model.load_state_dict(state['state_dict'],strict=True)
    if 'thresholds' not in state:raise ValueError('Checkpoint has no training-derived thresholds')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    predictions=[]
    with torch.inference_mode():
        for j in range(0,len(take),a.batch_size):
            predictions.extend(torch.sigmoid(bank.forward(model,take[j:j+a.batch_size],a.device)).cpu().numpy().tolist())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['candidate_id','cell','RNA_position1','probability','threshold','prediction','seed','ablation']);w.writeheader()
        for i,probability in zip(take,predictions):
            r=records[i];t=float(state['thresholds'][str(r['cell'])])
            w.writerow(dict(candidate_id=r['candidate_id'],cell=['HeLa','H9'][r['cell']],RNA_position1=r['RNA_position1'],
                probability=probability,threshold=t,prediction=int(probability>=t),seed=state['seed'],ablation=state.get('ablation','full')))


if __name__=='__main__':main()
