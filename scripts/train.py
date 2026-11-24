"""E selection -> fresh E+S refit -> C-only classification thresholds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.io import read_json,write_json,now,sha256
from glycocontext.model import GlycoContext
from glycocontext.data import read_dataset,roles,FeatureBank
from glycocontext.metrics import mcc_threshold


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--features',type=Path,default=Path('features'))
    p.add_argument('--split',type=Path,default=Path('splits/final.json'))
    p.add_argument('--recipe',type=Path,default=Path('configs/train.json'))
    p.add_argument('--seed',type=int,choices=[0,1,2,3,4],default=0)
    p.add_argument('--ablation',choices=GlycoContext.ABLATIONS,default='full')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--max-epochs',type=int,help='Explicit recipe override; recorded as non-reference setting')
    a=p.parse_args();cfg=read_json(a.recipe)
    if a.max_epochs is not None:
        if a.max_epochs<1:raise ValueError('Positive epoch count required')
        cfg['max_epochs']=a.max_epochs
    a.output.mkdir(parents=True,exist_ok=False)
    records,_,manifest=read_dataset(a.data);split=read_json(a.split);r=roles(records,split)
    bank=FeatureBank(a.features,records)
    ids=np.array([x['candidate_id'] for x in records]);y=np.array([x['y'] for x in records],dtype=np.float32)
    cells=np.array([x['cell'] for x in records])
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    report=dict(status='TRAINING',started=now(),seed=a.seed,ablation=a.ablation,recipe=cfg,
        split_sha256=sha256(a.split),data_manifest_sha256=sha256(a.data/'manifest.json'),
        feature_manifest_sha256=sha256(a.features/'manifest.json'),fits=[],cell_names=['HeLa','H9'])
    def save():write_json(a.output/'training.json',report)
    save()
    def reset():
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    @torch.no_grad()
    def predict(model,take):
        model.eval()
        return np.concatenate([torch.sigmoid(bank.forward(model,take[j:j+cfg['prediction_batch']],a.device)).cpu().numpy()
            for j in range(0,len(take),cfg['prediction_batch'])])
    def fit(stage,epochs):
        from sklearn.metrics import roc_auc_score
        reset();model=GlycoContext(a.ablation).to(a.device)
        opt=torch.optim.AdamW(model.parameters(),**cfg['optimizer'])
        fitting=r['E'] if stage=='selector' else r['refit'];rng=np.random.default_rng(a.seed)
        ratios={c:float(np.sum(y[fitting[cells[fitting]==c]]==0)/np.sum(y[fitting[cells[fitting]==c]]==1)) for c in (0,1)}
        entry=dict(stage=stage,fitting_ids=ids[fitting].tolist(),history=[])
        report['fits'].append(entry);save();best=-1.;chosen=0;beststate=None
        for epoch in range(epochs):
            events=[]
            for c in (0,1):
                order=rng.permutation(fitting[cells[fitting]==c])
                events.extend((c,order[j:j+cfg['batch_sites']]) for j in range(0,len(order),cfg['batch_sites']))
            rng.shuffle(events);model.train();losses=[]
            for c,take in events:
                opt.zero_grad(set_to_none=True);logits=bank.forward(model,take,a.device)
                loss=F.binary_cross_entropy_with_logits(logits,torch.from_numpy(y[take]).to(a.device),
                    pos_weight=torch.tensor(ratios[c],device=a.device))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                opt.step();losses.append(float(loss.detach()))
            event=dict(epoch=epoch+1,mean_batch_BCE=float(np.mean(losses)))
            if stage=='selector':
                q=predict(model,r['S']);ss=cells[r['S']];yy=y[r['S']]
                score=float(np.mean([roc_auc_score(yy[ss==c],q[ss==c]) for c in (0,1)]))
                event['S_macro_AUROC']=score
                if score>best:
                    best=score;chosen=epoch+1;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            entry['history'].append(event);save();print(stage,a.seed,a.ablation,event,flush=True)
        if stage=='selector':model.load_state_dict(beststate,strict=True)
        else:chosen=epochs
        packed=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=a.seed,
            ablation=a.ablation,epochs=chosen,stage=stage,cell_names=['HeLa','H9'],
            fitting_ids=ids[fitting].tolist(),recipe=cfg,split_sha256=sha256(a.split),architecture='GlycoContext-0.1')
        path=a.output/(stage+'.pt');torch.save(packed,path)
        entry.update(selected_epoch=chosen,checkpoint=path.name,checkpoint_sha256=sha256(path));save()
        return model,packed,chosen
    try:
        selector,_,chosen=fit('selector',cfg['max_epochs']);del selector
        if torch.cuda.is_available():torch.cuda.empty_cache()
        model,packed,_=fit('refit',chosen)
        probability=predict(model,r['C'])
        thresholds={str(c):mcc_threshold(y[r['C']][cells[r['C']]==c],probability[cells[r['C']]==c]) for c in (0,1)}
        packed['thresholds']=thresholds;packed['threshold_ids']=ids[r['C']].tolist()
        path=a.output/'model.pt';torch.save(packed,path)
        fresh=GlycoContext(a.ablation).to(a.device)
        fresh.load_state_dict(torch.load(path,map_location=a.device,weights_only=True)['state_dict'],strict=True)
        error=float(np.max(np.abs(predict(fresh,r['C'])-probability)))
        if error>1e-7:raise ValueError('Saved-model prediction mismatch')
        np.savez_compressed(a.output/'C_predictions.npz',ids=ids[r['C']],probabilities=probability)
        report.update(status='COMPLETE',completed=now(),selected_epoch=chosen,thresholds=thresholds,
            checkpoint='model.pt',checkpoint_sha256=sha256(path),reload_maxabs=error)
        save()
    except Exception as exc:
        report.update(status='FAILED_PRESERVE_PARTIAL',error=repr(exc),updated=now());save();raise


if __name__=='__main__':main()
