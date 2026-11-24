"""Sequential, finite reference/ablation queue; no automatic architecture search."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['full'])
    p.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2,3,4],default=[0,1,2,3,4])
    p.add_argument('--partitions',nargs='+',default=['dev0','dev1','dev2','final'])
    p.add_argument('--data',default='data');p.add_argument('--features',default='features')
    p.add_argument('--output',type=Path,default=Path('runs/reference'));p.add_argument('--device',default='cuda')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    for arm in a.arms:
        for partition in a.partitions:
            for seed in a.seeds:
                out=a.output/arm/partition/f'seed{seed}'
                subprocess.run([sys.executable,str(root/'scripts/train.py'),'--data',a.data,'--features',a.features,
                    '--split',str(root/'splits'/(partition+'.json')),'--recipe',str(root/'configs/train.json'),
                    '--seed',str(seed),'--ablation',arm,'--output',str(out),'--device',a.device],cwd=root,check=True)
                prediction=out/'evaluation_predictions.csv'
                subprocess.run([sys.executable,str(root/'scripts/predict.py'),'--data',a.data,'--features',a.features,
                    '--checkpoint',str(out/'model.pt'),'--split',str(root/'splits'/(partition+'.json')),
                    '--output',str(prediction),'--device',a.device],cwd=root,check=True)
                subprocess.run([sys.executable,str(root/'scripts/evaluate.py'),'--data',a.data,
                    '--predictions',str(prediction),'--output',str(out/'metrics.json')],cwd=root,check=True)


if __name__=='__main__':main()
