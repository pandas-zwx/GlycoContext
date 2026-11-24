"""Finite CPU baseline queue. Training and scoring remain separate entry points."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', choices=['rf', 'rnafm_lgbm'], default=['rf', 'rnafm_lgbm'])
    p.add_argument('--seeds', nargs='+', type=int, choices=[0, 1, 2, 3, 4], default=[0, 1, 2, 3, 4])
    p.add_argument('--partitions', nargs='+', choices=['dev0', 'dev1', 'dev2', 'final'], default=['dev0', 'dev1', 'dev2', 'final'])
    p.add_argument('--data', type=Path, default=Path('data')); p.add_argument('--features', type=Path, default=Path('features'))
    p.add_argument('--mean-features', type=Path, default=Path('features_baselines'))
    p.add_argument('--recipe', type=Path, default=Path('configs/baselines.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); scripts = Path(__file__).resolve().parent
    for method in a.methods:
        for part in a.partitions:
            for seed in a.seeds:
                out = a.output/method/part/f'seed{seed}'
                if out.exists(): raise FileExistsError('Preserve existing/partial run: '+str(out))
                subprocess.run([sys.executable, str(scripts/'train_baseline.py'), '--method', method,
                    '--data', str(a.data), '--features', str(a.features), '--mean-features', str(a.mean_features),
                    '--split', f'splits/{part}.json', '--seed', str(seed), '--recipe', str(a.recipe), '--output', str(out)], check=True)
                prediction = out/'predictions.csv'
                subprocess.run([sys.executable, str(scripts/'predict_baseline.py'), '--data', str(a.data),
                    '--features', str(a.features), '--mean-features', str(a.mean_features),
                    '--checkpoint', str(out/'model.joblib'), '--split', f'splits/{part}.json', '--output', str(prediction)], check=True)
                subprocess.run([sys.executable, str(scripts/'evaluate.py'), '--data', str(a.data),
                    '--predictions', str(prediction), '--output', str(out/'metrics.json')], check=True)


if __name__ == '__main__': main()
