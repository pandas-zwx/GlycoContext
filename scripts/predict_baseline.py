"""Predict with a trusted saved baseline and fixed C-derived thresholds."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset, roles
from glycocontext.io import read_json
from glycocontext.baselines import rf_predict, tree_inputs, assemble_rnafm


def main():
    import joblib
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--features', type=Path, default=Path('features'))
    p.add_argument('--mean-features', type=Path, default=Path('features_baselines'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', type=Path); p.add_argument('--role', choices=['E', 'S', 'C', 'test'], default='test')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); records, native, _ = read_dataset(a.data)
    take = roles(records, read_json(a.split))[a.role] if a.split else np.arange(len(records))
    if not len(take): raise ValueError('Empty requested role')
    records = [records[i] for i in take]
    bundle = joblib.load(a.checkpoint)  # Only load files from a trusted source.
    if bundle['schema'] != 'GlycoContext-baseline-v1': raise ValueError('Unsupported checkpoint')
    if bundle['method'] == 'rf': probability = rf_predict(bundle, records, native)
    else:
        import lightgbm as lgb
        if bundle['method'] != 'rnafm_lgbm': raise ValueError('Unsupported baseline')
        if lgb.__version__ != bundle['recipe']['LightGBM']['version']: raise ValueError('LightGBM version mismatch')
        inputs = tree_inputs(a.features, a.mean_features, records, native)
        x = assemble_rnafm(*inputs, bundle['preprocessing'])
        probability = lgb.Booster(model_str=bundle['model_text']).predict(x, num_threads=bundle['recipe']['threads'])
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)): raise ValueError('Invalid probability')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'cell', 'RNA_position1', 'probability', 'threshold', 'prediction', 'seed', 'ablation'])
        writer.writeheader()
        for row, value in zip(records, probability):
            threshold = float(bundle['thresholds'][str(row['cell'])])
            writer.writerow(dict(candidate_id=row['candidate_id'], cell=['HeLa', 'H9'][row['cell']],
                RNA_position1=row['RNA_position1'], probability=float(value), threshold=threshold,
                prediction=int(value >= threshold), seed=bundle['seed'], ablation=bundle['method']))


if __name__ == '__main__': main()
