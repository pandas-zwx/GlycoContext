"""Train RF or RNA-FM-only LightGBM; thresholds fitted only on C."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(name, '2')
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.io import read_json, write_json, sha256, now
from glycocontext.data import read_dataset, roles
from glycocontext.baselines import (encode_windows, PositionDinucleotideProfile, fusion_grid,
    inner_folds, class_weights, tree_inputs, assemble_rnafm, rf_predict)
from glycocontext.metrics import mcc_threshold


def main():
    import joblib
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['rf', 'rnafm_lgbm'])
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--features', type=Path, default=Path('features'))
    p.add_argument('--mean-features', type=Path, default=Path('features_baselines'))
    p.add_argument('--split', type=Path, default=Path('splits/final.json'))
    p.add_argument('--recipe', type=Path, default=Path('configs/baselines.json'))
    p.add_argument('--seed', type=int, choices=[0, 1, 2, 3, 4], default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); cfg = read_json(a.recipe)
    records, native, _ = read_dataset(a.data); r = roles(records, read_json(a.split))
    y = np.array([v['y'] for v in records]); cells = np.array([v['cell'] for v in records])
    groups = np.array([v['group'] for v in records]); ids = np.array([v['candidate_id'] for v in records])
    for role in ('E', 'S', 'C', 'refit'):
        for c in (0, 1):
            if set(y[r[role]][cells[r[role]] == c]) != {0, 1}: raise ValueError('Both classes required: '+role)
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='TRAINING', method=a.method, seed=a.seed, started=now(), recipe=cfg,
        split_sha256=sha256(a.split), data_manifest_sha256=sha256(a.data/'manifest.json'), fits=[])
    def save(): write_json(a.output/'training.json', report)
    save()
    bundle = dict(method=a.method, seed=a.seed, schema='GlycoContext-baseline-v1',
        fitting_ids=ids[r['refit']].tolist(), split_sha256=sha256(a.split), recipe=cfg)
    try:
        if a.method == 'rf':
            rc = cfg['RF']; onehot, knf, codes = encode_windows(records, native)
            def fit(fitting, query, leaf, seed):
                if set(groups[fitting]) & set(groups[query]): raise ValueError('RF fitting/query group overlap')
                if set(y[fitting]) != {0, 1}: raise ValueError('Both RF fitting classes required')
                profile = PositionDinucleotideProfile().fit(codes[fitting], y[fitting], groups[fitting])
                xx = [profile.transform(codes[fitting], groups[fitting]), onehot[fitting], knf[fitting]]
                zz = [profile.transform(codes[query]), onehot[query], knf[query]]
                models, predictions = [], []
                for x, z in zip(xx, zz):
                    model = RandomForestClassifier(n_estimators=rc['n_estimators'], max_features=rc['max_features'],
                        min_samples_leaf=leaf, class_weight=rc['class_weight'], n_jobs=cfg['threads'], random_state=seed)
                    model.fit(x, y[fitting]); models.append(model); predictions.append(model.predict_proba(z)[:, 1])
                return profile, models, np.stack(predictions)
            heads = {}
            for c in (0, 1):
                e = r['E'][cells[r['E']] == c]
                assignment = inner_folds(groups[e], a.split.stem, rc['inner_folds']); options = []
                for leaf in rc['min_samples_leaf']:
                    oof = np.full((3, len(e)), np.nan)
                    for fold in range(rc['inner_folds']):
                        take = assignment == fold
                        if set(y[e[take]]) != {0, 1}: raise ValueError('RF inner validation requires both classes')
                        _, _, prediction = fit(e[~take], e[take], leaf, rc['inner_seed']); oof[:, take] = prediction
                    if not np.isfinite(oof).all(): raise ValueError('Incomplete RF OOF')
                    table = [dict(weights=w, AUROC=float(roc_auc_score(y[e], np.array(w) @ oof))) for w in fusion_grid()]
                    best = max(table, key=lambda v: (v['AUROC'], -sum((w-1/3)**2 for w in v['weights']), tuple(v['weights'])))
                    options.append(dict(leaf=leaf, **best))
                best = max(options, key=lambda v: (v['AUROC'], -v['leaf']))
                fitting, query = r['refit'][cells[r['refit']] == c], r['C'][cells[r['C']] == c]
                profile, models, _ = fit(fitting, query, best['leaf'], a.seed)
                heads[str(c)] = dict(profile=profile, models=models, weights=best['weights'])
                report['fits'].append(dict(cell=c, options=options, selected=best,
                    fitting_ids=ids[fitting].tolist(), OOF_selection_ids=ids[e].tolist()))
                save(); print('RF cell completed', c, flush=True)
            bundle['heads'] = heads
            probability = rf_predict(bundle, [records[i] for i in r['C']], native)
        else:
            import lightgbm as lgb
            tc = cfg['LightGBM']
            if lgb.__version__ != tc['version']: raise ValueError('LightGBM version must be '+tc['version'])
            inputs = tree_inputs(a.features, a.mean_features, records, native)
            x = {processing: assemble_rnafm(*inputs, processing) for processing in tc['preprocessing']}
            report['feature_manifest_sha256'] = sha256(a.features/'manifest.json')
            report['mean_feature_manifest_sha256'] = sha256(a.mean_features/'manifest.json')
            best = None
            e, s = r['E'], r['S']
            def score(prediction, _):
                return 'macro_AUROC', float(np.mean([roc_auc_score(y[s][cells[s] == c], prediction[cells[s] == c]) for c in (0, 1)])), True
            for processing in tc['preprocessing']:
                for leaves in tc['num_leaves']:
                    params = dict(tc['params'], num_leaves=leaves, seed=a.seed,
                        feature_fraction_seed=a.seed+1109, data_random_seed=a.seed+2049)
                    fitting = lgb.Dataset(x[processing][e], label=y[e], weight=class_weights(y[e], cells[e]))
                    valid = lgb.Dataset(x[processing][s], label=y[s], reference=fitting)
                    model = lgb.train(params, fitting, num_boost_round=tc['max_rounds'], valid_sets=[valid], feval=score,
                        callbacks=[lgb.early_stopping(tc['early_stopping_rounds'], first_metric_only=True, verbose=False)])
                    rounds = int(model.best_iteration)
                    value = score(model.predict(x[processing][s], num_iteration=rounds, num_threads=cfg['threads']), None)[1]
                    choice = dict(processing=processing, leaves=leaves, params=params, rounds=rounds, score=value)
                    report['fits'].append(choice)
                    if best is None or value > best['score']: best = choice
                    save(); print('LightGBM selector completed', processing, leaves, rounds, flush=True)
            fitting = lgb.Dataset(x[best['processing']][r['refit']], label=y[r['refit']],
                weight=class_weights(y[r['refit']], cells[r['refit']]))
            model = lgb.train(best['params'], fitting, num_boost_round=best['rounds'])
            probability = model.predict(x[best['processing']][r['C']], num_threads=cfg['threads'])
            # Store text, not a pickle of the native LightGBM runtime.
            bundle.update(model_text=model.model_to_string(), preprocessing=best['processing'])
            report['selected'] = best
        bundle['thresholds'] = {str(c): mcc_threshold(y[r['C']][cells[r['C']] == c], probability[cells[r['C']] == c]) for c in (0, 1)}
        bundle['threshold_ids'] = ids[r['C']].tolist()
        path = a.output/'model.joblib'; joblib.dump(bundle, path)
        clone = joblib.load(path)
        if a.method == 'rf': reloaded = rf_predict(clone, [records[i] for i in r['C']], native)
        else:
            reloaded = lgb.Booster(model_str=clone['model_text']).predict(
                x[clone['preprocessing']][r['C']], num_threads=cfg['threads'])
        error = float(np.max(np.abs(reloaded-probability)))
        if error > 1e-12: raise ValueError('Saved-model prediction mismatch')
        report.update(status='COMPLETE', completed=now(), thresholds=bundle['thresholds'],
            checkpoint='model.joblib', checkpoint_sha256=sha256(path), reload_maxabs=error)
        save()
    except Exception as exc:
        report.update(status='FAILED_PRESERVE_PARTIAL', error=repr(exc), updated=now()); save(); raise


if __name__ == '__main__': main()
