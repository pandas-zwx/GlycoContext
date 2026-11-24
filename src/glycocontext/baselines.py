"""Portable sequence RF and single-encoder RNA-FM/LightGBM comparators."""
import hashlib
import itertools
from pathlib import Path
import numpy as np
from .io import site_window, read_json, checked

BASES = 'ACGU'
WORDS = {k: [''.join(x) for x in itertools.product(BASES, repeat=k)] for k in (2, 3, 4)}


def encode_windows(records, native):
    """PSDSP word codes, 51nt one-hot and central17nt k2/3/4 frequencies."""
    onehot, knf, codes = [], [], []
    pairs = {w: i for i, w in enumerate(WORDS[2])}
    for row in records:
        w = site_window(native[row['sequence_sha256']], row['rna_index0'])
        sequence, valid = w['sequence'], w['valid_mask']
        onehot.append([float(ok and b == a) for b, ok in zip(sequence, valid) for a in BASES])
        codes.append([pairs.get(sequence[i:i+2], -1) if all(valid[i:i+2]) else -1 for i in range(50)])
        central, mask = sequence[17:34], valid[17:34]
        frequencies = []
        for k in (2, 3, 4):
            lookup = {word: j for j, word in enumerate(WORDS[k])}
            counts = np.zeros(len(lookup))
            for i in range(17-k+1):
                word = central[i:i+k]
                if all(mask[i:i+k]) and word in lookup: counts[lookup[word]] += 1
            frequencies.extend(counts/max(float(counts.sum()), 1.))
        knf.append(frequencies)
    return (np.asarray(onehot, dtype=np.float32), np.asarray(knf, dtype=np.float32),
            np.asarray(codes, dtype=np.int64))


class PositionDinucleotideProfile:
    """Label-derived position profiles; remove the whole own group for fit rows."""
    def fit(self, codes, labels, groups):
        self.counts = np.zeros((2, 50, 16), dtype=np.float64)
        self.by_group = {}
        if codes.shape != (len(labels), 50) or len(groups) != len(labels): raise ValueError('Profile alignment')
        for words, label, group in zip(codes, labels, groups):
            if int(label) not in (0, 1): raise ValueError('Binary labels required')
            key = str(group)
            if key not in self.by_group: self.by_group[key] = np.zeros_like(self.counts)
            positions = np.flatnonzero(words >= 0)
            self.counts[int(label), positions, words[positions]] += 1
            self.by_group[key][int(label), positions, words[positions]] += 1
        return self

    @staticmethod
    def profile(counts):
        frequencies = counts/np.maximum(counts.sum(axis=2, keepdims=True), 1.)
        return frequencies[1]-frequencies[0]

    def transform(self, codes, groups=None):
        result = np.zeros(codes.shape, dtype=np.float32)
        global_profile, cache = self.profile(self.counts), {}
        for i, words in enumerate(codes):
            profile = global_profile
            if groups is not None:
                key = str(groups[i])
                if key not in self.by_group: raise ValueError('Unknown fitting group')
                if key not in cache: cache[key] = self.profile(self.counts-self.by_group[key])
                profile = cache[key]
            at = np.flatnonzero(words >= 0)
            result[i, at] = profile[at, words[at]]
        return result


def fusion_grid():
    return [(a/10, b/10, (10-a-b)/10) for a in range(11) for b in range(11-a)] + [(1/3, 1/3, 1/3)]


def inner_folds(groups, partition, folds=5):
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(
        f'publication_RF_inner|{partition}|{g}'.encode()).digest())
    if len(unique) < folds: raise ValueError('Insufficient groups for fixed RF inner folds')
    assignment = {g: j % folds for j, g in enumerate(unique)}
    return np.array([assignment[g] for g in groups])


def class_weights(y, cells):
    result = np.ones(len(y), dtype=float)
    for c in (0, 1):
        mask = cells == c
        positive, negative = sum(y[mask] == 1), sum(y[mask] == 0)
        if not positive or not negative: raise ValueError('Both classes required per fitting cell')
        result[mask & (y == 1)] = negative/positive
    return result


def assemble_rnafm(views, means, lengths, indices, cells, processing):
    views, means = np.asarray(views), np.asarray(means)
    lengths, indices, cells = map(np.asarray, (lengths, indices, cells))
    n = len(lengths)
    if (views.shape != (n, 3, 640) or means.shape != (n, 640)
        or indices.shape != (n,) or cells.shape != (n,) or np.any(lengths < 1)
        or np.any(indices < 0) or np.any(indices >= lengths) or not np.isin(cells, [0, 1]).all()
        or processing not in ('raw', 'token_LN') or not np.isfinite(views).all()
        or not np.isfinite(means).all()): raise ValueError('Invalid RNA-FM features')
    if processing == 'token_LN':
        def ln(x):
            x = np.asarray(x, dtype=np.float64)
            return ((x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+1e-5)).astype(np.float32)
        views, means = ln(views), ln(means)
    metadata = np.column_stack([np.log1p(lengths), indices/np.maximum(lengths-1, 1), np.eye(2)[cells]])
    result = np.column_stack([views.reshape(n, -1), means, metadata]).astype(np.float32)
    if result.shape != (n, 2564) or not np.isfinite(result).all(): raise ValueError('Tree feature dimension/value')
    return result


def tree_inputs(features, mean_directory, records, native):
    features, mean_directory = Path(features), Path(mean_directory)
    m = read_json(features/'manifest.json')
    for name in ('candidate_ids.json', 'rnafm_views.npy'): checked(features/name, m['sha256'][name])
    ids = read_json(features/'candidate_ids.json')
    lookup = {v: i for i, v in enumerate(ids)}
    if len(lookup) != len(ids): raise ValueError('Duplicate feature candidate ID')
    order = [lookup[r['candidate_id']] for r in records]
    views = np.load(features/'rnafm_views.npy', allow_pickle=False)[order]
    m = read_json(mean_directory/'manifest.json')
    if m.get('RNAFM_checkpoint_sha256') != '5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99':
        raise ValueError('Unexpected RNA-FM mean checkpoint')
    for name, digest in m['sha256'].items(): checked(mean_directory/name, digest)
    keys = read_json(mean_directory/'sequence_keys.json')
    means = np.load(mean_directory/'rnafm_native_mean.npy', allow_pickle=False)
    if len(set(keys)) != len(keys) or means.shape != (len(keys), 640): raise ValueError('Native mean identity')
    lookup = {key: j for j, key in enumerate(keys)}
    means = np.stack([means[lookup[r['sequence_sha256']]] for r in records])
    return views, means, np.array([len(native[r['sequence_sha256']]) for r in records]), np.array(
        [r['rna_index0'] for r in records]), np.array([r['cell'] for r in records])


def rf_predict(bundle, records, native):
    onehot, knf, codes = encode_windows(records, native)
    cells = np.array([r['cell'] for r in records]); result = np.empty(len(records))
    for c in (0, 1):
        take = cells == c
        if not take.any(): continue
        head = bundle['heads'][str(c)]
        blocks = [head['profile'].transform(codes[take]), onehot[take], knf[take]]
        predictions = np.stack([model.predict_proba(x)[:, 1] for model, x in zip(head['models'], blocks)])
        result[take] = np.array(head['weights']) @ predictions
    return result
