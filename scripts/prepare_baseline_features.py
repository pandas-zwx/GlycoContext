"""Frozen RNA-FM native-RNA mean vectors for the tree baseline (no labels)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from argparse import Namespace
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset
from glycocontext.io import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('features_baselines'))
    p.add_argument('--rnafm-repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/preparation.json'))
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); cfg = read_json(a.config)
    if a.output.exists(): raise FileExistsError('Preserve existing cache; use a new directory')
    _, native, _ = read_dataset(a.data)
    if max(map(len, native.values())) > cfg['max_RNA_nt']: raise ValueError('RNA exceeds fixed length limit')
    if sha256(a.checkpoint) != cfg['checkpoint_sha256']: raise ValueError('RNA-FM weight digest mismatch')
    sys.path.insert(0, str(a.rnafm_repo.resolve()))
    import fm
    with torch.serialization.safe_globals([Namespace]): payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model, alphabet = fm.pretrained.load_model_and_alphabet_core('rna-fm', payload, theme='rna'); del payload
    model = model.float().requires_grad_(False).eval().to(a.device)
    converter = alphabet.get_batch_converter(); offset = int(alphabet.prepend_bos)
    torch.set_num_threads(2)
    keys = sorted(native); means = {}
    with torch.inference_mode():
        for key in sorted(native, key=lambda k: (-len(native[k]), k)):
            sequence = native[key]
            _, strings, tokens = converter([(key, sequence)])
            if strings != [sequence] or tokens[0, offset:offset+len(sequence)].tolist() != [alphabet.get_idx(b) for b in sequence]:
                raise ValueError('Token/base alignment mismatch')
            values = model(tokens.to(a.device), repr_layers=[12], return_contacts=False)['representations'][12]
            values = values[0, offset:offset+len(sequence)].float().cpu().numpy()
            if values.shape != (len(sequence), 640) or not np.isfinite(values).all(): raise ValueError('Invalid native tokens')
            means[key] = values.mean(0, dtype=np.float64).astype(np.float32)
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output/'sequence_keys.json', keys)
    np.save(a.output/'rnafm_native_mean.npy', np.stack([means[k] for k in keys]), allow_pickle=False)
    write_json(a.output/'manifest.json', dict(schema='GlycoContext-baseline-features-v1',
        RNAFM_checkpoint_sha256=sha256(a.checkpoint), layer=12, dimension=640,
        native_batch=1, precision='FP32', pooling='Mean over nucleotide tokens only; float64 accumulation, float32 storage',
        data_manifest_sha256=sha256(a.data/'manifest.json'),
        sha256={name: sha256(a.output/name) for name in ('sequence_keys.json', 'rnafm_native_mean.npy')}))
    print('Prepared native RNA-FM means', len(keys))


if __name__ == '__main__': main()
