import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def now(): return datetime.now(timezone.utc).isoformat()


def read_json(path): return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f: return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f, gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as z:
        for row in rows: z.write((json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def checked(path, expected):
    path = Path(path)
    if sha256(path) != expected: raise ValueError('SHA256 mismatch: '+str(path))
    return path


def sequence_key(sequence):
    if not sequence or set(sequence)-set('ACGUN') or sequence != sequence.upper():
        raise ValueError('Require normalized nonempty A/C/G/U/N RNA sequence')
    return hashlib.sha256(sequence.encode()).hexdigest()


def site_window(sequence, index0, radius=25):
    if not isinstance(index0, int) or not 0 <= index0 < len(sequence): raise ValueError('Invalid RNA index0')
    indices = range(index0-radius, index0+radius+1)
    valid = [0 <= i < len(sequence) for i in indices]
    return dict(sequence=''.join(sequence[i] if ok else 'N' for i, ok in zip(indices, valid)),
        valid_mask=valid, center_index0=radius, parent_site_index0=index0, parent_site_base=sequence[index0])
