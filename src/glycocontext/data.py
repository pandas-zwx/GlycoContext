from pathlib import Path
import numpy as np
import torch
from .io import read_json,read_rows,checked,sequence_key


def read_dataset(directory):
    directory = Path(directory)
    manifest = read_json(directory/'manifest.json')
    for rel,h in manifest['sha256'].items(): checked(directory/rel,h)
    records = read_rows(directory/'sites.jsonl.gz')
    seqrows = read_rows(directory/'sequences.jsonl.gz')
    native = {r['sequence_sha256']:r['sequence'] for r in seqrows}
    if len(native) != len(seqrows): raise ValueError('Duplicate RNA sequence key')
    for k,v in native.items():
        if sequence_key(v) != k: raise ValueError('RNA sequence digest mismatch')
    if len({r['candidate_id'] for r in records}) != len(records): raise ValueError('Duplicate candidate ID')
    for r in records:
        sequence = native[r['sequence_sha256']]
        if (r['cell'] not in (0,1) or not 0 <= r['rna_index0'] < len(sequence)
            or r['RNA_position1'] != r['rna_index0']+1
            or r['center_base'] != sequence[r['rna_index0']]): raise ValueError('Site coordinate/base mismatch')
        if 'y' in r and r['y'] not in (0,1): raise ValueError('Invalid binary proxy label')
    return records,native,manifest


def roles(records,split):
    by_id = {r['candidate_id']:i for i,r in enumerate(records)}
    result = {k:np.array([by_id[v] for v in split[k]],dtype=np.int64) for k in ('E','S','C','test')}
    flat = np.concatenate(list(result.values()))
    if len(set(flat.tolist())) != len(flat): raise ValueError('Candidate crosses roles')
    for j,a in enumerate(result):
        for b in list(result)[j+1:]:
            for field in ('group','sequence_sha256'):
                if {records[i][field] for i in result[a]} & {records[i][field] for i in result[b]}:
                    raise ValueError(field+' crosses '+a+'/'+b)
    result['refit'] = np.sort(np.r_[result['E'],result['S']])
    return result


class FeatureBank:
    def __init__(self,directory,records):
        directory = Path(directory); m = read_json(directory/'manifest.json')
        for rel,h in m['sha256'].items(): checked(directory/rel,h)
        self.ids = np.asarray(read_json(directory/'candidate_ids.json'))
        wanted = [r['candidate_id'] for r in records]
        by_id = {v:i for i,v in enumerate(self.ids)}
        if len(by_id) != len(self.ids) or not set(wanted)<=set(by_id): raise ValueError('Feature candidate identity')
        order = np.array([by_id[v] for v in wanted])
        self.x = torch.from_numpy(np.load(directory/'cnn.npy',allow_pickle=False)[order].copy())
        self.views = torch.from_numpy(np.load(directory/'rnafm_views.npy',allow_pickle=False)[order].copy())
        with np.load(directory/'graphs.npz',allow_pickle=False) as g:
            keys = g['keys'].tolist(); lookup = {k:i for i,k in enumerate(keys)}
            self.nodes = torch.from_numpy(g['nodes'].copy()); self.neighbours = torch.from_numpy(g['neighbours'].copy()).long()
            self.masks = torch.from_numpy(g['masks'].copy()); self.edges = torch.from_numpy(g['edges'].copy())
            self.confidence = torch.from_numpy(g['confidence'].copy())
        self.parent = torch.tensor([lookup[r['sequence_sha256']] for r in records])
        self.centers = torch.tensor([r['rna_index0'] for r in records]); self.cells = torch.tensor([r['cell'] for r in records])
        if self.x.shape != (len(records),8,51) or self.views.shape != (len(records),3,640): raise ValueError('Feature shape')
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.views).all(): raise ValueError('Nonfinite features')
        self.manifest = m

    def args(self,take,device='cuda'):
        p = self.parent[take]
        graph = tuple(v.to(device) for v in [self.nodes[p],self.neighbours[p],self.masks[p],self.centers[take],self.edges[p],self.views[take]])
        return self.x[take].to(device),self.cells[take].to(device),graph,self.confidence[p].to(device)

    def forward(self,model,take,device='cuda'): return model(*self.args(take,device))
