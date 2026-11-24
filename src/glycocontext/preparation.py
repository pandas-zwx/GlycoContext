from pathlib import Path
import numpy as np
from .features import graph_arrays,cnn_window
from .io import write_json,sha256


def write_feature_bank(directory,records,native,structures,views,provenance):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    keys=sorted(native);maximum=max(map(len,native.values()));n=len(keys)
    if set(structures)!=set(keys):raise ValueError('Structure scope')
    if views.shape!=(len(records),3,640) or views.dtype!=np.float32 or not np.isfinite(views).all():raise ValueError('RNA-FM view shape/value')
    arrays=dict(nodes=np.zeros((n,maximum,8),dtype=np.float32),neighbours=np.zeros((n,maximum,3),dtype=np.int64),
        masks=np.zeros((n,maximum,3),dtype=np.float32),edges=np.zeros((n,maximum,3,7),dtype=np.float32),
        confidence=np.zeros((n,maximum),dtype=np.float32))
    partners={}
    for j,key in enumerate(keys):
        seq=native[key];s=structures[key]
        if s['sequence_sha256']!=key:raise ValueError('Structure identity')
        values=graph_arrays(seq,s['structure'],s['probabilities']);partners[key]=values[-1]
        for name,value in zip(arrays,values[:5]):arrays[name][j,:len(seq)]=value
    cnn=np.stack([cnn_window(native[r['sequence_sha256']],r['rna_index0'],partners[r['sequence_sha256']]) for r in records])
    np.save(directory/'cnn.npy',cnn,allow_pickle=False);np.save(directory/'rnafm_views.npy',views,allow_pickle=False)
    np.savez_compressed(directory/'graphs.npz',keys=np.array(keys),**arrays)
    write_json(directory/'candidate_ids.json',[r['candidate_id'] for r in records])
    write_json(directory/'manifest.json',dict(schema='GlycoContext-features-v1',records=len(records),RNAs=len(keys),
        RNAFM_dimension=640,RNAFM_views=['native','crop51','crop101'],provenance=provenance,
        sha256={name:sha256(directory/name) for name in ('cnn.npy','rnafm_views.npy','graphs.npz','candidate_ids.json')}))
