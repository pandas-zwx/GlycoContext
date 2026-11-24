"""Whole-RNA RNAfold + frozen RNA-FM native/crop51/crop101 center vectors."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from argparse import Namespace
from pathlib import Path
import sys
import subprocess
import tempfile
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset
from glycocontext.features import fold
from glycocontext.io import read_json,sha256,sequence_key,write_json
from glycocontext.preparation import write_feature_bank


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,default=Path('features'))
    p.add_argument('--rnafm-repo',type=Path,help='Official RNA-FM checkout, if not already importable')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fold-python',type=Path,help='Optional separate Python with ViennaRNA2.7.2')
    p.add_argument('--config',type=Path,default=Path('configs/preparation.json'));p.add_argument('--device',default='cuda')
    a=p.parse_args();cfg=read_json(a.config)
    if a.output.exists():raise FileExistsError('Preserve completed/partial features; use a new output directory')
    records,native,_=read_dataset(a.data)
    if max(map(len,native.values()))>cfg['max_RNA_nt']:raise ValueError('RNA exceeds1022nt: no silent truncation or unvalidated window stitching')
    if sha256(a.checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('RNA-FM weight digest mismatch')
    if a.rnafm_repo:sys.path.insert(0,str(a.rnafm_repo.resolve()))
    import fm
    with torch.serialization.safe_globals([Namespace]):payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model,alphabet=fm.pretrained.load_model_and_alphabet_core('rna-fm',payload,theme='rna');del payload
    model=model.float().requires_grad_(False).eval().to(a.device);converter=alphabet.get_batch_converter();offset=int(alphabet.prepend_bos)
    torch.set_num_threads(2)
    @torch.inference_mode()
    def encode(sequences):
        _,strings,tokens=converter([(str(i),s) for i,s in enumerate(sequences)])
        if strings!=sequences:raise ValueError('Tokenizer changed input')
        for i,s in enumerate(sequences):
            if tokens[i,offset:offset+len(s)].tolist()!=[alphabet.get_idx(b) for b in s]:raise ValueError('Token/base alignment')
        output=model(tokens.to(a.device),repr_layers=[12],return_contacts=False)['representations'][12].float()
        result=[output[i,offset:offset+len(s)].cpu().numpy().copy() for i,s in enumerate(sequences)]
        if any(x.shape!=(len(s),640) or not np.isfinite(x).all() for s,x in zip(sequences,result)):raise ValueError('RNA-FM output')
        return result
    full={}
    for key in sorted(native,key=lambda k:(-len(native[k]),k)):full[key]=encode([native[key]])[0]
    views=np.full((len(records),3,640),np.nan,dtype=np.float32);jobs={};bindings={}
    for j,r in enumerate(records):
        sequence=native[r['sequence_sha256']];center=r['rna_index0'];views[j,0]=full[r['sequence_sha256']][center]
        for col,width in ((1,51),(2,101)):
            left,right=max(0,center-width//2),min(len(sequence),center+width//2+1)
            if left==0 and right==len(sequence):views[j,col]=full[r['sequence_sha256']][center]
            else:
                fragment=sequence[left:right];key=sequence_key(fragment);jobs[key]=fragment
                bindings.setdefault(key,[]).append((j,col,center-left))
    order=sorted(jobs,key=lambda k:(-len(jobs[k]),k))
    for start in range(0,len(order),16):
        keys=order[start:start+16]
        for key,value in zip(keys,encode([jobs[k] for k in keys])):
            for row,col,index in bindings[key]:views[row,col]=value[index]
    if a.fold_python:
        with tempfile.TemporaryDirectory(prefix='glycocontext-fold-') as temporary:
            path=Path(temporary)/'structures.json'
            subprocess.run([str(a.fold_python),str(Path(__file__).with_name('fold_sequences.py')),
                '--data',str(a.data.resolve()),'--output',str(path)],check=True)
            structures=read_json(path)
    else:
        structures={key:fold(sequence) for key,sequence in native.items()}
    write_feature_bank(a.output,records,native,structures,views,dict(RNAFM_checkpoint_sha256=sha256(a.checkpoint),
        RNAFM_layer=12,precision='FP32',native_batch=1,crop_batch=16,ViennaRNA='2.7.2',temperature_C=37,
        data_manifest_sha256=sha256(a.data/'manifest.json')))
    write_json(a.output/'structures.json',structures)
    print('Feature preparation complete',len(records),len(native))


if __name__=='__main__':main()
