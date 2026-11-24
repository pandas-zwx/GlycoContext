"""Label-free exact/global-edit/local-window overlap screen for new cohorts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset
from glycocontext.io import site_window,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=Path('data'))
    p.add_argument('--query',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-similarity',type=float,default=.8);a=p.parse_args()
    import edlib
    rr,rn,_=read_dataset(a.reference);qr,qn,_=read_dataset(a.query)
    links=[]
    for q,qs in qn.items():
        for r,rs in rn.items():
            maximum=max(len(qs),len(rs));cap=int((1-a.minimum_similarity)*maximum+1e-9)
            if abs(len(qs)-len(rs))>cap:continue
            d=edlib.align(qs,rs,mode='NW',task='distance',k=cap)['editDistance']
            if d>=0:links.append(dict(query=q,reference=r,edit_distance=d,similarity=1-d/maximum,exact=q==r))
    windows={}
    for r in rr:
        w=site_window(rn[r['sequence_sha256']],r['rna_index0']);key=(w['sequence'],tuple(w['valid_mask']))
        windows.setdefault(key,set()).add(r['sequence_sha256'])
    shared=[]
    for q in qr:
        w=site_window(qn[q['sequence_sha256']],q['rna_index0']);key=(w['sequence'],tuple(w['valid_mask']))
        if key in windows:shared.append(dict(query_candidate=q['candidate_id'],reference_RNAs=sorted(windows[key])))
    if a.output.exists():raise FileExistsError('Preserve earlier overlap screen')
    write_json(a.output,dict(sequence_links=links,identical_candidate_windows=shared,labels_used_for_matching=False,
        limits='Screens sequence/global-edit and exact51nt-mask overlap only. Study provenance, genomic/exon overlap, source IDs and biological label/assay compatibility require separate review. This script does not silently delete records.'))
    print('RNA similarity links:',len(links),'shared candidate windows:',len(shared))


if __name__=='__main__':main()
