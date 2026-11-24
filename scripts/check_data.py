"""Check portable data hashes, source classes, coordinate roundtrips and splits."""
import argparse
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.data import read_dataset,roles
from glycocontext.io import read_json,read_rows,checked,site_window


def rna_to_genome(index,reference):
    exons=sorted(reference['exons0'],reverse=reference['strand']=='-')
    for a,b in exons:
        if index<b-a:return a+index if reference['strand']=='+' else b-1-index
        index-=b-a
    raise ValueError('RNA position outside reference exon path')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--splits',type=Path,default=Path('splits'))
    a=p.parse_args();records,native,m=read_dataset(a.data);by_id={r['candidate_id']:r for r in records}
    paths=read_json(a.data/'provenance/reference_paths.json');observations=read_rows(a.data/'provenance/source_observations.jsonl.gz')
    if len(observations)!=len(records):raise ValueError('Source records incomplete')
    checked_sources=0
    for item in observations:
        r=by_id[item['candidate_id']];sequence=native[r['sequence_sha256']]
        states={s['raw_observation_state'] for s in item['source_observations']}
        expected='ENRICHED_STORED_ONLY' if r['y']==1 else 'INPUT_STORED_ONLY'
        if states!={expected}:raise ValueError('Source disagreement or proxy-label mismatch')
        for s in item['source_observations']:
            reference=paths[s['RNA_ID']]
            if reference['sequence_sha256']!=r['sequence_sha256'] or len(reference['maps'])!=1:raise ValueError('Unresolved reference')
            ref=reference['maps'][0]
            if (s['RNA_position1']!=r['RNA_position1'] or s['RNA_index0']!=r['rna_index0']
                or s['RNA_base']!=sequence[r['rna_index0']]
                or s['strand']!=ref['strand'] or s['genome_chr']!=ref['chrom']
                or rna_to_genome(r['rna_index0'],ref)!=s['genome_position0']):raise ValueError('Coordinate roundtrip/base mismatch')
            enriched=s['enriched_observation'] is not None;input_=s['input_observation'] is not None
            if (enriched,input_)!=((True,False) if r['y']==1 else (False,True)):raise ValueError('Library-exclusive state mismatch')
            checked_sources+=1
    sm=read_json(a.splits/'manifest.json')
    for name,h in sm['sha256'].items():
        split=read_json(checked(a.splits/name,h));rr=roles(records,split)
        print(name,{k:dict(rows=len(v),RNAs=len({records[i]['sequence_sha256'] for i in v}),groups=len({records[i]['group'] for i in v})) for k,v in rr.items()})
    print('PASS',len(records),'candidates',len(native),'RNAs',checked_sources,'source-coordinate records',dict(Counter((r['cell_name'],r['y']) for r in records)))
    print('Scope: portable-file/source-coordinate consistency; not a new reference-genome, track or chemical validation.')


if __name__=='__main__':main()
