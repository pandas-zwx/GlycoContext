"""Import user-supplied RNA-position candidates; never invent negative labels."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.io import sequence_key,write_rows,write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--fasta',type=Path,required=True)
    p.add_argument('--sites',type=Path,required=True,help='TSV: RNA_ID, RNA_position1, cell; optional label0/1')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    native={};name=None
    for line in a.fasta.read_text().splitlines():
        if line.startswith('>'):
            name=line[1:].split()[0]
            if name in native:raise ValueError('Duplicate FASTA ID')
            native[name]=''
        elif line.strip():
            if name is None:raise ValueError('Sequence before FASTA header')
            native[name]+=line.strip().upper().replace('T','U')
    keys={name:sequence_key(s) for name,s in native.items()};records=[];seen=set()
    with a.sites.open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['cell'] not in ['HeLa','H9']:raise ValueError('Unknown cell: no automatic mapping or transfer claim')
            seq=native[row['RNA_ID']];position=int(row['RNA_position1']);key=keys[row['RNA_ID']]
            if not 1<=position<=len(seq):raise ValueError('RNA position outside provided sequence')
            identity=(row['cell'],key,position)
            if identity in seen:raise ValueError('Duplicate sequence-position-cell')
            seen.add(identity)
            r=dict(candidate_id=hashlib.sha256(f'GlycoContext|{row["cell"]}|{key}|{position}'.encode()).hexdigest(),
                sequence_sha256=key,cell=['HeLa','H9'].index(row['cell']),cell_name=row['cell'],
                rna_index0=position-1,RNA_position1=position,center_base=seq[position-1],group=key)
            if row.get('label','')!='':
                label=int(row['label'])
                if label not in (0,1):raise ValueError('Label must be0/1; leave unobserved labels absent')
                r['y']=label
            records.append(r)
    used={r['sequence_sha256'] for r in records};seqs={keys[k]:v for k,v in native.items() if keys[k] in used}
    a.output.mkdir(parents=True,exist_ok=False)
    write_rows(a.output/'sites.jsonl.gz',records)
    write_rows(a.output/'sequences.jsonl.gz',[dict(sequence_sha256=k,sequence=v) for k,v in sorted(seqs.items())])
    write_json(a.output/'manifest.json',dict(schema='GlycoContext-data-v1',candidate_count=len(records),unique_RNAs=len(seqs),
        imported_labels_are_user_supplied=True,grouping='Exact RNA identity only; not a new training split.',
        sha256={n:sha256(a.output/n) for n in ['sites.jsonl.gz','sequences.jsonl.gz']}))


if __name__=='__main__':main()
