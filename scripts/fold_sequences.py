"""Sequence-only RNAfold worker; supports a separate ViennaRNA Python."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from glycocontext.io import read_json,read_rows,write_json,checked,sequence_key
from glycocontext.features import fold


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=read_json(a.data/'manifest.json')
    seqs=read_rows(checked(a.data/'sequences.jsonl.gz',m['sha256']['sequences.jsonl.gz']))
    if a.output.exists():raise FileExistsError('Do not overwrite folds')
    structures={}
    for r in seqs:
        key=sequence_key(r['sequence'])
        if key!=r['sequence_sha256']:raise ValueError('Sequence identity')
        structures[key]=fold(r['sequence'])
    write_json(a.output,structures)


if __name__=='__main__':main()
