"""Run the prespecified development-only incremental component path."""
import argparse
import subprocess
import sys
from pathlib import Path


ARMS = ('inc_cnn_seq','inc_cnn_structure','inc_rnafm','inc_gnn_backbone')
PARTITIONS = ('dev0','dev1','dev2')
SEEDS = (0,1,2,3,4)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=Path('runs/development/incremental'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/run_protocol.py'),
        '--arms',*ARMS,'--seeds',*(str(v) for v in SEEDS),'--partitions',*PARTITIONS,
        '--output',str(args.output),'--device',args.device],cwd=root,check=True)


if __name__=='__main__':main()
