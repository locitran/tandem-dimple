import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run

SAVs = [
    # 'O94761 1105 G D', # 1208, 617 # nopfam
    # 'O60260 33 R Q', # 465, 384
    'O15305 69 P S', # 246, 484
    # 'O14958 335 N K', # 399, 1974
    ## 'O14717 101 H Y', # 391, 656
    # 'Q99720 211 R Q', # 223, 653
    # # 'Q99720 2 Q P', # 223 # nopfam
    # 'Q9H227 172 M I', # 469, 466
    # 'Q9BYC5 267 T K', # 575, 935
    # # 'Q9H6W3 239 Q H', # 641 # nopfam
    # 'Q9H6W3 364 V A', # 641, 942
    # 'Q99575 675 E Q', # 1024, 2595
    # # 'Q9UIQ6 730 N S', # 1025 # MEM # pfam
    # 'Q8TD43 830 R P', # 1214, 3908
    # # 'O15118 971 V G', # 1278 # MEM # nopfam
]

for s in SAVs:
    acc = s.split()[0]
    job_name = f'execution_time/{acc}'
    run(query=[s], job_name=job_name, refresh=True, log_time=True,
        uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
    )