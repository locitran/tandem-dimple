import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run

acc = 'Q46897'

DV6b_query = [
    'Q46897 9 A K',
    'Q46897 10 R K',
    'Q46897 43 K R',
    'Q46897 60 P A',
    'Q46897 63 T S',
    'Q46897 78 L F',
    'Q46897 140 H K',
]

run(
    query=DV6b_query,
    job_name=f'CasE/May21/DV6b_query',
    custom_PDB='4DZD',
    refresh=True,
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
)

run(
    query=acc,
    job_name=f'CasE/May21/saturation_mutagenesis',
    custom_PDB='4DZD',
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
)