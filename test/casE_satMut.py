import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run

acc = 'Q46897'
run(
    query=acc,
    job_name=f'CasE-Mar24/{acc}',
    custom_PDB='4DZD',
    refresh=True,
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
)
