import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run

from src.features.Uniprot import mapSAVs2PDB, SAV2SAV_coord

SAVs = ["Q8TDI8 S2P",
"Q8TDI8 K4Q",
"Q8TDI8 I8V",]
SAVs = SAV2SAV_coord(SAVs)

td = run(
    query=SAVs, # List of SAVs to be analyzed
    job_name='test/test_website', # Define where the job will be saved
    refresh=True, # Set to True to refresh the calculation
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
