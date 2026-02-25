import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import run
# from src.main import run

sav_list = ["P29033 170 N K"]
td = run(
    query=sav_list, # List of SAVs to be analyzed
    job_name='test_GJB2', # Define where the job will be saved
    refresh=True, # Set to True to refresh the calculation
    custom_PDB="/home/loci/main/tandem_website_dev/tandem/data/GJB2/structures/8qa2_opm_25Apr03.pdb",
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
