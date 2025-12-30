import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import run

query = ['O14508 52 S N', 'P29033 217 Y D']
query = ['P98161 1047 S L']
# query = ['P29033 76 L S']

td = run(
    query=query, # List of SAVs to be analyzed
    job_name='PKD1_test', # Define where the job will be saved
    custom_PDB='/home/loci/tandem_website/tandem/data/PKD1/pkd_full_final_fix_OPM_con.pdb', # Path to the custom PDB file (if any)
    refresh=False, # Set to True to refresh the calculation
    uniref90='/home/loci/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
