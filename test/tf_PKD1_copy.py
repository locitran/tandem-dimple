import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /home/newloci
os.chdir(addpath)
from src.main import run
from src.features.Uniprot import SAV2SAV_coord
import pandas as pd 

file = '/home/loci/main/tandem_website_dev/tandem/data/PKD1/PKD1-PKD2-SAVs-with-labels.txt'
df = pd.read_csv(file, header=None)

# Convert # P98161 A1227S 0 --> P98161 1227 A S
SAVs = df[0].tolist()
labels = [s[-1] for s in SAVs]
SAVs = [s[:-2] for s in SAVs]
SAVs = SAV2SAV_coord(SAVs)

SAVs = ['P98161 1312 R Q']
td = run(
    query=SAVs, # List of SAVs to be analyzed
    job_name='PKD1_test/test_March05', # Define where the job will be saved
    custom_PDB='/home/loci/main/tandem_website_dev/tandem/data/PKD1/pkd_v20260123.pdb',
    refresh=False, # Set to True to refresh the calculation
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
