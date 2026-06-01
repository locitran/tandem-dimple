import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
import pandas as pd 
from src.main import run

file = os.path.join(addpath, "data/GJB2/SAVs_20260509.csv")
df = pd.read_csv(file)
SAVs = df.SAVs.to_list()

td = run(
    query=SAVs, # List of SAVs to be analyzed
    pretrained_model_folder=os.path.join(addpath, "models/TANDEM_GJB2"),
    custom_PDB=os.path.join(addpath, "data/GJB2/structures/8qa2_opm_25Apr03.pdb"),
    job_name='GJB2/SAVs_20260509', # Define where the job will be saved
    refresh=False, # Set to True to refresh the calculation
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   