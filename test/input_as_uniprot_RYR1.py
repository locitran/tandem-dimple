import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath)

from tandem.src.main import tandem_dimple

query = 'P21817'

td = tandem_dimple(
    query=query, # List of SAVs to be analyzed
    job_name='input_as_uniprot_RYR1', # Define where the job will be saved
    custom_PDB='7M6L', # Path to the custom PDB file (if any)
    refresh=False, # Set to True to refresh the data
    )