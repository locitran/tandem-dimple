import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import run

query = ['O14508 52 S N', 'P29033 217 Y D']
query = ['P29033 100 H Q']
# query = ['P29033 76 L S']

td = run(
    query=query, # List of SAVs to be analyzed
    job_name='test1', # Define where the job will be saved
    custom_PDB='2ZW3', # Path to the custom PDB file (if any)
    refresh=False, # Set to True to refresh the calculation
)   
