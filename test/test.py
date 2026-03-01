import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import run
# from src.main import run

sav_list = [
    "O00189 271 R H",
    "O00194 138 P L",
    "O00194 92 A T",
    "O00204 240 V I",
    "O00204 51 L S",
    "O00206 175 T A",
    "O00206 188 Q R",
    "O00206 246 C S",
    "O00206 287 E D",
    "O00206 287 E G",
    "O00206 306 C W",
]
td = run(
    query=sav_list, # List of SAVs to be analyzed
    job_name='test', # Define where the job will be saved
    refresh=False, # Set to True to refresh the calculation
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
