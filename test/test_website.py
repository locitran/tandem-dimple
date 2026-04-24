import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run
from src.features.Uniprot import SAV2SAV_coord

SAVs = ["O00189 271 R H", "O00194 138 P L", "O00194 92 A T", "O00204 240 V I", "O00204 51 L S", "O00206 175 T A", "O00206 188 Q R", "O00206 246 C S",]

# Test error
SAVs = [
"Q8TDI8 S2P",
"Q8TDI8 K4Q",
"Q8TDI8 I8V",
"Q8TDI8 I8N",
"O00255 R176Q",
"O00255 D177Y",
"Q9P2D1 Y72C ",
"Q9P2D1 P86R",]

td = run(
    query=SAV2SAV_coord(SAVs), # List of SAVs to be analyzed
    job_name='test/test_website', # Define where the job will be saved
    refresh=True, # Set to True to refresh the calculation
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)   
