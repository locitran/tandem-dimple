import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run

case_wt_complete = "/home/loci/main/tandem_website_dev/tandem/data/casE/case_wt_complete.pdb"
_4tvx_chainA_wRNA = '/home/loci/main/tandem_website_dev/tandem/data/casE/4tvx_chainA_wRNA.pdb'
DV1_query = [
    'Q46897 9 A K',
    'Q46897 43 K R',
    'Q46897 60 P A',
    'Q46897 63 T S',
    'Q46897 78 L F',
]

DV2_query = [
    'Q46897 10 R K',
    'Q46897 16 L T',
    'Q46897 43 K R',
    'Q46897 60 P A',
    'Q46897 63 T S',
    'Q46897 68 V T',
    'Q46897 72 K R',
    'Q46897 78 L F',
    'Q46897 85 Y V',
]

DV6b_query = [
    'Q46897 9 A K',
    'Q46897 10 R K',
    'Q46897 43 K R',
    'Q46897 60 P A',
    'Q46897 63 T S',
    'Q46897 78 L F',
    'Q46897 140 H K',
]

td_DV1 = run(
    query=DV1_query,
    job_name='CasE-Mar24/CasE_DV1_4TVX',
    custom_PDB='4TVX',
    refresh=False,
    uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
)

# td_DV2 = run(
#     query=DV2_query,
#     job_name='CasE_DV2_4DZD',
#     custom_PDB='4DZD',
#     refresh=True,
#     uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
# )

# td_DV6b = run(
#     query=DV6b_query,
#     job_name='CasE_DV6b_4DZD',
#     custom_PDB='4DZD',
#     refresh=True,
#     uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta' # 
# )

