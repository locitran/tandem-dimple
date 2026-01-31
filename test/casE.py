import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, addpath) # /home/newloci

from tandem.src.main import run

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
    job_name='CasE_DV1',
    refresh=False,
)

td_DV2 = run(
    query=DV2_query,
    job_name='CasE_DV2',
    refresh=False,
)

td_DV6b = run(
    query=DV6b_query,
    job_name='CasE_DV6b',
    refresh=False,
)

