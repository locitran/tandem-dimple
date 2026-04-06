import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run
# uniref90=f'{addpath}/data/consurf/uniref90.fasta'
SAVs = [
    # 'O94761 1105 G D', # 1208, 617 # nopfam
    'O60260 33 R Q', # 465, 384
    'Q9H227 172 M I', # 469, 466
    'O15305 69 P S', # 246, 484
    'Q99720 211 R Q', # 223, 653
    'Q9BYC5 267 T K', # 575, 935
    'Q9H6W3 364 V A', # 641, 942
    'O14958 335 N K', # 399, 1974
    'Q99575 675 E Q', # 1024, 2595
    'Q8TD43 830 R P', # 1214, 3908

    ## 'O14717 101 H Y', # 391, 656
    # # 'Q99720 2 Q P', # 223 # nopfam
    # 'Q9H6W3 239 Q H', # 641 # nopfam
    # # 'Q9UIQ6 730 N S', # 1025 # MEM # pfam
    # # 'O15118 971 V G', # 1278 # MEM # nopfam
]

for s in SAVs:
    acc = s.split()[0]
    job_name = f'execution_time/April6/singleSAVs/{acc}'
    run(query=[s], job_name=job_name, refresh=True, log_time=True,
        uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
    )

import random
aa_list = 'ACDEFGHIKLMNPQRSTVWY'
Q9H227 = "MAFPAGFGWAAATAAYQVEGGWDADGKGPCVWDTFTHQGGERVFKNQTGDVACGSYTLWEEDLKCIKQLGLTHYRFSLSWSRLLPDGTTGFINQKGIDYYNKIIDDLLKNGVTPIVTLYHFDLPQTLEDQGGWLSEAIIESFDKYAQFCFSTFGDRVKQWITINEANVLSVMSYDLGMFPPGIPHFGTGGYQAAHNLIKAHARSWHSYDSLFRKKQKGMVSLSLFAVWLEPADPNSVSDQEAAKRAITFHLDLFAKPIFIDGDYPEVVKSQIASMSQKQGYPSSRLPEFTEEEKKMIKGTADFFAVQYYTTRLIKYQENKKGELGILQDAEIEFFPDPSWKNVDWIYVVPWGVCKLLKYIKDTYNNPVIYITENGFPQSDPAPLDDTQRWEYFRQTFQELFKAIQLDKVNLQVYCAWSLLDNFEWNQGYSSRFGLFHVDFEDPARPRVPYTSAKEYAKIIRNNGLEAHL"
SAVs = [
    f"Q9H227 {i} {wt} {random.choice([aa for aa in aa_list if aa != wt])}"
    for i, wt in enumerate(Q9H227, start=1)
]

n_test = [10, 50, 100, 200, 300, len(SAVs)]
acc = 'Q9H227'
for n in n_test:
    query = SAVs[:n]
    job_name = f'execution_time/April6/multipleSAVs/{acc}/{n}'
    run(query=query, job_name=job_name, refresh=True, log_time=True,
        uniref90='/home/loci/main/tandem_website/tandem/data/consurf/uniref90.fasta'
    )


"""
Conclusion:
For Structure and Dynamics feautures (except delta disulfide bond), calculation is performed once at the protein level rather than separately for each SAV. 
Therefore, the execution time is independent of SAV numbers.


"""