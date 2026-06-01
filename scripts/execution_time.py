import os
import sys
import random
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)
from src.main import run
from src.features.UniProt_API import searchUniprot
from src.features.Uniprot import mapSAVs2PDB

proteins = ['Q9H227', 'O94761', 'O60260', 'Q9H227', 'O15305', 'Q99720', 'Q9BYC5', 'Q9H6W3', 'O14958', 'Q99575', 'Q8TD43', 'O14717', 'Q9UIQ6', 'O15118']
aa_list = 'ACDEFGHIKLMNPQRSTVWY'
for acc in proteins:
    u = searchUniprot(acc)
    s = u.getSequence()
    SAVs = [
        f"{acc} {i} {wt} {random.choice([aa for aa in aa_list if aa != wt])}"
        for i, wt in enumerate(s, start=1)
    ]
    
    allSAVs = [
        f"{acc} {i} {wt} {aa}"
        for i, wt in enumerate(s, start=1)
        for aa in aa_list
        if aa != wt
    ]

    mapped_SAVs, custom_PDB = mapSAVs2PDB(SAVs)
    mapped_allSAVs, custom_PDB = mapSAVs2PDB(allSAVs)

    # I want to extract the pdb id that has the most SAVs mapped to it.
    pdbid_counts = {}
    for asu in mapped_allSAVs['Asymmetric_PDB_coords']:
        if "Cannot map" in asu:
            continue
        pdbid = asu.split()[0]
        if pdbid not in pdbid_counts:
            pdbid_counts[pdbid] = 0
        pdbid_counts[pdbid] += 1

    pdbid_the_most = max(pdbid_counts, key=pdbid_counts.get)

    for i in range(len(mapped_allSAVs)):
        if mapped_allSAVs[i]['is_alphafold']:
            mapped_allSAVs[i]['Asymmetric_PDB_coords'] = "Cannot map"
        if pdbid_the_most not in mapped_allSAVs[i]['Asymmetric_PDB_coords']:
            mapped_allSAVs[i]['Asymmetric_PDB_coords'] = "Cannot map"

    nValidSAVs = pdbid_counts[pdbid_the_most]

    # Extract SAVs that does not have "Cannot map" in their 'Asymmetric_PDB_coords' and save them in a list.
    valid_SAVs = []
    for i in range(len(mapped_allSAVs)):
        if "Cannot map" not in mapped_allSAVs[i]['Asymmetric_PDB_coords']:
            valid_SAVs.append(mapped_allSAVs[i]['SAV_coords'])

    _1SAV = [random.choice(valid_SAVs)]
            
    td = run(
        query=_1SAV, # List of SAVs to be analyzed
        job_name=f'execution_time/April30/{acc}_1',
        refresh=True, # Set to True to refresh the calculation
        uniref90='/tandem/data/consurf/uniref90.fasta', # 
        log_time=True,
    )  
    td = run(
        query=valid_SAVs, # List of SAVs to be analyzed
        job_name=f'execution_time/April30/{acc}_{nValidSAVs}',
        refresh=True, # Set to True to refresh the calculation
        uniref90='/tandem/data/consurf/uniref90.fasta', # 
        log_time=True,
    )  

"""
Conclusion:
For Structure and Dynamics feautures (except delta disulfide bond), calculation is performed once at the protein level rather than separately for each SAV. 
Therefore, the execution time is independent of SAV numbers.

"""