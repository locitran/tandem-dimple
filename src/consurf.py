import prody
from prody.measure.contacts import findNeighbors
from Bio.Align import PairwiseAligner

import json
import os
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

from .download import get_content
from .LociFixer import one2three
from .utils.settings import ROOT_DIR

CONSURFDB_URL = 'https://consurfdb.tau.ac.il/'

_LOGGER = logging.getLogger(__name__)
workingDir = Path.cwd()
pdbDir = Path(ROOT_DIR) / 'pdbfile/raw'
consurfDir = Path(ROOT_DIR) / 'consurf'
dataDir = Path(consurfDir) / 'data'
alnDir = Path(consurfDir) / 'alignments'
consurfLookup = Path(consurfDir) / '2024-10-08.json'
map_idxDir = Path(consurfDir) / 'map_idx'

with open(consurfLookup) as f:
    consurfLookup = json.load(f)

MATCH_SCORE = 1.0
MISMATCH_SCORE = 0.0
GAP_PENALTY = -1.
GAP_EXT_PENALTY = -0.1

def mapIndices(targetSeq, querySeq, targetName='target', queryName='query'):
    """
    Map the indices of the target and query sequences

    Args:
        targetSeq (str): target sequence
        querySeq (str): query sequence
        targetName (str, optional): target name. Defaults to None.
        queryName (str, optional): query name. Defaults to None.

    Returns:
        np.array: target indices
        np.array: query indices

    Gap is represented by -1, and the indices start from 0

    Example:
    >>> mapIndices('ACGT', 'ACGT')
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> mapIndices('ACGT', 'CGTA', 'target', 'query')
    (array([ 0,  1,  2,  3, -1]), array([-1,  0,  1,  2,  3]))
    
    target-query.aln
    -----------------
    target indices      0123-
    target            0 ACGT- 4
                      0 -|||- 5
    query             0 -CGTA 4
    query indices       -0123
    """
    map_idxFile = os.path.join(map_idxDir, f'{targetName}-{queryName}.pkl')
    alnFile = os.path.join(alnDir, f'{targetName}-{queryName}.aln')

    if os.path.exists(map_idxFile):
        with open(map_idxFile, 'rb') as f:
            map_pkl = pickle.load(f)
        return np.array(map_pkl['target']), np.array(map_pkl['query'])

    aln = align2Sequence(targetSeq, querySeq, fileName=alnFile)
    try:
        target_seq, align_dash, query_seq, _ = aln.split('\n')
    except:
        split_aln = aln.split('\n')
        target_seq = '' ; query_seq = ''
        for i, p in enumerate(split_aln):
            p_split = p.split()
            target_seq += p_split[2] if i % 4 == 0 else ''
            query_seq += p_split[2] if i % 4 == 2 else ''

    # Get the indices of the target and query sequences correspondingly
    target_indices = [] ; idx = 0
    for i in range(len(target_seq)):
        if target_seq[i] != '-':
            target_indices.append(idx)
            idx += 1
        else:
            target_indices.append(-1)

    query_indices = [] ; idx = 0
    for i in range(len(query_seq)):
        if query_seq[i] != '-':
            query_indices.append(idx)
            idx += 1
        else:
            query_indices.append(-1)

    map_pkl = {'target': target_indices, 'query': query_indices}
    with open(map_idxFile, 'wb') as f:
        pickle.dump(map_pkl, f)
    return np.array(target_indices), np.array(query_indices)

def _align(target, query) -> str:
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = MATCH_SCORE
    aligner.mismatch_score = MISMATCH_SCORE
    aligner.internal_open_gap_score = GAP_PENALTY
    aligner.internal_extend_gap_score = GAP_EXT_PENALTY
    alns = aligner.align(target, query)
    for i, aln in enumerate(alns):
        if i == 1:
            break
    return aln.format()

def align2Sequence(target, query, fileName='alignment.aln'):
    """Align two sequences using PairwiseAligner from Bio.Align and save the alignment to a file

    Args:
        target (str): target sequence
        query (str): query sequence
        fileName (str, optional): file name to save the alignment. Defaults to None.

    Returns:
        str: alignment in clustal format

    Example:
    >>> align2Sequence('ACGT', 'ACGT')
    target            0 ACGT 4
                      0 |||| 4
    query             0 ACGT 4
    """
    if os.path.exists(fileName):
        with open(fileName, 'r') as f:
            aln = f.read()
    else:
        aln = _align(target, query)
        with open(fileName, 'w') as f:
            f.write(aln)
    return aln

def getConsurf(PDB_coords, to_file='consurf.tsv'):
    """Retrieve consurf and ACNR scores for the target and contact residues
    Loop through the PDB_coords and do the following:
    1. Load the PDB file
    """
    os.chdir(pdbDir) # Change directory to pdbDir to load/save pdb files
    _dtype = np.dtype([
        ('consurf', 'f8'),
        ('ACNR', 'f8')
    ])
    featM = np.full(len(PDB_coords), np.nan, dtype=_dtype)
    df = pd.DataFrame(columns=['PDB_coords', 'info', 'chainID', 'resID', 'resIndex', 'resName', 'score'])
    for i in range(len(PDB_coords)):
        _LOGGER.info(f'Processing {PDB_coords[i]}')
        PDB_coord = PDB_coords[i].split()
        if len(PDB_coord) != 4:
            _LOGGER.warning('PDB_coord %s is not in the correct format', PDB_coord)
            continue
        pdbID, chainID, resNum, resName = PDB_coord

        if pdbID not in consurfLookup:
            _LOGGER.warning(f'{pdbID} not in consurfLookup')
            continue
       
        pdbPath = os.path.join(pdbDir, f'{pdbID.lower()}.pdb.gz')
        if os.path.exists(pdbPath):
            pdbCA = prody.parsePDB(pdbPath, subset='ca')
        else:
            pdbCA = prody.parsePDB(pdbID, subset='ca')

        savCA = pdbCA.select(f'chain {chainID} and resnum {resNum}').copy()
        savContacts = findNeighbors(atoms=savCA, atoms2=pdbCA, radius=7.3)
        
        consurf_score = {'target': [], 'contacts': []}
        for (target, contact, distance) in savContacts:
            key = 'target' if int(distance) == 0 else 'contacts'
            key_ca = pdbCA.select('chain ' + contact.getChid()).copy() # Select CA from contact chain
            key_resIndex = key_ca.select('resnum ' + str(contact.getResnum())).getIndices()[0] # get resID of contact residue in chainCA starting from 0
            key_chainID = contact.getChid() # chainID of contact residue e.g. 'A'
            key_resID = contact.getResnum() # resNum of contact residue e.g. recorded by PDB
            key_resName = contact.getResname() # resName of contact residue e.g. 'ALA'
            key_seq = key_ca.getSequence() # sequence of chainCA
            # Refer to this link for more information on how to get the indices of the residues
            # http://prody.csb.pitt.edu/manual/reference/atomic/atomgroup.html
            # Map the indices
            # Check chainID, consurfFile exists
            if key_chainID not in consurfLookup[pdbID]:
                _LOGGER.warning('%s %s not in consurfLookup', pdbID, chainID)
                consurf_score[key].append(np.nan)
                continue
            
            uniqueChain = consurfLookup[pdbID][key_chainID]
            df_consurf = get_consurf(uniqueChain, dataDir)
            consurf_seq = df_consurf.SEQ.to_string(index=False).replace('\n', '').strip()
            key_indices, consurf_indices = mapIndices(key_seq, consurf_seq, f'{pdbID}{key_chainID}', uniqueChain)
            idx = np.where(key_indices == key_resIndex)[0][0]
            map_idx = consurf_indices[idx]
            consurf_resName = one2three[df_consurf.loc[map_idx]['SEQ']]
            if consurf_resName == key_resName:
                consurf_score[key].append(float(df_consurf.loc[map_idx]['SCORE']))
            else:
                logging.warning('Mismatch: resID %s - resName %s, but consurf resID %s - resName %s', key_resIndex, key_resName, map_idx, consurf_resName)
                consurf_score[key].append(np.nan)
            
            new_row = pd.DataFrame({
                'PDB_coords': [PDB_coords[i]],
                'info': [key],
                'chainID': [key_chainID],
                'resID': [key_resID],
                'resIndex': [key_resIndex],
                'resName': [key_resName],
                'score': [consurf_score[key][-1]]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        featM['consurf'][i] = np.nanmean(consurf_score['target'])
        featM['ACNR'][i] = np.nanmean(consurf_score['contacts'])
    os.chdir(workingDir) # Change directory back to currentDir
    if to_file:
        df.to_csv(to_file, sep='\t', index=False)
    return featM

def _parse(unique_chain):
    url = f'{CONSURFDB_URL}DB/{unique_chain}/{unique_chain}_consurf_summary.txt'
    content = get_content(url)
    if content is None:
        return None

    try:
        lines = content.split('\n')
        lines = [line.strip() for line in lines[:-5]]
        for i, line in enumerate(lines):
            if line == '(normalized)':
                cols = lines[i-1]
                lines = lines[i+1:]
                break
        cols = [col.strip() for col in cols.split('\t') if col != '']

        data = []
        for line in lines:
            line = line.split('\t')
            line = [txt.strip() for txt in line if txt.strip() != '']
            if len(line) == 9 and line[0].isdigit():
                if line[2] != '-':
                    line[2] = line[2].split(':')[0]
            data.append(line)
        df = pd.DataFrame(data, columns=cols)
        return df
    except:
        _LOGGER.info(f'Error parsing {unique_chain}')
        return None

def get_consurf(unique_chain, save_dir='.') -> pd.DataFrame:
    """Run the Consurf database for a protein.

    Returns:
    --------
    df: pd.DataFrame, the conservation data for the protein
    """

    outpath = f'{save_dir}/{unique_chain}.tsv'
    if Path(outpath).exists():
        return pd.read_csv(outpath, sep='\t')
    else:
        df = _parse(unique_chain)
        if df is None:
            return None
        df.to_csv(outpath, sep='\t', index=False)
        return df

if __name__ == '__main__':
    logFile = os.path.join(workingDir, 'log/consurf.log')
    logging.basicConfig(filename=logFile, level=logging.ERROR, format='%(message)s')
    PDB_coords = ['5XTC l 17 P', '5XTC w 142 Q'] 
    PDB_coords = ['4JKQ A 77 A']
    featM = getConsurf(PDB_coords)
    print(featM)
    



