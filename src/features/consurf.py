
import json, traceback
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import more_itertools as mit

from prody import parsePDB
from prody.measure.contacts import findNeighbors
from Bio.Align import PairwiseAligner

from ..dynamics.ENM import GNM
from ..utils.logger import LOGGER
from ..download import get_content, fetchPDB
from ..utils.settings import ROOT_DIR,RAW_PDB_DIR
from ..utils.timer import getTimer
from ..stand_alone_consurf.main import run

__all__ = ['calcConSurf', 'get_consurf', 'mapIndices']

CONSURFDB_URL = 'https://consurfdb.tau.ac.il/'
pdbDir = ROOT_DIR + '/pdbfile/raw'
consurfDir = ROOT_DIR + '/data/consurf'
dataDir = consurfDir + '/db/2024-10-08'
consurfLookup = consurfDir + '/2024-10-08.json'
customDir = consurfDir + '/db/custom'
uniref90_2022_05 = os.path.join(consurfDir, 'uniref90.fasta')
os.makedirs(customDir, exist_ok=True)

timer = getTimer('tandem', verbose=True)
with open(consurfLookup) as f:
    consurfLookup = json.load(f)

MATCH_SCORE = 1.0
MISMATCH_SCORE = 0.0
GAP_PENALTY = -1.
GAP_EXT_PENALTY = -0.1

def mapIndices(targetSeq, querySeq):
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
    aln = _align(targetSeq, querySeq)
    try:
        target_seq, align_dash, query_seq, _ = aln.split('\n')
    except:
        split_aln = aln.split('\n')
        target_seq = ''
        query_seq = ''
        align_dash = ''
        for i, p in enumerate(split_aln):
            p_split = p.split()
            target_seq += p_split[2] if i % 4 == 0 else ''
            query_seq += p_split[2] if i % 4 == 2 else ''
            align_dash += p_split[2]  if i % 4 == 1 else ''

    # Get the indices of the target and query sequences correspondingly
    target_indices = [] ; idx = 0
    for i in range(len(target_seq)):
        if target_seq[i] not in ['-', '.']: # != '-':
            target_indices.append(idx)
            idx += 1
        else:
            target_indices.append(-1)

    query_indices = [] ; idx = 0
    for i in range(len(query_seq)):
        if query_seq[i] not in ['-', '.']: # != '-':
            query_indices.append(idx)
            idx += 1
        else:
            query_indices.append(-1)

    search = lambda x: x == "|"
    exact_match_indices = np.array(list(mit.locate(align_dash, search)))

    return np.array(target_indices), np.array(query_indices), exact_match_indices

def _align(target, query):
    """Align two sequences using PairwiseAligner from Bio.Align
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

def _parse(unique_chain):
    url = f'{CONSURFDB_URL}DB/{unique_chain}/{unique_chain}_consurf_summary.txt'
    # url = f'{CONSURFDB_URL}DB_NEW/{pdbid}/{unique_chain}/{pdbid}_{unique_chain}_consurf_grades.txt'
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
    except Exception as e:
        LOGGER.warning(f'Error parsing {unique_chain}, {e}')
        return None

def get_consurf(unique_chain, folder='.'):
    """Run the Consurf database for a protein.

    Returns:
    --------
    df: pd.DataFrame, the conservation data for the protein
    """

    outpath = os.path.join(folder, f'{unique_chain}.tsv')
    if os.path.exists(outpath):
        return pd.read_csv(outpath, sep='\t')
    else:
        df = _parse(unique_chain)
        if df is None:
            return None
        df.to_csv(outpath, sep='\t', index=False)
        return df

def getConSurffile(pdb, chid, folder='.', uniref90=uniref90_2022_05):
    """Get the consurf file for a given PDB ID and chain ID.
    pdb: PDB ID or PDB file
        - pdb: PDB ID 
            1. Check consurfLookup for PDB ID + chid
            2. Run stand_alone_consurf if not found
        
        - pdb: PDB file
            1. Check existence of the file f'{pdb}_{chid}.tsv' in customDir
            2. Run stand_alone_consurf for given chain if not found

    folder provided in case of running stand_alone_consurf
    Returns:
        df: DataFrame of output consurffile

    Example:        
        from src.features.consurf import getConSurffile
        pdb = '4xr8'
        pdb = '/home/newloci/tandem/src/stand_alone_consurf/pkd1/fold_1xpkd1_model_0_A.pdb'
        chid = 'A'
        consurf = getConSurffile(pdb, chid)
    """
    if not os.path.isfile(pdb):
        pdbID = pdb.upper()
        if (pdbID in consurfLookup) and (chid in consurfLookup[pdbID]):
            uniqueChain = consurfLookup[pdbID][chid]
            consurffile = os.path.join(dataDir, f'{uniqueChain}.tsv')
            return pd.read_csv(consurffile, sep='\t')
        else:
            pdb = fetchPDB(pdbID, format='pdb', compressed=False, folder=RAW_PDB_DIR)
            if pdb is None:
                raise ValueError(f'Cannot download {pdbID}')
    else:
        pdbID = os.path.basename(pdb).split('.')[0]

    # Search uniqueChain in customDir
    uniqueChain = f'{pdbID}_{chid}'
    consurffile = os.path.join(customDir, f'{uniqueChain}.tsv')
    if os.path.isfile(consurffile):
        return pd.read_csv(consurffile, sep='\t')
    
    LOGGER.info(f'Running consurf for {pdb} {pdbID} {chid}')
    # If not found, run stand_alone_consurf
    out = run(
        query=pdbID,
        structure=pdb,
        chain=chid,
        DB=uniref90,
        work_dir=folder,
        algorithm="HMMER"
    )
    # Parse the consurf file
    data = []
    with open(out, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip() != '']
        for i, line in enumerate(lines):
            if line.startswith('POS'):
                cols = lines[i]
                lines = lines[i+1:]
                break
        cols = [col.strip() for col in cols.split('\t') if col != '']
        for line in lines:
            line = line.split('\t')
            if len(line) == 10 and line[0].isdigit():
                data.append(line)
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(consurffile, sep='\t', index=False)
    return df

def calcConSurf(pdb, chid, folder='.', uniref90=uniref90_2022_05):
    _dtype = np.dtype([
        ('consurf', 'f4'), 
        ('ACNR', 'f4'), # Average contact neighbouring residues
        ('consurf_color', 'i4'),
    ])
    # Read the PDB file
    # custom or alphafold
    if os.path.isfile(pdb):
        pdbID = pdb
        pdb = parsePDB(pdb, model=1)
    else: # pdbID 
        pdbID = pdb
        pdbpath = fetchPDB(pdbID, format='pdb', folder=RAW_PDB_DIR, refresh=True)
        if pdbpath is not None:
            pdb = parsePDB(pdbpath, model=1)
        else:
            raise ValueError(f'Cannot download {pdbID}')
        
    LOGGER.timeit('_calcConSurf')
    ca = pdb.protein.ca
    tgt_chain = ca.select(f'chain {chid}').copy()
    if not tgt_chain:
        raise ValueError(f'Cannot find chain {chid} in {pdb}')
    
    tgt_seq = tgt_chain.getSequence()
    features = np.full(len(tgt_chain), np.nan, dtype=_dtype)
    
    df_consurf = getConSurffile(pdbID, chid, folder=folder, uniref90=uniref90)
    # Replace color* --> color+10
    if df_consurf['COLOR'].dtype != int:
        df_consurf['COLOR'] = df_consurf['COLOR'].apply(
            lambda x: int(x.replace('*', '')) + 10 if '*' in x else int(x)
        )
    consurf_seq = df_consurf.SEQ.to_string(index=False).replace('\n', '').replace(' ', '')

    consurf_indices, target_indices, exact_match_indices = mapIndices(consurf_seq, tgt_seq)
    # Extract exact match indices 
    consurf_match = consurf_indices[exact_match_indices]
    target_match  = target_indices[exact_match_indices]
    
    # Extract scores
    consurf_scores = df_consurf.iloc[consurf_match].SCORE.values
    consurf_colors = df_consurf.iloc[consurf_match].COLOR.values

    features['consurf'][target_match] = consurf_scores
    features['consurf_color'][target_match] = consurf_colors

    # Build Kirchhoff matrix 
    gnm = GNM()
    gnm.buildKirchhoff(tgt_chain, cutoff=7.3)
    kirchhoff = gnm.getKirchhoff()
    
    # Replace contact buy consurf score
    minus1 = np.argwhere(kirchhoff==-1) # Contact , row: target; column: contact
    kirchhoff[minus1[:, 0], minus1[:, 1]] = features['consurf'][minus1[:, 1]]
    diag_kirchhoff = np.diag(kirchhoff) # Extract diagonal : number of contacts
    numnan_kirchhoff = np.sum(np.isnan(kirchhoff), axis=1) # Eliminate nan
    diag_kirchhoff = diag_kirchhoff - numnan_kirchhoff

    kirchhoff = np.ma.array(kirchhoff)
    kirchhoff.mask = np.eye(kirchhoff.shape[0], dtype=bool)
    col_sum_excl_diag = kirchhoff.sum(axis=1) / diag_kirchhoff
    features['ACNR'][target_match] = col_sum_excl_diag
    
    LOGGER.report('ConSurf features calculated in %.2fs.', label='_calcConSurf')
    return features