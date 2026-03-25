
import json
import os
import pandas as pd
import numpy as np
import more_itertools as mit

from prody import parsePDB, writePDB
from Bio.Align import PairwiseAligner

from ..dynamics.ENM import GNM
from ..utils.logger import LOGGER
from ..download import fetchPDB
from ..utils.settings import ROOT_DIR,RAW_PDB_DIR
from ..utils.timer import getTimer
from ..stand_alone_consurf.main import run
from .. import download

__all__ = ['calcConSurf', 'calcConSurf_v2', 'mapIndices', 'getConSurffile_v2']

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

    target = 'MYLSKVIIARAWSRDLYQLHQGLWHLFPNRPDAARDFLFHVEKRNTPEGCHVLLQSAQMPVSTAVATVIKTKQVEFQLQVGVPLYFRLRANPIKTILDNQKRLDSKGNIKRCRVPLIKEAEQIAWLQRKLGNAARVEDVHPISERPQYFSGDGKSGKIQTVCFEGVLTINDAPALIDLVQQGIGPAKSMGCGLLSLAPL'
    query = 'EIDAMALYRAWQQLDNGSCAQIRRVSEPDELRDIPAFYRLVQPFGWENPRHQQALLRMVFCLSAGKNVIRHQDKKSEQTTGISLGRALANSGRINERRIFQLIRADRTADMVQLRRLLTHAEPVLDWPLMARMLTWWGKRERQQLLEDFVLTTNKNA'
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

def mapIndices_v2(targetSeq, querySeq):
    """biopython==1.81"""
    aln = _align(targetSeq, querySeq)
    lines = [line.rstrip() for line in aln.splitlines() if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f'Unexpected alignment block structure: {lines}')

    target_seq = ''
    query_seq = ''
    align_dash = ''

    for i in range(0, len(lines), 3):
        target_line = lines[i]
        match_line = lines[i + 1]
        query_line = lines[i + 2]

        target_parts = target_line.split()
        match_parts = match_line.split()
        query_parts = query_line.split()

        if (target_parts[0].lower() != 'target') and (len(target_parts) not in [3, 4]):
            print('\n'.join(lines))
            raise ValueError(f'Expected target line (len={len(target_parts)}), got: {target_line}')
        
        if (query_parts[0].lower() != 'target') and (len(query_parts) not in [3, 4]):
            print('\n'.join(lines))
            raise ValueError(f'Expected target line (len={len(query_parts)}), got: {query_line}')
        
        if len(match_parts) not in [2, 3]:
            print('\n'.join(lines))
            raise ValueError(f'Unexpected match alignment line (len={len(match_parts)}): {match_line}')

        target_seq += target_parts[2]
        align_dash += match_parts[1]
        query_seq += query_parts[2]

    if not target_seq or not query_seq or not align_dash:
        raise ValueError('Unable to parse alignment text from PairwiseAligner output.')

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
    target = 'ACGT'
    query = 'ACGT'

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
            if os.path.isfile(consurffile):
                return pd.read_csv(consurffile, sep='\t')
                
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
        algorithm="HMMER",
        cif=str(pdb).lower().endswith(".cif"),
    )
    df = parseConSurf(out)
    df.to_csv(consurffile, sep='\t', index=False)
    return df

def parseConSurf(out):
    rows = []
    cols = None

    with open(out, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]

    for i, line in enumerate(lines):
        if line.lstrip().startswith('POS'):
            cols = [col.strip() for col in line.split('\t')]
            lines = lines[i + 1:]
            break

    if cols is None:
        LOGGER.warn(f'Could not find ConSurf header in {out}')
        return pd.DataFrame()

    ncols = len(cols)
    for line in lines:
        parts = [part.strip() for part in line.split('\t')]
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) < ncols:
            continue
        if len(parts) > ncols:
            parts = parts[:ncols - 1] + [' '.join(parts[ncols - 1:]).strip()]
        rows.append(parts)

    return pd.DataFrame(rows, columns=cols)

def getConSurffile_v2(id, chid, folder='.', uniref90=uniref90_2022_05):
    """Run stand-alone ConSurf for a PDB ID or UniProt accession and cache the
    parsed output under customDir.

    Args:
        id: Either a 4-character PDB ID or a UniProt accession number.
        chid: Chain identifier for PDB IDs, don't care in case of UniProt accession number.
        folder: Working directory for stand-alone ConSurf.
        uniref90: Sequence database path used by stand-alone ConSurf.

    Returns:
        pd.DataFrame: Parsed ConSurf output.
    """
    assert isinstance(id, str), "id must be a string."
    assert isinstance(chid, str), "chid must be a non-empty string."

    id = id.strip().upper()
    chid = chid.strip()
    if len(id) == 4:
        if (id in consurfLookup) and (chid in consurfLookup[id]):
            uniqueChain = consurfLookup[id][chid]
            consurffile = os.path.join(dataDir, f'{uniqueChain}.tsv')
            if os.path.isfile(consurffile):
                return pd.read_csv(consurffile, sep='\t')
    
        pdb = fetchPDB(id, format='pdb', compressed=False, folder=RAW_PDB_DIR)
        if pdb is None:
            raise ValueError(f'Cannot download {id}')

        cache_name = f"{id}_{chid}.tsv"
        consurffile = os.path.join(customDir, cache_name)
        if os.path.isfile(consurffile):
            return pd.read_csv(consurffile, sep='\t')

        LOGGER.info(f'Running stand-alone ConSurf for PDB {id} chain {chid}')
        out = run(query=id, structure=pdb, chain=chid, DB=uniref90, work_dir=folder, algorithm="HMMER", cif=str(pdb).lower().endswith(".cif"),)
    else:
        cache_name = f"{id}.tsv"
        consurffile = os.path.join(customDir, cache_name)
        if os.path.isfile(consurffile):
            return pd.read_csv(consurffile, sep='\t')

        seq = download.uniprot_sequence(id, folder=folder)
        LOGGER.info(f'Running stand-alone ConSurf for UniProt accession {id} {seq}')
        out = run(query=id, seq=seq, DB=uniref90, work_dir=folder, algorithm="HMMER")

    df = parseConSurf(out)
    df.to_csv(consurffile, sep='\t', index=False)
    return df
def calcConSurf(
        pdbfile,
        pdbid, 
        chid, 
        folder='.', 
        uniref90=uniref90_2022_05
    ):

    _dtype = np.dtype([
        ('consurf', 'f4'), 
        ('ACNR', 'f4'), # Average contact neighbouring residues
        ('consurf_color', 'i4'),
    ])
    assert os.path.isfile(pdbfile), "Must be a file."
    # assert isinstance(pdbid, str) and len(pdbid) == 4, "pdbid must be a 4-character string."

    LOGGER.timeit('_calcConSurf')
    pdb = parsePDB(pdbfile, model=1)
    ca = pdb.protein.ca
    tgt_chain = ca.select(f'chain {chid}').copy()
    if not tgt_chain:
        raise ValueError(f'Cannot find chain {chid} in {pdb}')
    
    tgt_seq = tgt_chain.getSequence()
    features = np.full(len(tgt_chain), np.nan, dtype=_dtype)
    
    try:
        df_consurf = getConSurffile(pdbid, chid, folder=folder, uniref90=uniref90)
    except Exception as e:
        LOGGER.warn(f"Error with getConSurffile pdbid {pdbid} chid {chid}: {str(e)}")
        return features

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

    # Average ConSurf scores over valid contacting neighbours only.
    # Missing neighbour scores remain NaN, but they should not poison
    # the whole row average.
    # (kirchhoff == -1): contact && valid_consurf_mask: not nan ConSurf
    valid_consurf_mask = ~np.isnan(features['consurf']) # 
    contact_mask = (kirchhoff == -1) & valid_consurf_mask[:, None] 
    score_matrix = np.where(contact_mask, features['consurf'][None, :], np.nan)

    valid_counts = np.sum(~np.isnan(score_matrix), axis=1) # contact number with a valid ConSurf score - (n, )
    score_sums = np.nansum(score_matrix, axis=1) # Sum neighboring ConSurf scores - (n, )
    valid_rows = valid_counts > 0
    features['ACNR'][valid_rows] = score_sums[valid_rows] / valid_counts[valid_rows]
    LOGGER.report('ConSurf features calculated in %.2fs.', label='_calcConSurf')
    return features

def calcConSurf_v2(pdbfile,id,chid,folder='.',uniref90=uniref90_2022_05,write_consurf=False):
    """Calculate ConSurf-based residue features using getConSurffile_v2.

    Args:
        pdbfile: Structure file used for patching scores to residues.
        id: Either a 4-character PDB ID or a UniProt accession.
        chid: Target structure chain for pdbfile.
        folder: Working directory for stand-alone ConSurf.
        uniref90: Sequence database path used by stand-alone ConSurf.
        write_consurf: If True, write a PDB file with ConSurf scores in B-factors.
    """
    _dtype = np.dtype([
        ('consurf', 'f4'),
        ('ACNR', 'f4'),
        ('consurf_color', 'i4'),
    ])

    assert os.path.isfile(pdbfile), "pdbfile must be an existing file."

    LOGGER.timeit('_calcConSurf')
    pdb = parsePDB(pdbfile, model=1)
    ca = pdb.protein.ca
    if ca is None:
        raise ValueError(f'Cannot find protein C-alpha atoms in {pdbfile}')

    tgt_chain = ca.select(f'chain {chid}')
    if tgt_chain is None:
        raise ValueError(f'Cannot find chain {chid} in {pdbfile}')
    tgt_chain = tgt_chain.copy()

    tgt_seq = tgt_chain.getSequence()
    features = np.full(len(tgt_chain), np.nan, dtype=_dtype)

    try:
        df_consurf = getConSurffile_v2(id=id, chid=chid, folder=folder, uniref90=uniref90)
    except Exception as e:
        LOGGER.warn(str(e))
        return features

    if df_consurf is None or df_consurf.empty:
        LOGGER.warn(f'Empty ConSurf output for id={id} chain={chid}')
        return features

    # Replace color* --> color+10
    if df_consurf['COLOR'].dtype != int:
        df_consurf['COLOR'] = df_consurf['COLOR'].apply(
            lambda x: int(x.replace('*', '')) + 10 if '*' in x else int(x)
        )

    # Extract exact match indices 
    consurf_seq = df_consurf.SEQ.to_string(index=False).replace('\n', '').replace(' ', '') # consurf_seq = ''.join(df_consurf['SEQ'].astype(str).tolist())
    consurf_indices, target_indices, exact_match_indices = mapIndices(consurf_seq, tgt_seq)
    consurf_match = consurf_indices[exact_match_indices]
    target_match = target_indices[exact_match_indices]

    valid = (consurf_match >= 0) & (consurf_match < len(df_consurf)) & (target_match >= 0)
    consurf_match = consurf_match[valid]
    target_match = target_match[valid]
    if len(consurf_match) == 0:
        LOGGER.warn(f'No valid ConSurf residue mapping for id={id} chain={chid}')
        return features

    # Extract scores
    consurf_scores = df_consurf.iloc[consurf_match].SCORE.values
    consurf_colors = df_consurf.iloc[consurf_match].COLOR.values
    features['consurf'][target_match] = consurf_scores
    features['consurf_color'][target_match] = consurf_colors
    
    # Build Kirchhoff matrix
    gnm = GNM()
    gnm.buildKirchhoff(tgt_chain, cutoff=7.3)
    kirchhoff = gnm.getKirchhoff()

    # Average ConSurf scores over valid contacting neighbours only.
    # Missing neighbour scores remain NaN, but they should not poison
    # the whole row average.
    # (kirchhoff == -1): contact && valid_consurf_mask: not nan ConSurf
    valid_consurf_mask = ~np.isnan(features['consurf'])
    contact_mask = (kirchhoff == -1) & valid_consurf_mask[None, :]
    score_matrix = np.where(contact_mask, features['consurf'][None, :], np.nan)

    valid_counts = np.sum(~np.isnan(score_matrix), axis=1) # contact number with a valid ConSurf score - (n, )
    score_sums = np.nansum(score_matrix, axis=1) # Sum neighboring ConSurf scores - (n, )
    valid_rows = valid_counts > 0
    features['ACNR'][valid_rows] = score_sums[valid_rows] / valid_counts[valid_rows]

    if write_consurf:
        os.makedirs(folder, exist_ok=True)
        model = pdb.protein.copy()
        chain = model.select(f'chain {chid}')

        betas = model.getBetas()
        if betas is None or len(betas) != model.numAtoms():
            betas = np.zeros(model.numAtoms(), dtype=float)
        else:
            betas = np.asarray(betas, dtype=float).copy()

        chain_ca = chain.ca

        ca_resindices = chain_ca.getResindices()
        atom_resindices = model.getResindices()
        for resindex, score in zip(ca_resindices, features['consurf']):
            if np.isnan(score):
                continue
            betas[atom_resindices == resindex] = float(score)

        model.setBetas(betas)
        outfile = os.path.join(folder, f'{id}_{chid}_consurf.pdb')
        writePDB(outfile, model)
        LOGGER.info(f'ConSurf B-factor file written to {outfile}')

    LOGGER.report('ConSurf features calculated in %.2fs.', label='_calcConSurf')
    return features
