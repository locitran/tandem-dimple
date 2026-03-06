import requests
import os
import json
import re
import subprocess
import multiprocessing
from uuid import uuid1

from prody.utilities import makePath, gunzip, relpath

from ..utils.logger import LOGGER
from ..utils.settings import TMP_DIR, ROOT_DIR

__all__ = ['searchPfam', 'fetchPfamMSA', 'run_hmmscan', 'parse_hmmscan', 'read_pfam_data']

HMMDB = f'{ROOT_DIR}/data/pfamdb/Pfam-A.hmm'
PFAMDATA = f'{ROOT_DIR}/data/pfamdb/Pfam-A.hmm.dat'

base_url = "https://www.ebi.ac.uk/interpro" # Base URL
def searchPfam(acc):
    """
    acc: str, UniProt accession code
    """
    endpoint = "/api/entry/pfam/protein/UniProt/{}"
    url = base_url + endpoint.format(acc) + "?format=json" # Full URL
    try: 
        response = requests.get(url).content
    except Exception as e:
        raise ValueError(f"Failed to retrieve data from {url}: {e}")
    if not response:
        raise IOError('Pfam search failed to parse results JSON, check URL: ' + url)
    
    data = response.decode('utf-8')
    if data.find('There was a system error on your last request.') > 0:
        return None
    elif data.find('No valid UniProt accession or ID') > 0:
        raise ValueError('No valid UniProt accession or ID for: ' + acc)
    
    try:
        data = json.loads(data)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON data: {e}")

    matches = {}
    for entry in data['results']:
        metadata = entry['metadata']
        accession = metadata['accession']
        name = metadata['name']
        proteins = entry.get('proteins', [])
        for protein in proteins:
            locations = protein.get('entry_protein_locations', [])
            match = []
            for location in locations:
                fragments = location.get('fragments', [])
                score = location.get('score', None)
                for fragment in fragments:
                    start = fragment.get('start', None)
                    end = fragment.get('end', None)
                    if start is None or end is None:
                        continue
                    match.append({'start': start, 'end': end, 'score': score})
            if match:
                matches[accession] = {'name': name, 'locations': match}
    return matches

def fetchPfamMSA(acc, alignment='full', compressed=False, folder=TMP_DIR, outname=None):
    """
    acc: str, Pfam ID or Accession Code

    alignment: str, alignment type, 'full' or 'seed
    """

    if not re.search('(?<=PF)[0-9]{5}$', acc):
        raise ValueError('{0} is not a valid Pfam ID or Accession Code'.format(repr(acc)))
    assert alignment in ['full', 'seed'], 'alignment must be one of full or seed'

    endpoint = "/wwwapi/entry/pfam/{}/?annotation=alignment:{}"
    url = base_url + endpoint.format(acc, alignment) + "&download"
    extension = '.sth'

    try:
        response = requests.get(url).content
    except Exception as e:
        raise ValueError(f"Failed to retrieve data from {url}: {e}")    

    outname = acc if outname is None else outname
    filepath = os.path.join(makePath(folder), outname + '_' + alignment + extension)
    if compressed:
        filepath = filepath + '.gz'
        f_out = open(filepath, 'wb')
        f_out.write(response)
        f_out.close()
    else:
        gunzip(response, filepath)

    filepath = relpath(filepath)
    LOGGER.info('Pfam MSA for {0} is written as {1}.'.format(acc, filepath))
    return filepath

def run_hmmscan(fasta_file, hmm_db=HMMDB, folder=TMP_DIR, name=None, cpu=None):
    """Run hmmscan search using the supplied arguments."""
    # Use FASTA filename stem for outputs, e.g. O00187.fasta -> O00187_hmmscan_out
    fasta_stem = os.path.splitext(os.path.basename(fasta_file))[0]
    prefix = fasta_stem if fasta_stem else (name if name is not None else str(uuid1()))
    out = os.path.join(folder, f'{prefix}_hmmscan_out')
    stdout_out = os.path.join(folder, f'{prefix}_hmmscan_stdout')
    if cpu is None:
        cpu = min(multiprocessing.cpu_count(), 16)
    cpu = str(cpu)
    cmd = ['hmmscan', '--notextw', '--cpu', cpu,  '--cut_ga', '--domtblout', out, hmm_db, fasta_file]
    result = subprocess.run(cmd,
        stdout=open(stdout_out, 'w'), # Keep stdout separate from domtbl output
        stderr=subprocess.PIPE, # Suppress stderr
        text=True # Decode stdout to text
    )
    if result.returncode != 0:
        raise ValueError(result.stderr)
    return out

def parse_hmmscan(filename: str, pfam_data: dict):
    """Parse hmmscan output file and return a dictionary of domain hits.
    Reference: https://github.com/aziele/pfam_scan.git
    """
    data = {}
    with open(filename) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            # Alignments for each domain
            if "Domain annotation for each model (and alignments):" in line or \
                "Internal pipeline statistics summary:" in line or \
                "Alignments for each domain:" in line:
                break
            cols = line.split()
            if len(cols) < 21:
                continue
            name = cols[0]
            acc = cols[1]
            seq_id = cols[3]
            # Defensive parse: skip malformed rows accidentally mixed into output
            # (for example alignment annotation lines that are not domtbl rows).
            try:
                score_dom = float(cols[13])
                score_seq = float(cols[7])
            except (TypeError, ValueError, IndexError):
                continue
            # Determine which domain hits are significant. The significance 
            # value is 1 if the bit scores for a domain and a sequence are 
            # greater than or equal to the curated gathering thresholds for 
            # the matching domain, 0 otherwise. 
            significance = 0
            if (score_dom >= pfam_data[name]['ga_dom'] and 
                score_seq >= pfam_data[name]['ga_seq']):
                # Since both conditions are true, the domain hit is significant.
                significance = 1
            if seq_id not in data:
                data[seq_id] = {}
            if acc not in data[seq_id]:
                data[seq_id][acc] = {
                    'name': name,
                    'type': pfam_data[name]['type'],
                    'clan': pfam_data[name]['clan'],
                    'ga_seq': pfam_data[name]['ga_seq'],
                    'ga_dom': pfam_data[name]['ga_dom'],
                    'locations': []
                }
            dom = {
                # alignment coordinates in the query sequence (positions in your protein where the HMM alignment starts/ends).
                'ali_start': int(cols[17]), 'ali_end': int(cols[18]),
                # envelope coordinates in the query sequence (the full domain region on the sequence, often slightly wider than the aligned core).
                'seq_start': int(cols[19]), 'seq_end': int(cols[20]),
                # alignment coordinates in the Pfam HMM (positions along the model that align to the sequence).
                'hmm_start': int(cols[15]), 'hmm_end': int(cols[16]),
                'hmm_length': int(cols[2]), # total length of the Pfam HMM (model length).
                'score': score_dom, # domain bit score (strength of the domain match).
                'evalue': float(cols[12]), # domain E‑value.
                'significance': significance, # 1 if both domain and sequence scores pass Pfam GA thresholds, else 0.
            }
            data[seq_id][acc]['locations'].append(dom)
    os.remove(filename)
    return data

def read_pfam_data(filename: str = PFAMDATA):
    """Reads the Pfam data file to dictionary.

    Args:
        filename: Name/Path of the Pfam data file (Pfam-A.hmm.dat).

    Returns:
        A dict mapping HMM profile name to the corresponding information.
        For example:

        {'1-cysPrx_C': {type='Domain', clan=None, ga_seq=21.1, ga_dom=21.1},
         'RRM': {type='Domain', clan=None, ga_seq=21.0, ga_dom=21.0},
         'SOXp': {type='Family', clan=None, ga_seq=22.1, ga_dom=22.1}}

    Reference: https://github.com/aziele/pfam_scan.git
    """
    data = {}
    with open(filename) as fh:
        clan = None   # Not all domains have clan assigned.
        for line in fh:
            if line.startswith('#=GF ID'):
                hmm_name = line[10:-1]
            elif line.startswith('#=GF TP'):
                typ = line[10:-1]
            elif line.startswith('#=GF CL'):
                clan = line[10:-1]
            elif line.startswith('#=GF GA'):
                scores = line[10:-1].strip().rstrip(';').split(';')
                ga_seq = float(scores[0])
                ga_dom = float(scores[1])
            elif line.startswith('//'):
                data[hmm_name] = {
                    'type': typ,
                    'clan': clan,
                    'ga_seq': ga_seq,
                    'ga_dom': ga_dom
                }
                clan = None
    return data
