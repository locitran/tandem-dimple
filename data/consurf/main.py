import requests
import bs4
import os
import warnings, logging
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd

# Obtained RCSB by Oct 8, 2024
# https://www.rcsb.org/docs/programmatic-access/file-download-services
ENTRY_IDS_URL = 'https://data.rcsb.org/rest/v1/holdings/current/entry_ids' # Obtained from  Oct 8, 2024
CONSURFDB_URL = 'https://consurfdb.tau.ac.il/'

# Suppress the warning about insecure HTTPS request
warnings.simplefilter('ignore', InsecureRequestWarning)

_LOGGER = logging.getLogger(__name__)

def get_content(url):
    """
    Makes a request to a URL. Returns the content of the results
    :param str url:
    :return str:
    """
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        return response.text
    else:
        print("No data retrieved - %s" % response.status_code)
    return None

def get_all_pdb_ids(url=ENTRY_IDS_URL) -> list:
    """Get all PDB IDs from the RCSB database.

    Returns:
    --------
    all_ids: list, [pdb_id, ...]
    """

    response = requests.get(url)
    data = response.json()
    return data

def get_unique_chains(pdbID) -> list:
    """Get the unique chain for a protein from the Consurf database.

    Returns:
    --------
    unique_chains: list, [(chain, unique_chain), ...]
        chain: str, the chain ID in the PDB file
        unique_chain: str, the unique chain ID in the Consurf database (could be different pdbID)
    """
    url = f'{CONSURFDB_URL}scripts/chain_selection.php'

    payload = {
        'pdb_ID': pdbID,
    }
    response = requests.post(url, data=payload,  verify=False)
    
    try:
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        chains = soup.find_all('option')
        chains = [chain['value'] for chain in chains if chain['value'] != '']
        unique_chains = []
        for chain in chains:
            chain, unique_chain = chain.split()
            unique_chains.append((chain, unique_chain))
    except:
        unique_chains = None
    return unique_chains

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

def get_consurf(unique_chain, save_dir=None) -> pd.DataFrame:
    """Run the Consurf database for a protein.

    Returns:
    --------
    df: pd.DataFrame, the conservation data for the protein
    """
    # If save_dir is not None, Check if the file exists
    if save_dir is not None:
        outpath = f'{save_dir}/{unique_chain}.tsv'
        if os.path.exists(outpath):
            print(f'File exists: {outpath}')
            return pd.read_csv(outpath, sep='\t')

    print(f'Getting conservation data for {unique_chain}')
    df = _parse(unique_chain)
    if df is None:
        return None
    df.to_csv(outpath, sep='\t', index=False) if save_dir is not None else None
    return df
