import os 
import logging
import requests
import urllib.request

_LOGGER = logging.getLogger(__name__)
pdbe_prefix = 'https://www.ebi.ac.uk/pdbe'

__func__ = ['get_url', 'cif_biological_assembly', 'opm', 'pdb_asymmetric_unit', 'cif_asymmetric_unit', 'uniprot_sequence']
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_dir = os.path.join(parent_dir, 'pdbfile/raw')

def get_url(url):
    """
    Makes a request to a URL. Returns a JSON of the results
    :param str url:
    :return dict:
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        _LOGGER.warning("No data retrieved - %s" % response.status_code)
        print("[No data retrieved - %s] %s" % (response.status_code, response.text))
    return {}

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
        _LOGGER.warning("No data retrieved - %s" % response.status_code)
    return ''

def pdb_summary(pdbID: str):
    """This call provides a summary of properties of a PDB entry, 
    such as the title of the entry, list of depositors, date of deposition, 
    date of release, date of latest revision, experimental method, 
    list of related entries in case split entries, etc.

    Ref: https://www.ebi.ac.uk/pdbe/api/doc/pdb.html
    """
    summary_url = f'{pdbe_prefix}/api/pdb/entry/summary/{pdbID}'
    print(f'> Parse the summary data of {pdbID} from {summary_url}...')
    data = get_url(summary_url)
    try:
        assemblies = data[pdbID.lower()][0]['assemblies']
        n_assemblies = len(assemblies)
        print(f'{pdbID} has {n_assemblies} assembly(assemblies).')
    except KeyError:
        _LOGGER.error(f'{pdbID} has no assembly data.')
        print(f'{pdbID} has no assembly data.')
        n_assemblies = 0
    return n_assemblies

def cif_biological_assembly(pdbID: str, assemblyID: int, outdir: str = download_dir):
    "Download a cif file from RCSB PDB database."
    pdbID = pdbID.lower()
    cif_url = f'{pdbe_prefix}/static/entry/download/{pdbID}-assembly{assemblyID}.cif.gz'
    outpath = os.path.join(outdir, f'{pdbID}-assembly{assemblyID}.cif.gz')

    if os.path.exists(outpath):
        print(f'{pdbID}-assembly{assemblyID}.cif.gz already exists.')
        return outpath
    else:
        print(f'> Download {pdbID}-assembly{assemblyID}.cif.gz...')
        try:
            urllib.request.urlretrieve(cif_url, outpath)
            print(f'{pdbID}-assembly{assemblyID}.cif.gz is downloaded.')
            return outpath
        except urllib.error.HTTPError:
            msg = f'{pdbID}-assembly{assemblyID}.cif.gz does not exist.'
            _LOGGER.warning(msg)
            print(msg)
            return None

def opm(pdbID, outdir: str = download_dir):
    """Download a PDB file from OPM database.

    Returns:
        1: opm file is downloaded.
        0: opm file is not downloaded.
    """
    pdbID = pdbID.lower()
    outpath = os.path.join(outdir, f'{pdbID}-opm.pdb')
    url = f"https://opm-assets.storage.googleapis.com/pdb/{pdbID}.pdb"
    if os.path.exists(outpath):
        return outpath
    else:
        print(f'> Download {pdbID}-opm.pdb...')
        try:
            urllib.request.urlretrieve(url, outpath)
            print(f'{pdbID}-opm.pdb is downloaded.')
            return outpath
        except urllib.error.HTTPError:
            _LOGGER.warning(f'{url} does not exist.')
            return None

def pdb_asymmetric_unit(pdbID, outdir: str = download_dir):
    "Download a PDB file from RCSB PDB database."
    pdbID = pdbID.lower()
    outpath = os.path.join(outdir, f'{pdbID}.pdb.gz')
    url = f"https://files.rcsb.org/download/{pdbID}.pdb.gz"

    if os.path.exists(outpath):
        return outpath
    print(f'> Download {pdbID}.pdb.gz...')
    try:
        urllib.request.urlretrieve(url, outpath)
        print(f'{pdbID}.pdb.gz is downloaded.')
        return outpath
    except urllib.error.HTTPError:
        _LOGGER.warning(f'{url} does not exist.')
        return None

def cif_asymmetric_unit(pdbID, outdir: str = download_dir):
    "Download a cif file from RCSB PDB database."
    pdbID = pdbID.lower()
    outpath = os.path.join(outdir, f'{pdbID}.cif.gz')
    url = f"https://files.rcsb.org/download/{pdbID}.cif.gz"
    
    if os.path.exists(outpath):
        return outpath
    print(f'> Download {pdbID}.cif...')
    try:
        urllib.request.urlretrieve(url, outpath)
        print(f'{pdbID}.cif is downloaded.')
        return outpath
    except urllib.error.HTTPError:
        _LOGGER.warning(f'{url} does not exist.')
        return None

def uniprot_sequence(uniprotACC, outdir: str = None):
    "Download a sequence file from UniProt database."
    url = f"https://rest.uniprot.org/uniprotkb/{uniprotACC}.fasta"
    
    if outdir is not None:
        outpath = os.path.join(outdir, f'{uniprotACC}.fasta')
    
        if os.path.exists(outpath):
            return outpath
        print(f'> Download {uniprotACC}.fasta...')
        try:
            urllib.request.urlretrieve(url, outpath)
            print(f'{uniprotACC}.fasta is downloaded.')
            return outpath
        except urllib.error.HTTPError:
            _LOGGER.warning(f'{url} does not exist.')
            return None
    else:
        fasta =  get_content(url)
        if fasta is None:
            return None
        fasta_lines = fasta.split('\n')
        fasta_lines = [line.strip() for line in fasta_lines if line.strip() != '']
        seq = ''.join(fasta_lines[1:])
        return seq
   