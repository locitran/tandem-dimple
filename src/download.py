import os 
import re
import logging
import requests
import urllib.request
from urllib.parse import urlparse
from .utils.logger import LOGGER
import prody
import numpy as np
from prody import parsePDB

__all__ = ['pdb_summary', 'fetchPDB', 'fetchPDB_BiologicalAssembly']

pdbe_prefix = 'https://www.ebi.ac.uk/pdbe'

def _url_exists(url: str, timeout=10) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code == 405:  # some servers disallow HEAD
            r = requests.get(url, stream=True, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False
    
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
        LOGGER.warning("No data retrieved - %s" % response.status_code)
        LOGGER.info("[No data retrieved - %s] %s" % (response.status_code, response.text))
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
        LOGGER.warning("No data retrieved - %s" % response.status_code)
    return ''

def pdb_summary(pdbID: str):
    """This call provides a summary of properties of a PDB entry, 
    such as the title of the entry, list of depositors, date of deposition, 
    date of release, date of latest revision, experimental method, 
    list of related entries in case split entries, etc.

    Ref: https://www.ebi.ac.uk/pdbe/api/doc/pdb.html
    """
    summary_url = f'{pdbe_prefix}/api/pdb/entry/summary/{pdbID}'
    LOGGER.info(f'> Parse the summary data of {pdbID} from {summary_url}...')
    data = get_url(summary_url)
    try:
        assemblies = data[pdbID.lower()][0]['assemblies']
        n_assemblies = len(assemblies)
        LOGGER.info(f'{pdbID} has {n_assemblies} assembly(assemblies).')
    except KeyError:
        LOGGER.info(f'{pdbID} has no assembly data.')
        LOGGER.info(f'{pdbID} has no assembly data.')
        n_assemblies = 0
    return n_assemblies

def uniprot_sequence(uniprotACC, folder: str = None):
    "Download a sequence file from UniProt database."
    url = f"https://rest.uniprot.org/uniprotkb/{uniprotACC}.fasta"
    
    if folder is not None:
        outpath = os.path.join(folder, f'{uniprotACC}.fasta')
    
        if os.path.exists(outpath):
            return outpath
        LOGGER.info(f'Download {uniprotACC}.fasta...')
        try:
            urllib.request.urlretrieve(url, outpath)
            LOGGER.info(f'{uniprotACC}.fasta is downloaded.')
            return outpath
        except urllib.error.HTTPError:
            LOGGER.warning(f'{url} does not exist.')
            return None
    else:
        fasta =  get_content(url)
        if fasta is None:
            return None
        fasta_lines = fasta.split('\n')
        fasta_lines = [line.strip() for line in fasta_lines if line.strip() != '']
        seq = ''.join(fasta_lines[1:])
        return seq

def fetch_fasta(accs, **kwargs):
    """
    Fetch one or multiple UniProt sequences and save into a single FASTA file.
    
    Parameters:
    - accs: str or list of str, UniProt accession(s)
    - folder: str, output folder (default: '.')
    - outname: str, output filename (default: 'output.fasta')
    - refresh: bool, whether to force re-download even if file exists (default: False)
    
    Returns:
    - Output FASTA file path
    """
    
    if isinstance(accs, str):
        accs = [accs]
    
    folder = kwargs.get('folder', '.')
    filename = kwargs.get('filename', 'output')
    refresh = kwargs.get('refresh', False)
    os.makedirs(folder, exist_ok=True)
    
    output_path = os.path.join(folder, f'{filename}.fasta')
    if os.path.exists(output_path) and not refresh:
        LOGGER.info(f"{output_path} already exists (use refresh=True to overwrite).")
        return output_path
    
    with open(output_path, 'w') as f_out:
        for acc in accs:
            url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
            response = requests.get(url)
            if response.status_code == 200:
                f_out.write(response.text)
                LOGGER.info(f"Fetched: {acc}")
            else:
                LOGGER.info(f"Failed: {acc} (HTTP {response.status_code})")
    LOGGER.info(f"\n✅ Saved to: {output_path}")
    return output_path

def fetchAF2(afid, **kwargs):
    """Fetch AlphaFold structure file.
    Only two formats: cif and pdb.

    Example:
    fetchAF2("AF-O00189-F1-model_v6") --> fetches AF model v6 if available, else raises ValueError
    """
    folder = kwargs.get("folder", ".")
    refresh = kwargs.get("refresh", False)
    prefer_format = str(kwargs.get("prefer_format", "pdb")).lower()
    timeout = kwargs.get("timeout", 20)

    afid = (afid or "").strip()
    os.makedirs(folder, exist_ok=True)

    if not afid:
        raise ValueError("Empty AlphaFold ID.")
    if not re.fullmatch(r"AF-[A-Z0-9]{6,10}-F\d+-MODEL_V\d+", afid, re.I):
        raise ValueError(f"Invalid normalized AlphaFold ID: {afid}")

    if prefer_format not in ("pdb", "cif"):
        prefer_format = "pdb"

    def _download(url: str):
        filename = os.path.basename(urlparse(url).path)
        outpath = os.path.abspath(os.path.join(folder, filename))
        if (not refresh) and os.path.exists(outpath):
            return outpath

        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return None
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return outpath

    url_pdb = f"https://alphafold.ebi.ac.uk/files/{afid}.pdb"
    url_cif = f"https://alphafold.ebi.ac.uk/files/{afid}.cif"
    if prefer_format == "pdb":
        out = _download(url_pdb) or _download(url_cif)
    else:
        out = _download(url_cif) or _download(url_pdb)
    if out is None:
        raise ValueError(f"AlphaFold file not found for {afid} (.pdb/.cif).")
    return out

def customPDB2AFID(customPDB: str, version=None, strict_version=True) -> str:
    """
    Returns canonical AF filename stem: AF-<ACC>-F1-model_vN

    strict_version=True:
      - if version specified but unavailable -> raise ValueError
    strict_version=False:
      - if version specified but unavailable -> fallback to latest API record
    Example:
    print(customPDB2AFID("O00189"))
    print(customPDB2AFID("AF-O00189-F1"))
    print(customPDB2AFID("AF-O00189-F1-model_v6"))
    print(customPDB2AFID("AF-O00189-F1-model_v4"))
    """
    r_AF_w_model = re.compile(r'^AF-([A-Za-z0-9]{6,10})-F\d+-model_v(\d+)$', re.I) # AF-O00189-F1-model_v6
    r_AF_wo_model = re.compile(r'^AF-([A-Za-z0-9]{6,10})-F\d+$', re.I) # AF-O00189-F1
    r_UNIPROT_ACC = re.compile(r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$', re.I) # O00189

    # acc, version_in_input = _extract_acc_and_version(customPDB)
    # Extracted version from input
    version_in_input = None
    if m := r_AF_w_model.fullmatch(customPDB):
        acc = m.group(1).upper()
        version_in_input = int(m.group(2))
    elif m := r_AF_wo_model.fullmatch(customPDB):
        acc = m.group(1).upper()
    elif m := r_UNIPROT_ACC.fullmatch(customPDB):
        acc = m.group(0).upper()
    else: 
        raise ValueError(f"Unsupported customPDB format: {customPDB}")
    
    # If version specified, verify availability
    target_version = version if version is not None else version_in_input
    if target_version is not None:
        afid = f"AF-{acc}-F1-model_v{int(target_version)}"
        pdb_url = f"https://alphafold.ebi.ac.uk/files/{afid}.pdb"
        cif_url = f"https://alphafold.ebi.ac.uk/files/{afid}.cif"
        if _url_exists(pdb_url) or _url_exists(cif_url):
            return afid
        if strict_version:
            raise ValueError(f"Requested AlphaFold version v{target_version} not available for {acc}")

    # Resolve latest via API
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
    r = requests.get(api_url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"No AlphaFold prediction found for {acc}")

    rec = data[0]
    entry = rec.get("entryId") or rec.get("modelEntityId")
    if entry:
        # entry may be AF-...-F1; normalize to model_vN if possible from URLs
        # Prefer extracting exact model from pdbUrl/cifUrl when present
        for key in ("pdbUrl", "cifUrl"):
            u = rec.get(key) or ""
            m = re.search(r'(AF-[A-Za-z0-9]{6,10}-F\d+-model_v\d+)\.(?:pdb|cif)$', u)
            if m:
                return m.group(1)
        # fallback
        return f"{entry}-model_v4" if "-model_v" not in entry else entry

    raise ValueError(f"Unable to resolve AlphaFold ID for {acc}")

def verifyAF(pdbpath, return_message=False):
    """Check if a structure file is likely from AlphaFold.

    Rules:
    1) Text marker rule: file content contains "alphafold" or
       "alphafoldserver.com/output-terms" (any extension) -> AlphaFold.
    2) CIF rule: >=20% atoms have B-factor >= 50 -> AlphaFold.
    3) Generic confidence rule: all B-factors < 100 and >=50% atoms have
       B-factor >= 50 -> AlphaFold.
    """

    def _out(flag: bool, msg: str):
        return (flag, msg) if return_message else flag

    try:
        raw_path = os.fspath(pdbpath)
    except TypeError:
        return _out(False, "Input is not a valid filesystem path.")

    path = os.path.abspath(raw_path)
    title = os.path.basename(path)
    stem = os.path.splitext(title)[0]
    ext = os.path.splitext(path)[1].lower()

    # Fast path by AlphaFold-style filename (case-insensitive)
    # Accepts: AF-O00189-F1, AF-O00189-F1-model_v6 (+ any extension)
    if re.match(r'^AF-[A-Za-z0-9]{6,10}-F\d+(?:-model_v\d+)?$', stem, re.I):
        return _out(True, "Filename matches AlphaFold naming pattern.")

    if not os.path.isfile(path):
        return _out(False, f"File not found: {path}")

    # Rule 1: marker text in file content (any extension)
    try:
        with open(path, 'r', errors='ignore') as f:
            head = f.read(65536).lower()
        if ("alphafoldserver.com/output-terms" in head) or ("alphafold" in head):
            return _out(True, "Detected AlphaFold marker text `alphafoldserver.com/output-terms` in file content.")
    except Exception as e:
        LOGGER.warn(f"verifyAF: unable to read file text from {path}: {e}")

    # Parse B-factors for rules 2 and 3
    ag = None
    try:
        if ext in (".cif", ".mmcif") and hasattr(prody, "parseMMCIF"):
            ag = prody.parseMMCIF(path)
        if ag is None:
            ag = parsePDB(path, model=1)
    except Exception as e:
        return _out(False, f"Failed to parse structure for B-factor analysis: {e}")

    if ag is None:
        return _out(False, "Unable to parse structure (AtomGroup is None).")

    betas = ag.getBetas()
    if betas is None or len(betas) == 0:
        return _out(False, "No B-factor data available.")

    betas = np.asarray(betas, dtype=float)
    frac_ge50 = float(np.mean(betas >= 50.0))
    all_lt100 = bool(np.all(betas < 100.0))

    # Rule 2: CIF-specific threshold
    if ext in (".cif", ".mmcif") and frac_ge50 >= 0.20:
        return _out(True, f"CIF rule matched: {frac_ge50:.1%} atoms have B-factor >= 50.")

    # Rule 3: generic confidence rule
    if all_lt100 and frac_ge50 >= 0.50:
        return _out(True, f"Confidence rule matched: all B-factors < 100 and {frac_ge50:.1%} atoms >= 50.")

    return _out(False, f"Rules not matched (fraction>=50: {frac_ge50:.1%}, all<100: {all_lt100}).")

def fetchPDB(pdbID, **kwargs):
    """Fetch a PDB file from RCSB PDB database."""
    
    pdbID = pdbID.lower()
    folder = kwargs.get('folder', '.')
    compressed = kwargs.get('compressed', True)
    format = kwargs.get('format', 'pdb')
    refresh = kwargs.get('refresh', False)
    assert format in ['pdb', 'cif', 'opm'], f"format should be 'pdb', 'cif', or 'opm'."
    os.makedirs(folder, exist_ok=True)

    # Define the URL and output path
    if format == 'pdb':
        outpath = os.path.join(folder, f'{pdbID}.pdb.gz')
        url = f"https://files.rcsb.org/download/{pdbID}.pdb.gz"
    elif format == 'cif':
        outpath = os.path.join(folder, f'{pdbID}.cif.gz')
        url = f"https://files.rcsb.org/download/{pdbID}.cif.gz"
    else: # format == 'opm'
        outpath = os.path.join(folder, f'{pdbID}-opm.pdb')
        url = f"https://opm-assets.storage.googleapis.com/pdb/{pdbID}.pdb"
    
    outpath = os.path.abspath(outpath)
    # Check compressed
    if not compressed and format != 'opm':
        outpath = outpath[:-3] 
        url = url[:-3]
    # Check refresh
    if not refresh:
        if os.path.exists(outpath):
            return outpath
        
    # Fetch the file
    try:
        urllib.request.urlretrieve(url, outpath)
        if format == 'opm':
            # Remove 'END' lines from OPM file
            # > This helps PDBFixer recognize the Dummy atoms in some OPM files 
            with open(outpath, 'r') as file:
                lines = file.readlines()
            lines = [line for line in lines 
                if not line.startswith('END') and not line.startswith('CRYST1')]
            with open(outpath, 'w') as file:
                file.writelines(lines)
        if format == 'pdb' and not compressed:
            # Remove 'CRYST1' lines from PDB file
            # This keeps the PDBFixer from raising an error
            with open(outpath, 'r') as file:
                lines = file.readlines()
            lines = [line for line in lines if not line.startswith('CRYST1')]
            with open(outpath, 'w') as file:
                file.writelines(lines)
        elif format == 'pdb' and compressed:
            # Remove 'CRYST1' lines from PDB file
            # This keeps the PDBFixer from raising an error
            with open(outpath, 'rb') as file:
                lines = file.readlines()
            lines = [line for line in lines if not line.startswith(b'CRYST1')]
            with open(outpath, 'wb') as file:
                file.writelines(lines)
        return outpath
    except Exception as e:
        # msg = traceback.format_exc()
        # LOGGER.info(msg)
        LOGGER.info(f"Failed to fetch {pdbID} from RCSB PDB database {e}.")
        if format != 'cif':
            LOGGER.info(f"Fetch cif file instead.")
            return fetchPDB(pdbID, format='cif', folder=folder, compressed=compressed, refresh=refresh)
        return None

def fetchPDB_BiologicalAssembly(pdbID, assemblyID=1, **kwargs):
    """Fetch a PDB file from RCSB PDB database."""
    
    pdbID = pdbID.lower()
    assemblyID = int(assemblyID)
    folder = kwargs.get('folder', '.')
    compressed = kwargs.get('compressed', True)
    format = kwargs.get('format', 'pdb')
    assert format in ['pdb', 'cif'], f"format should be 'pdb' or 'cif'."
    assert assemblyID > 0, f"assemblyID should be greater than 0."
    os.makedirs(folder, exist_ok=True)  

    if format == 'pdb':
        outpath = os.path.join(folder, f'{pdbID}.pdb{assemblyID}.gz')
        url = f"https://files.rcsb.org/download/{pdbID}.pdb{assemblyID}.gz"
    else: # format == 'cif'
        outpath = os.path.join(folder, f'{pdbID}-assembly{assemblyID}.cif.gz')
        url = f"https://files.rcsb.org/download/{pdbID}-assembly{assemblyID}.cif.gz"
    outpath = os.path.abspath(outpath)

    # Check if the file already exists
    if compressed:
        if os.path.exists(outpath):
            return outpath
    else:
        # Remove the '.gz' extension
        outpath = outpath[:-3] 
        url = url[:-3]
        if os.path.exists(outpath):
            return outpath
        
    # Fetch the file
    try:
        urllib.request.urlretrieve(url, outpath)
        return outpath
    except Exception as e:
        logging.error(f"Failed to fetch {pdbID} from RCSB PDB database.")
        logging.error(e)
        return None
