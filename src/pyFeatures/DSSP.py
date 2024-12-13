"""
Modified by Dr. Yuan-Yu Chang for the NativeEnsemble project.
Later modified by Mr. Loci for IMPROVE project.
Version: DSSP 4.4 https://github.com/PDB-REDO/dssp/tree/trunk
Website: https://pdb-redo.eu/dssp
Date: 2023-09-07
"""
import pandas as pd
import subprocess, logging
import traceback, logging
from pathlib import Path
from ..utils.settings import ROOT_DIR
from ..utils.timer import getTimer

timer = getTimer('tandem', verbose=True)
logger = logging.getLogger(__name__)
DSSPEXE = ROOT_DIR / 'src' / 'pyFeatures' / 'bin' / 'mkdssp'
tempFolder = ROOT_DIR / 'src' / 'pyFeatures' / 'temp'

@timer.track
class DSSP:
    """Get DSSP data from PDB file
    Example:
    >>> pdbPath = '1PFA_A_E23K.pdb'
    >>> dssp = DSSP.DSSP(pdbPath)
    >>> dsspPdb = dssp.getPDB()
    >>> dsspData = dssp.getDSSP(dsspPdb)
    """
    def __init__(self, pdbPath, tempFolder=tempFolder):
        self.pdbPath = pdbPath
        self.tempFolder = tempFolder

    def getPDB(self):
        "Get DSSP data from PDB file"
        command = [DSSPEXE, self.pdbPath, '--output-format', 'dssp']
        try:
            dsspOut = subprocess.check_output(command).decode('utf-8')
        except subprocess.CalledProcessError:
            logger.exception(f"> TANDEM:DSSP: {traceback.format_exc()}")
            return None
        
        lines = dsspOut.splitlines()
        search_string = '  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA'
        name_index = lines.index(search_string)
        lines = lines[name_index+1:]
        one_to_three = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
                        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
                        'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
        dsspData = []
        # Ref: https://swift.cmbi.umcn.nl/gv/dssp/DSSP_3.html
        # '!': discontinuity of backbone coordinates
        for line in lines:
            subData = []
            if line[13] == '!':
                continue
            subData.append(int(line[5:10].strip()))     # resID
            subData.append(line[10].strip())            # iCode
            subData.append(line[11:12].strip())         # chainID
            subData.append(one_to_three[line[13]])      # resName (one letter a.a code)
            subData.append(line[15:17].strip())         # type
            subData.append(line[18:25])                 # structure
            subData.append(line[26:29].strip())         # BP1
            subData.append(line[30:34].strip())         # BP2
            subData.append(line[35:38].strip())         # ACC
            dsspData.append(subData)
        return pd.DataFrame(dsspData, columns=['resID', 'iCode', 'chainID', 'resName', 'type', 'structure', 'BP1', 'BP2', 'ACC'])

    def getDSSP(self, dsspData):
        """Get DSSP data from PDB file
        H = a-helix
        B = residue in isolated β-bridge
        E = extended strand, participates in β ladder
        G = 310-helix
        I = π-helix
        P = κ-helix (poly-proline II helix)
        T = hydrogen-bonded turn
        S = bend
        """
        if dsspData is None:
            return None
        n_residues = len(dsspData)
        secondary2score = {'H':0.5, 'I':0.5, 'G':0.5, 'P': 0.5, 'T':1, 'S':1, 'B':0, 'E':0, '':1}
        dsspData = dsspData.replace({'type': secondary2score})
        dsspData.loc[:, 'loop_percent'] = (sum(dsspData['type'] == 1) * 100) / float(n_residues)
        dsspData.loc[:, 'sheet_percent'] = (sum(dsspData['type'] == 0) * 100) / float(n_residues)
        dsspData.loc[:, 'helix_percent'] = (sum(dsspData['type'] == 0.5) * 100) / float(n_residues)
        return dsspData
