"""
Modified by Dr. Yuan-Yu Chang for the NativeEnsemble project.
Later modified by Mr. Loci for IMPROVE project.
Version: ...
Date: 2023-09-07
"""

import pandas as pd
from shutil import copyfile
from subprocess import Popen
from uuid import uuid1
import os, logging
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)
HBPLUSEXE = Path(__file__).parent / 'bin' / 'hbplus'
tempFolder = Path(__file__).parent / 'temp'

class HBplus:
    """
    Get HBplus data from PDB file

    Example:
    >>> pdbPath = '1G0D.pdb'
    >>> hbplus = HBplus.HBplus(pdbPath, disCutoff=3.5)
    >>> hbPdb = hbplus.getPDB()
    >>> hbondData = hbplus.getHbond(hbPdb, skipNeighbor=1)
    """
    def __init__(self, pdbPath, hbplusDisCutOff=3.9, tempFolder=tempFolder):
        self.pdbPath = pdbPath
        self.hbplusDisCutOff = hbplusDisCutOff
        self.tempFolder = tempFolder

    def getPDB(self):
        """Get HBplus data from PDB file
        Running HBplus will produce 2 files: .hb2 and hbdebug.dat. 
        We save it in temporary folder and remove it after getting data.
        
        Return:
        -------
        HPdata: list of list
            HBplus data
        """
        tempName = str(uuid1())                                     # random string for temp file name
        temppdbPath = '{}/{}.pdb'.format(self.tempFolder, tempName) # temp pdb file path
        temphb2Path = '{}/{}.hb2'.format(self.tempFolder, tempName) # temp hb2 file path
        temphbdebugPath = '{}/hbdebug.dat'.format(self.tempFolder)  # temp hbdebug file path
        copyfile(self.pdbPath, temppdbPath)                         # copy pdb file to temp folder
        tempFiles = [temppdbPath, temphb2Path, temphbdebugPath]     # list of temp files

        command = [HBPLUSEXE, '-d', str(self.hbplusDisCutOff), '-A', '120', '0', '0', '-e', 'HOH', '" O "', '0', temppdbPath]
        try:
            proc = Popen(command, cwd=self.tempFolder)
            proc.wait()
        except Exception:
            logger.exception(f"> TANDEM:HBplus: {traceback.format_exc()}")
            return None

        with open(temphb2Path, 'r') as hbf:
            lines = hbf.readlines()[8:] # skip headers
            HPdata = []
            for line in lines:
                subData = []
                subData.append(line[0].strip())                # 0 chainID-D
                subData.append(int(line[1:5].strip()))         # 1 resID-D
                subData.append(line[5].strip('-'))             # 2 iCode-D
                subData.append(line[6:9].strip())              # 3 resName-D
                subData.append(line[9:13].strip())             # 4 atomName-D
                subData.append(line[14].strip())               # 5 chainID-A
                subData.append(int(line[15:19].strip()))       # 6 resID-A
                subData.append(line[19].strip('-'))            # 7 iCode-A
                subData.append(line[20:23].strip())            # 8 resName-A
                subData.append(line[23:27].strip())            # 9 atomName-A
                subData.append(float(line[27:32].strip()))     # 10 distance
                subData.append(line[33:35])                    # 11 type
                subData.append(float(line[64:69]))             # 12 D-A-AA angle
                HPdata.append(subData)

        for file in tempFiles:
            if os.path.isfile(file):
                os.remove(file)
        return HPdata

    def getHbond(self, HPdata, skipNeighbor=1):
        """Get Hbond data from HBplus data
        Parameters:
        -----------
        HPdata: list of list
            HBplus data getted from getPDB()

        skipNeighbor: int, default=1
            Skip neighbor residues in Hbond calculation (0: not skip, 1: skip 1 neighbor, ...)
            In IMPROVE project, we set skipNeighbor=1 to remove Hbond formed by current residue and its neighbors (1 residue away).

        Return:
        -------
        Hbond: dataframe
            'resName', 'chainID', 'resID', 'iCode', 'h_bond_group'
        """
        if HPdata is None:
            return None

        Hbond = {}
        for pair in HPdata:
            # remove +1, -1 and self pairs
            if pair[0] == pair[5] and pair[2] == pair[7] and abs(pair[1] - pair[6]) > skipNeighbor:
                donor           = (pair[3], pair[0], pair[1], pair[2])  # resName, chainID, resID, iCode of donor
                acceptor        = (pair[8], pair[5], pair[6], pair[7])  # resName, chainID, resID, iCode of acceptor
                
                if pair[3] == 'HOH' or pair[8] == 'HOH':                # if donor or acceptor is water
                    continue      

                Hbond[donor]    = Hbond[donor] + 1 if donor in Hbond else 1         # count Hbond for donor
                Hbond[acceptor] = Hbond[acceptor] + 1 if acceptor in Hbond else 1   # count Hbond for acceptor

        Hbond = pd.DataFrame([(key[0], key[1], key[2], key[3], val) for key, val in Hbond.items()],
                             columns=['resName', 'chainID', 'resID', 'iCode', 'h_bond_group'])
        return Hbond
    
# pdbPath = '1G0D.pdb'
# hbplus = HBplus(pdbPath, hbplusDisCutOff=3.5)
# hbPdb = hbplus.getPDB()
# hbondData = hbplus.getHbond(hbPdb, skipNeighbor=1)
# print(hbondData)