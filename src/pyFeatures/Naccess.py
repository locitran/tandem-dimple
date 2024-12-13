"""
This file is part of molecular_container.py taken from the propka-3.1 package.
Modified by Dr. Yuan-Yu Chang for the NativeEnsemble project.
Version: ...
Date: 2023-09-07
    command = [naccessExec, temppdbPath]
    proc = Popen(command, cwd=tempFolder)
    proc.wait()
"""

import pandas as pd
from uuid import uuid1
from shutil import copyfile
from subprocess import Popen
import re, os, logging
import traceback
from pathlib import Path
from ..utils.settings import ROOT_DIR
from ..utils.timer import getTimer

timer = getTimer('tandem', verbose=True)
logger = logging.getLogger(__name__)
NACCESSEXE = ROOT_DIR / 'src' / 'pyFeatures' / 'bin' / 'naccess'
tempFolder = ROOT_DIR / 'src' / 'pyFeatures' / 'temp'

@timer.track
def getSASA(pdbPath, tempFolder=tempFolder):  
    "Get SASA data from PDB file"
    
    tempName = str(uuid1())
    temppdbPath = '{}/{}.pdb'.format(tempFolder, tempName)
    temprsaPath = '{}/{}.rsa'.format(tempFolder, tempName)
    templogPath = '{}/{}.log'.format(tempFolder, tempName)
    tempasaPath = '{}/{}.asa'.format(tempFolder, tempName)
    copyfile(pdbPath, temppdbPath)
    tempFiles = [temppdbPath, temprsaPath, templogPath, tempasaPath]
    command = [NACCESSEXE, temppdbPath]
    
    try:
        proc = Popen(command, cwd=tempFolder)
        proc.wait()
    except Exception:
        logger.exception(f"> TANDEM:Naccess: {traceback.format_exc()}")
        return None
    respattern = re.compile('^RES')                       
    
    with open(temprsaPath, 'r') as naccessData:    
        lines = naccessData.readlines()
        sasaData = []
        for line in lines:
            if respattern.search(line):
                subdata = []
                subdata.append(line[4:7].strip())   # resName
                subdata.append(line[7:9].strip())   # chainID
                subdata.append(int(line[9:13].strip()))  # resID
                subdata.append(line[13].strip())    # iCode
                subdata += map(float, line[14:].strip().split()) #add sasa data
                sasaData.append(subdata)
    
    RSAColumns = ['resName', 
                  'chainID', 
                  'resID', 
                  'iCode', 
                  'All-atoms-ABS', 
                  'All-atoms-REL', 
                  'Total-Side-ABS', 
                  'Total-Side-REL', 
                  'Main-Chain-ABS', 
                  'Main-Chain-REL', 
                  'Non-polar-ABS', 
                  'Non-polar-REL', 
                  'All polar-ABS', 
                  'All polar-REL']
    sasaData = pd.DataFrame(sasaData, columns=RSAColumns)
    for file in tempFiles:
        if os.path.isfile(file):
            os.remove(file)
    return sasaData                   
