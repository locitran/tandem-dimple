from pdbfixer import PDBFixer
import numpy as np
from openmm.app.element import hydrogen
import openmm.app as app
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from collections import OrderedDict
import traceback
import pandas as pd
import gzip
import logging
import os
import subprocess
from pathlib import Path
from .mpdbfile import PDBFile
from .utils.settings import one2three, three2one, standard_aa, ROOT_DIR

download_dir = ROOT_DIR / 'pdbfile/raw'
fix_dir = ROOT_DIR / 'pdbfile/fix'
_LOGGER = logging.getLogger(__name__)

class LociFixer(object):
    """
    This class is used to fix missing residues, nonstandard residues, and missing atoms in a PDB file.

    Parameters
    ----------
    pdbID : str
        The ID of the PDB file.
    pdbPath : str
        The path to the PDB file.

    Attributes
    ----------
    pdbID : str
        The ID of the PDB file.
    pdbPath : str
        The path to the PDB file.
    format : str
        The file format of the PDB file.
    name : str
        The name of the PDB file.
    fixer : PDBFixer
        The PDBFixer object used for fixing the PDB file.

    Methods
    -------
    _validateSAVs(extension, membrane=False, assemblyID=None)
        Validate SAVs in the PDB file using default or mapping PDB coordinates.
    _processWT(fix_dir, membrane=False, assemblyID=None)
        Process the wild type PDB file for membrane or non-membrane proteins.
    _processMT(SAV_coords, PDB_coords, fix_dir, assemblyID=None)
        Process the mutant PDB files based on the given SAV and PDB coordinates.
    readPDB()
        Read the PDB file and initialize the PDBFixer object.
    fix(fix_loop=True, replaceNonstandard=True, keepElement=[])
        Fix missing residues, nonstandard residues, and missing atoms in the PDB file.
    mutate(point_mutation=None, chain=None)
        Mutate a residue in the PDB file.

    Examples
    --------

    >>> fixer = LociFixer(pdbID, pdbPath)
    >>> fixer.fix(fix_loop=True, replaceNonstandard=True, keepElement=[])
    >>> fixer.saveTopology(savePath)
    """
    def __init__(self, pdbID, pdbPath):
        self.pdbID = pdbID
        self.pdbPath = pdbPath
        dir = os.path.split(pdbPath)[0]
        file = os.path.split(pdbPath)[1]
        name = file.split('.')
        if len(name) == 2:
            name = file[0:file.rfind('.')]
            self.format = pdbPath[pdbPath.rfind('.'):]
        else: # len(name) > 2: Compressed file
            if 'gz' and 'cif' in name:
                self.format = '.cif.gz'
                name = name[0]
            elif 'gz' and 'pdb' in name:
                self.format = '.pdb.gz'
                name = name[0]
        self.name = name
        self.readPDB()

    def _validateSAVs(self, extension: pd.DataFrame, file_format: str, keepIds: bool = True):
        # Store the topology in a dictionary: topology = {chainID: {'resID iCode': resName, ...}, ...}
        topology = dict()
        for chain in self.fixer.topology.chains():
            topology[chain] = dict()
            for residue in chain.residues():
                resID_iCode = f'{residue.id}{residue.insertionCode}'.strip()
                topology[chain][resID_iCode] = residue
        print(f'> {self.name} chains: {[chain.id for chain in topology.keys()]}')

        def _map(PDB_coord):
            "Map PDB_coord to the structure"
            return_value = [False, '', PDB_coord] # Default return value [valid status, error message, PDB_coord]
            pdbID, chainID, resID, wt = PDB_coord.split()

            chain = [chain for chain in topology.keys() if chain.id == chainID]
            if len(chain) == 0:
                return_value[1] = f'Chain {chainID} not found in the structure.'
                return return_value
            if len(chain) > 1:
                return_value[1] = f'Multiple chains {chainID} found in the structure.'
                return return_value
            
            chain = chain[0]
            possible_resID = [id for id in topology[chain] if id.startswith(resID)]
            if len(possible_resID) == 0:
                return_value[1] = f'resID {resID} not found in chain {chainID}.'
                return return_value
            
            for resID_iCode in possible_resID:
                try:
                    residue = topology[chain][resID_iCode]
                    if wt == three2one[residue.name]: # Correct residue

                        if keepIds and len(chain.id) == 1:
                            chainID = chain.id
                        elif keepIds and len(chain.id) > 1:
                            raise KeyError('Chain ID is too long. Please set keepIds=False to re-index chainID.')
                        else:
                            chainIndex = chain.index
                            chanName_after_61 = {
                                62: '~', 63: '!', 64: '@', 65: '#', 66: '%', 67: '-', 
                                68: '&', 69: '*', 70: '_', 71: '+', 72: '=', 73: '|',
                                74: '?', 75: '>', 76: '<', 77: '^',
                            }
                            if chainIndex in range(26):
                                chainID = chr(ord('A')+chainIndex)
                            elif chainIndex in range(26, 52):
                                chainID = chr(ord('a')+chainIndex-26)
                            elif chainIndex in range(52, 62):
                                chainID = str(chainIndex-52)
                            elif chainIndex in range(62, 78):
                                chainID = chanName_after_61[chainIndex]
                            else:
                                raise ValueError('Chain index is out of range [0, 87]. Please set keepIds=True to keep the original chainID.')
                            print(f'> ChainID {chain.id} -> {chainID}')

                        PDB_coord = f'{pdbID} {chainID} {resID_iCode} {wt}'
                        return_value = [True, chainID, PDB_coord]
                        break
                    else:
                        return_value[1] = f'WT residue {wt} not matched with {topology[chain][resID_iCode]} in chain {chainID}.'
                except:
                    continue

            return return_value

        def _validate(i, PDB_coord):
            if len(PDB_coord.split()) != 4:
                extension.at[i, 'indices'] = False
                extension.at[i, 'file_format'] = 'Invalid PDB_coord'
                extension.at[i, 'mapping_PDB_coords'] = PDB_coord
                return
            
            return_value = _map(PDB_coord)
            print(f'> Validating SAV {i}: {PDB_coord} - {return_value[0]}')

            if return_value[0]:
                extension.at[i, 'indices'] = True
                extension.at[i, 'file_format'] = file_format
            else:
                extension.at[i, 'indices'] = False
                extension.at[i, 'file_format'] = return_value[1]
            extension.at[i, 'mapping_PDB_coords'] = return_value[2]

        PDB_coords = extension['PDB_coords'].tolist()
        
        # Filter False indices in indices
        invalid_indices = extension[extension['indices'] == False]
        n_invalid_indices = len(invalid_indices)
        print(f'Before validation, {n_invalid_indices} SAVs need to be validated.')

        print(f'> Validating SAVs in {self.name} using default PDB_coords...')
        for i, row in invalid_indices.iterrows():
            PDB_coord = row['PDB_coords']
            _validate(i, PDB_coord)

        # Validate SAVs using mapping_PDB_coords if default PDB_coords does not work
        invalid_indices = extension[extension['indices'] == False]
        print(f'After validating, {len(invalid_indices)} SAVs cannot be validated.')
        if len(invalid_indices) == n_invalid_indices and self.format in ['.cif', '.cif.gz']:
            print(f'> Validating SAVs in {self.name} using mapping_PDB_coords...')
            mapping_PDB_coords = self.map_chainID(PDB_coords)
            for i in invalid_indices:
                PDB_coord = mapping_PDB_coords[i]
                _validate(i, PDB_coord)

            invalid_indices = extension[extension['indices'] == False]
            n_invalid_indices = len(invalid_indices)
            print(f'After validation, {n_invalid_indices} SAVs cannot be validated.')

        return extension

    def _processWT(self, 
                   fix_dir: str = fix_dir, 
                   membrane: bool = False,
                   assemblyID: int = None,
                   keepIds: bool = True, 
                   fix_loop: bool = True):
        """
        For membrane protein:
        Input:  raw opm file (with dummy atoms)      = self.pdbPath     download_dir/pdbID-opm.pdb
        Output: fixed opm file (with dummy atoms)    = opm_file         fix_dir/pdbID-opm.pdb
                fixed wt file (without dummy atoms)  = wt_file          fix_dir/pdbID.pdb
                fixed ne1 file (with NE11 atoms)     = ne1_file         fix_dir/pdbID-ne1.pdb

        For non-membrane protein:
        Input:  raw pdb file (or cif file)           = self.pdbPath     download_dir/pdbID.pdb
        Output: fixed wt file                        = wt_file          fix_dir/pdbID.pdb
        5ZM8, 2ZW3, 2LZL, 2K21, 1IKT, 6V00
        """
        assemblyID = f'-{assemblyID}' if assemblyID else '' # 1G0D.pdb or 1G0D-1.pdb
        wt_file = os.path.join(fix_dir, f'{self.pdbID.lower()}{assemblyID}.pdb')
        if membrane: # TODO Fix membrane protein
            print(f'> Fixing {self.pdbID} with OPM')
            opm_file = os.path.join(fix_dir, f'{self.pdbID.lower()}{assemblyID}-opm.pdb')
            ne1_file = os.path.join(fix_dir, f'{self.pdbID.lower()}{assemblyID}-ne1.pdb')           # Build ne1 molecule
                
            self.fix(fix_loop=fix_loop, replaceNonstandard=True, keepElement=['DUM'])   # Keep dummy atoms
            self.saveTopology(savePath=opm_file, keepIds=keepIds)
            buildNE1(opm_file, ne1_file)

            fixer = LociFixer(pdbID=self.pdbID, pdbPath=opm_file)
            fixer.fix(fix_loop=fix_loop, replaceNonstandard=True, keepElement=[])        # Remove dummy atoms
            fixer.saveTopology(savePath=wt_file, keepIds=keepIds)
            print(f'> {wt_file} created.')
        else:   # TODO Fix non-membrane protein
            self.fix(fix_loop=fix_loop, replaceNonstandard=True, keepElement=[])
            self.saveTopology(savePath=wt_file, keepIds=keepIds)
            print(f'> {wt_file} created.')

    def _processMT(self, 
                   extension: np.array,
                   file_format: str,
                   fix_dir: str = fix_dir,
                   assemblyID: int = None,
                   refresh: bool = True,
                   keepIds: bool = True,
                   fix_loop: bool = False,):
        """Process PDB file
        Input:  fixed pdb file  = self.pdbPath     fix_dir/{pdbID}.pdb
        Output: fixed mt file   = mt_file          fix_dir/{pdbID}_{chainID}_{wt}{resID_iCode}{mt}.pdb (e.g. 1A09_A_F57AG.pdb )
        """
        assemblyID = f'-{assemblyID}' if assemblyID else '' # 1G0D.pdb or 1G0D-1.pdb
        wt_file = os.path.join(fix_dir, f'{self.pdbID.lower()}{assemblyID}.pdb')

        valid_indices = extension[ (extension['indices'] == True) & (extension['file_format'] == file_format) ]
        for i, row in valid_indices.iterrows():
            PDB_coord_splitting = row['PDB_coords'].split() # pdbID, chainID, resID, wt
            SAV_coord_splitting = row['SAV_coords'].split() # uniprotACC, resID, wt, mt
            if str(PDB_coord_splitting[3]) != str(SAV_coord_splitting[2]):
                msg = (f"> TANDEM:LociFixer: {self.name}: {row.PDB_coords}, {row.SAV_coords} "
                        f"have different wt residues. {PDB_coord_splitting[3]} and {SAV_coord_splitting[2]}")
                _LOGGER.error(msg)
            else:
                mt                          = SAV_coord_splitting[3]    # MT
                pdbID, chainID, resID, wt   = PDB_coord_splitting   # resID could be resID_iCode
                mutate                      = f'{wt}{resID}{mt}'   # wt + resID + mt
                mt_file                     = os.path.join(fix_dir, f'{pdbID.lower()}{assemblyID}_{chainID}_{mutate}.pdb')
                if os.path.exists(mt_file) and not refresh:
                    print(f'> {mt_file} already exists.')
                else:
                    try:
                        fixer = LociFixer(pdbID=self.pdbID, pdbPath=wt_file)
                        fixer.mutate(point_mutation=mutate, chainID=chainID, fix_loop=fix_loop)
                        fixer.saveTopology(savePath=mt_file, keepIds=keepIds)
                        print(f'> {mt_file} created.')
                    except Exception as e:
                        e = traceback.format_exc()
                        extension.at[i, 'indices'] = False
                        extension.at[i, 'file_format'] = 'Cannot mutate'
                        msg = f"> TANDEM:LociFixer: {self.name}: {row.PDB_coords}, {row.SAV_coords} cannot be mutated. {e}"
                        _LOGGER.error(msg)
        return extension

    def readPDB(self):
        if self.format == '.cif':
            with open(self.pdbPath) as in_f:
                self.fixer = PDBFixer(pdbxfile=in_f)
        elif self.format == '.pdb':
            with open(self.pdbPath) as in_f:
                self.fixer = PDBFixer(pdbfile=in_f)
        elif self.format == '.cif.gz':
            with gzip.open(self.pdbPath, 'rt') as in_f:
                self.fixer = PDBFixer(pdbxfile=in_f)
        elif self.format == '.pdb.gz':
            with gzip.open(self.pdbPath, 'rb') as in_f:
                self.fixer = PDBFixer(pdbfile=in_f)
        else:
            _LOGGER.error('> TANDEM:LociFixer: Invalid file format. Only .cif, .pdb, .cif.gz, and .pdb.gz are supported.')
            raise ValueError('Invalid file format. Only .cif, .pdb, .cif.gz, and .pdb.gz are supported.')

    def fix(self, fix_loop=True, replaceNonstandard=True, keepElement=[]):
        """
        Fix missing residues, nonstandard residues, missing atoms
        Except N/C-terminus and loops with more than 12 missing residues
        Remove heterogens (including water)
        Add missing hydrogens (7.0 pH)

        Parameters
        ----------
        fix_loop : bool, optional
            Whether to fix loops with less than 12 missing residues. The default is True.
        
        replaceNonstandard : bool, optional
            Whether to replace nonstandard residues. The default is True.

        keepElement : list, optional
            The elements to keep. The default is [].

        missing_residues : dict
            The missing residues in the PDB file.
            Structure: {(chainIndex, residueIndex): [residueName, ...], ...}
            
        Example:
        >>> fixer = LociFixer(pdbPath)
        1. Fix missing loop (≤12 residues)
        2. Remove heterogens (including water)
        3. Replace nonstandard residues
        >>> fixer.fix() # For wild type
        
        1. Keep missing loop
        2. Remove heterogens (including water)
        3. Replace nonstandard residues
        >>> fixer.fix(fix_loop=False) # For membrane protein

        Notes:
        chain.id: asym_id
        
        """

        self.fixer.findMissingResidues()
        self.missing_residues = self.fixer.missingResidues.copy()
        chains = list(self.fixer.topology.chains())
        keys = list(self.fixer.missingResidues.keys())
        self.ms_Middle = {}          # {chainId: [[residueId, number of ms], ...], ...}
        self.ms_N = {}               # {chainId: [residueId, number of ms], ...}
        self.ms_C = {}               # {chainId: [residueId, number of ms], ...}
        for key in keys:
            chain = chains[key[0]]
            if key[1] == 0:
                self.ms_N[chain.index] = [key[1], len(self.fixer.missingResidues[key])]
                del self.fixer.missingResidues[key]
            elif key[1] == len(list(chain.residues())):
                self.ms_C[chain.index] = [key[1], len(self.fixer.missingResidues[key])]
                del self.fixer.missingResidues[key]
            else:
                if chain.index not in self.ms_Middle:
                    self.ms_Middle[chain.index] = []

                self.ms_Middle[chain.index].append([key[1], len(self.fixer.missingResidues[key])])
                if len(self.fixer.missingResidues[key]) > 12:
                    del self.fixer.missingResidues[key]

        if replaceNonstandard:
            self.fixer.findNonstandardResidues()
            self.fixer.replaceNonstandardResidues()
        
        self.fixer.removeHeterogens(keepElement=keepElement)
        self.fixer.findMissingAtoms()
        self.missing_atoms = self.fixer.missingAtoms.copy()
        self.missing_terminals = self.fixer.missingTerminals.copy()
        
        if not fix_loop:
            self.fixer.missingAtoms = {}     
            self.fixer.missingTerminals = {}
            self.fixer.missingResidues = {}    
        self.fixer.addMissingAtoms()

    def get_missing_residues(self):
        """Format
        self.missing_residues = {
            (chainIndex, residueIndex): [residueName, ...],
            (chainIndex, residueIndex): [residueName, ...],
            ...
        }
        Example:
        {(0, 0): ['MET', 'ALA', 'SER', 'TYR', 'LYS'],}
        => Chain 0, residue 0: Missing residues [MET, ALA, SER, TYR, LYS]
        """
        return self.missing_residues
    
    def get_missing_atoms(self):
        """Format
        self.missing_atoms = {
            residue: [atomName, ...],
        """
        return self.fixer.missingAtoms
    
    def get_missing_terminals(self):
        """Format
        self.missing_terminals = {
            residue: [atomName, ...],
        """
        return self.fixer.missingTerminals

    def mutate(self, point_mutation, chainID, fix_loop=True):
        """Mutate a residue
        GLY-6-GLU
        
        point_mutation = 'G6E'
        chainID = 'A'
        """
        wt, resID_iCode, mt = point_mutation[0], point_mutation[1:-1], point_mutation[-1]
        
        residue_map = dict()
        for chain in self.fixer.topology.chains():
            if chain.id == chainID:
                for residue in chain.residues():
                    residue_ID_iCode = f'{residue.id}{residue.insertionCode}'.strip()
                    if (residue_ID_iCode == resID_iCode) and (three2one[residue.name] == wt):
                        residue_map[residue] = one2three[mt]
                        break
            
        try:
            template = self.fixer.templates[one2three[mt]]
        except KeyError:
            raise(KeyError("Cannot find residue %s in template library!" % one2three[mt]))
        
        # Below is the code from pdbfixer
        # If there are mutations to be made, make them.
        # print(residue_map) # {<Residue 380 (PHE) of chain 0>: 'LEU'}
        if len(residue_map) > 0:
            deleteAtoms = [] # list of atoms to delete

            # Find atoms that should be deleted.
            for residue in residue_map.keys():
                replaceWith = residue_map[residue]
                residue.name = replaceWith
                template = self.fixer.templates[replaceWith]
                standardAtoms = set(atom.name for atom in template.topology.atoms())
                for atom in residue.atoms():
                    if atom.element in (None, hydrogen) or atom.name not in standardAtoms:
                        deleteAtoms.append(atom)

            # Delete atoms queued to be deleted.
            modeller = app.Modeller(self.fixer.topology, self.fixer.positions)
            modeller.delete(deleteAtoms)
            self.fixer.topology = modeller.topology
            self.fixer.positions = modeller.positions

        self.fixer.findMissingResidues()
        self.fixer.findMissingAtoms()

        if not fix_loop:
            try:
                self.fixer.missingTerminals = {}
                self.fixer.missingResidues = {}
                for residue in self.fixer.missingAtoms:
                    residue_id_ic = f'{residue.id}{residue.insertionCode}'.strip()
                    if residue_id_ic == resID_iCode and \
                        three2one[residue.name] == mt and \
                        residue.chain.id == chainID:
                        break
                self.fixer.missingAtoms = self.fixer.missingAtoms[residue]
            except:
                pass
            
        self.fixer.addMissingAtoms()

    def saveTopology(self, savePath, keepIds=True, modify_chain=None):
        """Save topology to a pdb file

        In our setting, we reset the chainID using ```chainName = chr(ord('A')+chainIndex%26)```
        """
        with open(savePath, 'w') as out_f:
            out_f.write('HEADER\n')
            PDBFile.writeFile(self.fixer.topology, self.fixer.positions, out_f, keepIds=keepIds, modify_chain=modify_chain)

    def map_chainID(self, PDB_coords):
        if self.format == '.cif':
            with open(self.pdbPath) as in_f:
                mmcif_dict = MMCIF2Dict(in_f)
        elif self.format == '.cif.gz':
            with gzip.open(self.pdbPath, 'rt') as in_f:
                mmcif_dict = MMCIF2Dict(in_f)
        else:
            _LOGGER.error('> TANDEM:LociFixer: Invalid file format. Only .cif and .cif.gz are supported.')
            raise ValueError('Invalid file format. Only .cif and .cif.gz are supported.')
        
        asym_id         = mmcif_dict["_pdbx_poly_seq_scheme.asym_id"]  # {'A', 'B'}
        pdb_strand_id   = mmcif_dict["_pdbx_poly_seq_scheme.pdb_strand_id"]  # {'D', 'F'}
        unique_asym_id = list(OrderedDict.fromkeys(asym_id))                # BAS file
        unique_pdb_strand_id = list(OrderedDict.fromkeys(pdb_strand_id))    # ASU file
        for i, PDB_coord in enumerate(PDB_coords):
            PDB_coord_splitting = PDB_coord.split() # Record the chainID in PDB file (ASU)
            for j, _chainID in enumerate(unique_pdb_strand_id): # asu-chainID recorded for ASU
                if _chainID == PDB_coord_splitting[1]:          # asu-chainID == SAVs-chainID
                    PDB_coord_splitting[1] = unique_asym_id[j]  # asu-chainID -> bas-chainID
                    PDB_coord = ' '.join(PDB_coord_splitting)
                    print(f'> Mapping ASU-chainID to BAS-chainID: {PDB_coords[i]} to {PDB_coord}')
                    PDB_coords[i] = PDB_coord
                    break
        return PDB_coords
    
def cif2pdb(pdbID, keepIds=True, assemblyID: int = None, cifPath=None, pdbPath=None):
    if assemblyID:
        if cifPath: # Raise error if cifPath provided
            raise ValueError('cifPath is not required if assemblyID is provided.')
        import download 
        cifPath = download.cif_biological_assembly(pdbID, assemblyID)
        if cifPath is None:
            return

    fixer = LociFixer(pdbID, cifPath)
    if fixer.format != '.cif' or fixer.format != '.cif.gz':
        raise ValueError('Invalid file format. Only .cif and .cif.gz are supported.')
    fixer.saveTopology(pdbPath, keepIds=keepIds)
    return print(f'> {pdbPath} created.')

def buildNE1(OPM_file, NE1_file, tempFolder=None):
    if tempFolder is None:
        command = ['bash', f'{ROOT_DIR}/src/OPM_build.sh', OPM_file, NE1_file]
    else:
        command = ['bash', f'{ROOT_DIR}/src/OPM_build.sh', OPM_file, NE1_file, tempFolder]
    subprocess.run(command)
    print(f'> BuildNE1: {OPM_file} -> {NE1_file}')
