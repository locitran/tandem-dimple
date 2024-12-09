import pandas as pd
from io import StringIO
import numpy as np
import os
import traceback
import logging
from .LociFixer import one2three, standard_aa
from .pyFeatures import PropKa, DSSP, HBplus, Naccess

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))
matFeatures_dir = os.path.join(dir_path, 'matFeatures')

class Features:
    """
    Get features from pdbPath.
    
    Example:
    >>> pdbID = '1G0D'
    >>> pdbPath = '1G0D.pdb'
    >>> import matlab.engine
    >>> eng = matlab.engine.start_matlab() 
    >>> gf = Features(pdbPath, pdbID, eng)
    >>> gf.getModes(eng=eng)
    >>> gf.matFeatures(residue_features=True, protein_features=True)
    >>> gf.pyFeatures()
    >>> gf.mergeFeatures()
    >>> gf.features
    """
    def __init__(self, pdbPath, pdbID, eng=None, GJB2=False):
        self.pdbPath = pdbPath
        self.pdbID = pdbID
        self.eng = Features.start_matlab_engine(eng)
        Features.readPDB(eng=self.eng, pdbPath=self.pdbPath, pdbID=self.pdbID, GJB2=GJB2)

    @staticmethod
    def start_matlab_engine(eng=None):
        "Start matlab engine and add path to MATLAB"
        import matlab.engine
        # MATLAB test
        try:
            eng.eval("1+1;", nargout=0)
        except:
            eng = matlab.engine.start_matlab()

        eng.eval(f"addpath('{matFeatures_dir}');", nargout=0)        # Add path to MATLAB
        eng.eval("importlib;", nargout=0)
        return eng

    @staticmethod
    def readPDB(pdbPath, pdbID, eng=None, GJB2=False):
        eng = Features.start_matlab_engine(eng)
        eng.eval(f'data = getFeatures("{pdbPath}");', nargout=0)
        if GJB2:
            eng.eval(f'pdb_data = getFeatures.parsePDB("{pdbPath}");', nargout=0)
            eng.eval("membrane = as('resn DOPC', pdb_data.full);", nargout=0)
            eng.eval("membrane = as('name P or name N or name C31 or name C21', membrane);", nargout=0)

            # Add DOPC CGs to environment in MATLAB
            eng.eval("data.environment = [data.environment, membrane];", nargout=0)

    @staticmethod
    def getModes(cutOff=7.3, eng=None):
        eng = Features.start_matlab_engine(eng)
        eng.eval(f'data = data.get_GVecs_GVals({cutOff});', nargout=0)

    def matFeatures(self, 
                    residue_features: bool = True,
                    protein_features: bool = True,
    ):
        """
        Compute features from MATLAB scripts: ./matFeatures/getFeatures

        Parameters:
        -----------
        self.matfeatures: dataframe
            'resName', 'chainID', 'resID', 'iCode', ... matFeatures ...
        """
        self.eng.eval("data = data.protein_level_features(data.GVecs, data.GVals);", nargout=0) if protein_features else None
        self.eng.eval("data = data.residue_level_features();", nargout=0) if residue_features else None
        self.eng.eval("data = data.removeField();", nargout=0)
        self.eng.eval("ca   = jsonencode(data.system);", nargout=0)    # Get ca in json format
        json_string = self.eng.workspace['ca']
        json_io = StringIO(json_string)
        matfeatures = pd.read_json(json_io)
        self.eng.eval("clear", nargout=0)                        # Clear workspace

        # Rename columns: resno -> resID, subunit -> chainID, resname -> resName
        matfeatures = matfeatures.rename(columns={'resno': 'resID', 'subunit': 'chainID', 'resname': 'resName'})
        matfeatures['resName'] = matfeatures['resName'].astype(str)
        matfeatures['chainID'] = matfeatures['chainID'].astype(str)
        matfeatures['iCode'] = matfeatures['iCode'].apply(lambda x: x.strip())
        matfeatures['iCode'] = matfeatures['iCode'].astype(str)
        self.matfeatures = matfeatures
        self.matfeatures_columns = self.matfeatures.columns
        # column names: ['resName', 'chainID', 'resID', 'iCode', ... matFeatures ...]
    
    def pyFeatures(self, pHCondition=7.0, hbplusDisCutOff=3.5, skipNeighbor=1, **kwargs):
        """
        Compute features from Python scripts: ./pyFeatures

        Parameters:
        -----------
        pHCondition: float, default=7.0
            pH condition for pKa calculation

        hbplusDisCutOff: float, default=3.5
            Distance cutoff for Hbond calculation

        skipNeighbor: int, default=1
            Skip neighbor residues in Hbond calculation

        self.pyfeatures: dataframe
            'resName', 'chainID', 'resID', 'iCode', sasa, dssp_result, loop_percent, sheet_percent, helix_percent, pka_charge, h_bond_group
        """
        addFile = kwargs.get('addFile', None)
        pdbPath = addFile if addFile else self.pdbPath
        assert len(self.matfeatures) != 0
        
        # propkaData = PropKa.getPropKa(pdbPath, pHCondition) # column names: ['resName', 'chainID', 'resID', 'iCode', 'pka_charge']---
        dssp = DSSP.DSSP(pdbPath)
        dsspPdb = dssp.getPDB()
        dsspData = dssp.getDSSP(dsspPdb)    # column names: ['resID', 'iCode', 'chainID', 'resName', 'type', 'structure', 'BP1', 'BP2', 'ACC', 'loop_percent', 'sheet_percent', 'helix_percent']

        hbplus = HBplus.HBplus(pdbPath, hbplusDisCutOff)
        hbPdb = hbplus.getPDB()
        hbondData = hbplus.getHbond(hbPdb, skipNeighbor)    # column names: ['resName', 'chainID', 'resID', 'iCode', 'h_bond_group']---
        
        sasaData = Naccess.getSASA(pdbPath) # column names: ['resName', 'chainID', 'resID', 'iCode', 'All-atoms-ABS', 'All-atoms-REL', 'Total-Side-ABS', 'Total-Side-REL', 'Main-Chain-ABS', 'Main-Chain-REL', 'Non-polar-ABS', 'Non-polar-REL', 'All polar-ABS', 'All polar-REL']
        
        pyfeatures = self.matfeatures[['resName', 'chainID', 'resID', 'iCode']].copy()  # Merge all features
        if sasaData is not None:
            sasaData    = sasaData.rename(columns={'All-atoms-REL': 'sasa'})    # Change column names: 'All-atoms-REL' -> 'sasa'
            pyfeatures  = pd.merge(pyfeatures, sasaData[['resName', 'chainID', 'resID', 'iCode', 'sasa']], 
                                   how='outer', on=['resName', 'chainID', 'resID', 'iCode'])
        else:
            pyfeatures.loc[:, 'sasa'] = np.nan
        if dsspData is not None:
            dsspData    = dsspData.rename(columns={'type': 'dssp_result'})  # Change column names: 'type' -> 'dssp_result'
            pyfeatures  = pd.merge(pyfeatures, dsspData[['resName', 'chainID', 'resID', 'iCode', 'dssp_result', 'loop_percent', 'sheet_percent', 'helix_percent']],
                                   how='outer', on=['resName', 'chainID', 'resID', 'iCode'])
        else:
            pyfeatures.loc[:, ['dssp_result', 'loop_percent', 'sheet_percent', 'helix_percent']] = np.nan
        # if propkaData is not None:
        #     pyfeatures  = pd.merge(pyfeatures, propkaData[['resName', 'chainID', 'resID', 'iCode', 'pka_charge']],
        #                            how='outer', on=['resName', 'chainID', 'resID', 'iCode'])
        #     pyfeatures['pka_charge'].fillna(0, inplace=True)
        # else:
        #     pyfeatures.loc[:, 'pka_charge'] = np.nan
        if hbondData is not None:
            pyfeatures  = pd.merge(pyfeatures, hbondData[['resName', 'chainID', 'resID', 'iCode', 'h_bond_group']],
                                   how='outer', on=['resName', 'chainID', 'resID', 'iCode'])
            pyfeatures['h_bond_group'] = pyfeatures['h_bond_group'].fillna(0)
        else:
            pyfeatures.loc[:, 'h_bond_group'] = np.nan
        pyfeatures = pyfeatures[pyfeatures['resName'].isin(standard_aa)]    # If resName is not a standard amino acid, remove it
        self.pyfeatures = pyfeatures
    
    def mergeFeatures(self):
        "Merge features from MATLAB and Python, and remove Nan values, and save to self.features"
        features = pd.merge(self.matfeatures, self.pyfeatures, how='outer', on=['resName', 'chainID', 'resID', 'iCode'])
        self.features = features

class getFeatures(Features):
    def __init__(self, eng=None, **kwargs):
        self.eng = eng
        self.cutOff = kwargs.get('cutOff', 7.3)
        self.pHCondition = kwargs.get('pHCondition', 7.0)
        self.hbplusDisCutOff = kwargs.get('hbplusDisCutOff', 3.5)
        self.skipNeighbor = kwargs.get('skipNeighbor', 1)

    def valid_indices(self, SAV_coords: list, PDB_coords: list, pdbID: str, 
                      fix_dir: str, 
                      assemblyID: int = None,
                      membrane: bool = False):

        pdbID = pdbID.lower()
        assemblyID = f'-{assemblyID}' if assemblyID else ''
        if membrane:
            WT_pdbPath = os.path.join(fix_dir, f'{pdbID}{assemblyID}-ne1.pdb')
            addFile = os.path.join(fix_dir, f'{pdbID}{assemblyID}.pdb')
        else:
            WT_pdbPath = os.path.join(fix_dir, f'{pdbID}{assemblyID}.pdb')
            addFile = None
        if not os.path.exists(WT_pdbPath):
            raise FileNotFoundError(f'TANDEM:Features.valid_indices: Error {WT_pdbPath} not exists')
        
        wt_features = self.WT_Features(pdbID, WT_pdbPath, addFile=addFile)

        # Create numpy array to store features
        n_samples = len(SAV_coords)
        n_features = len(self._get_feat_dtype())
        features = np.zeros((n_samples, n_features))
        for i, SAV_coord, PDB_coord in zip(range(n_samples), SAV_coords, PDB_coords):
            print(f'> Computing valid {i} {SAV_coord} {PDB_coord} ...')
            PDB_coord_splitting = PDB_coord.split()
            SAV_coord_splitting = SAV_coord.split()
            assert len(PDB_coord_splitting) == 4 and len(SAV_coord_splitting) == 4

            _, chainID, pos, wt = PDB_coord_splitting # 6CXI A 318 A
            mt                  = SAV_coord_splitting[3]
            MT_pdbPath = os.path.join(fix_dir, f'{pdbID}{assemblyID}_{chainID}_{wt}{pos}{mt}.pdb')
            if not os.path.exists(MT_pdbPath):
                logger.error(f'TANDEM:Features.valid_indices: Error {SAV_coord} {PDB_coord}\n{MT_pdbPath} not exists')
                features[i] = np.nan
            else:
                try:
                    mt_features = self.MT_Features(pdbID, MT_pdbPath)
                    features_i = self.merge_Features(SAV_coord, PDB_coord, wt_features, mt_features)
                    features[i] = features_i
                except Exception as e:
                    e = traceback.format_exc()
                    features[i] = np.nan
                    logger.error(f'TANDEM:Features.valid_indices: Error {SAV_coord} {PDB_coord}\n{e}')
                    continue
        return features
    
    def WT_Features(self, pdbID, WT_pdbPath, GJB2=False, **kwargs):
        """
        """
        print('> Computing WT features...')
        WT_data = Features(pdbPath=WT_pdbPath, pdbID=pdbID, eng=self.eng, GJB2=GJB2)
        WT_data.getModes(eng=self.eng, cutOff=self.cutOff)
        WT_data.matFeatures(residue_features=True, protein_features=True)
        # **kwargs add another file to compute pyFeatures: "addFile"
        addFile = kwargs.get('addFile', None)
        if addFile:
            WT_data.pyFeatures(pHCondition=self.pHCondition, hbplusDisCutOff=self.hbplusDisCutOff, skipNeighbor=self.skipNeighbor, addFile=addFile)
        else:
            WT_data.pyFeatures(pHCondition=self.pHCondition, hbplusDisCutOff=self.hbplusDisCutOff, skipNeighbor=self.skipNeighbor)

        WT_data.mergeFeatures()
        self.wt_features = WT_data.features
        self.wt_features_columns = self.wt_features.columns
        # ['resName', 'resID', 'iCode', 'chainID', 'charge', 'entropy_v', 'rmsf_overall', 'eig_first', 'eig_sec', 'SEG_all', 'SEG_20', 'eig5_eig1', 'rank_1', 'rank_2', 'vector_1', 'vector_2', 'GNM_co', 'co_rank', 'eig_vv_1', 'eig_vv_2', 'ca_len', 'gyradius', 'RPA_1', 'RPA_2', 'RPA_3', 'side_chain_length', 'IDRs', 'Dcom', 'phobic_percent', 'philic_percent', 'contact_per_res', 'polarity', 'ssbond_matrix', 'atomic_1', 'atomic_3', 'atomic_5', 'sasa', 'dssp_result', 'loop_percent', 'sheet_percent', 'helix_percent', 'pka_charge', 'h_bond_group']

        return self.wt_features
    
    def MT_Features(self, pdbID, MT_pdbPath=None, **kwargs):
        """
        """
        print('> Computing MT features...')
        MT_data = Features(pdbPath=MT_pdbPath, pdbID=pdbID, eng=self.eng)
        MT_data.matFeatures(residue_features=True, protein_features=False)
        
        addFile = kwargs.get('addFile', None)
        if addFile:
            MT_data.pyFeatures(pHCondition=self.pHCondition, hbplusDisCutOff=self.hbplusDisCutOff, skipNeighbor=self.skipNeighbor, addFile=addFile)
        else:
            MT_data.pyFeatures(pHCondition=self.pHCondition, hbplusDisCutOff=self.hbplusDisCutOff, skipNeighbor=self.skipNeighbor)
        MT_data.mergeFeatures()
        self.mt_features = MT_data.features
        self.mt_features_columns = self.mt_features.columns
        # ['resName', 'resID', 'iCode', 'chainID', 'charge', 'ca_len', 'gyradius', 'RPA_1', 'RPA_2', 'RPA_3', 'side_chain_length', 'IDRs', 'Dcom', 'phobic_percent', 'philic_percent', 'contact_per_res', 'polarity', 'ssbond_matrix', 'atomic_1', 'atomic_3', 'atomic_5', 'sasa', 'dssp_result', 'loop_percent', 'sheet_percent', 'helix_percent', 'pka_charge', 'h_bond_group']

        return self.mt_features

    def merge_Features(
            self,
            SAV_coord: str,
            PDB_coord: str,
            wt_features: pd.DataFrame,
            mt_features: pd.DataFrame
    ):
        PDB_coord_splitting = PDB_coord.split()
        SAV_coord_splitting = SAV_coord.split()
        if len(PDB_coord_splitting) != 4 or len(SAV_coord_splitting) != 4:
            msg = '> TANDEM:getFeatures: Invalid PDB_coord or SAV_coord!'
            raise Exception(msg)
        
        _, chainID, resID, WT_resName   = PDB_coord_splitting # 6CXI A 318 A
        MT_resName                          = SAV_coord_splitting[3]
        WT_resName = one2three[WT_resName]
        MT_resName = one2three[MT_resName]

        # if float(resID) ok => No iCode, if not ok => iCode = resID[-1], resID = resID[:-1]
        try: 
            resID = int(resID)
            iCode = ''
        except ValueError:
            resID_numeric = int(resID[:-1])  # Extract numeric part and convert to integer
            iCode = resID[-1]                # Extract the last character as insertion code
            resID = resID_numeric            # Use the numeric part as the new value of resID

        wt_features, mt_features = self.wt_features, self.mt_features
        mt_features_i = mt_features[(mt_features['resID']   == resID)       & \
                                    (mt_features['chainID'] == chainID)     & \
                                    (mt_features['resName'] == MT_resName)  & \
                                    (mt_features['iCode']   == iCode)]
        wt_features_i = wt_features[(wt_features['resID']   == resID)       & \
                                    (wt_features['chainID'] == chainID)     & \
                                    (wt_features['resName'] == WT_resName)  & \
                                    (wt_features['iCode']   == iCode)]
        mt_features_i = mt_features_i.reset_index(drop=True)
        wt_features_i = wt_features_i.reset_index(drop=True)
        if len(mt_features_i) == 0 or len(wt_features_i) == 0:
            msg = (f'> TANDEM:getFeatures: Fail in locating'
                    f'resID-{resID} chainID-{chainID} wt-{WT_resName} mt-{MT_resName} iCode-{iCode} '
                     'in mt_features_i or wt_features_i!')
            raise Exception(msg)

        features_i = np.array([
            (
            # wt_features_i.entropy_v.values[0],        # Entropyv
            wt_features_i.rmsf_overall.values[0],     # rmsf_overall
            wt_features_i.eig_first.values[0],        # eig_1
            wt_features_i.eig_sec.values[0],          # eig_sec
            # wt_features_i.SEG_all.values[0],          # SEG_all
            # wt_features_i.SEG_20.values[0],           # SEG_20
            # wt_features_i.eig5_eig1.values[0],        # log10(eig5) - log(eig1)
            wt_features_i.rank_1.values[0],           # rank V(1,i)
            wt_features_i.rank_2.values[0],           # rank V(2,i)
            wt_features_i.vector_1.values[0],         # V(1,i)
            wt_features_i.vector_2.values[0],         # V(2,i)
            # wt_features_i.GNM_co.values[0],           # C(i,i)
            wt_features_i.co_rank.values[0],          # rank C(i,i)
            # wt_features_i.eig_vv_1.values[0],         # (1/eig1) * V(1,i)
            # wt_features_i.eig_vv_2.values[0],         # (1/eig2) * V(2,i)
            # Structure Features 12
            # mt_features_i.ca_len.values[0],           # Protein Size
            wt_features_i.gyradius.values[0],         # Radius of gyration
            # mt_features_i.RPA_1.values[0],            # Shape factor 1
            # mt_features_i.RPA_2.values[0],            # Shape factor 2
            # mt_features_i.RPA_3.values[0],            # Shape factor 3
            mt_features_i.loop_percent.values[0],     # %Loop
            mt_features_i.helix_percent.values[0],    # %Helix
            mt_features_i.sheet_percent.values[0],    # %Sheet
            mt_features_i.side_chain_length.values[0],# Side chain length
            # mt_features_i.IDRs.values[0],             # Disorderliness
            # mt_features_i.dssp_result.values[0],      # DSSP
            mt_features_i.Dcom.values[0],             # Dcom
            # Chemical Features 14
            # mt_features_i.philic_percent.values[0],        # % Hydrophilic
            wt_features_i.phobic_percent.values[0],        # % Hydrophobic
            # mt_features_i.contact_per_res.values[0],       # Average contact per Residues
            # mt_features_i.polarity.values[0],              # polarity
            # mt_features_i.charge.values[0],                # Charge
            # mt_features_i.pka_charge.values[0],            # ChargepH7
            mt_features_i.sasa.values[0],                  # SASA
            # mt_features_i.ssbond_matrix.values[0],         # Number disulfide bond formed by a residue
            # mt_features_i.h_bond_group.values[0],          # Number of H bond formed by a residue
            mt_features_i.atomic_1.values[0],              # [Atomic contact] Group 1
            mt_features_i.atomic_3.values[0],              # [Atomic contact] Group 3
            mt_features_i.atomic_5.values[0],              # [Atomic contact] Group 5
            # wt_features_i.consurf.values[0],               # ConSurf
            # wt_features_i.ACNR.values[0],                  # ACNR
            # Delta Features 11
            # mt_features_i.gyradius.values[0]            - wt_features_i.gyradius.values[0],         # Delta Radius of gyration
            mt_features_i.side_chain_length.values[0]   - wt_features_i.side_chain_length.values[0],# Delta Side chain length
            # mt_features_i.sasa.values[0]                - wt_features_i.sasa.values[0],             # Delta SASA
            mt_features_i.phobic_percent.values[0]      - wt_features_i.phobic_percent.values[0],   # Delta % Hydrophobic
            # mt_features_i.philic_percent.values[0]      - wt_features_i.philic_percent.values[0],   # Delta % Hydrophilic
            # mt_features_i.contact_per_res.values[0]     - wt_features_i.contact_per_res.values[0],  # Delta Average contact per Residues
            mt_features_i.polarity.values[0]            - wt_features_i.polarity.values[0],         # Delta polarity
            bool(mt_features_i.charge.values[0])        - bool(wt_features_i.charge.values[0]),     # Delta Charge # Error
            # mt_features_i.pka_charge.values[0]          - wt_features_i.pka_charge.values[0],       # Delta ChargepH7
            # mt_features_i.ssbond_matrix.values[0]       - wt_features_i.ssbond_matrix.values[0],    # Delta Number disulfide bond formed by a residue
            mt_features_i.h_bond_group.values[0]        - wt_features_i.h_bond_group.values[0]      # Delta Number of H bond formed by a residue
            )
        ])
        return features_i

    def _get_feat_dtype(self, featSet=None):
        feat_dtype = np.dtype([
            # ('entropy_v', 'f8'),                # Start from Dynamic Features
            ('rmsf_overall', 'f8'),
            ('eig_first', 'f8'),
            ('eig_sec', 'f8'),
            # ('SEG_all', 'f8'),
            # ('SEG_20', 'f8'),
            # ('eig5_eig1', 'f8'),
            ('rank_1', 'f8'),
            ('rank_2', 'f8'),
            ('vector_1', 'f8'),
            ('vector_2', 'f8'),
            # ('GNM_co', 'f8'),
            ('co_rank', 'f8'),
            # ('eig_vv_1', 'f8'),
            # ('eig_vv_2', 'f8'),                 # End of Dynamic Features : 15 features
            # ('ca_len', 'f8'),                   # Start from Structure Features
            ('gyradius', 'f8'),
            # ('RPA_1', 'f8'),
            # ('RPA_2', 'f8'),
            # ('RPA_3', 'f8'),
            ('loop_percent', 'f8'),
            ('helix_percent', 'f8'),
            ('sheet_percent', 'f8'),
            ('side_chain_length', 'f8'),
            # ('IDRs', 'f8'),
            # ('dssp_result', 'f8'),
            ('Dcom', 'f8'),                     # End of Structure Features : 12 features
            # ('philic_percent', 'f8'),           # Start from Chemical Features
            ('phobic_percent', 'f8'),
            # ('contact_per_res', 'f8'),
            # ('polarity', 'f8'),
            # ('charge', 'f8'),
            # ('pka_charge', 'f8'),
            ('sasa', 'f8'),
            # ('ssbond_matrix', 'f8'),
            # ('h_bond_group', 'f8'),
            ('atomic_1', 'f8'),
            ('atomic_3', 'f8'),
            ('atomic_5', 'f8'),                 
            # ('consurf', 'f8'),                  
            # ('ACNR', 'f8'),                     # End of Chemical Features : 14 features
            # ('delta_gyradius', 'f8'),           # Start from Delta Features
            ('delta_side_chain_length', 'f8'),
            # ('delta_sasa', 'f8'),
            ('delta_phobic_percent', 'f8'),
            # ('delta_philic_percent', 'f8'),
            # ('delta_contact_per_res', 'f8'),
            ('delta_polarity', 'f8'),
            ('delta_charge', 'f8'),
            # ('delta_pka_charge', 'f8'),
            # ('delta_ssbond_matrix', 'f8'),
            ('delta_h_bond_group', 'f8')        # End of Delta Features : 11 features
        ])

        # Remove features that are not in the feature set
        if featSet is not None:
            feat_dtype = np.dtype([(name, dtype) for name, dtype in feat_dtype.descr if name in featSet])
        return feat_dtype
