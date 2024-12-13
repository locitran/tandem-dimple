import os
import logging
import numpy as np
import pandas as pd
from .features import Features, getFeatures
from .consurf import getConsurf
from .utils.timer import getTimer
from .utils.settings import ROOT_DIR
from . import download
from .LociFixer import LociFixer
from pathlib import Path

logger = logging.getLogger(__name__)
DOWNLOAD_DIR = ROOT_DIR / 'pdbfile' / 'raw'
FIX_DIR = ROOT_DIR / 'pdbfile' / 'fix'
CURRENT_DIR = ROOT_DIR / 'src'
timer = getTimer('tandem', verbose=True)

class Pipeline:
    def __init__(self, 
        SAV_coords: list, 
        download_dir: str = DOWNLOAD_DIR, 
        fix_dir: str = FIX_DIR,
        refresh: bool = False,
        file_format: str = 'opm', # Either 'opm', 'asu, or 'bas'
        fix_loop: bool = False,
        keepIds: bool = True,
        output_dir: str = None,
        timer=None
        ):
        # Calculate Rhapsody features
        (PDB_coords, PDB_IDs, PDB_sizes), rhapsody_featMatrix = self.getRhapsody(SAV_coords)
        extension = self._form_extension(SAV_coords, PDB_coords, PDB_IDs, PDB_sizes, to_file=output_dir / 'extension.txt')

        # Calculate Consurf features
        cs = getConsurf(extension['PDB_coords'], to_file=Path(output_dir) / 'consurf_map.tsv' ) # Get consurf features
        consurf_featMatrix = cs.view(float).reshape(len(cs), 2)

        self.start_matlab_engine(eng=None)
        gf = getFeatures(eng=self.eng)
        n_SAVs = len(SAV_coords)
        n_features = len(gf._get_feat_dtype())
        features = np.zeros((n_SAVs, n_features))

        grouped_id = extension.groupby('PDB_IDs')
        for i, (pdbID, gp_id_extension) in enumerate(grouped_id):
            logger.info(f'> Processing {pdbID} in {i+1}/{len(grouped_id)}...')

            if pdbID == '?':
                features[gp_id_extension.index] = np.zeros((len(gp_id_extension), n_features)) * np.nan
                continue

            p = ProcessPDB(pdbID, gp_id_extension, download_dir=download_dir, fix_dir=fix_dir, refresh=refresh, file_format=file_format, fix_loop=fix_loop, keepIds=keepIds)
            extension.loc[gp_id_extension.index] = p.extension

            grouped_format = p.extension.groupby('file_format')
            # for file_format, gp_format_extension in grouped_format:
            for j, (file_format, gp_format_extension) in enumerate(grouped_format):
                logger.info(f'> Processing {file_format} in {j+1}/{len(grouped_format)} of {pdbID} in {i+1}/{len(grouped_id)}...')
                gp_sav_coords = gp_format_extension['SAV_coords'].tolist()
                gp_pdb_coords = gp_format_extension['mapping_PDB_coords'].tolist()
                if file_format == 'opm':
                    gp_features = gf.valid_indices(gp_sav_coords, gp_pdb_coords, pdbID, fix_dir, membrane=True)
                elif file_format.startswith('assembly'):
                    assemblyID = int(file_format.split('assembly')[1])
                    gp_features = gf.valid_indices(gp_sav_coords, gp_pdb_coords, pdbID, fix_dir, assemblyID=assemblyID)
                elif file_format == 'pdb':
                    gp_features = gf.valid_indices(gp_sav_coords, gp_pdb_coords, pdbID, fix_dir)
                else: # file_format == 'Cannot validate' or 'Cannot mutate'
                    gp_features = np.zeros((len(gp_sav_coords), n_features)) * np.nan
                    logger.error('> TANDEM:pipeline: cannot validate %s!', gp_sav_coords)

                # Add gp_features to features
                features[gp_format_extension.index] = gp_features

        # Concatenate features
        featSet = list(self.rhapsody_featSet) + list(cs.dtype.names) + list(gf._get_feat_dtype().names)
        featM = np.concatenate((rhapsody_featMatrix, consurf_featMatrix, features), axis=1)

        # Save features to text file with 'tab' delimiter
        fileName = output_dir / 'features.txt'
        np.savetxt(fileName, featM, fmt='%.3e', delimiter='\t', header='\t'.join(featSet), comments='')

        self.extension = extension
        self.featM = featM
        self.featSet = featSet
        self.stop_matlab_engine()

    def _get_features(self):
        # Save features to text file with 'tab' delimiter
        featSet = self._get_featSet()
        return pd.DataFrame(self.featM, columns=featSet)
    
    def _get_extension(self):
        return self.extension
    
    def _get_featSet(self):
        return self.featSet

    @timer.track
    def getRhapsody(self, SAV_coords, custom_PDB=None):
        import rhapsody as rd
        self.rhapsody_featSet = ['wt_PSIC', 'Delta_PSIC', 'BLOSUM', 'entropy', 'ranked_MI', 'ANM_effectiveness-chain', 'stiffness-chain']
        os.chdir(DOWNLOAD_DIR)
        r = rd.Rhapsody(query=SAV_coords)
        r.setCustomPDB(custom_PDB) if custom_PDB else None
        r.setFeatSet(self.rhapsody_featSet)
        rhapsody_featMatrix = r._calcFeatMatrix()

        # Get PDB mapping
        pdb = r.getUniprot2PDBmap(filename=None)
        fields = [
            row['PDB SAV coords'].split() if row['PDB size'] > 0
            else ['?', '?', -999, '?'] for row in pdb
        ]
        PDB_IDs = [r[0] for r in fields]
        PDB_coords = pdb['PDB SAV coords']
        PDB_sizes = pdb['PDB size']

        os.chdir(CURRENT_DIR)
        return (PDB_coords, PDB_IDs, PDB_sizes), rhapsody_featMatrix

    def _form_extension(self, SAV_coords, PDB_coords, PDB_IDs, PDB_sizes, to_file=None):
        extention = pd.DataFrame({
            'SAV_coords': SAV_coords,
            'PDB_coords': PDB_coords,
            'PDB_IDs': PDB_IDs,
            'mapping_PDB_coords': None,
            'file_format': None,
            'indices': False,
            'PDB_size': PDB_sizes
        })
        if to_file:
            extention.to_csv(to_file, sep='\t', index=False)
        return extention

    @timer.track
    def start_matlab_engine(self, eng):
        import matlab.engine
        eng = Features.start_matlab_engine(eng)
        self.eng = eng

    @timer.track
    def stop_matlab_engine(self):
        logger.info('> Stopping MATLAB engine...')
        self.eng.quit()
    
@timer.track
class ProcessPDB:
    def __init__(
            self, 
            pdbID: str,
            extension: pd.DataFrame,
            download_dir: str = DOWNLOAD_DIR,
            fix_dir: str = FIX_DIR,
            refresh: bool = False,
            file_format: str = 'opm', # Either 'opm', 'asu, or 'bas'
            fix_loop: bool = False,
            keepIds: bool = True,
            ):
        
        self.pdbID = pdbID
        self.extension = extension # Grouped extension
        self.download_dir = download_dir
        self.fix_dir = fix_dir
        self.refresh = refresh
        self.file_format = file_format
        self.fix_loop = fix_loop
        self.keepIds = keepIds

        opm = self.process_opm()
        self.process_bas(n_assemblies=20) if not opm else None

    def process_opm(self):
        opm = download.opm(self.pdbID, self.download_dir)    # Download OPM file
        if opm is None:
            return False
        
        opm_path = os.path.join(self.download_dir, f'{self.pdbID.lower()}-opm.pdb')

        # Remove 'END' lines from OPM file
        # > This helps PDBFixer recognize the Dummy atoms in some OPM files 
        with open(opm_path, 'r') as file:
            lines = file.readlines()
        lines = [line for line in lines if not line.startswith('END')]
        with open(opm_path, 'w') as file:
            file.writelines(lines)

        self._process_file(opm_path, membrane=True)
        return True
    
    def process_bas(self, n_assemblies: int):
        for assemblyID in range(1, n_assemblies+1):
            cis_bas = download.cif_biological_assembly(self.pdbID, assemblyID, self.download_dir)
            if cis_bas is None: # No cis-biological assemblyID found => use asymmetric unit if necessary
                break
            cifPath = os.path.join(self.download_dir, f'{self.pdbID.lower()}-assembly{assemblyID}.cif.gz')
            self._process_file(cifPath, assemblyID=assemblyID)

            # Check if there are invalid indices
            invalid_indices = self.extension[self.extension['indices'] == False]
            if len(invalid_indices) == 0: 
                logger.info('> Yeah all indices are valid for %s!' % self.pdbID)
                break

        # Use PDB file if all cif files are invalid  
        invalid_indices = self.extension[self.extension['indices'] == False]
        if len(invalid_indices) != 0:
            self.process_asu()

    def process_asu(self):
        pdbPath = os.path.join(self.download_dir, f'{self.pdbID.lower()}.pdb.gz')
        cifPath = f'{self.download_dir}/{self.pdbID.lower()}.cif.gz'
        try:
            pdb_asu = download.pdb_asymmetric_unit(self.pdbID, self.download_dir)
            self._process_file(pdbPath) if pdb_asu is not None else None
        except Exception as e:
            msg = f'> IMPROVE:pipeline: cannot process {pdbPath} {e}'
            logger.error(msg, exc_info=True)
            
            cif_asu = download.cif_asymmetric_unit(self.pdbID, self.download_dir)
            if cif_asu is not None:
                self._process_file(cifPath)
            else:
                msg = f'> IMPROVE:pipeline: cannot find structure for {self.pdbID}!'
                logger.error(msg)

    @timer.track
    def _process_file(
            self, 
            pdb_path: str,
            membrane: bool = False,
            assemblyID: int = None
    ):
        if membrane:
            file_format = 'opm'
        elif assemblyID:
            file_format = f'assembly{assemblyID}'
        else:
            file_format = 'pdb'

        fixer = LociFixer(self.pdbID, pdb_path)
        fixer._processWT(self.fix_dir, membrane=membrane, assemblyID=assemblyID, keepIds=self.keepIds, fix_loop=self.fix_loop)
        self.extension = fixer._validateSAVs(extension=self.extension, file_format=file_format, keepIds=self.keepIds)
        self.extension = fixer._processMT(
            extension=self.extension, file_format=file_format, fix_dir=self.fix_dir, assemblyID=assemblyID, 
            refresh=self.refresh, keepIds=self.keepIds, fix_loop=self.fix_loop
        )