import os
import re
import numpy as np

from . import TANDEM_FEATS
from .Uniprot import seqScanning, mapSAVs2PDB, SAV_coord2SAV
from .PDB import calcPDBfeatures
from .SEQ import calcSEQfeatures
from ..utils.logger import LOGGER
from ..utils.user_log import UserLog, USERLOG_MESSAGES, FEATURE_STAGE

class Features:

    def __init__(self, query, refresh=False, **kwargs):
        
        # global shape: (nSAVs, )
        # individual shape
        self.model2pred_dtype = np.dtype([
            ('prob', object), # (n_models, ) 
            ('pred', object), # (n_models, )
            ('mode', 'i4'),
            ('classification', 'U20'),
            ('ratio', 'f4'),
            ('path_prob', 'f4'),
            ('path_prob_sem', 'f4'), 
            ('shap', object), # (n_models, n_features)
        ])

        # masked NumPy array that will contain all info about SAVs
        self.data = None
        self.data_dtype = np.dtype([
            # <UniProtID> <mutation site>
            ('SAVs', 'U50'),
            # original Uniprot SAV_coords, extracted from
            # PolyPhen-2's output or imported directly
            ('SAV_coords', 'U50'),
            # Report whether SAV_coords is in training set
            ('is_train', 'U1'),
            # "official" Uniprot SAV identifiers and corresponding
            # PDB coords (if found, otherwise message errors)
            ('Unique_SAV_coords', 'U50'),
            ('Uniprot_sequence_length', 'i4'),
            ('Asymmetric_PDB_coords', 'U100'),
            ('BioUnit_PDB_coords', 'U100'),
            ('OPM_PDB_coords', 'U100'),
            # number of residues in PDB structure (0 if not found)
            ('Asymmetric_PDB_resolved_length', 'i4'),
            # labels for SAVs if available
            ('labels', 'i4'),
            # Predictions from TANDEM and TANDEM transfer learning
            ('tandem', self.model2pred_dtype),
            ('tandem_dimple', self.model2pred_dtype),
        ])

        # number of SAVs
        self.nSAVs = None
        # NumPy array (num_SAVs)x(num_features)
        self.featMatrix = None
        # standardize --> fill nan and standardize
        self.standardize = None
        # classifiers and main feature set
        self.featSet = None
        # custom PDB structure used for PDB features calculation
        self.custom_PDB = None
        # options
        self.options = kwargs
        self.job_directory = kwargs["job_directory"] if "job_directory" in kwargs else '.'
        self.userlog: UserLog = kwargs.get("userlog") or UserLog(path=f"{self.job_directory}/log.jsonl")
        self.refresh = refresh
        self.saturation_mutagenesis = None
        self.setSAVs(query)
        # map SAVs to PDB structures
        self.Uniprot2PDBmap = None
        self.config = None

    def setSAVs(self, query):
        assert self.data is None, 'SAV list already set.'
        SAV_dtype = [
            ('acc', 'U10'),
            ('pos', 'i'),
            ('wt_aa', 'U1'),
            ('mut_aa', 'U1')
        ]
        if isinstance(query, str):
            if os.path.isfile(query):
                # 'query' is a filename, with line format 'P17516 135 G E'
                SAVs = np.loadtxt(query, dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*s).upper() for s in SAVs]
            elif len(query.split()) < 3:
                # single Uniprot acc (+ pos), e.g. 'P17516' or 'P17516 135'
                SAV_list = seqScanning(query)
                self.saturation_mutagenesis = True
            else:
                # single SAV
                SAV = np.array(query.upper().split(), dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*SAV)]
        else:
            query = list(query)
            assert len(query) > 0, 'Empty SAV query.'

            if all(len(str(s).split()) < 3 for s in query):
                # 'query' is a list or tuple of single Uniprot acc (+ pos),
                # e.g. ['P17516', 'P17516 135']
                SAV_list = []
                for s in query:
                    SAV_list.extend(seqScanning(str(s)))
                self.saturation_mutagenesis = True
            else:
                # 'query' is a list or tuple of SAV coordinates
                SAVs = np.array([tuple(str(s).upper().split()) for s in query], dtype=SAV_dtype)
                SAV_list = ['{} {} {} {}'.format(*s) for s in SAVs]

        # store SAV coordinates
        nSAVs = len(SAV_list)
        data = np.ma.masked_all(nSAVs, dtype=self.data_dtype)
        # Assign nan to all columns
        data['SAV_coords'] = SAV_list
        data['SAVs'] = SAV_coord2SAV(SAV_list)
        self.data = data
        self.nSAVs = nSAVs
        self.userlog.emit(level="important", stage="Validating SAVs", message=f"Validating {nSAVs} SAVs",)
    
    def setLabels(self, labels):
        if labels is None:
            return
        assert self.data is not None, 'SAVs not set.'
        assert len(labels) == self.nSAVs, 'Labels do not match SAVs.'
        assert set(labels).issubset({0, 1}), 'Invalid labels.'
        self.data['labels'] = labels

    def setFeatSet(self, featset):
        assert self.featSet is None, 'Feature set already set.'
        if featset is None:
            featset = TANDEM_FEATS['v1.1']
        elif isinstance(featset, str):
            assert featset in TANDEM_FEATS.keys(), 'Unrecognized feature set.'
            featset = TANDEM_FEATS[featset]
        # check for unrecognized features
        known_feats = TANDEM_FEATS['all']
        for f in featset:
            if f not in known_feats:
                raise RuntimeError(f"Unknown feature: '{f}'")
        if len(set(featset)) != len(featset):
            raise RuntimeError('Duplicate features in feature set.')
        self.featSet = tuple(featset)
        LOGGER.info(f'Selected feature set: {self.featSet}')
        return self.featSet
    
    def setCustomPDB(self, custom_PDB):
        """Set custom PDB structure for PDB features calculation."""
        if custom_PDB is None:
            return
        assert self.custom_PDB is None, 'Custom PDB structure already set.'
        # check if file exists
        self.custom_PDB = custom_PDB
        LOGGER.info(f'Custom PDB structure set to {custom_PDB}')

    def setFeatureMatrix(self, fm):
        assert self.featMatrix is None, 'Feature matrix already set.'
        assert self.featSet is not None, 'Feature set not set.'
        assert self.data is not None, 'SAVs not set.'
        assert len(fm) == self.nSAVs, 'Wrong length.'
        self.featMatrix = fm

    def mapUniprot2PDB(self):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        assert self.data is not None, "SAVs not set."
        cols = ['SAV_coords', 'Unique_SAV_coords', 
                'Asymmetric_PDB_coords', 'Uniprot_sequence_length',
                'BioUnit_PDB_coords', 'OPM_PDB_coords', 
                'Asymmetric_PDB_resolved_length']

        Uniprot2PDBmap, custom_PDB = mapSAVs2PDB(
            self.data['SAV_coords'], custom_PDB=self.custom_PDB, 
            refresh=self.refresh, **self.options
        )
        for col in cols:
            self.data[col] = Uniprot2PDBmap[col]

        # Mapping SAVs to structure: summarize by failure pattern.
        pattern_no_hits = "Cannot map, no hits found"
        pattern_wt_mismatch = re.compile(r"^Cannot map, wild type residue is ([A-Z]) not ([A-Z])$")
        pattern_low_confidence = re.compile(r"^Cannot map, very low confidence region (\d+(?:\.\d+)?)$")

        no_hits_savs = []
        wt_mismatch_savs = []
        low_confidence_savs = []

        for i, row in enumerate(Uniprot2PDBmap):
            sav = self.data["SAVs"][i]
            asu = row["Asymmetric_PDB_coords"]
            if not isinstance(asu, str) or "Cannot map" not in asu:
                continue

            if asu == pattern_no_hits:
                no_hits_savs.append(sav)
                continue

            if pattern_wt_mismatch.fullmatch(asu):
                wt_mismatch_savs.append(sav)
                continue

            if pattern_low_confidence.fullmatch(asu):
                low_confidence_savs.append(sav)
                continue

        # Emit one message per pattern (if any).
        if no_hits_savs:
            self.userlog.emit(level="warning", stage="Mapping SAVs to structures",
                message=USERLOG_MESSAGES['SAV2PDB_NO_HITS']['message'],
                action=USERLOG_MESSAGES['SAV2PDB_NO_HITS']['action'],
                context={"savs": no_hits_savs, "reason": pattern_no_hits},
            )

        if wt_mismatch_savs:
            self.userlog.emit(level="warning", stage="Mapping SAVs to structures",
                message=USERLOG_MESSAGES['SAV2PDB_WT_MISMATCH']['message'],
                action=USERLOG_MESSAGES['SAV2PDB_WT_MISMATCH']['action'],
                context={"savs": wt_mismatch_savs, "reason": "Cannot map, wild type residue is X not Y"},
            )
        
        if low_confidence_savs:
            self.userlog.emit(level="warning", stage="Mapping SAVs to structures",
                message=USERLOG_MESSAGES['SAV2PDB_LOW_CONFIDENCE']['message'],
                action=USERLOG_MESSAGES['SAV2PDB_LOW_CONFIDENCE']['action'],
                context={"savs": low_confidence_savs, "reason": "Cannot map, low confidence region pLDDT<50"},
            )
        
        self.custom_PDB = custom_PDB
        self.Uniprot2PDBmap = Uniprot2PDBmap

        n = np.sum(Uniprot2PDBmap['Asymmetric_PDB_length'] != 0)
        s = np.unique([c.split()[0] for c in Uniprot2PDBmap['Asymmetric_PDB_coords'] if "Cannot map" not in c]).__len__()
        self.userlog.emit(level="important", stage="Mapping SAVs to structures",
            message=f"Mapping {n}/{self.nSAVs} SAVs to {s} structures",
        )
        if s == 0 and n == 0:
            self.userlog.emit(level="error", stage="Mapping SAVs to structures",
                message=USERLOG_MESSAGES["SAV2PDB_FAILED"]["message"]
            )

    def getUniprot2PDBmap(self, **kwargs):
        """Maps each SAV to the corresponding resid in a PDB chain.
        """
        if self.Uniprot2PDBmap is None:
            self.mapUniprot2PDB()
        Uniprot2PDBmap = self.Uniprot2PDBmap
        folder = kwargs.get('folder', '.')
        filename = kwargs.get('filename', None)
        os.makedirs(folder, exist_ok=True)
        if filename is not None: # print to file, if requested
            filepath = os.path.join(folder, filename)
            SAVs = self.data['SAVs']

            rows = []
            for i, s in enumerate(Uniprot2PDBmap):  # type: ignore
                pdbid = ""
                chid = ""
                resid = ""
                note = ""

                coord_text = str(s['Asymmetric_PDB_coords'])
                if coord_text.startswith("Cannot map"):
                    note = coord_text
                else:
                    parts = coord_text.split()
                    if len(parts) >= 4:
                        pdbid = parts[0]
                        chid = parts[1]
                        resid = parts[2]
                    else:
                        note = coord_text

                rows.append({"sav": SAVs[i], "pdbid": pdbid, "chid": chid, "resid": resid, "note": note,
                    "resolved_len": str(s['Asymmetric_PDB_resolved_length']),
                    "total_len": str(s['Uniprot_sequence_length']),
                })

                sav_width = max(len("SAV"), max(len(r["sav"]) for r in rows))
                pdbid_width = max(len("pdbid"), max(len(r["pdbid"]) for r in rows))
                chid_width = max(len("chid"), max(len(r["chid"]) for r in rows))
                resid_width = max(len("resid"), max(len(r["resid"]) for r in rows))
                note_width = max(len("note"), max(len(r["note"]) for r in rows))

                with open(filepath, 'w') as f:
                    f.write('   '.join([
                        f"{'SAV':<{sav_width}}",
                        f"{'pdbid':<{pdbid_width}}",
                        f"{'chid':<{chid_width}}",
                        f"{'resid':<{resid_width}}",
                        f"{'resolved_len':<12}",
                        f"{'total_len':<9}",
                        f"{'note':<{note_width}}",
                    ]) + '\n')

                    for r in rows:
                        f.write('   '.join([
                            f"{r['sav']:<{sav_width}}",
                            f"{r['pdbid']:<{pdbid_width}}",
                            f"{r['chid']:<{chid_width}}",
                            f"{r['resid']:<{resid_width}}",
                            f"{r['resolved_len']:<12}",
                            f"{r['total_len']:<9}",
                            f"{r['note']:<{note_width}}",
                        ]) + '\n')

            LOGGER.info(f'Uniprot2PDB map saved to {filepath}')
        return self.Uniprot2PDBmap

    def getSAVs(self, filename=None, folder='.'):
        SAVs = self.data['SAVs']
        if not filename:
            return SAVs
        else:
            filepath = os.path.join(folder, filename)
            with open(filepath, 'w', 1) as f:
                for s in SAVs:
                    f.write(f"{s}\n")
            LOGGER.info(f'SAVs saved to {filename}')
        return filepath
    
    def getFeatMatrix(self, withSAVs=False, withLabels=False, **kwargs):
        """Export feature matrix to a file."""
        if self.featMatrix is None:
            self._calcFeatMatrix()
        folder = kwargs.get('folder', '.')
        filename = kwargs.get('filename', None)
        os.makedirs(folder, exist_ok=True)
        # Concate SAV_coords, labels and features
        sav_coords = np.array(self.data['SAVs'])
        labels = np.array(self.data['labels'])
        # Create a new structured array with the desired columns
        dtype = [('SAV_coords', 'U50'), ('labels', 'f')] + \
                [(name, 'f') for name in self.featSet]
        arr = np.zeros(len(sav_coords), dtype=dtype)
        arr['SAV_coords'] = sav_coords
        arr['labels'] = labels
        arr[list(self.featSet)] = self.featMatrix

        if withLabels and withSAVs:
            # Keep all columns
            arr = arr[['SAV_coords', 'labels'] + list(self.featSet)]
        elif withLabels:
            # Remove the SAV_coords column from arr
            arr = arr[['labels'] + list(self.featSet)]
        elif withSAVs:
            # Remove the labels column from arr
            arr = arr[['SAV_coords'] + list(self.featSet)]
            LOGGER.info('SAV_coords column removed from feature matrix.')
        else:
            # Remove the labels and SAV_coords columns from arr
            arr = arr[list(self.featSet)]

        # Save the structured array to a TXT file
        if filename is not None:
            filepath = os.path.join(folder, filename)

            headers = list(arr.dtype.names)
            rows = []
            for row in arr:
                row_values = []
                for col in headers:
                    value = row[col]
                    row_values.append(str(value))
                rows.append(row_values)

            col_widths = []
            for i, header in enumerate(headers):
                max_value_width = max(len(r[i]) for r in rows) if rows else 0
                col_widths.append(max(len(header), max_value_width))

            with open(filepath, 'w') as f:
                f.write('   '.join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers)) + '\n')

                for row in rows:
                    f.write('   '.join(f"{row[i]:<{col_widths[i]}}" for i in range(len(headers))) + '\n')
            LOGGER.info(f'Feature matrix saved to {filepath}')
        return arr
    
    def _buildFeatMatrix(self, featset, all_features):
        _dtype = np.dtype([(f, 'f') for f in featset])
        features = np.full(self.nSAVs, np.nan, dtype=_dtype)
        for name in featset:
            # find structured array containing a specific feature
            arrays = [a for a in all_features if name in a.dtype.names]
            if len(arrays) == 0:
                raise RuntimeError(f'Invalid feature name: {name}')
            if len(arrays) > 1:
                LOGGER.warn(f'Multiple values for feature {name}')
            array = arrays[0]
            features[name] = array[name]
            # Report number of missings for each feature
            n_miss = np.sum(np.isnan(array[name]))
            if n_miss > 0:
                LOGGER.warn(f'{n_miss} missing values for feature {name}')
        return features

    def _calcFeatMatrix(self):
        assert self.data is not None, 'SAVs not set.'
        assert self.featSet is not None, 'Feature set not set.'
        # list of structured arrays that will contain all computed features
        all_feats = []
        sel_PDBfeats = TANDEM_FEATS['PDB'].intersection(self.featSet)
        if sel_PDBfeats:
            # compute dynamical features
            f = calcPDBfeatures(self.Uniprot2PDBmap, custom_PDB=self.custom_PDB,
                refresh=self.refresh, sel_feats=sel_PDBfeats, **self.options)
            all_feats.append(f)
        sel_SEQfeats = TANDEM_FEATS['SEQ'].intersection(self.featSet)
        if sel_SEQfeats:
            # compute sequence features
            f = calcSEQfeatures(self.Uniprot2PDBmap['SAV_coords'], 
                refresh=False, sel_feats=sel_SEQfeats, **self.options) # refresh=False because of mapSAVs2PDB
            all_feats.append(f)
        # build matrix of selected features
        self.featMatrix = self._buildFeatMatrix(self.featSet, all_feats)

        savs = np.asarray(self.data['SAVs'])
        no_structure_mask = np.asarray(self.data['Asymmetric_PDB_resolved_length'] == 0, dtype=bool)
        no_structure_savs = savs[no_structure_mask].tolist()
        if no_structure_savs:
            msg = USERLOG_MESSAGES["FEATURE_NO_STRUCTURE"]
            self.userlog.emit(level="warning", stage=FEATURE_STAGE,
                message=msg["message"], action=msg["action"],
                context={"savs": no_structure_savs, "reason": "No structure available"},
            )

        missing_feature_groups = {}
        for idx, sav in enumerate(savs):
            if no_structure_mask[idx]:
                continue
            missing_feats = tuple(
                feature_name for feature_name in self.featSet
                if np.isnan(self.featMatrix[feature_name][idx])
            )
            if not missing_feats:
                continue
            missing_feature_groups.setdefault(missing_feats, []).append(str(sav))

        for missing_feats, group_savs in missing_feature_groups.items():
            feat_text = "-".join(missing_feats)
            msg = USERLOG_MESSAGES["MISSING_FEATURE"]
            self.userlog.emit(level="warning", stage=FEATURE_STAGE,
                message=msg["message"].format(feature_text=feat_text), action=msg["action"],
                context={"savs": group_savs, "missing_features": list(missing_feats)},
            )
