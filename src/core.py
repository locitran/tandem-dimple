import os
import csv
import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import json

from scipy import stats
from dataclasses import asdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from .features import TANDEM_FEATS
from .features.features import Features
from .utils.settings import TANDEM_v1dot1, TANDEM_R20000
from .utils.logger import LOGGER
from .model.data_processing import Preprocessing, onehot_encoding, np2ds
from .model.train import TLConfig, train_model, evaluate_model
from .model.plot import plotSHAP_bar, plotLoss

import logging
logging.getLogger("shap").setLevel(logging.ERROR)

class Tandem(Features):
    
    def __init__(self, query, refresh=False, **kwargs):
        super().__init__(query, refresh, **kwargs)
        self.setR20000()
        self.models = self.setModels()
    #### -------- Calculate predictions ------- #####

    def setModels(self, folder=TANDEM_v1dot1):
        """Import models from the given folder.
        Args:
            folder (str): Folder containing the models.
        Returns:
            models (list): List of models.
        """
        models = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.h5'):
                    models.append(os.path.join(root, file))

        assert len(models) > 0, f"No models found in {folder}."
        LOGGER.info(f"Found {len(models)} models in {folder}.")
        models = [tf.keras.models.load_model(model) for model in models]
        return models

    def setR20000(self, data=TANDEM_R20000):
        df = pd.read_csv(data)
        fm = df[TANDEM_FEATS['v1.1']].values
        self.preprocess = Preprocessing(fm)
    
    def getPredictions(self, models, folder='.', filename=None):
        """
        Generate prediction results with hierarchical (MultiIndex) columns.

        Output columns:
        - SAV
        - TANDEM -> [probability, classification]
        - TANDEM-DIMPLE -> [probability, classification] (if models provided)

        Saves CSV only if filename is provided.
        """

        # 1. Calculate predictions
        self.calcPredictions(models)

        # 2. Base data
        savs = self.data["SAVs"]

        tandem_prob = self.data["tandem"]["path_prob"]
        tandem_cls  = self.data["tandem"]["classification"]

        # 3. Build column MultiIndex
        data = {
            ("SAV", "SAV"): savs,
            ("TANDEM", "probability"): tandem_prob,
            ("TANDEM", "classification"): tandem_cls,
        }

        # 4. Optional TANDEM-DIMPLE
        if models != TANDEM_v1dot1:
            td_prob = self.data["tandem_dimple"]["path_prob"]
            td_cls  = self.data["tandem_dimple"]["classification"]

            data[("TANDEM-DIMPLE", "probability")] = td_prob
            data[("TANDEM-DIMPLE", "classification")] = td_cls
        
        columns = data.keys()

        # 5. Create DataFrame with MultiIndex columns
        multi_cols = pd.MultiIndex.from_tuples(columns)
        df = pd.DataFrame(data, columns=multi_cols)

        # 7. Save CSV (flatten header for compatibility)
        if filename:
            filepath = os.path.join(folder, f"{filename}.csv")

            df_to_save = df.copy()
            df_to_save.columns = [
                col if isinstance(col, str) else f"{col[0]}::{col[1]}"
                for col in df_to_save.columns
            ]

            df_to_save.to_csv(filepath, index=False)
            LOGGER.info(f"Predictions saved to {filepath}")

        return df
    
    def calcPredictions(self, models):
        assert os.path.isdir(models), f"Folder {models} does not exist."
        assert self.featMatrix is not None, 'Feature matrix not set.'
        
        # Convert the feature matrix to a NumPy array
        feat_names = self.featMatrix.dtype.names
        fm = np.column_stack([self.featMatrix[name] for name in feat_names])
        fm = self.preprocess(fm)
        
        # Load foundation models
        fd_models = self.models
        shap_background = np.load(f"{TANDEM_v1dot1}/shap_background.npy")
        self.data['tandem'] = self._calcPredictions(
            fm, shap_background, fd_models
        )

        # if TANDEM-DIMPLE is provided (models not None)
        # Load tf models
        if models != TANDEM_v1dot1:
            tf_models = self.setModels(models)
            shap_background = np.load(f"{models}/shap_background.npy")
            self.data['tandem_dimple'] = self._calcPredictions(
                fm, shap_background, tf_models
            )

    def _calcPredictions(self, featMatrix, shap_background, models):
        "Voting average & SHAP"
        
        n_models = len(models)
        n_features = featMatrix.shape[1]
        probs = []
        for model in models:
            _pred = model.predict(featMatrix, verbose=False)
            _pred = _pred[:, 1] # Get the probability of class 1: pathogenic
            probs.append(_pred)
        probs = np.column_stack(probs) # (nSAVs, n_models): 2D array

        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        N, M = probs.shape
        out = np.full(N, np.nan, dtype=self.model2pred_dtype)

        # Convert probabilities to binary predictions
        preds = (probs > 0.5).astype(int) # (nSAVs, n_models)
        
        # Get mode and count across the whole dataset
        mode = stats.mode(preds, axis=1) 
        mode_val = mode[0] # (nSAVs, )
        mode_count = mode[1] # (nSAVs, )
        classification = np.array([
            "pathogenic" if val == 1 else "benign" for val in mode_val
        ])
        ratio = mode_count / M
        mode_arr = np.repeat(mode_val[:, None], n_models, axis=1) # (nSAVs, n_models)
        mask_2d  = np.abs(mode_arr-preds) # Mask the difference: # (nSAVs, n_models)
        mask_3d  = np.repeat(mask_2d[:, :, None], n_features, axis=2) # (nSAVs, n_models, n_features)

        # Calculate SHAP
        featImp = self._calcSHAP(shap_background, featMatrix, models) # (nSAVs, n_models, n_features)
        featImp_masked = np.ma.masked_array(featImp, mask=mask_3d) # (nSAVs, n_models, n_features)

        # Broadcast mode_val to match vote shape
        probs_masked = np.ma.array(probs, mask=mask_2d) # (nSAVs, n_models)
        path_probs = probs_masked.mean(1) # (nSAVs, )
        path_probs_sem = stats.sem(probs_masked, axis=1) # (nSAVs, )

        for i in range(probs.shape[0]):
            out['prob'][i] = probs[i]
            out['pred'][i] = preds[i]
            out['shap'][i] = featImp_masked[i]
        out['mode'] = mode_val
        out['classification'] = classification
        out['ratio'] = ratio
        out['path_prob'] = path_probs
        out['path_prob_sem'] = path_probs_sem
        return out

    def _calcSHAP(self, trainSet, testSet, models):
        """
        Input: 
            trainSet: nSAVs X n_feats
            testSet: nSAVs X n_feats
            models: model for inferencing
            featSet: feature order

        Output:
            featImp: nSAVs X n_models X n_feats
        """
        np.random.seed(1)
        n_clusters = 100
        if len(trainSet) > n_clusters:
            background = shap.kmeans(trainSet, n_clusters)
            background = background.data
        else:
            background = trainSet
        
        if testSet.ndim == 1:
            testSet = testSet[None, :] 

        featImp = []
        for model in models:
            explainer = shap.KernelExplainer(model.predict, background, link="logit")
            shap_values = explainer.shap_values(testSet, nsamples=100, silent=True)
            featImp.append(shap_values[1])
        featImp = np.array(featImp) # n_models X nSAVs X n_feats
        return np.transpose(featImp, (1, 0, 2)) # nSAVs X n_models X n_feats

    ### --------- Visualization     ------- #####
    def plotSHAP(self, folder='.'):
        assert not np.ma.is_masked(self.data['tandem']['shap']), 'SHAP has not been calculated' 

        SAVs = self.data['SAVs']
        globalSHAP_title = 'Global feature contribution to model prediction'
        individualSHAP_title = 'Feature contribution to model prediction on {} ({})'

        # Create folder(s) to store SHAP figure(s)
        tandem_shap = os.path.join(folder, 'tandem_shap')
        os.makedirs(tandem_shap, exist_ok=True)
        # Plot global SHAP values in case more than 1 SAV being calculated
        # if self.nSAVs > 1:
        #     _featImp = self.data['tandem']['shap']
        #     plotSHAP_bar(_featImp, globalSHAP_title, tandem_shap, 'globalSHAP', globalshap=True)
            
        # Plot SHAP values for individual SAVs
        for i in range(self.nSAVs):
            sav = str(SAVs[i])
            _featImp = self.data['tandem']['shap'][i]
            _classif = self.data['tandem']['classification'][i]
            plotSHAP_bar(_featImp, individualSHAP_title.format(sav, _classif), 
                         tandem_shap, sav, globalshap=False)

        # This is for TANDEM-DIMPLE in case models is not default
        if not np.ma.is_masked(self.data['tandem_dimple']['shap']):
            tandem_dimple_shap = os.path.join(folder, 'tandem_dimple_shap')
            os.makedirs(tandem_dimple_shap, exist_ok=True)
            # if self.nSAVs > 1:
            #     _featImp = self.data['tandem_dimple']['shap']
            #     plotSHAP_bar(_featImp, globalSHAP_title, tandem_dimple_shap, 'globalSHAP')

            for i in range(self.nSAVs):
                sav = str(SAVs[i])
                _featImp = self.data['tandem_dimple']['shap'][i]
                _classif = self.data['tandem_dimple']['classification'][i]
                plotSHAP_bar(_featImp, individualSHAP_title.format(sav, _classif), 
                            tandem_dimple_shap, sav, globalshap=False)
            
    #### -------- Transfer learning ------- #####

    def setConfig(self, config=None):
        default = TLConfig()
        cfg = asdict(default)
        if config:
            cfg.update({k: v for k, v in config.items() if k in cfg})
        self.config = TLConfig(**cfg)

    def history_avg(self, history):
        metrics = ["loss", "accuracy", "auc", "precision", "recall", "f1"]
        summary = {}
        for model_type in ["fd", "tf"]:
            summary[model_type] = {}
            for split in ["val", "test"]:
                arr = np.array(history[model_type][split])  # shape (n_runs, n_metrics)
                means = arr.mean(axis=0)
                stds  = arr.std(axis=0, ddof=1)
                sems  = stds / np.sqrt(arr.shape[0])  # SEM
                mins  = arr.min(axis=0)
                maxs  = arr.max(axis=0)

                summary[model_type][split] = {
                    m: {
                        "mean": float(mu),
                        "std": float(sd),
                        "sem": float(se),
                        "min": float(mn),
                        "max": float(mx)
                    }
                    for m, mu, se, sd, mn, mx in zip(metrics, means, sems, stds, mins, maxs)
                }
        # --- Print: Foundation models ---
        fd_title = "Foundation models"
        LOGGER.info(fd_title)
        LOGGER.info(f"{'val':>15}{'std':>6}{'sem':>6}{'min':>6}{'max':>6}"
                    f"{'test':>9}{'std':>5}{'sem':>6}{'min':>6}{'max':>6}")

        for metric in metrics:
            v = summary["fd"]["val"][metric]
            t = summary["fd"]["test"][metric]
            line = (f"{metric:>10}: "
                    f"{v['mean']:.3f} {v['std']:.3f} {v['sem']:.3f} {v['min']:.3f} {v['max']:.3f}   "
                    f"{t['mean']:.3f} {t['std']:.3f} {t['sem']:.3f} {t['min']:.3f} {t['max']:.3f}")
            LOGGER.info(line)

        # --- Print: Transfer learning models ---
        tf_title = "Transfer learning models"
        LOGGER.info(tf_title)
        LOGGER.info(f"{'val':>15}{'std':>6}{'sem':>6}{'min':>6}{'max':>6}"
                    f"{'test':>9}{'std':>5}{'sem':>6}{'min':>6}{'max':>6}")

        for metric in metrics:
            v = summary["tf"]["val"][metric]
            t = summary["tf"]["test"][metric]
            line = (f"{metric:>10}: "
                    f"{v['mean']:.3f} {v['std']:.3f} {v['sem']:.3f} {v['min']:.3f} {v['max']:.3f}   "
                    f"{t['mean']:.3f} {t['std']:.3f} {t['sem']:.3f} {t['min']:.3f} {t['max']:.3f}")
            LOGGER.info(line)

    def history_to_csv(self, history, filename="history.csv"):
        """
        Save training history of foundation vs transfer models into CSV format.

        history: dict
            {
                'fd': {'val': [...], 'test': [...]},
                'tf': {'val': [...], 'test': [...]}
            }
        filename: str
            Output CSV file name
        """
        metrics = ["loss", "accuracy", "auc", "precision", "recall", "f1"]
        filepath = os.path.join(self.options['job_directory'], filename)
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            # header
            writer.writerow([
                "fold", "model_type", "split", *metrics
            ])
            n_folds = len(history["fd"]["val"])
            for fold in range(n_folds):
                # foundation model
                writer.writerow([fold+1, "foundation", "val",  *history['fd']['val'][fold]])
                writer.writerow([fold+1, "foundation", "test", *history['fd']['test'][fold]])
                # transfer model
                writer.writerow([fold+1, "transfer", "val",  *history['tf']['val'][fold]])
                writer.writerow([fold+1, "transfer", "test", *history['tf']['test'][fold]])
        LOGGER.info(f"[INFO] History saved to {filepath}")

    def train(self, name, filename, smin=47):
        assert self.featMatrix is not None, "Feature matrix not set."
        assert self._isColSet("labels"), "Labels not set."
        assert self.config is not None, "Config not set."
        
        LOGGER.timeit('_train')
        job_dir = self.options['job_directory']

        cfg = self.config
        feat_names = self.featMatrix.dtype.names
        # Data from All indices
        X = np.column_stack([self.featMatrix[name] for name in feat_names])
        X = self.preprocess(X)  # ensure scaler # (nSAVs, nfeat)
        y = np.asarray(self.data["labels"], dtype=int) # (nSAVs, )
        SAVs = self.data["SAVs"] # (nSAVs, )

        # Check indices no mapping -> ignore these SAVs
        # If resolved length is 0  -> no structure model
        accept_idx = self.data['Asymmetric_PDB_resolved_length'] != 0
        all_idx    = np.arange(self.nSAVs)[accept_idx]
        X = X[all_idx]
        y = y[all_idx]
        SAVs = SAVs[all_idx]
        assert SAVs.shape[0] >= smin, f"Does not meet minimum SAVs {smin} for transfer learning"

        # ----- hold-out test split first -----
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y
        )
        x_te, y_te, sav_te  = X[test_idx],  y[test_idx],  SAVs[test_idx]

        # Save train data (train+val) for shap analysis
        shap_background = X[train_idx]
        np.save(f'{job_dir}/shap_background.npy', shap_background)
        
        fd_models = self.models
        # ----- CV on training set -----
        skf = StratifiedKFold(n_splits=cfg.val_splits, shuffle=True, random_state=cfg.seed)
        models = []

        test_evaluation = {'TANDEM': [], 'TANDEM-DIMPLE': []}
        SAV_cv = {}
        for fold_idx, (inner_tr, inner_va) in enumerate(skf.split(train_idx, y[train_idx]), start=1):
            x_tr, y_tr, sav_tr = X[inner_tr], y[inner_tr], SAVs[inner_tr]
            x_va, y_va, sav_va = X[inner_va], y[inner_va], SAVs[inner_va]
            
            model_dir = os.path.join(job_dir, f'TD_{fold_idx}') # TD: TANDEM-DIMPLE
            os.makedirs(model_dir, exist_ok=True)

            # log the folds
            pos_tr  = int(np.sum(y_tr))
            neg_tr  = int(len(y_tr) - np.sum(y_tr))
            pos_va  = int(np.sum(y_va))
            neg_va  = int(len(y_va) - np.sum(y_va))
            pos_te  = int(np.sum(y_te))
            neg_te  = int(len(y_te) - np.sum(y_te))

            SAV_cv[fold_idx] = {'train': sav_tr.tolist(), 'val': sav_va.tolist(), 'test': sav_te.tolist()}
            LOGGER.info(
                f"Fold {fold_idx} - Train: {pos_tr}pos + {neg_tr}neg, "
                f"Val: {pos_va}pos + {neg_va}neg, "
                f"Test: {pos_te}pos + {neg_te}neg"
            )
            LOGGER.info(f"Train: {sav_tr}")
            LOGGER.info(f"Val: {sav_va}")
            LOGGER.info(f"Test: {sav_te}")
            
            y_tr_1h = onehot_encoding(y_tr, 2)
            y_va_1h = onehot_encoding(y_va, 2)
            y_te_1h = onehot_encoding(y_te, 2)

            train_ds = np2ds(x_tr, y_tr_1h, shuffle=True,  batch_size=cfg.batch_size, seed=cfg.seed)
            val_ds   = np2ds(x_va, y_va_1h, shuffle=False, batch_size=cfg.batch_size, seed=cfg.seed)
            test_ds  = np2ds(x_te, y_te_1h, shuffle=False, batch_size=cfg.batch_size, seed=cfg.seed)

            # ----- build/train model -----
            # Load foundation model
            for model_idx, fd_model in enumerate(fd_models, start=1):
                fd_model_cp = tf.keras.models.clone_model(fd_model)
                fd_model_cp.set_weights(fd_model.get_weights())
                # Transfer learning 
                tf_model = train_model(
                    train_ds, 
                    val_ds, 
                    cfg=cfg,
                    folder=model_dir, 
                    filename=f'{model_idx}',
                    model_input=fd_model_cp,
                )
                tf_model.name = 'TANDEM-DIMPLE'
                models.append(tf_model)
                tf_model.save(f"{model_dir}/model_{model_idx}.h5")

                # ----- Evaluation -----
                fd_test_eval = evaluate_model(fd_model_cp, x_te, y_te_1h)
                tf_test_eval = evaluate_model(tf_model, x_te, y_te_1h)

                test_evaluation['TANDEM'].append(fd_test_eval)
                test_evaluation['TANDEM-DIMPLE'].append(tf_test_eval)

        plotLoss(folder=job_dir, filename='loss.png')
        # Convert each model's list of dicts into a DataFrame
        dfs = {}
        for model, runs in test_evaluation.items():
            df = pd.DataFrame(runs)          # shape: (n_runs, n_metrics)
            dfs[model] = df.mean()           # or df.mean(), df.std()
        # Combine into final table
        pd.DataFrame(dfs).to_csv(f'{job_dir}/test_evaluation.csv', index=False)
        
        # Save SAVs splitting schemes for cross-validation
        with open(f'{job_dir}/cross_validation_SAVs.json', 'w') as f:
            json.dump(SAV_cv, f, indent=4)

        LOGGER.report('train in %.1fs.', '_train')
