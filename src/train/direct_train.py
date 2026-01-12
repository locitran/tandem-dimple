import os 
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# from ..olds.run import use_all_gpus, get_config, get_seed
# from ..olds.run import getR20000, getTestset
# # from ..utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
# from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, TANDEM_PKD1
# from ..utils.settings import ROOT_DIR, CLUSTER
# from ..utils.logger import LOGGER
# from ..features import TANDEM_FEATS

from .train import use_all_gpus, evaluate, train_model
from .modules import Preprocessing, DelayedEarlyStopping, Callback_CSVLogger
from .modules import BinaryF1Score, GradientLoggingModel, GradientLogger
from .modules import build_model, np_to_dataset, plot_acc_loss, plot_acc_loss_3fold_CV, build_optimizer
from ..utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, CLUSTER, ROOT_DIR
from ..utils.logger import LOGGER
from ..features import TANDEM_FEATS
from .config import get_config
from .process_data import getR20000, getTestset, onehot_encoding

LOGGER = logging.getLogger(__name__)

def reproduce_direct_learning_model(TANDEM_testSet, name, seed=73):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(ROOT_DIR, 'logs', name, f'{current_time}-seed-{seed}')
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, 'log.txt')
    LOGGER.start(logfile)
    LOGGER.info(f"Start Time = {current_time}")
    use_all_gpus()

    # R20000_folds, R20000, preprocess_feat, test_knw, test_unk, input_shape = import_data(TANDEM_testSet)
    ##################### 1. Set up feature set #####################
    t_sel_feats = TANDEM_FEATS['v1.1']
    LOGGER.info(f"Feature set: {t_sel_feats}")
    R20000_folds, R20000, preprocess_feat, df_clstr = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats)
    test_knw, test_unk = getTestset(TANDEM_testSet, t_sel_feats, preprocess_feat) 

    SAV_coords, labels, features = test_knw
    VUS_coords, VUS_labels, VUS_features = test_unk
    labels = np.argmax(labels, axis=1)

    ##################### 3. Set up model configuration #####################
    patience = 50
    n_hidden = 5
    cfg = get_config(33, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)
    cfg.training.callbacks.EarlyStopping.start_from_epoch = 10
    cfg.training.n_epochs = 10000
    LOGGER.info(f"Start from epoch: {cfg.training.callbacks.EarlyStopping.start_from_epoch}")

    ##################### 5. Split test data #####################
    # 1. Split 3 folds (60% – 30% – 10%)
    train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.1, random_state=seed, stratify=labels)
    # Save train data (train+val) for shap analysis
    testset_train = test_knw[2][train_indices]
    np.save(f'{log_dir}/shap_background.npy', testset_train)

    kf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
    folds = []
    for i, (train_idx, val_idx) in enumerate(kf.split(train_indices, labels[train_indices])):
        train, val = train_indices[train_idx], train_indices[val_idx]
        test = test_indices
        # Save the folds
        element = {
            'train': {'x': features[train], 'y': labels[train], 'SAV_coords': SAV_coords[train]},
            'val': {'x': features[val], 'y': labels[val], 'SAV_coords': SAV_coords[val]},
            'test': {'x': features[test], 'y': labels[test], 'SAV_coords': SAV_coords[test]}
        }
        folds.append(element)
        # log the folds
        LOGGER.info(
            f"Fold {i+1} - "
            f"Train: {np.sum(labels[train])}pos + {len(train)-np.sum(labels[train])}neg, "
            f"Val: {np.sum(labels[val])}pos + {len(val)-np.sum(labels[val])}neg, "
            f"Test: {np.sum(labels[test])}pos + {len(test)-np.sum(labels[test])}neg"
        )
        LOGGER.info(f"Train: {SAV_coords[train]}")
        LOGGER.info(f"Val: {SAV_coords[val]}")
        LOGGER.info(f"Test: {SAV_coords[test]}")

    evaluations = {}
    for fold_idx in range(3):
        fold = folds[fold_idx]
        train, val, test = fold['train'], fold['val'], fold['test']
        x_train, y_train, SAVs_train = train['x'], train['y'], train['SAV_coords']
        x_val, y_val, SAVs_val = val['x'], val['y'], val['SAV_coords']
        x_test, y_test, SAVs_test  = test['x'], test['y'], test['SAV_coords']

        y_train = onehot_encoding(y_train, 2)
        y_val = onehot_encoding(y_val, 2)
        y_test = onehot_encoding(y_test, 2)

        train_ds = np_to_dataset(x_train, y_train, shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
        val_ds = np_to_dataset(x_val, y_val, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
        test_ds = np_to_dataset(x_test, y_test, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

        y_knw = onehot_encoding(labels, 2)
        knw_ds  = np_to_dataset(features, y_knw, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

        ##################### 5. Train model on test data #####################
        model = train_model(train_ds, val_ds, cfg=cfg, folder=log_dir, filename=f'fold_{fold_idx+1}')
        val_eval  = evaluate(model, x_val,    y_val)
        test_eval = evaluate(model, x_test,   y_test)
        knw_eval  = evaluate(model, features, y_knw)
        # return accuracy, auc, precision, recall, f1
        evaluations[fold_idx] = {
            'val_accuracy': val_eval[0], 'val_auc': val_eval[1], 'val_precision': val_eval[2], 'val_recall': val_eval[3], 'val_f1': val_eval[4],
            'test_accuracy': test_eval[0], 'test_auc': test_eval[1], 'test_precision': test_eval[2], 'test_recall': test_eval[3], 'test_f1': test_eval[4],
            'knw_accuracy': knw_eval[0], 'knw_auc': knw_eval[1], 'knw_precision': knw_eval[2], 'knw_recall': knw_eval[3], 'knw_f1': knw_eval[4],
        }

    df_evaluations = pd.DataFrame(evaluations).T
    df_evaluations.to_csv(f'{log_dir}/evaluations.csv')

    import matplotlib.pyplot as plt    
    folds_history = [pd.read_csv(f'{log_dir}/history_fold_{j}.csv') for j in range(3)]
    fig = plot_acc_loss_3fold_CV(folds_history, 'Training History')
    fig.savefig(f'{log_dir}/training_history.png')
    plt.close(fig)



    # for split_idx in range(3):

    #     get_seed(seed)
    #     fold = folds[split_idx]
    #     train, val, test = fold['train'], fold['val'], fold['test']
    #     x_train, y_train, SAVs_train = train['x'], train['y'], train['SAV_coords']
    #     x_val, y_val, SAVs_val = val['x'], val['y'], val['SAV_coords']
    #     x_test, y_test, SAVs_test  = test['x'], test['y'], test['SAV_coords']

    #     y_train = Preprocessing.one_hot_encoding_labels(y_train, 2)
    #     y_val = Preprocessing.one_hot_encoding_labels(y_val, 2)
    #     y_test = Preprocessing.one_hot_encoding_labels(y_test, 2)

    #     train_ds = np_to_dataset(x_train, y_train, shuffle=True, batch_size=cfg.training.batch_size, seed=seed)
    #     val_ds = np_to_dataset(x_val, y_val, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    #     test_ds = np_to_dataset(x_test, y_test, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)

    #     y_knw = Preprocessing.one_hot_encoding_labels(labels, 2)
    #     knw_ds  = np_to_dataset(features, y_knw, shuffle=False, batch_size=cfg.training.batch_size, seed=seed)
    #     ##################### 5. Train model on test data #####################
    #     csv_logger = Callback_CSVLogger(
    #         data=[train_ds, val_ds], 
    #         name=['train', 'val'],
    #         log_file=f'{log_dir}/history_fold_{split_idx}.csv',
    #     )
    #     early_stopping = DelayedEarlyStopping(**cfg.training.callbacks.EarlyStopping)

    #     model = build_model(cfg, verbose=False)
    #     optimizer = build_optimizer(cfg)
        
    #     init_weight = f'/mnt/nas_1/YangLab/loci/tandem/models/Direct_train_RYR1/RYR1-20250502-1546-seed-0/model_fold_{split_idx+1}_init.weights.h5'
    #     model.load_weights(init_weight)
    #     model.compile(optimizer=optimizer, loss=cfg.training.loss, 
    #                 metrics=['accuracy', 
    #                         tf.keras.metrics.AUC(name='auc'), 
    #                         tf.keras.metrics.Precision(name='precision'), 
    #                         tf.keras.metrics.Recall(name='recall'),
    #                         BinaryF1Score(name='f1_score')
    #                 ])
    #     # Save model weights
    #     model.save_weights(f'{log_dir}/model_fold_{split_idx+1}_init.weights.h5')
    
    #     # Train model on GJB2 data
    #     model.fit(
    #         # GJB2_train_ds,
    #         # validation_data=GJB2_val_ds,
    #         train_ds,
    #         validation_data=val_ds,
    #         epochs=cfg.training.n_epochs,
    #         callbacks=[csv_logger, early_stopping],
    #         verbose=1,
    #         batch_size=cfg.training.batch_size,
    #     )

    #     # Evaluation after training
    #     val_performance = model.evaluate(val_ds)
    #     test_performance = model.evaluate(test_ds)
        
    #     evaluations[split_idx] = {
    #         'val_loss': val_performance[0], 'val_accuracy': val_performance[1], 'val_auc': val_performance[2], 'val_precision': val_performance[3], 'val_recall': val_performance[4], 'val_f1': val_performance[5],
    #         'test_loss': test_performance[0], 'test_accuracy': test_performance[1], 'test_auc': test_performance[2], 'test_precision': test_performance[3], 'test_recall': test_performance[4], 'test_f1': test_performance[5],
    #     }
    #     msg = "Fold %d - val_loss: %.1f, val_accuracy: %.1f%%, val_auc: %.1f, val_precision: %.1f, val_recall: %.1f, val_f1: %.1f, " + \
    #             "test_loss: %.1f, test_accuracy: %.1f%%, test_auc: %.1f, test_precision: %.1f, test_recall: %.1f, test_f1: %.1f"
    #     logging.error(msg, split_idx+1, val_performance[0], val_performance[1] * 100, val_performance[2], val_performance[3], val_performance[4], val_performance[5],
    #                             test_performance[0], test_performance[1] * 100, test_performance[2], test_performance[3], test_performance[4], test_performance[5])
    #     model.save(f'{log_dir}/model_fold_{split_idx+1}.h5')