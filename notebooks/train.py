from src.train.run import train_model, use_all_gpus, get_config
from src.train.run import getR20000, getTestset
import os 
import logging
import datetime
import pandas as pd
from src.utils.settings import FEAT_STATS, dynamics_feat, structure_feat, seq_feat
from src.utils.settings import TANDEM_R20000, TANDEM_GJB2, TANDEM_RYR1, TANDEM_PKD1, CLUSTER

##################### 1. Set up logging and experiment name #####################
NAME_OF_EXPERIMENT = 'test'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
n_hidden=5 ; patience=50
log_dir = os.path.join('logs', NAME_OF_EXPERIMENT, current_time)
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.ERROR, format='%(message)s')

##################### 2. Set up feature files #####################
logging.error("*"*50)
logging.error("Feature files")
logging.error(f"R20000: {TANDEM_R20000}")
logging.error(f"GJB2: {TANDEM_GJB2}")
logging.error(f"RYR1: {TANDEM_RYR1}")
logging.error(f"Cluster path: {CLUSTER}")

### T-test
##################### 2. Set up feature set #####################
df_stats = pd.read_csv(FEAT_STATS)
t_sel_feats = df_stats.sort_values('ttest_rank').head(33)['feature'].values
sel_DYNfeats = [feat for feat in t_sel_feats if feat in dynamics_feat.keys()]
sel_STRfeats = [feat for feat in t_sel_feats if feat in structure_feat.keys()]
sel_SEQfeats = [feat for feat in t_sel_feats if feat in seq_feat.keys()]
t_sel_feats = sel_DYNfeats + sel_STRfeats + sel_SEQfeats
# Randomly choose 33 features 
# t_sel_feats = df_stats.sample(33, random_state=0)['feature'].values

logging.error("*"*50)
logging.error("# Feature selection based on randomly selections")
logging.error(FEAT_STATS)
logging.error(f"Feature set: {t_sel_feats} {len(t_sel_feats)}")
##################### 3. Set up data #####################
use_all_gpus()
folds, R20000, preprocess_feat = getR20000(TANDEM_R20000, CLUSTER, feat_names=t_sel_feats, folder=log_dir)
GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, t_sel_feats, preprocess_feat) 
RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, t_sel_feats, preprocess_feat)
input_shape = R20000[2].shape[1]
##################### 4. Set up model configuration #####################
cfg = get_config(input_shape, n_hidden=n_hidden, patience=patience, dropout_rate=0.0)

cfg['training']['batch_size'] = 300
logging.error(f"Batch size = 300")
###################### 5. Train model #####################

for i in range(400, 500):
    train_dir = os.path.join('logs', NAME_OF_EXPERIMENT, current_time, f'seed{i}')
    os.makedirs(train_dir, exist_ok=True)
    train_model(folds, cfg, train_dir, GJB2_knw, GJB2_unk, RYR1_knw, RYR1_unk, seed=i)