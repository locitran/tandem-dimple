import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# from src.train.optimization import test_numberOflayers_TANDEM, test_numberOflayers_RHAPSODY, test_ranking_method, simple_training
# from src.train.optimization import test_batch_size, test_different_numberOfneurons, visualization_optimization
from src.train.train import train_foundation_model, train_transfer_learning_model
from src.train.train import reproduce_foundation_model, reproduce_transfer_learning_model, reproduce_direct_learning_model
from src.utils.settings import TANDEM_GJB2, TANDEM_RYR1, TANDEM_v1dot1, TANDEM_R20000, CLUSTER, TANDEM_RYR1_V2026
from src.utils.settings import RHAPSODY_R20000, RHAPSODY_GJB2, RHAPSODY_RYR1, RHAPSODY_FEATS, RHAPSODY_RYR1_V2026
from src.features import TANDEM_FEATS

from src.model.plot import pl_gene_general_performance, pl_gene_specific_performance
from src.utils.settings import ROOT_DIR

if __name__ == "__main__":
#     # train_model(
    #     base_models="/mnt/nas_1/YangLab/loci/tandem/logs/Optimization_Tandem_NumberOfLayers/20250627-1012/n_hidden-5",
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="RYR1",
    #     seed=100,
    # )
    # featset = list(dynamics_feat.keys())
    # reproduce_foundation_model(
    #     name='reproduce_foundation_model_RYR1_v2026',
    #     featds=TANDEM_R20000,
    #     gjb2ds=TANDEM_GJB2,
    #     ryr1ds=TANDEM_RYR1_V2026,
    #     clstr=CLUSTER,
    # )
    # train_foundation_model(
    #     name='train_plus_val_foundation_model',
    #     featds=TANDEM_R20000,
    #     featset=TANDEM_FEATS['v1.1'],
    #     gjb2ds=TANDEM_GJB2,
    #     ryr1ds=TANDEM_RYR1,
    #     clstr=CLUSTER,
    # )

    # reproduce_foundation_model(
    #     name='RhapsodyDNN',
    #     featds=RHAPSODY_R20000,
    #     featset=RHAPSODY_FEATS,
    #     gjb2ds=RHAPSODY_GJB2,
    #     ryr1ds=RHAPSODY_RYR1,
    #     clstr=CLUSTER,
    # )

    # train_transfer_learning_model(
    #     base_model=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="TANDEM-DIMPLE-RYR1-90",
    #     seed=0,
    # )
    # train_transfer_learning_model(
    #     base_model=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="TANDEM-DIMPLE-RYR1-90",
    #     seed=100,
    # )
    # train_transfer_learning_model(
    #     base_model=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="TANDEM-DIMPLE-RYR1-90",
    #     seed=73,
    # )
    train_transfer_learning_model(base_model=TANDEM_v1dot1, TANDEM_testSet=TANDEM_GJB2, name="TANDEM-DIMPLE-GJB2-90", seed=73)
    train_transfer_learning_model(base_model=TANDEM_v1dot1, TANDEM_testSet=TANDEM_RYR1, name="TANDEM-DIMPLE-RYR1-90", seed=0)
    
    # train_transfer_learning_model(
    #     base_model='/home/loci/tandem_website/tandem/logs/train_plus_val_foundation_model/20251230-1711/model_train_plus_val.h5',
    #     TANDEM_testSet=TANDEM_GJB2,
    #     name="TANDEM_GJB2_train_plus_val",
    #     seed=73,
    # )

# RYR1
##############################################################################################################################
    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1_V2026,
    #     name="TANDEM_RYR1_V2026",
    #     seed=21,
    # )
    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1_V2026,
    #     name="TANDEM_RYR1_V2026",
    #     seed=73,
    # )
    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1_V2026,
    #     name="TANDEM_RYR1_V2026",
    #     seed=0,
    # )
    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1_V2026,
    #     name="TANDEM_RYR1_V2026",
    #     seed=100,
    # )
    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1_V2026,
    #     name="TANDEM_RYR1_V2026",
    #     seed=5,
    # )
    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet=TANDEM_RYR1_V2026,
    #     name="TANDEM_RYR1_V2026",
    #     seed=2004,
    # )
##############################################################################################################################

    # reproduce_transfer_learning_model(
    #     base_models=TANDEM_v1dot1,
    #     TANDEM_testSet='/home/loci/main/tandem_website_dev/tandem/jobs/PKD1_test/features_high_confidence.csv',
    #     name="TANDEM_PKD",
    #     seed=73,
    # )

    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_RYR1, name="Direct_train_RYR1_2026Jan12", nHidden=2, nNeurons=16, seed=0)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_RYR1, name="Direct_train_RYR1_2026Jan12", nHidden=2, nNeurons=33, seed=0)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_RYR1, name="Direct_train_RYR1_2026Jan12", nHidden=2, nNeurons=16, seed=100)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_RYR1, name="Direct_train_RYR1_2026Jan12", nHidden=2, nNeurons=33, seed=100)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_RYR1, name="Direct_train_RYR1_2026Jan12", nHidden=2, nNeurons=16, seed=73)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_RYR1, name="Direct_train_RYR1_2026Jan12", nHidden=2, nNeurons=33, seed=73)

    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2_2026Jan12", nHidden=2, nNeurons=16, seed=27)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2_2026Jan12", nHidden=2, nNeurons=33, seed=27)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2_2026Jan12", nHidden=2, nNeurons=16, seed=100)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2_2026Jan12", nHidden=2, nNeurons=33, seed=100)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2_2026Jan12", nHidden=2, nNeurons=16, seed=73)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2_2026Jan12", nHidden=2, nNeurons=33, seed=73)

    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2", nHidden=2, nNeurons=16, seed=100)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2", nHidden=2, nNeurons=33, seed=100)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2", nHidden=2, nNeurons=16, seed=27)
    # reproduce_direct_learning_model(TANDEM_testSet=TANDEM_GJB2, name="Direct_train_GJB2", nHidden=2, nNeurons=33, seed=27)

# from src.train.train import reproduce_transfer_learning_model, reproduce_foundation_model
# import pandas as pd 
# from src.features import TANDEM_FEATS
# feat_names = TANDEM_FEATS['v1.1']
# feat_path = '/mnt/nas_1/YangLab/loci/tandem/data/GJB2/final_features.csv'
# df = pd.read_csv(feat_path)
# df = df[~df['labels'].isna()]
# features = df[feat_names].values
# labels = df['labels'].values

# reproduce_transfer_learning_model(
#     features, 
#     labels, 
#     name='reproduce_transfer_learned_model', 
#     model_input=None, 
#     seed=73, 
#     patience = 50
# )

# reproduce_foundation_model(name='weight_initialization')

"""
write new function(s) / module. 

Take three inputs 



goal: execute the file automatically. 

input: SAVs and labels
hyper-parameters (default) advance options



"""

