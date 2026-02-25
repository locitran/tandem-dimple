import shap
import os 
import numpy as np


import numpy as np
import os 
import sys
from src.utils.settings import TANDEM_R20000, CLUSTER, TANDEM_GJB2, TANDEM_RYR1, ROOT_DIR, TANDEM_v1dot1_GJB2, TANDEM_v1dot1_RYR1
from src.features import TANDEM_FEATS, dynamics_feat, structure_feat, sequence_feat, all_feat
from src.train.process_data import getR20000, getTestset
import tensorflow as tf
from sklearn.model_selection import train_test_split


np.random.seed(1)

folds, R20000, preprocess_feat, df_clstr = getR20000()
R20000_train_and_val = np.vstack(
    (folds[1]['train']['x'],
    folds[1]['val']['x'])
)


GJB2_knw, GJB2_unk = getTestset(TANDEM_GJB2, TANDEM_FEATS['v1.1'], preprocess_feat) 
RYR1_knw, RYR1_unk = getTestset(TANDEM_RYR1, TANDEM_FEATS['v1.1'], preprocess_feat) 

seed = 73
GJB2_train_indices, GJB2_test_indices = train_test_split(np.arange(GJB2_knw[0].shape[0]), test_size=0.1, random_state=seed, stratify=np.argmax(GJB2_knw[1], axis=1))
GJB2_train_and_val = GJB2_knw[2][GJB2_train_indices]
GJB2_test = GJB2_knw[2][GJB2_test_indices]

seed = 0
RYR1_train_indices, RYR1_test_indices = train_test_split(np.arange(RYR1_knw[0].shape[0]), test_size=0.1, random_state=seed, stratify=np.argmax(RYR1_knw[1], axis=1))
RYR1_train_and_val = RYR1_knw[2][RYR1_train_indices]
RYR1_test = RYR1_knw[2][RYR1_test_indices]

tandem = os.path.join(ROOT_DIR, 'models/TANDEM')
tandem_gjb2 = os.path.join(ROOT_DIR, 'models/TANDEM_GJB2')
tandem_ryr1 = os.path.join(ROOT_DIR, 'models/TANDEM_RYR1')

tandem_models = [tf.keras.models.load_model(os.path.join(tandem, f'model_fold_{i}.h5')) for i in range(1, 6)]
tandem_gjb2_folders = [os.path.join(tandem_gjb2, f'model_{i}') for i in range(5)]
tandem_gjb2_models = [
    [os.path.join(folder, f'model_fold_{i+1}.h5') for i in range(3)] for folder in tandem_gjb2_folders
]

tandem_gjb2_models = [
    [tf.keras.models.load_model(f, compile=False) for f in files] for files in tandem_gjb2_models
]

tandem_ryr1_folders = [os.path.join(tandem_ryr1, f'model_{i}') for i in range(5)]
tandem_ryr1_models = [
    [os.path.join(folder, f'model_fold_{i+1}.h5') for i in range(3)] for folder in tandem_ryr1_folders
]
tandem_ryr1_models = [
    [tf.keras.models.load_model(f, compile=False) for f in files] for files in tandem_ryr1_models
]


split = 1
background_1 = shap.kmeans(R20000_train_and_val, 100)
explainer_1 = shap.KernelExplainer(tandem_models[split-1].predict, background_1.data, link="logit")
shap_values_1 = explainer_1.shap_values(folds[split-1]['test']['x'], nsamples=100)

split = 2
background_2 = shap.kmeans(R20000_train_and_val, 100)
explainer_2 = shap.KernelExplainer(tandem_models[split-1].predict, background_2.data, link="logit")
shap_values_2 = explainer_2.shap_values(folds[split-1]['test']['x'], nsamples=100)

split = 3
background_3 = shap.kmeans(R20000_train_and_val, 100)
explainer_3 = shap.KernelExplainer(tandem_models[split-1].predict, background_3.data, link="logit")
shap_values_3 = explainer_3.shap_values(folds[split-1]['test']['x'], nsamples=100)

split = 4
background_4 = shap.kmeans(R20000_train_and_val, 100)
explainer_4 = shap.KernelExplainer(tandem_models[split-1].predict, background_4.data, link="logit")
shap_values_4 = explainer_2.shap_values(folds[split-1]['test']['x'], nsamples=100)

split = 5
background_5 = shap.kmeans(R20000_train_and_val, 100)
explainer_5 = shap.KernelExplainer(tandem_models[split-1].predict, background_5.data, link="logit")
shap_values_5 = explainer_2.shap_values(folds[split-1]['test']['x'], nsamples=100)

np.save(f"{tandem}/shap_values_1.npy", shap_values_1)
np.save(f"{tandem}/shap_values_2.npy", shap_values_2)
np.save(f"{tandem}/shap_values_3.npy", shap_values_3)
np.save(f"{tandem}/shap_values_4.npy", shap_values_4)
np.save(f"{tandem}/shap_values_5.npy", shap_values_5)


import numpy as np
np.random.seed(1)

for i in range(5):
    for j in range(3):
        model = tandem_gjb2_models[i][j]
        explainer = shap.KernelExplainer(model.predict, GJB2_train_and_val, link="logit")
        shap_values = explainer.shap_values(GJB2_test, nsamples=100)
        np.save(f"{tandem_gjb2}/shap_values_{i}_{j}.npy", shap_values)

for i in range(5):
    for j in range(3):
        model = tandem_ryr1_models[i][j]
        explainer = shap.KernelExplainer(model.predict, RYR1_train_and_val, link="logit")
        shap_values = explainer.shap_values(RYR1_test, nsamples=100)
        np.save(f"{tandem_ryr1}/shap_values_{i}_{j}.npy", shap_values)