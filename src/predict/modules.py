import numpy as np
import tensorflow as tf
from pathlib import Path
from .config import get_feature_names
from ..utils.settings import ROOT_DIR
import pandas as pd
import numpy as np

MODEL_DIR = ROOT_DIR / 'models'
DATA_DIR = ROOT_DIR / 'data'
feat_names = get_feature_names()

class ModelInference:

    def __init__(self, models=None):
        if models is None:
            models = self.import_model()    
        self.models = models
        self.p = self.load_R20000()

    def import_model(self):
        models = [Path(MODEL_DIR, f'tandem_fold_{i}.h5').resolve() for i in range(1, 6)]
        models = [tf.keras.models.load_model(model) for model in models]
        return models
    
    def load_R20000(self):
        R20000_data = Path(DATA_DIR, 'R20000.tsv').resolve()
        df = pd.read_csv(R20000_data, sep='\t')
        fm = df[feat_names].values
        p = Preprocessing(fm)
        return p
    
    def get_prediction(self, test_fm, models):
        # Preprocess the test_fm
        preds = np.zeros((len(test_fm), len(models)))
        for i, model in enumerate(models):
            pred = model.predict(test_fm)
            preds[:, i] = pred[:, 1]
        return preds
    
    def __call__(self, test_fm):
        test_fm = test_fm[feat_names].values
        test_fm = self.p.fill_na_mean(test_fm)
        test_fm = self.p.normalize(test_fm)
        self.preds = self.get_prediction(test_fm, self.models)
        return self.preds
    
class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)

    def fill_na_mean(self, new_data):
        """Fill missing values with the mean of the column (only for numerical features)
        """
        for i in range(new_data.shape[1]): # Iterate through each column
            mask = np.isnan(new_data[:, i])  # Find the indices of NaN values
            new_data[mask, i] = self.mean[i] # Replace NaN values with the mean of the column
        return new_data

    def normalize(self, new_data):
        """Normalizes the new input data based on the mean and std of the training data
        """
        return (new_data - self.mean) / self.std
    
    def one_hot_encode(self, new_data):
        """One hot encodes the new input data based on the one hot encoding of the training data

        Example:
        data = np.array(
            [
                [1, -1, 1],         # 0 -> [1, 0]; 1 -> [0, 1]
                [0, 1, 0],          # -1 -> [1, 0]; 1 -> [0, 1]
                [1, -1, 0],         # 0 -> [1, 0]; 1 -> [0, 1]
            ]
        )
        one_hot = One_Hot_Encoding(data)
        new_data = one_hot.encoding(data)

        new_data = np.array(
            [
                [0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 1, 0],
            ]
        )
        """
        categories = [np.unique(self.data[:, i]) for i in range(self.data.shape[1])]
        n_categories = [len(categories[i]) for i in range(self.data.shape[1])]
        n_features = self.data.shape[1]

        # Check new_data is continuous or categorical
        for feature_idx in range(n_features):
            # if there is any value in feature that is not in the training data, raise error
            if not set(new_data[:, feature_idx]).issubset(set(categories[feature_idx])):
                raise ValueError(f"Feature {feature_idx} has values that are not in the training data")

        one_hot = np.zeros((new_data.shape[0], sum(n_categories)))
        start = 0
        for feature_idx in range(n_features):
            for category_idx in range(n_categories[feature_idx]):
                category = categories[feature_idx][category_idx]
                mask = new_data[:, feature_idx] == category
                one_hot[mask, start + category_idx] = 1
            start += n_categories[feature_idx]
        return one_hot
    
    @staticmethod
    def one_hot_encoding_labels(labels, n_classes):
        """One hot encodes the labels

        """
        labels = np.asarray(labels, dtype=int)
        if n_classes != len(np.unique(labels)):
            raise ValueError(f"n_classes is not equal to the number of unique labels: {len(np.unique(labels))}")
        
        one_hot = np.zeros((len(labels), n_classes))
        for i, label in enumerate(labels):
            one_hot[i, label] = 1
        return one_hot