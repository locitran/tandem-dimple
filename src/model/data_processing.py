from scipy import stats
import numpy as np

def np2ds(x, y, shuffle=True, batch_size=32, seed=150):
    """Convert a numpy array to a tf.data dataset
    Args:
    data: numpy array
    shuffle: shuffle the dataset
    batch_size: batch size
        No. samples in each batch

    Returns:
    ds: tf.data.Dataset
    ds = (features, labels)
    """
    
    import sys
    if "tensorflow" not in sys.modules:
        import tensorflow as tf
    else:
        tf = sys.modules["tensorflow"]
        
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def onehot_encoding(labels, n_classes):
    """One hot encodes the labels
    class 0 --> [1, 0]
    class 1 --> [0, 1]
    """
    labels = np.asarray(labels, dtype=int)
    onehot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        onehot[i, label] = 1
    return onehot
    
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

    def denormalize(self, z_data):
        """Inverse of normalize: returns values on the original scale."""
        return z_data * self.std + self.mean

    def __call__(self, new_data):
        new_data = self.fill_na_mean(new_data)
        new_data = self.normalize(new_data)
        return new_data