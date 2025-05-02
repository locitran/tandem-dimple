import logging, os
import tensorflow as tf
# from tensorflow import keras
import numpy as np
# from config import model_config
# from oldfile.func import plot_acc_loss
# from tensorflow.data import Dataset
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Assign using 10 CPUs
# tf.config.threading.set_intra_op_parallelism_threads(10)

# Assign using all GPUs
tf.config.set_soft_device_placement(True)

_LOGGER = logging.getLogger(__name__)

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

def build_model(cfg, output_bias=None):
    """Build a neural network model based on the configuration file
    Args:
        cfg: dict
            Configuration file
        output_bias: np.array
            Initial bias for the output layer

    Returns:
        model: tf.keras.Model
            Neural network model
    """
    input_shape = (cfg.model.input.n_neurons,)  # Define input shape as a tuple
    X = tf.keras.Input(shape=input_shape)
    Y = X
    for hid_name, hid_config in cfg['model']['hidden'].items():
        Y = tf.keras.layers.Dense(units=hid_config['n_neurons'], activation=hid_config['activation'], kernel_initializer=hid_config['initializer'],
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hid_config['l1'], l2=hid_config['l2']),
                name=hid_name)(Y)
        if hid_config['batch_norm']:
            Y = tf.keras.layers.BatchNormalization()(Y)
        Y = tf.keras.layers.Dropout(hid_config['dropout_rate'])(Y)

    # Add bias to the output layer
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(value=output_bias)
        Y = tf.keras.layers.Dense(units=cfg['model']['output']['n_neurons'], activation=cfg['model']['output']['activation'], bias_initializer=output_bias, name='Output')(Y)
    else:
        Y = tf.keras.layers.Dense(units=cfg['model']['output']['n_neurons'], activation=cfg['model']['output']['activation'], name='Output')(Y)

    model = tf.keras.Model(inputs=X, outputs=Y, name="TANDEM-DIMPLE")
    return model

def build_optimizer(cfg):
    opt_name = cfg.optimizer.name
    lr = cfg.optimizer.learning_rate
    if opt_name == 'Nadam':
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif opt_name == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    # Add other optimizers if needed
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def np_to_dataset(x, y, shuffle=True, batch_size=32, seed=150):
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
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_mean_and_sem(evaluations, entity='val', key='accuracy'):
    mean = np.mean([evaluations[i][entity][key] for i in evaluations.keys()])
    sem = np.std([evaluations[i][entity][key] for i in evaluations.keys()]) / np.sqrt(len(evaluations))
    return mean, sem

class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, 
                 verbose=0, mode='auto', baseline=None, 
                 restore_best_weights=False, start_from_epoch=50):
        super(DelayedEarlyStopping, self).__init__(monitor=monitor, min_delta=min_delta, 
                                                   patience=patience, verbose=verbose, 
                                                   mode=mode, baseline=baseline, 
                                                   restore_best_weights=restore_best_weights)
        self.patience = patience
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.best_epoch = 0  # Initialize the best_epoch attribute

    def on_train_begin(self, logs=None):
        # Initialize variables at the beginning of training
        self.wait = 0  # Number of epochs waited after the best epoch
        self.stopped_epoch = 0  # The epoch where training stops
        self.best = np.inf  # Initialize the best as infinity

    def on_epoch_end(self, epoch, logs=None):
        # Only start early stopping after a certain number of epochs
        if epoch > self.start_from_epoch:
            current = logs.get("val_loss")
            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                self.best_epoch = epoch
                # Record the best weights if the current result is better (less loss)
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    # _LOGGER.error(f"Restoring model weights from the end of the best epoch: {self.best_epoch}.")
                    # _LOGGER.error(f"Epoch {self.best_epoch + 1}: best epoch")
                    self.model.set_weights(self.best_weights)

    # def on_train_end(self, logs=None):
    #     if self.stopped_epoch > 0:
    #         _LOGGER.error(f"Epoch {self.stopped_epoch + 1}: early stopping")
            
# Create customized callbacks to record test accuracy and loss
class Callback_CSVLogger(tf.keras.callbacks.Callback):
    """A custom callback to record performance on dataset at each epoch
    Args:
    -----
    data: list of tf.data.Dataset, each element is a dataset for each fold
        E.g. [train_set, val_set, test_set]
    name: list of str, each element is corresponding name for each dataset
        E.g. ['train', 'val', 'test']
    log_file: str
        Path to save the log file in csv format

    Returns:
    --------
    log_file: csv file
        epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
    """
    def __init__(self, data, name, log_file):
        super(Callback_CSVLogger, self).__init__()
        self.data = data
        self.name = name
        self.log_file = log_file

    def on_train_begin(self, logs=None):
        self.metric_names = [] # [<Mean name=loss>, <CompileMetrics name=compile_metrics>]
        self.metric_names.append(self.model.metrics[0].name)
        for metric in self.model.metrics[1]._user_metrics:
            # self.metric_names.append(metric.name)
            if isinstance(metric, str):
                self.metric_names.append(metric)
            else:
                self.metric_names.append(metric.name)

    def on_epoch_end(self, epoch, logs=None):
        metrics_names = self.metric_names
        header = 'epoch,'
        for name in self.name:
            for metric in metrics_names:
                header += f"{name}_{metric},"
        header = header[:-1] + '\n'
        if epoch == 0:
            with open(self.log_file, 'w') as f:
                f.write(header)
        
        row = f"{epoch+1},"
        for ds in self.data:
            ds_logs = self.model.evaluate(ds, verbose=0)
            ds_logs = dict(zip(metrics_names, ds_logs))
            for metric in metrics_names:
                row += f"{ds_logs[metric]},"
        row = row[:-1] + '\n'
        with open(self.log_file, 'a') as f:
            f.write(row)

def plot_grid_search(df):
    df = df.sort_values(by='val_accuracy_mean', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['patience'], df['n_hidden'], c=df['val_accuracy_mean'], cmap='coolwarm', s=100)
    ax.set_xlabel('Early stop', fontsize=20)
    ax.set_ylabel('No. hidden layers', fontsize=20)
    ax.set_title('Validation Accuracy', fontsize=20)
    fig.colorbar(scatter, ax=ax)

    ax.set_xticks(range(10, 51, 10))
    # ax.set_yticks(range(6, 13, 2))
    # From 6 to 20
    ax.set_yticks(range(6, 21, 2))

    best = df.loc[df['val_accuracy_mean'].idxmax()]
    value = best['val_accuracy_mean'] * 100
    ax.annotate(f'Best model\n{value:.2f}%', # Percentage of accuracy
                xy=(best['patience'], best['n_hidden']),
                xytext=(best['patience']-10, best['n_hidden']+1),   
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                position=(43, 11),
                fontsize=15
                )
    plt.show()
    return plt

def plot_acc_loss(folds, title, labels=[r'R20000$_{train}$', r'R20000$_{val}$']):
    # Row 1: Loss, Row 2: Accuracy
    # 5 folds for each
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(title, fontsize=35)
    # fig.supxlabel('Epoch', fontsize=25)
    # fig

    for i, ax in enumerate(axes[0]):
        ax.plot(folds[i]['train_loss'], color='blue', linestyle='dashed', linewidth=2, label=labels[0])
        ax.plot(folds[i]['val_loss'], color='red', linestyle='solid', linewidth=2, label=labels[1])
        # ax.plot(folds[i]['test_loss'], color='green', linestyle='solid', linewidth=2, label=r'R20000$_{test}$')
        # ax.plot(folds[i]['GJB2_loss'], color='purple', linestyle='solid', linewidth=2, label=r'GJB2$_{test}$')
        # ax.plot(folds[i]['RYR1_loss'], color='orange', linestyle='solid', linewidth=2, label=r'RYR1$_{test}$')

        ax.set_title('Split ' + str(i + 1), fontsize=20)

    for i, ax in enumerate(axes[1]):
        ax.plot(folds[i]['train_accuracy'], color='blue', linestyle='dashed', linewidth=2, label=labels[0])
        ax.plot(folds[i]['val_accuracy'], color='red', linestyle='solid', linewidth=2, label=labels[1])
        # ax.plot(folds[i]['test_accuracy'], color='green', linestyle='solid', linewidth=2, label=r'R20000$_{test}$')
        # ax.plot(folds[i]['GJB2_accuracy'], color='purple', linestyle='solid', linewidth=2, label=r'GJB2$_{test}$')
        # ax.plot(folds[i]['RYR1_accuracy'], color='orange', linestyle='solid', linewidth=2, label=r'RYR1$_{test}$')

    # Set y labels
    axes[0, 0].set_ylabel('Loss', fontsize=25)
    axes[1, 0].set_ylabel('Accuracy', fontsize=25)
    axes[1, 2].legend(fontsize=25, loc='best') # Set legends
    for ax in axes[0]: # y-loss range
        ax.set_ylim([0.4, 0.65])
    for ax in axes[1]: # y-accuracy range
        ax.set_ylim([0.4, 0.9])
    for ax in axes.flatten(): # Size of tick labels
        ax.tick_params(labelsize=15)
    for ax in axes.flatten(): # Grid y
        ax.grid(axis='y')
    plt.show()
    return fig

def plot_acc_loss_3fold_CV(folds, title, name='GJB2'):
    # Row 1: Loss, Row 2: Accuracy
    # 5 folds for each
    fig, axes = plt.subplots(2, 3, figsize=(25, 10))
    fig.suptitle(title, fontsize=35)


    # labels=[r'GJB2$_{train}$', r'GJB2$_{val}$']
    labels = [f'{name}$_{{train}}$', f'{name}$_{{val}}$']

    for i, ax in enumerate(axes[0]):
        ax.plot(folds[i]['train_loss'], color='blue', linestyle='dashed', linewidth=2, label=labels[0])
        ax.plot(folds[i]['val_loss'], color='red', linestyle='solid', linewidth=2, label=labels[1])
        ax.set_title('Fold ' + str(i + 1), fontsize=20)

    for i, ax in enumerate(axes[1]):
        ax.plot(folds[i]['train_accuracy'], color='blue', linestyle='dashed', linewidth=2, label=labels[0])
        ax.plot(folds[i]['val_accuracy'], color='red', linestyle='solid', linewidth=2, label=labels[1])

    axes[0, 0].set_ylabel('Loss', fontsize=25)
    axes[1, 0].set_ylabel('Accuracy', fontsize=25)
    axes[1, 2].legend(fontsize=25, loc='best') # Set legends
    for ax in axes.flatten(): # Size of tick labels
        ax.tick_params(labelsize=15)
    for ax in axes.flatten(): # Grid y
        ax.grid(axis='y')
    return fig


# def plot_acc_loss_3fold_CV(folds, title, gene='GJB2'):
#     # Row 1: Loss, Row 2: Accuracy
#     # 5 folds for each
#     fig, axes = plt.subplots(2, 3, figsize=(25, 10))
#     fig.suptitle(title, fontsize=35)

#     if gene == 'GJB2':
#         labels = [r'GJB2$_{train}$', r'GJB2$_{val}$']
#     elif gene == 'RYR1':
#         labels = [r'RYR1$_{train}$', r'RYR1$_{val}$']
#     else:
#         labels = [r'train', r'val']

#     for i, ax in enumerate(axes[0]):
#         ax.plot(folds[i][f'{gene}train_loss'], color='blue', linestyle='dashed', linewidth=2, label=labels[0])
#         ax.plot(folds[i][f'{gene}val_loss'], color='red', linestyle='solid', linewidth=2, label=labels[1])

#         ax.set_title('Fold ' + str(i + 1), fontsize=20)

#     for i, ax in enumerate(axes[1]):
#         ax.plot(folds[i][f'{gene}train_accuracy'], color='blue', linestyle='dashed', linewidth=2, label=labels[0])
#         ax.plot(folds[i][f'{gene}val_accuracy'], color='red', linestyle='solid', linewidth=2, label=labels[1])

#     # Set y labels
#     axes[0, 0].set_ylabel('Loss', fontsize=25)
#     axes[1, 0].set_ylabel('Accuracy', fontsize=25)
#     axes[1, 2].legend(fontsize=25, loc='best') # Set legends
#     # for ax in axes[0]: # y-loss range
#         # ax.set_ylim([0.4, 0.65])
#     # for ax in axes[1]: # y-accuracy range
#         # ax.set_ylim([0.4, 0.9])
#     for ax in axes.flatten(): # Size of tick labels
#         ax.tick_params(labelsize=15)
#     for ax in axes.flatten(): # Grid y
#         ax.grid(axis='y')
#     plt.show()
#     return fig

# def main():
#     pass

