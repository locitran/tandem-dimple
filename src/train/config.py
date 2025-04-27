import copy
import ml_collections

def model_config() -> ml_collections.ConfigDict:
    cfg = copy.deepcopy(CONFIG_IMPROVE)
    return cfg

# Try the architecture of hinden layers can be 
# 33-node by 33-node by 33-node by 33-node by 10 node and by 2-node of DNN first
CONFIG_IMPROVE = ml_collections.ConfigDict({
    'model': {
        # Input layer
        'input': {
            'n_neurons': 33,
            'dropout_rate': 0.0,
        },

        # Hidden layers
        'hidden': {
            'hidden_0': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_1': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_2': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_3': {
                'n_neurons': 33,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
            'hidden_4': {
                'n_neurons': 10,
                'activation': 'gelu',
                'initializer': 'glorot_uniform',
                'batch_norm': False,
                'dropout_rate': 0,
                'l1': 0,
                'l2': 1e-4,
            },
        },

        # Output layer
        'output': {
            'n_neurons': 2,
            'activation': 'softmax',
        },
    },
    'optimizer': {
        'name': 'Nadam',
        'learning_rate': 5e-5,
        # Drop to 1e-5 after 300 epochs
        # 'schedule': {
        #     'start_epoch': 50,
        #     'learning_rate': 1e-5,
        # },

    },
    'training': {
        'batch_size': 300,
        'n_epochs': 300,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy', 
                    'AUC', 
                    # 'AUCPR',
                    # 'F1Score',
                    # 'Precision', 
                    # 'Recall', 
                    # 'TruePositives', 
                    # 'TrueNegatives', 
                    # 'FalsePositives', 
                    # 'FalseNegatives'
                    ],
        'callbacks': {
            'EarlyStopping': {
                'monitor': 'val_loss',
                'patience': 30,
                'mode': 'min',
                'restore_best_weights': True,
                'start_from_epoch': 50,
                'verbose': 1,
            },
            'ModelCheckpoint': {
                'monitor': 'val_loss',
                'save_best_only': True,
                'mode': 'min',
            },
            # 'ReduceLROnPlateau': {
            #     'monitor': 'val_loss',
            #     'factor': 0.1,
            #     'patience': 10,
            #     'mode': 'min',
            # },
        },
        'class_weight': None, # 'auto' or None; 'auto': adjust weights inversely proportional to class frequencies
    },
})
