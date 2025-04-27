def get_config():
    return {
    'model': {
        # Input layer
        'input': {
            'n_neurons': 33, 'dropout_rate': 0.0,
        },

        # Hidden layers
        'hidden': {
            'hidden_0': {'n_neurons': 33, 'activation': 'gelu', 'initializer': 'glorot_uniform', 'batch_norm': False, 'dropout_rate': 0, 'l1': 0, 'l2': 1e-4},
            'hidden_1': {'n_neurons': 33, 'activation': 'gelu', 'initializer': 'glorot_uniform', 'batch_norm': False, 'dropout_rate': 0, 'l1': 0, 'l2': 1e-4},
            'hidden_2': {'n_neurons': 33, 'activation': 'gelu', 'initializer': 'glorot_uniform', 'batch_norm': False, 'dropout_rate': 0, 'l1': 0, 'l2': 1e-4},
            'hidden_3': {'n_neurons': 33, 'activation': 'gelu', 'initializer': 'glorot_uniform', 'batch_norm': False, 'dropout_rate': 0, 'l1': 0, 'l2': 1e-4},
            'hidden_4': {'n_neurons': 10, 'activation': 'gelu', 'initializer': 'glorot_uniform', 'batch_norm': False, 'dropout_rate': 0, 'l1': 0, 'l2': 1e-4},
        },
        
        # Output layer
        'output': {
            'n_neurons': 2, 'activation': 'softmax'
        },
    },
    'optimizer': {
        'name': 'Nadam', 'learning_rate': 5e-5,
    },
    'training': {
        'batch_size': 300,
        'n_epochs': 300,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy', 'AUC'],
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
        },
        'class_weight': None, # 'auto' or None; 'auto': adjust weights inversely proportional to class frequencies
    },
}

def get_feature_names():
    return [
        'consurf', 'wt_PSIC', 'Delta_PSIC', 'entropy', 'ACNR', 'sasa', 'BLOSUM', 'stiffness-chain', 
        'loop_percent', 'atomic_1', 'vector_2', 'co_rank', 'atomic_3', 'atomic_5', 'Dcom', 'vector_1', 
        'rank_2', 'eig_first', 'ranked_MI', 'delta_h_bond_group', 'phobic_percent', 'eig_sec', 'sheet_percent', 
        'gyradius', 'delta_polarity', 'side_chain_length', 'helix_percent', 'delta_side_chain_length', 
        'ANM_effectiveness-chain', 'rank_1', 'rmsf_overall', 'delta_charge', 'delta_phobic_percent'
    ]

"""
Dynamic features: 10
    stiffness-chain*
    vector_2
    co_rank
    vector_1
    rank_2
    eig_first
    eig_sec
    ANM_effectiveness-chain*
    rank_1
    rmsf_overall

Structure features: 7
    loop_percent
    helix_percent
    sheet_percent
    gyradius
    side_chain_length
    delta_side_chain_length
    Dcom

Sequence & Chemical features: 16
    consurf
    wt_PSIC
    Delta_PSIC
    entropy
    ACNR
    sasa
    BLOSUM
    atomic_1
    atomic_3
    atomic_5
    ranked_MI
    delta_h_bond_group
    delta_polarity
    delta_charge
    phobic_percent
    delta_phobic_percent

"""