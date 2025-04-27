from src.main import tandem_dimple

td = tandem_dimple(
    query=['O14508 52 S N', 'O14522 1096 A P'],
    job_name='test',
    models='logs/final/different_number_of_layers/20250423-1234-tandem/n_hidden-5',
    r20000='data/R20000/final_features.csv',
    custom_PDB=None,
    refresh=True,
    folder='data',
    )