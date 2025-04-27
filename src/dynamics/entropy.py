import numpy as np
from prody.dynamics.nma import NMA
from prody.dynamics.modeset import ModeSet
from prody.dynamics.mode import Mode

def calcSpectralEntropy(model, n_modes="all"):
    if not isinstance(model, (NMA, ModeSet, Mode)):
        raise TypeError(f'model must be an NMA, ModeSet, or Mode instance, not {type(model)}')

    if isinstance(model, NMA) and len(model) == 0:
        raise ValueError('model must have normal modes calculated')

    if str(n_modes).lower() == 'all':
        n_modes = None
    assert n_modes is None or isinstance(n_modes, int) and n_modes > 0, \
        f'n_modes must be a positive integer, or "all", not {n_modes}'
    if n_modes is None:
        n_modes = model.numModes()
    else:
        n_modes = min(n_modes, model.numModes())
    eigvals = model.getEigvals()
    eigvals = eigvals[:n_modes]
    # Calc bandwidth and bincounts
    binrange = np.linspace(eigvals[0], eigvals[-1], n_modes+1)
    bincounts, _ = np.histogram(eigvals, bins=binrange)
    bincounts = bincounts[bincounts != 0] # Remove zero counts
    # Calc entropy
    p = bincounts / n_modes
    se = -np.sum(p * np.log2(p)) / np.log2(n_modes)
    return se