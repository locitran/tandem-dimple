import numpy as np
from prody.dynamics.nma import NMA
from prody.dynamics.modeset import ModeSet
from prody.dynamics.mode import Mode
from prody.utilities import checkCoords, solveEig, ZERO
import time
from prody import LOGGER

class PAA:
    """
    Principal Axis Analysis (PAA) class.
    """

    def __init__(self,):
        pass

    def buildCovariance(self, coords, weights=None):
        try: 
            coords = (coords._getCoords() if hasattr(coords, '_getCoords') else
                        coords.getCoords())
        except AttributeError:
            try:
                checkCoords(coords)
            except TypeError:
                raise TypeError('coords must be a Numpy array or an object '
                                'with `getCoords` method')
        if coords.shape[1] != 3:
            raise ValueError('coords must have 3 columns')

        if weights is not None:
            if len(weights) != coords.shape[0]:
                raise ValueError('weights must have the same length as coords')
            if not np.isscalar(weights).all():
                raise ValueError('weights must be a scalar or an array of scalars')
        else:
            weights = np.ones(coords.shape[0], dtype=coords.dtype)
        n_atoms = coords.shape[0]
        dof = n_atoms * 3
        
        # Calculate covariance matrix
        w = np.sqrt(weights)  # sqrt of weights is used in covariance calculation
        w_coords = coords * w[:, None]
        w_coords = w_coords - np.mean(w_coords, axis=0)
        cov = np.dot(w_coords.T, w_coords) / np.sum(weights)    
        self._cov = cov
        self._n_atoms = n_atoms
        self._dof = dof

    def calcModes(self, turbo=True, **kwargs):
        if self._cov is None:
            raise ValueError('covariance matrix is not built or set')
        start = time.time()
        n_modes = 3
        values, vectors, _ = solveEig(self._cov, n_modes=n_modes, zeros=True, 
                                      turbo=turbo, reverse=True, **kwargs)
        which = values > ZERO
        self._eigvals = values[which]
        self._array = vectors[:, which]
        self._vars = values[which]
        self._n_modes = len(self._eigvals)
        if self._n_modes > 1:
            LOGGER.debug('{0} PAA modes were calculated in {1:.2f}s.'
                     .format(self._n_modes, time.time()-start))
        else:
            LOGGER.debug('{0} PAA mode was calculated in {1:.2f}s.'
                     .format(self._n_modes, time.time()-start))

def calcShapeFactors(coords, weights=None, turbo=True, **kwargs):
    paa = PAA()
    paa.buildCovariance(coords, weights)
    paa.calcModes(turbo=turbo, **kwargs)
    sf1 = np.sqrt(paa._eigvals[2] / paa._eigvals[0])
    sf2 = np.sqrt(paa._eigvals[1] / paa._eigvals[0])
    sf3 = np.sqrt(paa._eigvals[2] / paa._eigvals[1])
    return sf1, sf2, sf3
