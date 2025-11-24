# Reference
# 1. Ming, D. and Wall, M.E. (2005) Allostery in a coarse-grained model of protein dynamics. Phys. Rev. Lett., 95, 198103.
# 2. Bahar, I., Lezon, T.R., Bakan, A. and Shrivastava, I.H. (2010) Normal Mode Analysis of Biomolecular Structures: Functional Mechanisms of Membrane Proteins. Chem. Rev., 110, 1463-1497.

# -*- coding: utf-8 -*-
# """This module defines a class and a function for Gaussian network model
# (GNM) calculations."""

import time
from types import FunctionType

import numpy as np

from prody import LOGGER
from prody.utilities import checkCoords, solveEig, ZERO

from prody.dynamics.gamma import Gamma
from prody.dynamics.nma import NMA

class GNMBase(NMA):

    """Class for Gaussian Network Model analysis of proteins.
    _cov: covariance matrix
    _trace: trace of the covariance matrix
    _gamma: spring constant
    _kirchhoff: Kirchhoff matrix
    _cutoff: distance cutoff
    _is3d: flag for 3D or 2D
    _n_atoms: number of atoms
    _dof: number of degrees of freedom
    _eigvals: eigenvalues
    _array: eigenvectors
    _vars: variance of modes
    _n_modes: number of modes
    """

    def __init__(self, name='Unknown'):

        super(GNMBase, self).__init__(name)
        self._is3d = False
        self._cutoff = None
        self._kirchhoff = None
        self._gamma = None

    def __repr__(self):

        return ('<{0}: {1} ({2} modes; {3} nodes)>'
                .format(self.__class__.__name__, self._title, self.__len__(),
                        self._n_atoms))

    def __str__(self):

        return self.__class__.__name__ + ' ' + self._title

    def _reset(self):

        super(GNMBase, self)._reset()
        self._cutoff = None
        self._gamma = None
        self._kirchhoff = None
        self._is3d = False

    def _clear(self):
        self._trace = None
        self._cov = None
        
    def getCutoff(self):
        """Returns cutoff distance."""

        return self._cutoff

    def getGamma(self):
        """Returns spring constant (or the gamma function or :class:`Gamma`
        instance)."""

        return self._gamma

    def getKirchhoff(self):
        """Returns a copy of the Kirchhoff matrix."""

        if self._kirchhoff is None:
            return None
        return self._getKirchhoff().copy()

    def _getKirchhoff(self):
        """Returns the Kirchhoff matrix."""

        return self._kirchhoff

    def getEigvals(self):
        """Returns eigenvalues of the Kirchhoff matrix."""

        return self._eigvals
    
    def getEigvecs(self):
        """Returns eigenvectors of the Kirchhoff matrix."""

        return self._array

class envGNM(GNMBase):

    def __init__(self, name='Unknown'):
        super(envGNM, self).__init__(name)
        self._hessianbar = None
        self._hessian = None

    def buildHessian(self, system, environment, cutoff=10., gamma=1.):
        """Build Kirchhoff matrix using distance cutoff and gamma factor.
        
        system: a coordinate set or an object
        environment: a coordinate set or an object
        cutoff distance (Å) for pairwise interactions
            default is 7.3 Å, , minimum is 4.0 Å
        gamma: spring constant, default is 1.0
        """

        try:
            system = (system._getCoords() if hasattr(system, '_getCoords') else system.getCoords())
            environment = (environment._getCoords() if hasattr(environment, '_getCoords') else environment.getCoords())
        except AttributeError:
            try:
                checkCoords(system)
                checkCoords(environment)
            except TypeError:
                raise TypeError('system and environment must be a Numpy array or an object '
                                'with `getCoords` method')
        coords = np.concatenate((system, environment), axis=0)
        n_atoms = system.shape[0]

        cutoff, g, gamma = checkENMParameters(cutoff, gamma)
        self._cutoff = cutoff
        self._gamma = g

        hessian = GNM._buildKirchhoff(coords, cutoff=cutoff, gamma=gamma)
        len_s = system.shape[0]
        Hss = hessian[:len_s, :len_s]
        Hee = hessian[len_s:, len_s:]
        Hse = hessian[:len_s, len_s:]

        eeIval, eeIvec = np.linalg.eigh(Hee)
        eigZero = np.abs(eeIval) < 1e-10
        eeIval_inv = np.diag(1.0 / eeIval)
        eeIval_inv[eigZero, eigZero] = 0

        ee_inv = eeIvec @ eeIval_inv @ eeIvec.T
        HBar = Hss - Hse @ ee_inv @ Hse.T
        self._hessianbar = HBar
        self._hessian = np.block([[Hss, Hse], [Hse.T, Hee]])
        self._n_atoms = n_atoms
        self._dof = n_atoms

    def calcModes(self, n_modes=20, zeros=False, turbo=True):
        if self._hessianbar is None:
            raise ValueError('Hessian matrix is not built yet. Please run buildHessian method first.')
        if str(n_modes).lower() == 'all':
            n_modes = None
        assert n_modes is None or isinstance(n_modes, int) and n_modes > 0, \
            'n_modes must be a positive integer'
        assert isinstance(zeros, bool), 'zeros must be a boolean'
        assert isinstance(turbo, bool), 'turbo must be a boolean'
        
        self._trace = None
        self._cov = None
        LOGGER.timeit('_gnm_calc_modes')
        values, vectors, vars = solveEig(self._hessianbar, n_modes=n_modes, zeros=zeros, 
                                         turbo=turbo, expct_n_zeros=1)
        self._eigvals = values
        self._array = vectors
        self._vars = vars
        self._trace = self._vars.sum()
        self._n_modes = len(self._eigvals)
        if self._n_modes > 1:
            LOGGER.report('{0} envGNM modes were calculated in %.2fs.'
                        .format(self._n_modes), label='_gnm_calc_modes')
        else:
            LOGGER.report('{0} envGNM mode was calculated in %.2fs.'
                        .format(self._n_modes), label='_gnm_calc_modes')

class GNM(GNMBase):

    def __init__(self, name='Unknown'):
        super(GNM, self).__init__(name)

    def buildKirchhoff(self, coords, cutoff=10., gamma=1.):
        try: 
            coords = (coords._getCoords() if hasattr(coords, '_getCoords') else
                        coords.getCoords())
        except AttributeError:
            try:
                checkCoords(coords)
            except TypeError:
                raise TypeError('coords must be a Numpy array or an object '
                                'with `getCoords` method')
        
        n_atoms = coords.shape[0]
        cutoff, g, gamma = checkENMParameters(cutoff, gamma)
        self._cutoff = cutoff
        self._gamma = g
        kirchhoff = GNM._buildKirchhoff(coords, cutoff=cutoff, gamma=gamma)
        self._kirchhoff = kirchhoff
        self._n_atoms = n_atoms
        self._dof = n_atoms

    def calcModes(self, n_modes=20, zeros=False, turbo=True):
        if self._kirchhoff is None:
            raise ValueError('Kirchhoff matrix is not built or set')
        if str(n_modes).lower() == 'all':
            n_modes = None
        assert n_modes is None or isinstance(n_modes, int) and n_modes > 0, \
            'n_modes must be a positive integer'
        assert isinstance(zeros, bool), 'zeros must be a boolean'
        assert isinstance(turbo, bool), 'turbo must be a boolean'
        LOGGER.timeit('_gnm_calc_modes')
        values, vectors, vars = solveEig(self._kirchhoff, n_modes=n_modes, zeros=zeros, 
                                         turbo=turbo, expct_n_zeros=1)
        self._eigvals = values
        self._array = vectors
        self._vars = vars
        self._trace = self._vars.sum()
        self._n_modes = len(self._eigvals)
        if self._n_modes > 1:
            LOGGER.report('{0} GNM modes were calculated in %.2fs.'
                        .format(self._n_modes), label='_gnm_calc_modes')
        else:
            LOGGER.report('{0} GNM mode was calculated in %.2fs.'
                        .format(self._n_modes), label='_gnm_calc_modes')

    @staticmethod
    def _buildKirchhoff(coords, cutoff, gamma):
        assert isinstance(coords, np.ndarray), 'coords must be a Numpy array'
        n_atoms = coords.shape[0]
        start = time.time()
        kirchhoff = np.zeros((n_atoms, n_atoms), 'd')
        # Build Kirchhoff matrix
        LOGGER.info('Using slower method for building the Hessian.')
        cutoff2 = cutoff * cutoff
        mul = np.multiply
        for i in range(n_atoms):
            xyz_i = coords[i, :]
            i_p1 = i+1
            i2j = coords[i_p1:, :] - xyz_i
            mul(i2j, i2j, i2j)
            for j, dist2 in enumerate(i2j.sum(1)):
                if dist2 > cutoff2:
                    continue
                j += i_p1
                g = gamma(dist2, i, j)
                kirchhoff[i, j] = -g
                kirchhoff[j, i] = -g
                kirchhoff[i, i] = kirchhoff[i, i] + g
                kirchhoff[j, j] = kirchhoff[j, j] + g
        LOGGER.debug('Kirchhoff matrix was built in {0:.2f}s.'.format(time.time()-start))
        return kirchhoff
    
class envANM(GNMBase):

    def __init__(self, name='Unknown'):
        super(envANM, self).__init__(name)
        self._is3d = True
        self._hessian = None
        self._hessianbar = None

    def buildHessian(self, system, environment, cutoff=15., gamma=1.):
        try:
            system = (system._getCoords() if hasattr(system, '_getCoords') else system.getCoords())
            environment = (environment._getCoords() if hasattr(environment, '_getCoords') else environment.getCoords())
        except AttributeError:
            try:
                checkCoords(system)
                checkCoords(environment)
            except TypeError:
                raise TypeError('system and environment must be a Numpy array or an object '
                                'with `getCoords` method')
        coords = np.concatenate((system, environment), axis=0)
        n_atoms = system.shape[0]
        dof = n_atoms * 3

        cutoff, g, gamma = checkENMParameters(cutoff, gamma)
        self._cutoff = cutoff
        self._gamma = g
        hessian = ANM._buildHessian(coords, cutoff=cutoff, gamma=gamma)
        len_s = system.shape[0]
        Hss = hessian[:3*len_s, :3*len_s]
        Hee = hessian[3*len_s:, 3*len_s:]
        Hse = hessian[:3*len_s, 3*len_s:]

        eeIval, eeIvec = np.linalg.eigh(Hee)
        eigZero = np.abs(eeIval) < 1e-10
        eeIval_inv = np.diag(1.0 / eeIval)
        eeIval_inv[eigZero, eigZero] = 0

        ee_inv = eeIvec @ eeIval_inv @ eeIvec.T
        HBar = Hss - Hse @ ee_inv @ Hse.T
        self._hessianbar = HBar # Or Kirchhoff Bar
        self._hessian = np.block([[Hss, Hse], [Hse.T, Hee]])
        self._n_atoms = n_atoms
        self._dof = dof

    def calcModes(self, n_modes=20, zeros=False, turbo=True):
        if self._hessianbar is None:
            raise ValueError('Hessian matrix is not built yet. Please run buildHessian method first.')
        if str(n_modes).lower() == 'all':
            n_modes = None
        assert n_modes is None or isinstance(n_modes, int) and n_modes > 0, \
            'n_modes must be a positive integer'
        assert isinstance(zeros, bool), 'zeros must be a boolean'
        assert isinstance(turbo, bool), 'turbo must be a boolean'

        self._trace = None
        self._cov = None
        LOGGER.timeit('_anm_calc_modes')
        values, vectors, vars = solveEig(self._hessianbar, n_modes=n_modes, zeros=zeros,
                                            turbo=turbo, expct_n_zeros=6)
        self._eigvals = values
        self._array = vectors
        self._vars = vars
        self._trace = self._vars.sum()
        self._n_modes = len(self._eigvals)
        if self._n_modes > 1:
            LOGGER.report('{0} envANM modes were calculated in %.2fs.'
                        .format(self._n_modes), label='_anm_calc_modes')
        else:
            LOGGER.report('{0} envANM mode was calculated in %.2fs.'
                        .format(self._n_modes), label='_anm_calc_modes')
            
class ANM(GNMBase):

    def __init__(self, name='Unknown'):
        super(ANM, self).__init__(name)
        self._is3d = True
        self._hessian = None

    def buildHessian(self, coords, cutoff=15., gamma=1.):
        try:
            coords = (coords._getCoords() if hasattr(coords, '_getCoords') else
                      coords.getCoords())
        except AttributeError:
            try:
                checkCoords(coords)
            except TypeError:
                raise TypeError('coords must be a Numpy array or an object '
                                'with `getCoords` method')
        n_atoms = coords.shape[0]
        dof = n_atoms * 3
        cutoff, g, gamma = checkENMParameters(cutoff, gamma)
        self._cutoff = cutoff
        self._gamma = g
        n_atoms = coords.shape[0]
        hessian = ANM._buildHessian(coords, cutoff=cutoff, gamma=gamma)
        self._hessian = hessian
        self._n_atoms = n_atoms
        self._dof = dof

    def calcModes(self, n_modes=20, zeros=False, turbo=True):
        if self._hessian is None:
            raise ValueError('Hessian matrix is not built or set')
        if str(n_modes).lower() == 'all':
            n_modes = None
        assert n_modes is None or isinstance(n_modes, int) and n_modes > 0, \
            'n_modes must be a positive integer'
        assert isinstance(zeros, bool), 'zeros must be a boolean'
        assert isinstance(turbo, bool), 'turbo must be a boolean'
        LOGGER.timeit('_anm_calc_modes')
        values, vectors, vars = solveEig(self._hessian, n_modes=n_modes, zeros=zeros, 
                                         turbo=turbo, expct_n_zeros=6)
        self._eigvals = values
        self._array = vectors
        self._vars = vars
        self._trace = self._vars.sum()

        self._n_modes = len(self._eigvals)
        if self._n_modes > 1:
            LOGGER.report('{0} ANM modes were calculated in %.2fs.'
                        .format(self._n_modes), label='_anm_calc_modes')
        else:
            LOGGER.report('{0} ANM mode was calculated in %.2fs.'
                        .format(self._n_modes), label='_anm_calc_modes')

    @staticmethod
    def _buildHessian(coords, cutoff, gamma):
        n_atoms = coords.shape[0]
        dof = n_atoms * 3
        LOGGER.timeit('_anm_hessian')
        hessian = np.zeros((dof, dof), float)

        # Build Hessian matrix
        cutoff2 = cutoff * cutoff
        for i in range(n_atoms):
            res_i3 = i*3
            res_i33 = res_i3+3
            i_p1 = i+1
            i2j_all = coords[i_p1:, :] - coords[i]
            for j, dist2 in enumerate((i2j_all ** 2).sum(1)):
                if dist2 > cutoff2:
                    continue
                i2j = i2j_all[j]
                j += i_p1
                g = gamma(dist2, i, j)
                res_j3 = j*3
                res_j33 = res_j3+3
                super_element = np.outer(i2j, i2j) * (- g / dist2)
                hessian[res_i3:res_i33, res_j3:res_j33] = super_element
                hessian[res_j3:res_j33, res_i3:res_i33] = super_element
                hessian[res_i3:res_i33, res_i3:res_i33] = \
                    hessian[res_i3:res_i33, res_i3:res_i33] - super_element
                hessian[res_j3:res_j33, res_j3:res_j33] = \
                    hessian[res_j3:res_j33, res_j3:res_j33] - super_element

        LOGGER.report('Hessian was built in %.2fs.', label='_anm_hessian')
        return hessian

def checkENMParameters(cutoff, gamma):
    """Check type and values of *cutoff* and *gamma*."""

    if not isinstance(cutoff, (float, int)):
        raise TypeError('cutoff must be a float or an integer')
    elif cutoff < 4:
        raise ValueError('cutoff must be greater or equal to 4')
    if isinstance(gamma, Gamma):
        gamma_func = gamma.gamma
    elif isinstance(gamma, FunctionType):
        gamma_func = gamma
    else:
        if not isinstance(gamma, (float, int)):
            raise TypeError('gamma must be a float, an integer, derived '
                            'from Gamma, or a function')
        elif gamma <= 0:
            raise ValueError('gamma must be greater than 0')
        gamma = float(gamma)
        gamma_func = lambda dist2, i, j: gamma
    return cutoff, gamma, gamma_func