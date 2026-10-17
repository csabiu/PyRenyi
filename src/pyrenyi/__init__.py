"""PyRenyi: Renyi generalised dimensions D_q of point distributions.

Fixed-radius (point-centred and volume-weighted) and fixed-mass estimators,
for periodic boxes and for surveys with random catalogues and weights.
"""
from . import mocks, regions
from .catalogue import Catalogue
from .fixed_mass import FixedMassResult, fixed_mass
from .fixed_radius import DEFAULT_Q, measure
from .io import read_catalogue
from .results import RenyiResult, jackknife_covariance, load

__version__ = '0.1.0'

__all__ = ['Catalogue', 'measure', 'fixed_mass', 'RenyiResult', 'FixedMassResult', 'load',
           'jackknife_covariance', 'DEFAULT_Q', 'read_catalogue', 'mocks', 'regions', '__version__']
