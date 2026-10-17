"""Point catalogues: Cartesian positions with optional weights, marks and n(x)."""
import numpy as np

__all__ = ['Catalogue', 'as_catalogue']


class Catalogue:
    """A set of points in Cartesian coordinates.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Cartesian positions. In a periodic box they must lie in [0, L).
    weights : array_like, shape (N,), optional
        Non-negative weights, e.g. completeness or systematics weights for
        data and the corresponding weights for randoms. Default 1.
    marks : array_like, shape (N,), optional
        Non-negative marks that define the measure, e.g. halo mass (data only;
        ignored for randoms). A neighbour contributes ``weight * mark`` to the
        mass in a sphere, and centres are weighted by ``weight * mark``, so
        that the centres sample the marked measure. Default 1.
    nbar : array_like, shape (N,), optional
        Expected (weighted) number density of the *data* at each point. Only
        used for randoms, where it gives the volume element of each random
        (needed by ``edge_cut`` and by the volume-weighted estimator).
    """

    def __init__(self, positions, weights=None, marks=None, nbar=None):
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f'positions must have shape (N, 3), got {pos.shape}')
        if not np.all(np.isfinite(pos)):
            raise ValueError('positions contain non-finite values')
        self.positions = pos
        self.weights = self._column(weights, 'weights')
        self.marks = self._column(marks, 'marks')
        self.nbar = None
        if nbar is not None:
            nb = self._column(nbar, 'nbar')
            if np.any(nb <= 0):
                raise ValueError('nbar must be positive')
            self.nbar = nb

    def _column(self, v, name):
        n = len(self.positions)
        if v is None:
            return np.ones(n)
        v = np.ascontiguousarray(v, dtype=np.float64).reshape(-1)
        if v.shape != (n,):
            raise ValueError(f'{name} must have length {n}, got {v.shape[0]}')
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f'{name} must be finite and non-negative')
        return v

    def __len__(self):
        return len(self.positions)

    @property
    def mass(self):
        """Contribution of each point to the measure: weight * mark."""
        return self.weights * self.marks

    def subset(self, idx):
        """A new Catalogue with the points idx (index array or boolean mask)."""
        return Catalogue(self.positions[idx], self.weights[idx], self.marks[idx],
                         None if self.nbar is None else self.nbar[idx])

    def __repr__(self):
        extra = []
        if not np.all(self.weights == 1):
            extra.append('weights')
        if not np.all(self.marks == 1):
            extra.append('marks')
        if self.nbar is not None:
            extra.append('nbar')
        return f'Catalogue({len(self)} points' + (', ' + ', '.join(extra) if extra else '') + ')'


def as_catalogue(x, name='data'):
    """Accept a Catalogue or an (N, 3) array."""
    if x is None:
        return None
    if isinstance(x, Catalogue):
        return x
    try:
        return Catalogue(x)
    except ValueError as e:
        raise ValueError(f'{name}: {e}') from None
