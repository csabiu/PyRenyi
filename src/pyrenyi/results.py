"""Result containers: Renyi functionals, generalised dimensions, jackknife
covariances, and saving / loading."""
import json

import numpy as np

__all__ = ['RenyiResult', 'jackknife_covariance', 'load']


def jackknife_covariance(samples):
    """Delete-one jackknife covariance of samples[n_regions, ...] (flattened)."""
    x = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    n = len(x)
    d = x - x.mean(0)
    return (n - 1.0) / n * d.T @ d


def _dq(Y, radii, method):
    lr = np.log(radii)
    if method == 'adjacent':
        return np.diff(Y, axis=-1) / np.diff(lr), np.sqrt(radii[1:] * radii[:-1])
    if method == 'central':
        return (Y[..., 2:] - Y[..., :-2]) / (lr[2:] - lr[:-2]), radii[1:-1]
    raise ValueError("method must be 'adjacent' or 'central'")


class RenyiResult:
    """Renyi functionals Y_q(r) from the fixed-radius estimators.

    Y has shape (n_cases, n_q, n_r, n_wedges): case 0 counts every neighbour
    within r, case e + 1 only those beyond the minimum separation r_min[e]
    (NaN for r <= r_min[e]); the last wedge index is the isotropic statistic.
    D_q(r) = dY_q / d ln r (see `dq`). Y_q is defined up to an additive
    constant (the reference volume; see `meta['ref_volume']`), which cancels
    in D_q.

    Attributes
    ----------
    q, radii, r_min, mu_edges : the measurement grid (mu_edges is None
        without wedges)
    Y : ndarray, the functionals of the full sample
    Y_jk : ndarray (n_regions, ...) or None, delete-one jackknife functionals
    centre_weight : ndarray (n_cases, n_r, n_wedges), the summed weight of the
        centres that entered each radius (after any edge cut)
    meta : dict, the options of the measurement
    """

    def __init__(self, q, radii, r_min, mu_edges, Y, Y_jk=None, centre_weight=None, meta=None):
        self.q = np.asarray(q, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        self.r_min = np.asarray(r_min if r_min is not None else [], dtype=np.float64)
        self.mu_edges = None if mu_edges is None else np.asarray(mu_edges, dtype=np.float64)
        self.Y = np.asarray(Y)
        self.Y_jk = None if Y_jk is None else np.asarray(Y_jk)
        self.centre_weight = None if centre_weight is None else np.asarray(centre_weight)
        self.meta = dict(meta or {})

    # ---- selection ----
    @property
    def n_wedges(self):
        return 0 if self.mu_edges is None else len(self.mu_edges) - 1

    def _case(self, r_min):
        if r_min is None or r_min == 0:
            return 0
        k = np.flatnonzero(np.isclose(self.r_min, r_min))
        if not len(k):
            raise ValueError(f'r_min={r_min} was not measured (available: {list(self.r_min)})')
        return int(k[0]) + 1

    def _wedge(self, wedge):
        if wedge is None or wedge == 'iso':
            return self.Y.shape[-1] - 1
        if not 0 <= wedge < self.n_wedges:
            raise ValueError(f'wedge must be in 0..{self.n_wedges - 1} or None/"iso"')
        return int(wedge)

    def Yq(self, r_min=None, wedge=None, jackknife=False):
        """Y_q(r), shape (n_q, n_r) (or (n_regions, n_q, n_r) with jackknife=True)."""
        c, w = self._case(r_min), self._wedge(wedge)
        if jackknife:
            if self.Y_jk is None:
                raise ValueError('no jackknife regions were used')
            return self.Y_jk[:, c, :, :, w]
        return self.Y[c, :, :, w]

    def dq(self, r_min=None, wedge=None, method='adjacent', jackknife=False):
        """Generalised dimensions D_q(r) = dY_q / d ln r.

        method='adjacent': between consecutive radii, at their geometric
        mean; 'central': between the neighbours of each interior radius.
        Returns (D, r) with D of shape (n_q, n_r') (or (n_regions, n_q, n_r')).
        """
        return _dq(self.Yq(r_min, wedge, jackknife), self.radii, method)

    def covariance(self, quantity='D', r_min=None, wedge=None, method='adjacent'):
        """Jackknife covariance of the flattened (q, r) array of D_q or Y_q."""
        if quantity == 'D':
            x = self.dq(r_min, wedge, method, jackknife=True)[0]
        elif quantity == 'Y':
            x = self.Yq(r_min, wedge, jackknife=True)
        else:
            raise ValueError("quantity must be 'D' or 'Y'")
        return jackknife_covariance(x)

    def errors(self, quantity='D', r_min=None, wedge=None, method='adjacent'):
        """Jackknife standard errors, shaped like `dq` / `Yq`."""
        if quantity == 'D':
            x = self.dq(r_min, wedge, method, jackknife=True)[0]
        else:
            x = self.Yq(r_min, wedge, jackknife=True)
        return np.sqrt(np.diag(jackknife_covariance(x))).reshape(x.shape[1:])

    # ---- I/O ----
    def save(self, path):
        """Save to a .npz file (arrays and a JSON header)."""
        arrs = dict(kind='RenyiResult', q=self.q, radii=self.radii, r_min=self.r_min, Y=self.Y,
                    meta=json.dumps(self.meta, default=_json_default))
        if self.mu_edges is not None:
            arrs['mu_edges'] = self.mu_edges
        if self.Y_jk is not None:
            arrs['Y_jk'] = self.Y_jk
        if self.centre_weight is not None:
            arrs['centre_weight'] = self.centre_weight
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path):
        f = np.load(path, allow_pickle=False)
        g = lambda k: f[k] if k in f.files else None   # noqa: E731
        return cls(f['q'], f['radii'], f['r_min'], g('mu_edges'), f['Y'], g('Y_jk'),
                   g('centre_weight'), json.loads(str(f['meta'])))

    def __repr__(self):
        w = f', {self.n_wedges} wedges' if self.n_wedges else ''
        jk = f', {len(self.Y_jk)} jackknife regions' if self.Y_jk is not None else ''
        rm = f', r_min={list(self.r_min)}' if len(self.r_min) else ''
        return (f'RenyiResult({self.meta.get("estimator", "?")}, {len(self.q)} q, '
                f'{len(self.radii)} radii{rm}{w}{jk})')


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def load(path):
    """Load a RenyiResult or FixedMassResult saved with .save()."""
    kind = str(np.load(path, allow_pickle=False)['kind'])
    if kind == 'RenyiResult':
        return RenyiResult.load(path)
    if kind == 'FixedMassResult':
        from .fixed_mass import FixedMassResult
        return FixedMassResult.load(path)
    raise ValueError(f'unknown result kind {kind!r}')
