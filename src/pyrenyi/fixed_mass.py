"""Fixed-mass (nearest-neighbour) estimator of the generalised dimensions.

For each centre i (a data point) and number k, r_ik is the distance of the
k-th neighbour beyond the minimum separation r_min, and nu_ik the expected
number of neighbours in r_min < d <= r_ik for a homogeneous field:
analytic in a periodic box, from the randoms in a survey. For Poisson
sampling of a field, nu_ik follows a Gamma(k) distribution (the expected
count at the k-th arrival), whatever the geometry and selection. For a
measure with generalised dimensions D_q, the moments

    W(tau, k) = sum_i c_i nu_ik^(-tau / 3) / sum_i c_i

satisfy W(tau, k) ~ Gamma(k + 1 - q) / Gamma(k) * const with tau = (q - 1) D_q
(Grassberger 1985; Badii & Politi 1985; van de Water & Schram 1988). For two
masses k_a < k_b, q(tau) is therefore the root of

    ln W(tau, k_b) - ln W(tau, k_a)
        = [ln Gamma(k_b + 1 - q) - ln Gamma(k_b)] - [ln Gamma(k_a + 1 - q) - ln Gamma(k_a)],

and D_q = tau / (q - 1) (for q = 1, D_1 = 3 [psi(k_b) - psi(k_a)] /
[<ln nu_kb> - <ln nu_ka>]). A Poisson process gives D_q = 3 for every q and k
(up to O(k / N)), so the estimator has no discreteness bias at low counts, in
particular for q < 1. The moments exist for q < k_a + 1.
"""
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from . import _kernels
from ._common import (check_in_box, chunks, parse_box, parse_q, reduce_regions, region_labels,
                      set_threads)
from ._grid import Grid
from .catalogue import as_catalogue
from .results import _json_default, jackknife_covariance

__all__ = ['fixed_mass', 'FixedMassResult']

EDGE_QUANTILE = 0.9


def _neighbour_radii(tree, pos_c, k, rmin, workers, batch=20000, has_self=True):
    """r[i, j]: distance of the k[j]-th neighbour beyond rmin of centre i (NaN
    if the catalogue runs out). has_self: the centres are points of the tree,
    which are not their own neighbours."""
    nc, kmax, n = len(pos_c), int(k.max()), tree.n
    off = 1 if has_self else 0
    nin = np.zeros(nc, np.int64)
    if rmin > 0:
        nin = np.asarray(tree.query_ball_point(pos_c, rmin, return_length=True, workers=workers)) - off
    out = np.full((nc, len(k)), np.nan)
    order = np.argsort(nin, kind='stable')
    for b in range(0, nc, batch):
        ii = order[b:b + batch]
        kq = min(int(nin[ii].max()) + kmax + off, n)
        d = tree.query(pos_c[ii], k=kq, workers=workers)[0].reshape(len(ii), -1)
        col = nin[ii][:, None] + k[None, :] - 1 + off
        good = col < d.shape[1]
        r = np.take_along_axis(d, np.minimum(col, d.shape[1] - 1), 1)
        out[ii] = np.where(good & np.isfinite(r), r, np.nan)
    return out


def _gk(k, q):
    """ln Gamma(k + 1 - q) - ln Gamma(k), -inf ... NaN outside q < k + 1."""
    a = k + 1.0 - q
    return np.where(a > 0, gammaln(np.where(a > 0, a, 1.0)) - gammaln(k), np.nan)


def _q_of_tau(delta, ka, kb, n_iter=80):
    """Root q < ka + 1 of h(q) = _gk(kb, q) - _gk(ka, q) = delta (h decreases)."""
    delta = np.asarray(delta, dtype=np.float64)
    hi = np.full(delta.shape, ka + 1.0 - 1e-12)
    lo = np.full(delta.shape, -50.0)
    h = lambda q: _gk(kb, q) - _gk(ka, q)   # noqa: E731
    for _ in range(12):
        bad = np.isfinite(delta) & (h(lo) < delta)
        if not bad.any():
            break
        lo = np.where(bad, lo * 4.0, lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        up = h(mid) > delta
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    q = 0.5 * (lo + hi)
    return np.where(np.isfinite(delta) & (h(lo) >= delta - 1e-9), q, np.nan)


def _monotone_segment(qt, taus, i0):
    """The stretch of q(tau) about tau = 0 over which it increases strictly
    (at extreme tau the moments are dominated by single centres)."""
    n = len(taus)
    if not np.isfinite(qt[i0]):
        return np.empty(0), np.empty(0)
    lo = i0
    while lo > 0 and np.isfinite(qt[lo - 1]) and qt[lo - 1] < qt[lo]:
        lo -= 1
    hi = i0
    while hi < n - 1 and np.isfinite(qt[hi + 1]) and qt[hi + 1] > qt[hi]:
        hi += 1
    return qt[lo:hi + 1], taus[lo:hi + 1]


def _dq_pairs(S, C, L1, taus, k, pairs, q):
    """D[set, q, pair] from the moment sums of each set (full sample and
    delete-one regions)."""
    nset = S.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        lnW = np.log(S / C[:, None, :])
        lmean = L1 / C
    D = np.full((nset, len(q), len(pairs)), np.nan)
    kidx = {int(kk): j for j, kk in enumerate(k)}
    i1 = np.flatnonzero(np.abs(q - 1.0) < 1e-9)
    for p, (ka, kb) in enumerate(pairs):
        a, b = kidx[int(ka)], kidx[int(kb)]
        qt = _q_of_tau(lnW[:, :, b] - lnW[:, :, a], float(ka), float(kb))
        i0 = int(np.argmin(np.abs(taus)))
        for s in range(nset):
            qs_, ts_ = _monotone_segment(qt[s], taus, i0)
            if len(qs_) < 2:
                continue
            inside = (q >= qs_[0]) & (q <= qs_[-1]) & (q < ka + 1.0)
            tau_q = np.interp(q, qs_, ts_)
            with np.errstate(divide='ignore', invalid='ignore'):
                D[s, :, p] = np.where(inside, tau_q / (q - 1.0), np.nan)
        if len(i1):
            with np.errstate(divide='ignore', invalid='ignore'):
                D[:, i1, p] = (3.0 * (digamma(kb) - digamma(ka)) / (lmean[:, b] - lmean[:, a]))[:, None]
    return D


class FixedMassResult:
    """Generalised dimensions from the fixed-mass estimator.

    Attributes
    ----------
    q, k : the orders and the neighbour masses
    r_k : typical radius of each mass, exp(<ln r_ik>) over the centres
    pairs : (n_pair, 2) the mass pairs of D (adjacent masses by default)
    D : (n_q, n_pair) generalised dimensions of the full sample
    D_jk : (n_regions, n_q, n_pair) delete-one jackknife values, or None
    meta : dict
    """

    def __init__(self, q, k, taus, S, C, L1, lnr, pairs=None, meta=None):
        self.q = np.asarray(q, dtype=np.float64)
        self.k = np.asarray(k, dtype=np.int64)
        self.taus = np.asarray(taus, dtype=np.float64)
        self._S, self._C, self._L1, self._lnr = (np.asarray(a) for a in (S, C, L1, lnr))
        self.meta = dict(meta or {})
        self.pairs = (np.stack([self.k[:-1], self.k[1:]], 1) if pairs is None
                      else np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
        self.D, self.D_jk = self._compute(self.pairs)

    @property
    def n_regions(self):
        return self._S.shape[0]

    @property
    def r_k(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.exp(self._lnr.sum(0) / self._C.sum(0))

    @property
    def r_pair(self):
        """Typical radius of each pair (geometric mean of its two masses)."""
        r = dict(zip(self.k.tolist(), self.r_k))
        return np.array([np.sqrt(r[a] * r[b]) for a, b in self.pairs])

    def _sets(self):
        S, C, L1 = self._S.sum(0), self._C.sum(0), self._L1.sum(0)
        if self.n_regions == 1:
            return S[None], C[None], L1[None]
        return (np.concatenate([S[None], S - self._S]), np.concatenate([C[None], C - self._C]),
                np.concatenate([L1[None], L1 - self._L1]))

    def _compute(self, pairs):
        for a, b in pairs:
            if a not in self.k or b not in self.k or a >= b:
                raise ValueError(f'pair ({a}, {b}) must be two measured masses with k_a < k_b')
        S, C, L1 = self._sets()
        D = _dq_pairs(S, C, L1, self.taus, self.k, pairs, self.q)
        return D[0], (D[1:] if self.n_regions > 1 else None)

    def dq(self, pairs=None, jackknife=False):
        """D_q for mass pairs [(k_a, k_b), ...] (default: the adjacent pairs).
        Returns (D, r) with D of shape (n_q, n_pair) (or (n_regions, ...))."""
        if pairs is None:
            D, Djk, pr = self.D, self.D_jk, self.pairs
        else:
            pr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            D, Djk = self._compute(pr)
        r = dict(zip(self.k.tolist(), self.r_k))
        rp = np.array([np.sqrt(r[a] * r[b]) for a, b in pr])
        if jackknife:
            if Djk is None:
                raise ValueError('no jackknife regions were used')
            return Djk, rp
        return D, rp

    def covariance(self, pairs=None):
        """Jackknife covariance of the flattened (q, pair) array of D_q."""
        return jackknife_covariance(self.dq(pairs, jackknife=True)[0])

    def errors(self, pairs=None):
        x = self.dq(pairs, jackknife=True)[0]
        return np.sqrt(np.diag(jackknife_covariance(x))).reshape(x.shape[1:])

    def save(self, path):
        np.savez(path, kind='FixedMassResult', q=self.q, k=self.k, taus=self.taus, S=self._S,
                 C=self._C, L1=self._L1, lnr=self._lnr, pairs=self.pairs,
                 meta=json.dumps(self.meta, default=_json_default))

    @classmethod
    def load(cls, path):
        f = np.load(path, allow_pickle=False)
        return cls(f['q'], f['k'], f['taus'], f['S'], f['C'], f['L1'], f['lnr'], f['pairs'],
                   json.loads(str(f['meta'])))

    def __repr__(self):
        jk = f', {self.n_regions} jackknife regions' if self.n_regions > 1 else ''
        return f'FixedMassResult({len(self.q)} q, k={self.k.tolist()}{jk})'


def fixed_mass(data, randoms=None, *, k, q=None, box=None, r_min=0.0, edge_cut=None,
               random_correction=True, min_randoms=30, jackknife=None, centres=None,
               n_centres=None, tau=None, seed=0, nthreads=None):
    """Generalised dimensions D_q from the radii holding fixed neighbour masses.

    Parameters
    ----------
    data : Catalogue or array (N, 3)
    randoms : Catalogue or array, optional
        Required without a periodic box.
    k : sequence of int
        Increasing numbers of neighbours. D_q is returned for adjacent pairs;
        any pair can be requested later with `FixedMassResult.dq(pairs=...)`.
        Neighbours are counted, so that nu_ik ~ Gamma(k) exactly for Poisson
        sampling: data weights and marks weight only the centres, and the
        randoms (with their weights) must trace the density of the data
        points, e.g. include the completeness. A smooth mismatch multiplies
        each centre's nu by a factor independent of k and cancels in D_q.
    q : array, optional
        Orders (default -4 to 6 in steps of 0.5); D_q needs q < k_a + 1.
    box : float or (3,), optional
        Periodic box.
    r_min : float
        Count only neighbours beyond this separation.
    edge_cut : float in (0, 1], optional
        Drop the centre at mass k if less than this fraction of the sphere of
        radius R_k lies inside the survey, where R_k is the 90th percentile of
        the radii r_ik over the centres (needs randoms with nbar). A fixed R_k
        keeps the selection of centres independent of their own neighbours.
    random_correction : bool
        Replace nu^-t by its Poisson-unbiased estimate from the random count.
    min_randoms : int
        Where fewer randoms than this lie within r_ik, nu_ik is taken from the
        distance R to the min_randoms-th random, whose enclosed expected count
        is Gamma-distributed: nu^-t is estimated without bias as
        (A v_ik / v_R)^-t Gamma(m) / Gamma(m + t), assuming a locally uniform
        random density within R (the edge cut then uses the footprint
        fraction within R). 0 disables it. Default 30.
    jackknife, centres, n_centres, seed, nthreads : as for `measure`.
    tau : array, optional
        The grid of tau = (q - 1) D_q on which the moments are evaluated.

    Returns
    -------
    FixedMassResult
    """
    data = as_catalogue(data, 'data')
    randoms = as_catalogue(randoms, 'randoms')
    k = np.asarray(k, dtype=np.int64).reshape(-1)
    if len(k) < 2 or np.any(k < 1) or np.any(np.diff(k) <= 0):
        raise ValueError('k must hold at least two increasing positive integers')
    q = parse_q(q)
    L = parse_box(box)
    nthr = set_threads(nthreads)
    rng = np.random.default_rng(seed)
    rmin = float(r_min or 0.0)
    dpos = data.positions
    if L is not None:
        dpos = check_in_box(dpos, L, 'data')
    elif randoms is None:
        raise ValueError('without a periodic box, randoms are required')
    if edge_cut is not None and (randoms is None or randoms.nbar is None):
        raise ValueError('edge_cut needs randoms with nbar (the data density at each random)')

    u = data.mass
    unit_w = bool(np.all(u == u[0]))
    du = np.ones(len(data)) if unit_w else u

    idx = np.arange(len(data))
    if centres is not None:
        c = np.asarray(centres)
        idx = np.flatnonzero(c) if c.dtype == bool else np.unique(c.astype(np.int64))
    if n_centres is not None and n_centres < len(idx):
        idx = np.sort(rng.choice(idx, int(n_centres), replace=False))
    cen = dpos[idx]
    cu = du[idx]

    tree = cKDTree(dpos, boxsize=None if L is None else L)
    rad = _neighbour_radii(tree, cen, k, rmin, nthr)
    ok = np.isfinite(rad)
    if L is not None and np.nanmax(rad) >= 0.5 * L.min():
        raise ValueError('the neighbour radii reach half the box; use smaller k')
    vmin = 4.0 / 3.0 * np.pi * rmin ** 3
    # ascending per centre also where a mass was not reached (masked by ok)
    rad_f = np.maximum(np.maximum.accumulate(np.where(ok, rad, 0.0), axis=1), rmin + 1e-6)
    vk = 4.0 / 3.0 * np.pi * rad_f ** 3
    vstar = vk - vmin                                 # shell volume r_min < d <= r_ik
    norm = np.full(len(cen), len(data) - 1.0)         # number of other points

    # estimate of nu^-t per (centre, mass): A^-t Gamma(b) / Gamma(b + t), or A^-t (b <= 0)
    bpar = np.zeros_like(rad)
    corr_r = False
    if randoms is None:
        nu = norm[:, None] * vstar / float(np.prod(L))
        ok &= nu > 0
        lnA = np.log(np.where(ok, nu, 1.0))
    else:
        rpos = randoms.positions if L is None else check_in_box(randoms.positions, L, 'randoms')
        rw = randoms.weights
        rsplit = edge_cut is not None
        mu_r = (np.arange(len(randoms)) % 2 == 0) if rsplit else np.ones(len(randoms), bool)
        WR = float(rw[mu_r].sum())
        WV = float(rw[~mu_r].sum()) if rsplit else WR
        rdv = (np.empty(0) if randoms.nbar is None else data.weights.sum() / WV * rw / randoms.nbar)
        unit_rw = bool(np.all(rw[mu_r] == rw[mu_r][0]))
        wbar = float(rw[mu_r].mean())
        base = norm / WR                                  # nu = base * (random weight)
        # reference: the distance to the m0-th (expectation) random beyond r_min
        m0 = int(min_randoms)
        Rref = np.full(len(cen), np.nan)
        if m0 > 0:
            rtree = cKDTree(rpos[mu_r], boxsize=None if L is None else L)
            Rref = _neighbour_radii(rtree, cen, np.array([m0]), rmin, nthr, has_self=False)[:, 0]
        # footprint fraction for the edge cut at one radius per mass (independent
        # of each centre's own neighbours): the EDGE_QUANTILE of r_ik
        Rcut = np.array([np.percentile(rad[ok[:, j], j], 100 * EDGE_QUANTILE) if ok[:, j].any()
                         else rmin + 1e-6 for j in range(len(k))])
        nk_ = len(k)
        rad_all = np.concatenate([rad_f, np.where(np.isfinite(Rref) & (Rref > rmin), Rref,
                                                  rmin + 1e-6)[:, None],
                                  np.broadcast_to(np.maximum(Rcut, rmin + 1e-6), rad_f.shape)], 1)
        srt = np.argsort(rad_all, axis=1, kind='stable')
        rs = np.take_along_axis(rad_all, srt, 1)
        grid = Grid(float(np.percentile(rs[:, -1], 75)), [dpos, rpos, cen], None if L is None else L)
        r_order, r_start = grid.build(rpos)
        out4 = [np.zeros_like(rs) for _ in range(4)]
        _kernels.random_sums_at_radii(cen, np.full(len(cen), -1, np.int64), np.ascontiguousarray(rs),
                                      rmin * rmin, rpos, rw, rdv, r_order, r_start, grid.lo, grid.cell,
                                      grid.n, grid.periodic, np.ones(3) if L is None else L, rsplit, *out4)
        inv = np.argsort(srt, axis=1)
        R, Rn, R2, DV = (np.take_along_axis(a_, inv, 1) for a_ in out4)
        Rk, Rnk, R2k = R[:, :nk_], Rn[:, :nk_], R2[:, :nk_]
        use_ref = (m0 > 0) & (Rnk < m0) & np.isfinite(Rref)[:, None]
        ok &= (Rk > 0) | use_ref
        if edge_cut is not None:
            ok &= DV[:, nk_ + 1:] / (4.0 / 3.0 * np.pi * np.maximum(Rcut, rmin + 1e-6) ** 3) >= edge_cut
        with np.errstate(divide='ignore', invalid='ignore'):
            vref_sh = 4.0 / 3.0 * np.pi * np.where(np.isfinite(Rref), Rref, 1.0) ** 3 - vmin
            # the local mean random weight within the reference sphere
            wloc = np.where(Rn[:, nk_] > 0, R[:, nk_] / np.where(Rn[:, nk_] > 0, Rn[:, nk_], 1.0), wbar)
            lnA_ref = np.log(base[:, None] * wloc[:, None] * vstar / vref_sh[:, None])
            if random_correction:
                corr_r = True
                if unit_rw:
                    lnA_own = np.log(base[:, None] * wbar * np.ones_like(Rk))
                    b_own = Rnk + 1.0
                else:
                    lnA_own = np.log(np.where(Rk > 0, base[:, None] * R2k / np.where(Rk > 0, Rk, 1.0), 1.0))
                    b_own = np.where(R2k > 0, Rk * Rk / np.where(R2k > 0, R2k, 1.0), 0.0) + 1.0
            else:
                lnA_own = np.log(np.where(Rk > 0, base[:, None] * Rk, 1.0))
                b_own = np.zeros_like(Rk)
            lnA = np.where(use_ref, lnA_ref, lnA_own)
            bpar = np.where(use_ref, float(m0), b_own)
    lnA = np.where(ok, lnA, 0.0)
    bpar = np.where(ok, bpar, 0.0)

    if tau is None:
        lo = min(-1.0, 5.0 * (q.min() - 1.0))
        hi = max(1.0, 5.0 * (q.max() - 1.0))
        taus = np.unique(np.concatenate([np.linspace(lo, hi, 401), [0.0]]))
    else:
        taus = np.unique(np.asarray(tau, dtype=np.float64))

    if jackknife is not None and not np.isscalar(jackknife):
        lab_all = np.asarray(jackknife).reshape(-1)
        if len(lab_all) != len(data):
            raise ValueError('jackknife labels must have one entry per data point')
        lab, nreg = region_labels(lab_all[idx], cen, L, seed)
    else:
        lab, nreg = region_labels(jackknife, cen, L, seed)
    order = np.argsort(lab, kind='stable')
    chunk_start, chunk_region = chunks(lab[order], nreg, nthr)
    out = np.zeros((len(chunk_start) - 1, len(taus), len(k)))
    o = lambda a: np.ascontiguousarray(a[order])   # noqa: E731
    _kernels.tau_sums(o(lnA), o(bpar), o(ok), o(cu), chunk_start, taus, out)
    S = reduce_regions(out, chunk_region, nreg)
    w = cu[:, None] * ok
    lnnu = lnA + np.where(bpar > 0, digamma(np.where(bpar > 0, bpar, 1.0)), 0.0)
    C = np.zeros((nreg, len(k)))
    L1 = np.zeros((nreg, len(k)))
    lnr = np.zeros((nreg, len(k)))
    np.add.at(C, lab, w)
    np.add.at(L1, lab, w * np.where(ok, lnnu, 0.0))
    np.add.at(lnr, lab, w * np.log(rad_f))
    meta = dict(estimator='fixed-mass', periodic=L is not None,
                box=None if L is None else L.tolist(), n_data=len(data),
                n_randoms=0 if randoms is None else len(randoms), n_centres=len(cen),
                unit_weights=unit_w, r_min=rmin, edge_cut=edge_cut,
                random_correction=corr_r, min_randoms=int(min_randoms) if randoms is not None else 0,
                n_regions=nreg, seed=seed)
    return FixedMassResult(q, k, taus, S, C, L1, lnr, meta=meta)
