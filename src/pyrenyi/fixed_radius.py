"""Fixed-radius estimators of the Renyi generalised dimensions.

Point-centred (default): the centres are the data points, and

    Z_q(r) = sum_i c_i T_s(i, r) / mu_i(r)^s / sum_i c_i,     s = q - 1,
    Y_q(r) = ln Z_q(r) / (q - 1) + ln[f v(r) / V],
    D_q(r) = dY_q / d ln r,

where T_s is the finite-sample-corrected s-th power of the neighbour mass
around centre i, mu_i its expectation for a homogeneous field (from the
randoms, or analytic in a periodic box), v(r) the sphere volume, f the
solid-angle fraction of the wedge and V a reference volume. For q = 1,
Y_1 = sum_i c_i [psi-corrected ln(mass) - ln mu_i] / sum_i c_i + ln[f v / V].

Volume-weighted: the centres fill the volume (uniform points in a box, or the
randoms of a survey) and s = q, Z_q = sum_c T_q / mu^q over occupied spheres.

Finite-sample corrections (`correction`):

'grassberger'  Unit weights: T_s = Gamma(n + 1) / Gamma(n + 1 - s)
               (Grassberger 1988, 2003), exact for integer s >= 0. With
               weights, s! e_s(u), the elementary symmetric polynomial of the
               neighbour masses u_j (the sum over distinct s-tuples), which is
               exact for integer s: for Poisson sampling of a field its mean
               is the s-th power of the expected mass. Non-integer s uses the
               'effective' form.
'effective'    T_s = c^s Gamma(n_eff + 1) / Gamma(n_eff + 1 - s), with
               n_eff = (sum u)^2 / sum u^2 and c = sum u^2 / sum u (identical to
               'grassberger' for unit weights).
'none'         T_s = (sum u)^s; point-centred with the centre included (the
               sandbox estimator), so that n = 0 does not occur.
"""
import math
import warnings

import numpy as np

from . import _kernels
from ._common import (DEFAULT_Q, check_in_box, chunks, is_integer, parse_box, parse_q,
                      reduce_regions, region_labels, set_threads, unit_vector)
from ._grid import Grid
from .catalogue import as_catalogue
from .results import RenyiResult

__all__ = ['measure', 'DEFAULT_Q']

CORRECTIONS = {'grassberger': 0, 'effective': 1, 'none': 2}
AXES = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}


def _mu_edges(wedges):
    if wedges is None:
        return np.array([0.0, 1.0]), True
    if np.isscalar(wedges):
        n = int(wedges)
        if n < 1:
            raise ValueError('wedges must be >= 1')
        return np.linspace(0.0, 1.0, n + 1), True
    e = np.asarray(wedges, dtype=np.float64)
    if e.ndim != 1 or len(e) < 2 or e[0] != 0.0 or e[-1] != 1.0 or np.any(np.diff(e) <= 0):
        raise ValueError('wedge edges must increase from 0 to 1 (|mu| = |cos| to the line of sight)')
    return e, bool(np.allclose(e, np.linspace(0.0, 1.0, len(e)), rtol=0, atol=1e-15))


def measure(data, randoms=None, *, radii, q=None, box=None, estimator='point', r_min=None,
            wedges=None, los=None, observer=(0.0, 0.0, 0.0), correction='grassberger',
            edge_cut=None, random_correction=True, min_randoms=30, jackknife=None,
            centres=None, n_centres=None, seed=0,
            ref_volume=None, nthreads=None, chunk_size=None):
    """Renyi functionals Y_q(r) and generalised dimensions D_q(r) of a point set.

    Parameters
    ----------
    data : Catalogue or array (N, 3)
        Cartesian positions, with optional weights and marks.
    randoms : Catalogue or array (N_R, 3), optional
        Random catalogue with the geometry and selection of the data (and
        weights). Required without a periodic box; in a box the expectation is
        analytic and randoms are optional.
    radii : array
        Increasing radii at which the counts are evaluated.
    q : array, optional
        Orders; default -4 to 6 in steps of 0.5.
    box : float or (3,), optional
        Side length(s) of a periodic box with points in [0, L).
    estimator : 'point' or 'volume'
        Point-centred (centres = data) or volume-weighted (centres fill the
        volume; q >= 0 only).
    r_min : float or list, optional
        Minimum centre-neighbour separations: neighbours are counted in the
        shell r_min < d <= r. Each must be one of the radii.
    wedges : int or array, optional
        Number of equal-width |mu| wedges, or their edges in [0, 1]. The
        isotropic statistic is always returned as well.
    los : 'x', 'y', 'z', a 3-vector, 'centre' or 'midpoint'
        Line of sight for the wedges: a fixed direction (default 'z' in a
        box), the direction from the observer to the centre, or to the pair
        midpoint (default without a box).
    observer : (3,)
        Observer position for los='centre' or 'midpoint'.
    correction : 'grassberger', 'effective' or 'none'
        Finite-sample correction (see the module docstring).
    edge_cut : float in (0, 1], optional
        Drop, at each radius, the centres whose sphere has less than this
        fraction of its volume inside the survey. Needs randoms with nbar.
    random_correction : bool
        Correct the noise of the random counts: mu^-s is replaced by its
        Poisson-unbiased estimate from the random count m,
        Gamma(m + 1) / Gamma(m + 1 + s) (exact for integer s < 0, and up to
        terms of order exp(-m) for s > 0). Default True.
    min_randoms : int
        A sphere, shell or wedge with fewer randoms takes its expected count
        from the smallest sphere about the same centre with at least this many
        randoms (or the sphere of the same radius), scaled by the volume
        ratio; the edge cut uses the footprint fraction of that sphere. This
        removes the noise of sparse randoms at small scales for centres away
        from the edges. 0 disables it. Default 30.
    jackknife : int or array, optional
        Number of jackknife regions (sub-boxes in a box, which must be a cube
        number; k-means clusters of the centres otherwise) or a label per
        centre.
    centres : boolean mask or index array, optional
        The data points (point-centred) or randoms (volume-weighted) used as
        centres; all data remain neighbours. E.g. centres away from the
        survey edges, with neighbours from the whole survey.
    n_centres : int, optional
        A random subset of n_centres of the (selected) centres. For the
        volume-weighted estimator in a box without randoms: the number of
        uniform centres (default len(data)).
    seed : int
        Seed for centre subsets, uniform centres and k-means regions.
    ref_volume : float, optional
        Reference volume V in Y_q (default: the box volume, or 1).
    nthreads : int, optional
        Number of numba threads.

    Returns
    -------
    RenyiResult
    """
    data = as_catalogue(data, 'data')
    randoms = as_catalogue(randoms, 'randoms')
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(radii) < 1 or np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError('radii must be positive and strictly increasing')
    q = parse_q(q)
    L = parse_box(box)
    if estimator not in ('point', 'volume'):
        raise ValueError("estimator must be 'point' or 'volume'")
    if correction not in CORRECTIONS:
        raise ValueError(f'correction must be one of {list(CORRECTIONS)}')
    corr = CORRECTIONS[correction]
    volume = estimator == 'volume'
    nthr = set_threads(nthreads)

    dpos = data.positions
    rpos = None if randoms is None else randoms.positions
    if L is not None:
        dpos = check_in_box(dpos, L, 'data')
        if rpos is not None:
            rpos = check_in_box(rpos, L, 'randoms')
        if radii[-1] >= 0.5 * L.min():
            raise ValueError('in a periodic box the largest radius must be below half the box')
    elif randoms is None:
        raise ValueError('without a periodic box, randoms are required')

    # ---- minimum separations ----
    rmins = [] if r_min is None else list(np.atleast_1d(np.asarray(r_min, dtype=np.float64)))
    case_start = [0]
    for rm in rmins:
        k = np.flatnonzero(np.isclose(radii, rm, rtol=1e-10, atol=0.0))
        if not len(k):
            raise ValueError(f'r_min={rm} is not one of the radii; add it to radii')
        if k[0] >= len(radii) - 1:
            raise ValueError(f'r_min={rm} must be below the largest radius')
        case_start.append(int(k[0]) + 1)
    case_start = np.asarray(case_start, np.int64)
    ncase = len(case_start)

    # ---- wedges and line of sight ----
    mu_edges, equal_mu = _mu_edges(wedges)
    nmu = len(mu_edges) - 1
    nW = nmu + 1 if nmu > 1 else 1
    if los is None:
        los = 'z' if L is not None else 'midpoint'
    los_mid = isinstance(los, str) and los == 'midpoint'
    los_cen = isinstance(los, str) and los == 'centre'
    if (los_mid or los_cen) and L is not None and nmu > 1:
        raise ValueError("los='centre'/'midpoint' needs an observer, not a periodic box; "
                         "use a fixed axis in a box")
    observer = np.asarray(observer, dtype=np.float64).reshape(3)
    if isinstance(los, str) and not (los_mid or los_cen):
        if los not in AXES:
            raise ValueError("los must be 'x', 'y', 'z', a 3-vector, 'centre' or 'midpoint'")
        los_vec = np.asarray(AXES[los])
    elif not isinstance(los, str):
        los_vec = unit_vector(los)
    else:
        los_vec = None

    # ---- masses and weights ----
    u = data.mass
    if not np.any(u > 0):
        raise ValueError('all data weights * marks are zero')
    unit_w = bool(np.all(u == u[0]))
    du = np.ones(len(data)) if unit_w else u
    U = float(du.sum())
    u0 = float(du.mean())
    rng = np.random.default_rng(seed)

    have_rand = randoms is not None
    if edge_cut is not None:
        if not 0 < edge_cut <= 1:
            raise ValueError('edge_cut must be in (0, 1]')
        if randoms is None or randoms.nbar is None:
            raise ValueError('edge_cut needs randoms with nbar (the data density at each random)')
    rsplit = edge_cut is not None
    if have_rand:
        rw = randoms.weights
        # with an edge cut, even randoms give the expectation, odd ones the footprint
        mu_r = (np.arange(len(randoms)) % 2 == 0) if rsplit else np.ones(len(randoms), bool)
        WR = float(rw[mu_r].sum())
        if WR <= 0:
            raise ValueError('all random weights are zero')
        WV = float(rw[~mu_r].sum()) if rsplit else WR
        rdv = (np.empty(0) if randoms.nbar is None
               else data.weights.sum() / WV * rw / randoms.nbar)
        unit_rw = bool(np.all(rw[mu_r] == rw[mu_r][0]))
        rw_mean = float(rw[mu_r].mean())
        rw_self = np.where(mu_r, rw, 0.0)
    else:
        rw, WR, rdv = np.empty(0), 0.0, np.empty(0)
        unit_rw, rw_mean, rw_self = False, 1.0, np.empty(0)
    fmin = 0.0 if edge_cut is None else float(edge_cut)
    Vref = float(np.prod(L)) if (L is not None and ref_volume is None) else float(ref_volume or 1.0)
    Vbox = float(np.prod(L)) if L is not None else 0.0

    # ---- centres ----
    def _select(n):
        if centres is None:
            idx = np.arange(n)
        else:
            c = np.asarray(centres)
            idx = np.flatnonzero(c) if c.dtype == bool else np.unique(c.astype(np.int64))
            if (c.dtype == bool and len(c) != n) or (len(idx) and (idx[0] < 0 or idx[-1] >= n)):
                raise ValueError('centres must be a boolean mask or indices of the centre catalogue')
            if not len(idx):
                raise ValueError('no centres selected')
        if n_centres is not None and n_centres < len(idx):
            idx = np.sort(rng.choice(idx, int(n_centres), replace=False))
        return idx

    if not volume:
        idx = _select(len(data))
        cen = dpos[idx]
        cen_self = idx.astype(np.int64)
        cen_rself = np.full(len(idx), -1, np.int64)
        cen_u = du[idx]
        cen_c = cen_u.copy()
        mu_norm = (U - cen_u) / (WR if have_rand else Vbox)
        sb_fac = U / np.maximum(U - cen_u, 1e-300)
        lab_total = len(data)
        lab_idx = idx
    else:
        if have_rand:
            idx = _select(len(randoms))
            cen = rpos[idx]
            cen_rself = idx.astype(np.int64)
            cen_c = rdv[idx].copy() if len(rdv) else np.ones(len(idx))
            mu_norm = U / np.maximum(WR - rw_self[idx], 1e-300)
            lab_total, lab_idx = len(randoms), idx
        else:
            if centres is not None:
                raise ValueError('with uniform centres in a box, use n_centres instead of centres')
            nc = len(data) if n_centres is None else int(n_centres)
            cen = rng.uniform(0.0, 1.0, (nc, 3)) * L
            cen_rself = np.full(nc, -1, np.int64)
            cen_c = np.ones(nc)
            mu_norm = np.full(nc, U / Vbox)
            lab_total, lab_idx = None, None
            if jackknife is not None and not np.isscalar(jackknife):
                raise ValueError('with uniform centres in a box, jackknife must be a number of sub-boxes')
        cen_self = np.full(len(cen), -1, np.int64)
        cen_u = np.zeros(len(cen))
        sb_fac = np.ones(len(cen))
        if np.any(q < 0):
            warnings.warn('the volume-weighted estimator is undefined for q < 0; returning NaN there')

    if los_vec is not None:
        cen_los = np.broadcast_to(los_vec, cen.shape).copy()
    elif los_cen:
        d = cen - observer
        n = np.linalg.norm(d, axis=1, keepdims=True)
        cen_los = np.where(n > 0, d / np.where(n > 0, n, 1.0), np.array([0.0, 0.0, 1.0]))
    else:
        cen_los = np.zeros_like(cen)

    # ---- orders ----
    s_vals = q - 1.0 if not volume else q.copy()
    special = np.array([is_integer(x) and round(x) == 1 for x in q], np.int64) if volume else \
        np.array([abs(x) < 1e-12 for x in s_vals], np.int64)
    exact = (not unit_w) and corr == 0
    s_int = np.array([int(round(s)) if exact and is_integer(s) and round(s) >= 1 and not sp else -1
                      for s, sp in zip(s_vals, special)], np.int64)
    smax = int(max(s_int.max(), 0))
    fact = np.array([math.factorial(k) for k in range(smax + 1)], dtype=np.float64)

    # ---- grid, regions, chunks ----
    grid = Grid(radii[-1], [dpos, rpos, cen], None if L is None else L)
    d_order, d_start = grid.build(dpos)
    r_order, r_start = grid.build(rpos)
    if jackknife is not None and not np.isscalar(jackknife):
        lab_all = np.asarray(jackknife).reshape(-1)
        if lab_total is None or len(lab_all) != lab_total:
            raise ValueError(f'jackknife labels must have one entry per '
                             f'{"data point" if not volume else "random"}')
        lab, nreg = region_labels(lab_all[lab_idx], cen, L, seed)
    else:
        lab, nreg = region_labels(jackknife, cen, L, seed)
    order = np.lexsort((grid.cell_index(cen), lab))
    chunk_start, chunk_region = chunks(lab[order], nreg, nthr, chunk_size)

    vsph = 4.0 / 3.0 * np.pi * radii ** 3
    frac = np.ones(nW)
    if nmu > 1:
        frac[:nmu] = np.diff(mu_edges)
    vol = np.zeros((ncase, len(radii), nW))
    for c, b0 in enumerate(case_start):
        vin = 0.0 if c == 0 else vsph[b0 - 1]
        vol[c, b0:] = (vsph[b0:] - vin)[:, None] * frac[None, :]

    nch = len(chunk_start) - 1
    num = np.zeros((nch, ncase, len(q), len(radii), nW))
    den = np.zeros_like(num)
    e = np.empty((0, 3))
    o = lambda a: np.ascontiguousarray(a[order])   # noqa: E731
    _kernels.fixed_radius(
        o(cen), o(cen_self), o(cen_rself), o(cen_c), o(cen_u), o(mu_norm), o(sb_fac), o(cen_los),
        chunk_start, dpos, du, d_order, d_start,
        e if rpos is None else rpos, rw, rdv, r_order, r_start,
        grid.lo, grid.cell, grid.n, grid.reach, grid.periodic, np.ones(3) if L is None else L,
        radii ** 2, case_start, vol, vsph, mu_edges, equal_mu, los_mid, observer,
        s_vals, s_int, special, volume, unit_w, corr, u0, smax, fact,
        have_rand, bool(random_correction), unit_rw, rw_mean,
        int(min_randoms) if have_rand else 0, fmin, rsplit, num, den)

    ref = np.log(vsph[None, :, None] * frac[None, None, :] / Vref) * np.ones((ncase, 1, 1))
    reg_num = reduce_regions(num, chunk_region, nreg)
    reg_den = reduce_regions(den, chunk_region, nreg)
    tn, td = reg_num.sum(0), reg_den.sum(0)
    Y = _assemble(tn, td, q, special, ref, volume)
    Y_jk = None
    if nreg > 1:
        Y_jk = np.stack([_assemble(tn - reg_num[k], td - reg_den[k], q, special, ref, volume)
                         for k in range(nreg)])
    reg_q = int(np.flatnonzero(special == 0)[0]) if np.any(special == 0) else 0
    meta = dict(estimator=estimator, correction=correction, periodic=L is not None,
                box=None if L is None else L.tolist(), n_data=len(data),
                n_randoms=0 if randoms is None else len(randoms), n_centres=len(cen),
                unit_weights=unit_w, los=los if isinstance(los, str) else los_vec.tolist(),
                observer=observer.tolist(), edge_cut=edge_cut,
                random_correction=bool(random_correction and have_rand),
                min_randoms=int(min_randoms) if have_rand else 0, ref_volume=Vref,
                n_regions=nreg, seed=seed)
    return RenyiResult(q, radii, rmins, mu_edges if nmu > 1 else None, Y, Y_jk,
                       td[:, reg_q], meta)


def _assemble(num, den, q, special, ref, volume):
    """Y[case, q, r, w] from the summed moments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
        Y = np.full(num.shape, np.nan)
        for iq, qq in enumerate(q):
            if volume and qq < 0:
                continue
            good = den[:, iq] > 0
            if special[iq]:
                y = ratio[:, iq] + ref
            else:
                ok = good & (ratio[:, iq] > 0)
                y = np.where(ok, np.log(np.where(ok, ratio[:, iq], 1.0)) / (qq - 1.0) + ref, np.nan)
            Y[:, iq] = np.where(good, y, np.nan)
    return Y
