"""Shared input handling: geometry, q grid, jackknife regions, chunking."""
import math

import numba
import numpy as np

from . import regions as _regions

DEFAULT_Q = np.round(np.arange(-4.0, 6.0 + 1e-9, 0.5), 10)


def parse_box(box):
    """None (no periodic box) or the box side lengths (3,)."""
    if box is None:
        return None
    L = np.asarray(box, dtype=np.float64) * np.ones(3)
    if L.shape != (3,) or np.any(L <= 0) or not np.all(np.isfinite(L)):
        raise ValueError('box must be a positive scalar or three positive side lengths')
    return L


def check_in_box(pos, L, name):
    """Points must lie in [0, L); points at exactly L are mapped to 0."""
    if np.any(pos < 0) or np.any(pos > L):
        raise ValueError(f'{name}: positions must lie in [0, box) for a periodic box '
                         '(wrap them with np.mod(positions, box))')
    if np.any(pos == L):
        pos = pos.copy()
        pos[pos == np.broadcast_to(L, pos.shape)] = 0.0
    return pos


def parse_q(q):
    q = DEFAULT_Q if q is None else np.atleast_1d(np.asarray(q, dtype=np.float64))
    if q.ndim != 1 or not len(q) or not np.all(np.isfinite(q)):
        raise ValueError('q must be a non-empty 1-d array of finite values')
    return q


def is_integer(x, tol=1e-12):
    return abs(x - round(x)) < tol


def set_threads(n):
    if n is not None:
        numba.set_num_threads(int(n))
    return numba.get_num_threads()


def region_labels(jackknife, centres, L, seed):
    """Integer labels 0..n-1 of the centres, and n (1 without jackknife).
    jackknife: None, a number of regions, or one label per centre."""
    nc = len(centres)
    if jackknife is None or (np.isscalar(jackknife) and int(jackknife) <= 1):
        return np.zeros(nc, np.int64), 1
    if np.isscalar(jackknife):
        n = int(jackknife)
        if L is not None:
            side = int(round(n ** (1.0 / 3.0)))
            if side ** 3 != n:
                raise ValueError(f'in a periodic box, jackknife must be a cube number of sub-boxes '
                                 f'(e.g. {side ** 3} or {(side + 1) ** 3}), got {n}')
            lab = _regions.subboxes(centres, L, side)
        else:
            lab = _regions.assign(centres, _regions.kmeans(centres, n, seed=seed))
        return _regions.relabel(lab)
    lab = np.asarray(jackknife).reshape(-1)
    if len(lab) != nc:
        raise ValueError('jackknife labels must have one entry per centre')
    return _regions.relabel(lab)


def chunks(order_regions, nreg, nthr, chunk_size=None):
    """Chunk boundaries over centres sorted by region; each chunk lies in one
    region. Returns (chunk_start, chunk_region)."""
    nc = len(order_regions)
    cs = chunk_size or max(16, int(math.ceil(nc / (8 * max(nthr, 1)))))
    bounds = np.searchsorted(order_regions, np.arange(nreg + 1))
    start, creg = [], []
    for r in range(nreg):
        a, b = bounds[r], bounds[r + 1]
        if b <= a:
            continue
        for s in range(a, b, cs):
            start.append(s)
            creg.append(r)
    start.append(nc)
    return np.asarray(start, np.int64), np.asarray(creg, np.int64)


def reduce_regions(arr, chunk_region, nreg):
    """Sum per-chunk arrays into per-region arrays."""
    out = np.zeros((nreg,) + arr.shape[1:])
    np.add.at(out, chunk_region, arr)
    return out


def unit_vector(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError('line-of-sight vector must be non-zero')
    return v / n
