"""Jackknife regions: sub-boxes of a periodic box, or k-means clusters of
the centres (any geometry)."""
import numpy as np
from scipy.spatial import cKDTree

__all__ = ['subboxes', 'kmeans', 'assign', 'relabel']


def subboxes(positions, box, n_side):
    """Label of the sub-box (n_side^3 equal cubes of the box) of each point."""
    L = np.asarray(box, dtype=np.float64) * np.ones(3)
    c = np.clip(np.floor(np.asarray(positions) / L * n_side).astype(np.int64), 0, n_side - 1)
    return (c[:, 0] * n_side + c[:, 1]) * n_side + c[:, 2]


def kmeans(positions, n_regions, seed=0, n_sample=200_000, n_iter=30):
    """Centroids of n_regions k-means clusters of the points (k-means++ start,
    Lloyd iterations on a random subsample of at most n_sample points)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(positions, dtype=np.float64)
    if n_regions > len(x):
        raise ValueError('more regions than points')
    if len(x) > n_sample:
        x = x[rng.choice(len(x), n_sample, replace=False)]
    cen = np.empty((n_regions, 3))
    cen[0] = x[rng.integers(len(x))]
    d2 = ((x - cen[0]) ** 2).sum(1)
    for k in range(1, n_regions):
        cen[k] = x[rng.choice(len(x), p=d2 / d2.sum())]
        d2 = np.minimum(d2, ((x - cen[k]) ** 2).sum(1))
    for _ in range(n_iter):
        lab = cKDTree(cen).query(x)[1]
        cnt = np.bincount(lab, minlength=n_regions)
        new = np.stack([np.bincount(lab, x[:, d], minlength=n_regions) for d in range(3)], 1)
        ok = cnt > 0
        new[ok] /= cnt[ok, None]
        new[~ok] = cen[~ok]
        if np.allclose(new, cen):
            break
        cen = new
    return cen


def assign(positions, centroids):
    """Label of the nearest centroid of each point."""
    return cKDTree(centroids).query(np.asarray(positions))[1].astype(np.int64)


def relabel(labels):
    """Map arbitrary integer labels to 0..n-1; returns (labels, n)."""
    u, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64), len(u)
