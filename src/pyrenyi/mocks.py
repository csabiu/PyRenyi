"""Test point sets with known generalised dimensions."""
import numpy as np

__all__ = ['poisson', 'cascade', 'random_cascade', 'cascade_dq']


def poisson(n, box=1.0, seed=0):
    """n uniform points in a box of side(s) `box` (D_q = 3 for every q)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (int(n), 3)) * (np.asarray(box, dtype=np.float64) * np.ones(3))


def cascade(n, w, levels, box=1.0, seed=0):
    """n points sampled from a deterministic 3D multiplicative cascade.

    At each of `levels` levels a cube is split into octants that receive the
    fractions w[0..7] of its mass (the same assignment at every level); below
    the finest level the points are uniform. Self-similar for scales between
    box / 2**levels and box, with D_q given by `cascade_dq`. The cascade is
    periodic, so it can be analysed in a periodic box.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (8,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('w must hold 8 non-negative octant weights')
    rng = np.random.default_rng(seed)
    octs = rng.choice(8, size=(int(n), levels), p=w / w.sum())
    bits = np.stack([(octs >> b) & 1 for b in range(3)], axis=-1)
    scale = 1.0 / 2.0 ** (np.arange(levels) + 1)
    x = np.einsum('nld,l->nd', bits.astype(np.float64), scale)
    x += rng.uniform(0.0, 1.0 / 2.0 ** levels, size=(int(n), 3))
    return (x % 1.0) * (np.asarray(box, dtype=np.float64) * np.ones(3))


def _splitmix64(x):
    with np.errstate(over='ignore'):
        x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def random_cascade(n, w, levels, box=1.0, seed=0):
    """n points from a random 3D multiplicative cascade.

    As `cascade`, but every cell at every level assigns the weights w to its
    octants in an independent random order, so the measure is statistically
    homogeneous (and periodic) with the same D_q (`cascade_dq`). Points are
    drawn independently: each descends the levels choosing an octant with the
    probabilities of its current cell, whose permutation is a hash of the
    cell, so all points see the same cascade.
    """
    from itertools import permutations
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (8,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('w must hold 8 non-negative octant weights')
    if levels > 20:
        raise ValueError('levels must be at most 20')
    perms = np.array(list(permutations(range(8))), dtype=np.int64)
    cum = np.cumsum((w / w.sum())[perms], axis=1)
    rng = np.random.default_rng(seed)
    n = int(n)
    cell = np.zeros((n, 3), np.int64)
    salt = np.uint64(_splitmix64(np.array([seed], np.uint64))[0])
    for lev in range(levels):
        key = (((cell[:, 0] << 20) | cell[:, 1]) << 20 | cell[:, 2]).astype(np.uint64)
        h = _splitmix64(_splitmix64(key ^ salt) + np.uint64(lev))
        p = (h % np.uint64(len(perms))).astype(np.int64)
        o = (rng.random(n)[:, None] > cum[p]).sum(1)
        o = np.minimum(o, 7)
        cell = 2 * cell + np.stack([o & 1, (o >> 1) & 1, (o >> 2) & 1], axis=1)
    x = (cell + rng.random((n, 3))) / 2.0 ** levels
    return x * (np.asarray(box, dtype=np.float64) * np.ones(3))


def cascade_dq(w, q):
    """Exact D_q of the cascade with octant weights w (array of q allowed)."""
    w = np.asarray(w, dtype=np.float64)
    w = w[w > 0] / w.sum()
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    out = np.empty(len(q))
    for i, qq in enumerate(q):
        if abs(qq - 1.0) < 1e-12:
            out[i] = -np.sum(w * np.log2(w))
        else:
            out[i] = np.log2(np.sum(w ** qq)) / (1.0 - qq)
    return out if len(out) > 1 else out[0]
