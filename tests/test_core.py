"""Tests of the estimators: exact agreement with a brute-force reference and
with the estimator of Sabiu (2026), and statistical tests on Poisson
processes and multiplicative cascades in boxes and survey geometries."""
import os
import warnings

import numpy as np
import pytest

import pyrenyi
from reference import renyi_reference

HERE = os.path.dirname(os.path.abspath(__file__))
W8 = [0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09]
RADII = 5.0 * 2.0 ** (np.arange(14) / 3.0)


# ---------------------------------------------------------------- exactness

def test_matches_paper_estimator():
    """Box mode reproduces the estimator of Sabiu (2026) (spheres, shells
    beyond r_min, mu-wedges, sandbox)."""
    ref = np.load(os.path.join(HERE, 'data', 'paper_dq_reference.npz'))
    x = ref['positions'].astype(np.float64)
    res = pyrenyi.measure(x, radii=ref['radii'], q=ref['q'], box=float(ref['box']),
                          r_min=ref['r_min'], wedges=3, los='z')
    sb = pyrenyi.measure(x, radii=ref['radii'], q=ref['q'], box=float(ref['box']), correction='none')

    def same(a, b):
        fin = np.isfinite(b)
        assert np.allclose(a[fin], b[fin], rtol=0, atol=1e-10)
        assert np.all(np.isnan(a[~fin]))              # the paper's -inf / NaN

    same(res.Yq(), ref['Y'])
    for e, rm in enumerate(ref['r_min']):
        same(res.Yq(r_min=rm), ref['Yex'][e])
        for m in range(3):
            same(res.Yq(r_min=rm, wedge=m), ref['Ywex'][m, e])
    for m in range(3):
        same(res.Yq(wedge=m), ref['Yw'][m])
    same(sb.Yq(), ref['Ysb'])


@pytest.fixture(scope='module')
def small():
    g = np.random.default_rng(0)
    L = 100.0
    x = pyrenyi.mocks.random_cascade(400, [0.3, 0.2, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03], 5, box=L, seed=1)
    return dict(L=L, x=x, u=g.lognormal(0, 0.5, len(x)), r=g.uniform(0, L, (3000, 3)),
                rw=g.uniform(0.5, 1.5, 3000), radii=np.array([4.0, 7.0, 10.0, 16.0, 24.0]),
                q=np.array([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0]))


def _close(a, b):
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) < 1e-10


@pytest.mark.parametrize('marks', [False, True])
@pytest.mark.parametrize('correction', ['grassberger', 'effective', 'none'])
def test_box_matches_reference(small, marks, correction):
    s = small
    u = s['u'] if marks else np.ones(len(s['x']))
    got = pyrenyi.measure(pyrenyi.Catalogue(s['x'], marks=u), box=s['L'], radii=s['radii'], q=s['q'],
                          r_min=[7.0], wedges=3, correction=correction).Y
    _close(got, renyi_reference(s['x'], u, s['radii'], s['q'], box=s['L'], r_min=[7.0], wedges=3,
                                correction=correction))


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('los', ['midpoint', 'centre', (1.0, 2.0, 0.5)])
@pytest.mark.parametrize('rc', [True, False])
def test_survey_matches_reference(small, weighted, los, rc):
    s = small
    x, r = s['x'] + 50.0, s['r'] + 50.0
    u = s['u'] if weighted else np.ones(len(x))
    rw = s['rw'] if weighted else None
    got = pyrenyi.measure(pyrenyi.Catalogue(x, weights=u), pyrenyi.Catalogue(r, weights=rw),
                          radii=s['radii'], q=s['q'], r_min=[7.0], wedges=3, los=los,
                          observer=(-300.0, -200.0, -100.0), random_correction=rc, min_randoms=0).Y
    _close(got, renyi_reference(x, u, s['radii'], s['q'], randoms=r, rw=rw, r_min=[7.0], wedges=3,
                                los=los, observer=(-300.0, -200.0, -100.0), random_correction=rc))


@pytest.mark.parametrize('correction', ['grassberger', 'none'])
def test_volume_matches_reference(small, correction):
    s = small
    x, r = s['x'] + 50.0, s['r'] + 50.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = pyrenyi.measure(pyrenyi.Catalogue(x, weights=s['u']), pyrenyi.Catalogue(r, weights=s['rw']),
                              estimator='volume', radii=s['radii'], q=s['q'], r_min=[7.0], wedges=3,
                              correction=correction, min_randoms=0).Y
    _close(got, renyi_reference(x, s['u'], s['radii'], s['q'], randoms=r, rw=s['rw'], r_min=[7.0],
                                wedges=3, los='midpoint', estimator='volume', correction=correction))


def test_constant_marks_are_unit_weights(small):
    s = small
    a = pyrenyi.measure(s['x'], box=s['L'], radii=s['radii'], q=s['q']).Y
    b = pyrenyi.measure(pyrenyi.Catalogue(s['x'], marks=np.full(len(s['x']), 7.3)), box=s['L'],
                        radii=s['radii'], q=s['q']).Y
    _close(a, b)
    c = pyrenyi.measure(pyrenyi.Catalogue(s['x'], marks=s['u']), box=s['L'], radii=s['radii'], q=s['q']).Y
    d = pyrenyi.measure(pyrenyi.Catalogue(s['x'], marks=3 * s['u']), box=s['L'], radii=s['radii'], q=s['q']).Y
    _close(c, d)


def test_jackknife_equals_dropped_centres(small):
    s = small
    lab = pyrenyi.regions.subboxes(s['x'], s['L'], 2)
    res = pyrenyi.measure(s['x'], box=s['L'], radii=s['radii'], q=s['q'], wedges=2, jackknife=lab)
    for k in range(8):
        sub = pyrenyi.measure(s['x'], box=s['L'], radii=s['radii'], q=s['q'], wedges=2, centres=lab != k)
        _close(res.Y_jk[k], sub.Y)


# ------------------------------------------------------- Poisson processes

def test_poisson_box_is_three_dimensional():
    x = pyrenyi.mocks.poisson(100000, 1000.0, seed=2)
    res = pyrenyi.measure(x, radii=RADII, q=pyrenyi.DEFAULT_Q, box=1000.0, jackknife=27)
    D, r = res.dq()
    E = res.errors()
    lam = 1e-4 * 4 / 3 * np.pi * r ** 3
    q = res.q
    # every q once spheres hold > 40 neighbours
    assert np.all(np.abs(D[:, lam > 40] - 3) < 0.012)
    # integer q >= 2: exact factorial moments, unbiased at any occupancy
    for qq in (2.0, 3.0, 4.0):
        i = int(np.flatnonzero(q == qq)[0])
        k = lam > 1
        assert np.all(np.abs(D[i, k] - 3) < 4 * E[i, k] + 0.005), qq


def test_volume_estimator_box():
    x = pyrenyi.mocks.poisson(100000, 1000.0, seed=3)
    with pytest.warns(UserWarning):
        res = pyrenyi.measure(x, radii=RADII, q=[-1.0, 0.0, 1.0, 2.0, 3.0], box=1000.0,
                              estimator='volume', jackknife=27, seed=1)
    D, r = res.dq()
    assert np.all(np.isnan(D[0]))
    lam = 1e-4 * 4 / 3 * np.pi * r ** 3
    assert np.all(np.abs(D[1:, lam > 40] - 3) < 0.01)


def test_fixed_mass_poisson_every_q():
    """The fixed-mass estimator has no discreteness bias: D_q = 3 for every q
    and every mass, including q < 0 at k = 1."""
    x = pyrenyi.mocks.poisson(100000, 1000.0, seed=4)
    q = np.array([-4.0, -2.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    res = pyrenyi.fixed_mass(x, k=[1, 2, 4, 8, 16, 32, 64], q=q, box=1000.0, jackknife=27)
    D, _ = res.dq()
    E = res.errors()
    ok = np.isfinite(D)
    assert ok[:, 2:].all()
    assert np.all(np.abs(D[ok] - 3) < 4 * E[ok] + 0.01)


def test_inverse_completeness_weights_box():
    """Thinning by a completeness c(x) and weighting by 1/c leaves the
    integer-q moments unbiased (the weighted correction is exact)."""
    g = np.random.default_rng(5)
    x = pyrenyi.mocks.poisson(200000, 1000.0, seed=5)
    c = 0.2 + 0.8 * x[:, 0] / 1000.0
    keep = g.random(len(x)) < c
    cat = pyrenyi.Catalogue(x[keep], weights=1.0 / c[keep])
    res = pyrenyi.measure(cat, radii=RADII[3:], q=[2.0, 3.0], box=1000.0, jackknife=27)
    D, _ = res.dq()
    E = res.errors()
    assert np.all(np.abs(D - 3) < 4 * E + 0.005)


# ------------------------------------------------------ cascades in a box

@pytest.fixture(scope='module')
def cascade_box():
    return pyrenyi.mocks.random_cascade(300000, W8, 10, box=1000.0, seed=4)


def test_random_cascade_fixed_radius(cascade_box):
    q = np.array([-2.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    res = pyrenyi.measure(cascade_box, radii=RADII, q=q, box=1000.0)
    D, r = res.dq()
    th = pyrenyi.mocks.cascade_dq(W8, q)
    assert np.all(np.abs(D[:, r > 30] - th[:, None]) < 0.02)


def test_random_cascade_fixed_mass(cascade_box):
    q = np.array([-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    res = pyrenyi.fixed_mass(cascade_box, k=[4, 8, 16, 32, 64], q=q, box=1000.0, jackknife=27)
    D, r = res.dq()
    E = res.errors()
    th = pyrenyi.mocks.cascade_dq(W8, q)
    assert np.all(np.abs(D - th[:, None]) < 3 * E + 0.01)
    assert np.all(np.abs(D[:, 1:] - th[:, None]) < 0.02)


def test_cascade_theory():
    assert np.allclose(pyrenyi.mocks.cascade_dq([0.25] * 4 + [0] * 4, [-2.0, 0.0, 1.0, 3.0]), 2.0)
    assert np.allclose(pyrenyi.mocks.cascade_dq([0.125] * 8, [-2.0, 1.0, 4.0]), 3.0)
    a = pyrenyi.mocks.random_cascade(1000, W8, 6, seed=3)
    b = pyrenyi.mocks.random_cascade(1000, W8, 6, seed=3)
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() < 1


# ------------------------------------------------------ survey geometry

@pytest.fixture(scope='module')
def survey():
    """Poisson field in an octant of radius 700 about the observer at the
    origin, with a radial selection, and 20 times as many randoms."""
    g = np.random.default_rng(9)
    R, nbar0 = 700.0, 4e-4
    sel = lambda r: np.exp(-(r / 550.0) ** 2)   # noqa: E731

    def sample(nb):
        n = g.poisson(nb * R ** 3)
        p = g.uniform(0, R, (n, 3))
        rr = np.linalg.norm(p, axis=1)
        return p[(rr < R) & (g.random(n) < sel(rr))]

    d, r = sample(nbar0), sample(20 * nbar0)
    return pyrenyi.Catalogue(d), pyrenyi.Catalogue(r, nbar=nbar0 * sel(np.linalg.norm(r, axis=1)))


def test_survey_fixed_radius_poisson(survey):
    data, rand = survey
    res = pyrenyi.measure(data, rand, radii=RADII[:12], q=[2.0, 3.0], r_min=10.0, edge_cut=0.8,
                          jackknife=27)
    for rm in (None, 10.0):
        D, r = res.dq(r_min=rm)
        E = res.errors(r_min=rm)
        ok = np.isfinite(D)
        assert np.all(np.abs(D[ok] - 3) < 3.5 * E[ok] + 0.005)


def test_survey_fixed_mass_poisson(survey):
    data, rand = survey
    q = np.array([-2.0, 0.0, 1.0, 2.0, 3.0])
    res = pyrenyi.fixed_mass(data, rand, k=[2, 4, 8, 16, 32], q=q, edge_cut=0.8, jackknife=27)
    D, _ = res.dq()
    E = res.errors()
    ok = np.isfinite(D)
    assert np.all(np.abs(D[ok] - 3) < 3.5 * E[ok] + 0.01)


def test_survey_fixed_mass_completeness(survey):
    """Data thinned by an angular completeness, randoms weighted by it: the
    fixed-mass expectation (including the reference sphere at small k, which
    must use the local random weight) traces the data points."""
    data, rand = survey
    comp = lambda p: 0.5 + 0.45 * np.cos(4 * np.arctan2(p[:, 1], p[:, 0])) ** 2   # noqa: E731
    g = np.random.default_rng(3)
    x = data.positions
    thin = pyrenyi.Catalogue(x[g.random(len(x)) < comp(x)])
    pr = rand.positions
    rw = pyrenyi.Catalogue(pr, weights=comp(pr), nbar=rand.nbar * comp(pr))
    q = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    res = pyrenyi.fixed_mass(thin, rw, k=[1, 2, 4, 8, 16, 32], q=q, jackknife=27)
    D, _ = res.dq()
    E = res.errors()
    ok = np.isfinite(D)
    assert np.all(np.abs(D[ok] - 3) < 3.5 * E[ok] + 0.01)


def test_survey_without_box_needs_randoms(survey):
    data, _ = survey
    with pytest.raises(ValueError, match='randoms are required'):
        pyrenyi.measure(data, radii=RADII)


# ------------------------------------------------------------- interface

def test_input_errors(small):
    s = small
    x, L, radii = s['x'], s['L'], s['radii']
    with pytest.raises(ValueError, match='not one of the radii'):
        pyrenyi.measure(x, box=L, radii=radii, r_min=8.0)
    with pytest.raises(ValueError, match=r'\[0, box\)'):
        pyrenyi.measure(x + 60, box=L, radii=radii)
    with pytest.raises(ValueError, match='half the box'):
        pyrenyi.measure(x, box=L, radii=[10.0, 60.0])
    with pytest.raises(ValueError, match='nbar'):
        pyrenyi.measure(x + 50, s['r'] + 50, radii=radii, edge_cut=0.8)
    with pytest.raises(ValueError, match='observer'):
        pyrenyi.measure(x, box=L, radii=radii, wedges=3, los='midpoint')
    with pytest.raises(ValueError, match='edges'):
        pyrenyi.measure(x, box=L, radii=radii, wedges=[0.0, 0.7, 0.5, 1.0])
    with pytest.raises(ValueError, match='cube number'):
        pyrenyi.measure(x, box=L, radii=radii, jackknife=10)
    with pytest.raises(ValueError, match='shape'):
        pyrenyi.Catalogue(np.zeros((5, 2)))
    with pytest.raises(ValueError, match='non-negative'):
        pyrenyi.Catalogue(np.zeros((5, 3)), weights=-np.ones(5))
    with pytest.raises(ValueError, match='increasing'):
        pyrenyi.fixed_mass(x, k=[4, 2], box=L)


def test_save_load(small, tmp_path):
    s = small
    res = pyrenyi.measure(s['x'], box=s['L'], radii=s['radii'], q=s['q'], r_min=7.0, wedges=2,
                          jackknife=8)
    res.save(tmp_path / 'a.npz')
    back = pyrenyi.load(tmp_path / 'a.npz')
    assert isinstance(back, pyrenyi.RenyiResult)
    assert np.array_equal(back.Y, res.Y, equal_nan=True)
    assert np.array_equal(back.Y_jk, res.Y_jk, equal_nan=True)
    assert back.meta == res.meta
    D1, r1 = res.dq(r_min=7.0, wedge=1, method='central')
    D2, r2 = back.dq(r_min=7.0, wedge=1, method='central')
    assert np.array_equal(D1, D2, equal_nan=True) and np.array_equal(r1, r2)
    fm = pyrenyi.fixed_mass(s['x'], k=[2, 4, 8], q=[0.0, 2.0], box=s['L'], jackknife=8)
    fm.save(tmp_path / 'b.npz')
    fb = pyrenyi.load(tmp_path / 'b.npz')
    assert isinstance(fb, pyrenyi.FixedMassResult)
    assert np.array_equal(fb.D, fm.D, equal_nan=True)
    assert np.array_equal(fb.dq(pairs=[(2, 8)])[0], fm.dq(pairs=[(2, 8)])[0], equal_nan=True)


def test_result_shapes(small):
    s = small
    res = pyrenyi.measure(s['x'], box=s['L'], radii=s['radii'], q=s['q'], r_min=[7.0, 10.0],
                          wedges=3, jackknife=8)
    nq, nr = len(s['q']), len(s['radii'])
    assert res.Y.shape == (3, nq, nr, 4)
    assert res.Y_jk.shape == (8, 3, nq, nr, 4)
    assert res.dq()[0].shape == (nq, nr - 1)
    assert res.dq(method='central')[0].shape == (nq, nr - 2)
    assert res.covariance(r_min=10.0, wedge=0).shape == (nq * (nr - 1),) * 2
    assert np.all(np.isnan(res.Yq(r_min=10.0)[:, :3]))
    with pytest.raises(ValueError):
        res.Yq(r_min=5.0)


def test_regions():
    g = np.random.default_rng(0)
    x = g.uniform(0, 100, (5000, 3))
    lab = pyrenyi.regions.subboxes(x, 100.0, 3)
    assert set(np.unique(lab)) == set(range(27))
    c = pyrenyi.regions.kmeans(x, 20, seed=1)
    lab = pyrenyi.regions.assign(x, c)
    n = np.bincount(lab, minlength=20)
    assert len(np.unique(lab)) == 20 and n.min() > 0.4 * n.mean()
