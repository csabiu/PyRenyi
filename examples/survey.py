"""A mock survey: a random multiplicative cascade seen by an observer at the
origin, within 900 of it in one octant, with a radial selection and an
angular completeness. The randoms follow the same selection and completeness
and carry the data density nbar (for the edge cut). D_q is measured with the
fixed-radius estimator (isotropic and in three mu-wedges about the pair line
of sight) and the fixed-mass estimator, and compared with the exact values.

    python examples/survey.py                (~1 min, plus numba compilation)
"""
import numpy as np

import pyrenyi

L, RMAX = 1000.0, 900.0
w = [0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09]
rng = np.random.default_rng(2)
selection = lambda r: np.exp(-(r / 700.0) ** 2)                            # noqa: E731
completeness = lambda p: 0.5 + 0.45 * np.cos(4 * np.arctan2(p[:, 1], p[:, 0])) ** 2  # noqa: E731

# the "universe": a periodic cascade; the survey: an octant about the observer
x = pyrenyi.mocks.random_cascade(600_000, w, levels=10, box=L, seed=4)
r = np.linalg.norm(x, axis=1)
x = x[(r < RMAX) & (rng.random(len(x)) < selection(r))]
data = pyrenyi.Catalogue(x[rng.random(len(x)) < completeness(x)])

# randoms: 20 times the density, with the same selection and completeness
nbar0 = 600_000 / L ** 3
xr = rng.uniform(0, RMAX, (20 * int(nbar0 * RMAX ** 3), 3))
rr = np.linalg.norm(xr, axis=1)
xr = xr[(rr < RMAX) & (rng.random(len(xr)) < selection(rr) * completeness(xr))]
nbar = nbar0 * selection(np.linalg.norm(xr, axis=1)) * completeness(xr)
randoms = pyrenyi.Catalogue(xr, nbar=nbar)
print(data, randoms)

q = np.array([-2.0, 0.0, 1.0, 2.0, 3.0, 4.0])
exact = pyrenyi.mocks.cascade_dq(w, q)
radii = 5.0 * 2.0 ** (np.arange(14) / 3.0)
res = pyrenyi.measure(data, randoms, radii=radii, q=q, r_min=10.0, wedges=3, los='midpoint',
                      observer=(0.0, 0.0, 0.0), edge_cut=0.8, jackknife=32)
D, rad = res.dq()
E = res.errors()
fm = pyrenyi.fixed_mass(data, randoms, k=[1, 2, 4, 8, 16, 32, 64], q=q, edge_cut=0.8, jackknife=32)
Dm, rm = fm.dq()
Em = fm.errors()

print(f'\nfixed radius, isotropic, r = {rad[7]:.0f} and {rad[10]:.0f}; fixed mass at r = {rm[3]:.0f}')
for i, qq in enumerate(q):
    print(f'q = {qq:4.1f}  exact {exact[i]:.3f}   {D[i, 7]:.3f} +- {E[i, 7]:.3f}   '
          f'{D[i, 10]:.3f} +- {E[i, 10]:.3f}   {Dm[i, 3]:.3f} +- {Em[i, 3]:.3f}')
# the cascade is isotropic, so the wedges agree
Dw = [res.dq(r_min=10.0, wedge=m)[0][3, 10] for m in range(3)]
print('D_2 in the three wedges beyond r_min = 10 at r = %.0f:' % rad[10], np.round(Dw, 3))
