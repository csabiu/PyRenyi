"""Generalised dimensions of a random multiplicative cascade in a periodic box,
with the fixed-radius and fixed-mass estimators, against the exact D_q.

    python examples/periodic_box.py          (~30 s, plus numba compilation)
"""
import numpy as np

import pyrenyi

L = 1000.0
w = [0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09]       # octant weights
x = pyrenyi.mocks.random_cascade(300_000, w, levels=10, box=L, seed=1)
q = np.array([-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
exact = pyrenyi.mocks.cascade_dq(w, q)

# fixed radius: spheres, and shells beyond r_min = 10, with 27 jackknife sub-boxes
radii = 5.0 * 2.0 ** (np.arange(15) / 3.0)                # 5 to 127, three per octave
fr = pyrenyi.measure(x, radii=radii, q=q, box=L, r_min=10.0, jackknife=27)
D, r = fr.dq()
E = fr.errors()

# fixed mass: the radii holding k = 1, 2, 4, ... neighbours
fm = pyrenyi.fixed_mass(x, k=2 ** np.arange(8), q=q, box=L, jackknife=27)
Dm, rm = fm.dq()
Em = fm.errors()

j = int(np.argmin(np.abs(r - 40)))
jm = int(np.argmin(np.abs(rm - 15)))
print(f'{"q":>5s} {"exact":>7s}   fixed radius (r = {r[j]:.0f})   fixed mass (r = {rm[jm]:.0f})')
for i, qq in enumerate(q):
    print(f'{qq:5.1f} {exact[i]:7.3f}   {D[i, j]:7.3f} +- {E[i, j]:.3f}          '
          f'{Dm[i, jm]:7.3f} +- {Em[i, jm]:.3f}')
print('The fixed-radius estimator is biased at q < 1 where the spheres hold few points;')
print('the fixed-mass estimator is not:')
for i, qq in enumerate(q[:4]):
    print(f'  q = {qq:4.1f}: fixed radius at r = {r[2]:.1f}: {D[i, 2]:.3f}; '
          f'fixed mass at r = {rm[0]:.1f}: {Dm[i, 0]:.3f}; exact {exact[i]:.3f}')

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for i in (0, 3, 6, 8):
    ax[0].errorbar(r, D[i], E[i], label=f'q = {q[i]:g}', capsize=2)
    ax[0].axhline(exact[i], color='k', lw=0.5, ls=':')
    ax[1].errorbar(rm, Dm[i], Em[i], label=f'q = {q[i]:g}', capsize=2)
    ax[1].axhline(exact[i], color='k', lw=0.5, ls=':')
for a, t in zip(ax, ('fixed radius', 'fixed mass')):
    a.set_xscale('log')
    a.set_xlabel('r')
    a.set_title(t)
ax[0].set_ylabel('$D_q$')
ax[0].set_ylim(2.6, 3.4)
ax[1].legend()
fig.tight_layout()
fig.savefig('periodic_box.png', dpi=120)
print('saved periodic_box.png')
