# PyRenyi

Rényi generalised dimensions $D_q$ of point distributions, in periodic
simulation boxes and in surveys with random catalogues and weights.

The generalised dimensions describe how the moments of the mass around each
point scale with radius. With $p_i(r)$ the fraction of the points within $r$
of point $i$,

$$Z_q(r) = \langle p_i(r)^{q-1} \rangle_i, \qquad
Y_q(r) = \frac{\ln Z_q(r)}{q-1}, \qquad
D_q(r) = \frac{d Y_q}{d \ln r}.$$

Orders $q > 1$ weight the dense regions and $q < 1$ the empty ones; a
homogeneous distribution has $D_q = 3$ at every $q$. PyRenyi provides
estimators that are free of the discreteness bias of the naive moments where
that is possible, and that work in arbitrary survey geometries.

* **Fixed-radius estimators**: point-centred (the centres sample the measure)
  and volume-weighted (the centres fill the volume), in spheres, in shells
  beyond a minimum separation, and in $|\mu|$ wedges about the line of sight.
* **Fixed-mass estimator**: from the radii that hold $k$ neighbours; it has no
  discreteness bias at any $q$, including $q < 1$.
* **Periodic boxes** with analytic expectations, and **surveys** with a weighted
  random catalogue, an edge cut and a line of sight from an observer.
* **Weights and marks**, with a finite-sample correction that stays exact for
  integer $q$.
* **Jackknife errors and covariances**, file readers (ASCII, npy, npz, FITS,
  HDF5), a command-line interface, and test point sets with exact $D_q$.

All inputs are Cartesian positions: convert sky coordinates and redshifts to
comoving positions first.

## Installation

```bash
pip install pyrenyi                  # numpy, scipy, numba
pip install "pyrenyi[fits,hdf5]"     # + astropy and h5py for the file readers
```

From a clone of this repository:

```bash
pip install -e ".[test]"
pytest
```

The numerical kernels are compiled by numba on first use (about a minute) and
cached.

## Quick start

### A periodic box

```python
import numpy as np
import pyrenyi

L = 1000.0
w = [0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09]
x = pyrenyi.mocks.random_cascade(300_000, w, levels=10, box=L, seed=1)   # exact D_q known

radii = 5.0 * 2.0 ** (np.arange(15) / 3.0)          # 5 to 127, three per octave
q = np.arange(-4.0, 6.01, 0.5)
res = pyrenyi.measure(x, radii=radii, q=q, box=L, r_min=10.0, wedges=3, los='z',
                      jackknife=27)

D, r = res.dq()                          # isotropic D_q(r), shape (n_q, n_r - 1)
Dw, r = res.dq(r_min=10.0, wedge=2)      # beyond 10, in the wedge 2/3 <= |mu| <= 1
err = res.errors(r_min=10.0, wedge=2)    # jackknife errors
C = res.covariance(r_min=10.0)           # covariance of the flattened (q, r) array
print(pyrenyi.mocks.cascade_dq(w, q))    # the exact values
```

### A survey

```python
data = pyrenyi.Catalogue(galaxy_xyz, weights=w_sys)                # Cartesian, with weights
randoms = pyrenyi.Catalogue(random_xyz, weights=w_ran, nbar=nz_ran)  # nbar: data density at each random

res = pyrenyi.measure(data, randoms, radii=radii, q=q, r_min=10.0, wedges=3,
                      los='midpoint', observer=(0.0, 0.0, 0.0),
                      edge_cut=0.8, jackknife=64)
```

The randoms define the survey geometry and the expected counts; `nbar` gives
the volume of each random, which the edge cut needs.

### Fixed mass

```python
fm = pyrenyi.fixed_mass(data, randoms, k=[1, 2, 4, 8, 16, 32, 64, 128], q=q,
                        edge_cut=0.8, jackknife=64)
D, r = fm.dq()                  # D_q between adjacent k, at their typical radius
D, r = fm.dq(pairs=[(4, 64)])   # or over any range of k
```

### Saving

```python
res.save('dq.npz')
res = pyrenyi.load('dq.npz')
```

`examples/cascade_dq.ipynb` is a notebook that samples a multiplicative cascade,
measures $D_q$ with both estimators and plots it against the exact curve (it
needs matplotlib). `examples/periodic_box.py` and `examples/survey.py` are
complete, runnable scripts. The survey example builds a mock survey with a radial selection and
an angular completeness, and recovers the exact $D_q$ of the underlying cascade
within the errors. The fixed-radius estimator does so at $q < 1$ only where the
spheres are well occupied.

## The estimators

### Point-centred, fixed radius

`measure(..., estimator='point')` (the default) centres a sphere of radius $r$
on every data point $i$ and finds the mass of its neighbours (the centre
excluded). The partition function is the centre-weighted mean

$$Z_q(r) = \frac{\sum_i c_i\, T_s(i, r)\, /\, \mu_i(r)^s}{\sum_i c_i},
\qquad s = q - 1,$$

where $T_s$ is the finite-sample-corrected $s$-th power of the neighbour mass,
$\mu_i$ its expectation for a homogeneous field, and $c_i$ the weight times
mark of the centre. Then $Y_q = \ln Z_q / (q-1) + \ln[f v(r) / V]$, with $v(r)$
the sphere volume, $f$ the solid-angle fraction of the wedge and $V$ a reference
volume that cancels in $D_q$; for $q = 1$, $Y_1 = \langle \psi$-corrected
$\ln(\text{mass}) - \ln \mu_i \rangle + \ln[f v / V]$. In a periodic box with unit
weights this is exactly the estimator of Sabiu (2026).

### Finite-sample corrections

The moments of a discrete sample are biased: for a Poisson count $n$ with mean
$\lambda$, $\langle n^s \rangle \neq \lambda^s$. With `correction='grassberger'`
(the default):

* unit weights: $T_s = \Gamma(n+1) / \Gamma(n+1-s)$ (Grassberger 1988, 2003),
  whose Poisson mean is exactly $\lambda^s$ for integer $s \ge 0$;
* weights: $T_s = s!\, e_s(u_1, \dots, u_n)$, the elementary symmetric
  polynomial of the neighbour masses (the sum over distinct $s$-tuples). For
  Poisson sampling of any field its mean is exactly the $s$-th power of the
  expected mass, so the correction stays exact for integer $q \ge 2$ with
  weights;
* non-integer $s$ with weights: the Gamma-function form with the effective
  count $n_\text{eff} = (\sum u)^2 / \sum u^2$, scaled by $\sum u^2 / \sum u$.

`'effective'` uses the last form for every $q$, and `'none'` the plain powers with
the centre included (the sandbox estimator of Sabiu, Teodoro & Hendry 2007).

For $q < 1$ the correction is not exact: at mean counts of a few, the
point-centred moments of the emptiest spheres are biased low (for a Poisson
field, $\langle T_s \rangle = \lambda^s P(|s|, \lambda)$). Use radii where the spheres
hold tens of points, or the fixed-mass estimator.

### Expected counts: boxes and randoms

In a periodic box $\mu_i = (U - u_i)\, v_\text{region}/V$ is analytic ($U$ is the
total mass). In a survey it comes from the randoms,
$\mu_i = (U - u_i) / W_R \times$ (random weight in the region). Three refinements
keep the finite random catalogue from biasing the result:

* `random_correction=True`: $\mu^{-s}$ is replaced by its Poisson-unbiased estimate
  from the random count $m$, $\alpha^{-s}\,\Gamma(m+1)/\Gamma(m+1+s)$; this is exact for
  integer $s < 0$ and accurate to terms of order $e^{-m}$ for $s > 0$.
* `min_randoms=30`: a sphere, shell or wedge with fewer randoms takes its
  expectation from the smallest sphere about the same centre with at least 30,
  scaled by volume. This removes the random noise at small scales.
* `edge_cut=f`: at each radius, centres with less than a fraction $f$ of the
  sphere inside the survey are dropped. The footprint fraction is estimated from
  the odd-numbered randoms and the expectation from the even-numbered ones, so
  the cut does not select on the noise of the expectation. Without an edge cut,
  spheres that cross the survey boundary measure a truncated window, which
  biases $D_q$ at the 0.01–0.02 level at scales where most spheres are cut.

### Minimum separation

`r_min` counts neighbours only in the shell $r_\text{min} < d \le r$, which
removes the smallest separations (e.g. one-halo pairs, or scales affected by
redshift errors or fibre collisions) from every moment; each $r_\text{min}$ must
be one of the radii. Only the centre–neighbour separation is limited: for
$q \ge 3$ the moments still involve neighbour pairs closer than $r_\text{min}$.

### Wedges and the line of sight

`wedges=n` splits each sphere into $n$ equal-width bins of $|\mu|$, the cosine of
the angle between the separation and the line of sight (or `wedges=[0, 0.5, 0.8, 1]`
for custom edges); the isotropic statistic is always returned as well. The line
of sight is a fixed axis in a box (`los='z'` or any vector), and in a survey the
direction from the `observer` to the pair midpoint (`los='midpoint'`, the default)
or to the centre (`los='centre'`).

### Volume-weighted

`estimator='volume'` places the centres uniformly in the volume (in a box) or on
the randoms (in a survey, weighted by their volume element if `nbar` is given),
and averages $T_q / \mu^q$ over the occupied spheres: the box-counting analogue,
defined for $q \ge 0$.

### Fixed mass

`fixed_mass` uses, for each centre and number $k$, the distance $r_{ik}$ of the
$k$-th neighbour (beyond `r_min`) and the expected number of neighbours
$\nu_{ik}$ within it for a homogeneous field. For Poisson sampling of any field,
$\nu_{ik}$ follows a Gamma($k$) distribution, whatever the geometry and selection.
For a measure with generalised dimensions $D_q$,
$\langle \nu_{ik}^{-\tau/3} \rangle \propto \Gamma(k+1-q)/\Gamma(k)$ with $\tau = (q-1) D_q$
(Grassberger 1985; Badii & Politi 1985; van de Water & Schram 1988), so two
values of $k$ determine $q(\tau)$ and $D_q = \tau / (q-1)$. A Poisson process gives
$D_q = 3$ at every $q$ and $k$, so the estimator has no discreteness bias, in
particular for $q < 1$; it needs $q < k + 1$. In a survey, $\nu$ comes from the
randoms, with the same random-count correction; where fewer than `min_randoms`
randoms lie within $r_{ik}$, the random density is taken from the distance to the
`min_randoms`-th random, whose enclosed expected count is also Gamma-distributed.
The edge cut uses the footprint fraction at one radius per $k$ (the 90th
percentile of $r_{ik}$), so that it does not select on each centre's own
neighbours.

### Jackknife

`jackknife=n` defines $n$ regions of the centres (sub-boxes in a box, where $n$ must
be a cube number; k-means clusters otherwise), or takes one label per centre.
The delete-one estimates are computed in the same pass, from per-region sums;
the neighbours of the remaining centres include the deleted region.

## Weights, marks and randoms

* **weights** correct the data for selection (completeness, systematics) and
  the randoms carry the matching weights;
* **marks** (data only) define the measure, e.g. halo mass: each neighbour
  contributes weight × mark, and the centres are weighted by weight × mark;
* **nbar** (randoms) is the expected data density at each random, which gives
  its volume element for the edge cut and for the volume-weighted estimator.

For the fixed-radius estimators the expected mass is computed from the
weighted randoms, so either convention works:

1. unweighted data, and randoms that trace the observed density (including the
   completeness, by subsampling or by weights); or
2. data weighted by 1/completeness, and randoms that trace the density without
   the completeness.

The fixed-mass estimator counts neighbours, so it needs convention 1. There,
data weights and marks only weight the centres. A smooth mismatch between the
randoms and the data density multiplies each centre's expected count by a
factor independent of $k$, which cancels in $D_q$. A strong mismatch on the
scale of $r_{ik}$ does not cancel.

## Practical notes

* Integer $q \ge 2$ are exact at any occupancy (in expectation). The
  correction for non-integer $q$ is approximate: where the spheres hold a few
  points, non-integer orders are biased and alternate with the integer ones.
* $D_q$ is a scaling exponent of the measure, not the dimension of a set: for
  $q < 0$ it can exceed 3 where the density vanishes.
* At the smallest radii the moments of high $q$ rest on a few centres; check
  the jackknife errors.
* Use randoms with at least ~20 times the data density. The random-count
  correction and `min_randoms` handle the small scales.

## Command line

```bash
pyrenyi template > box.toml            # example configuration (add --survey for randoms)
pyrenyi run box.toml                   # measure, save, print a D_q table
pyrenyi show result.npz --errors       # D_q with jackknife errors
pyrenyi show result.npz --r-min 10 --wedge 2 --all
```

A configuration for a survey (paths are relative to the file):

```toml
[data]
file = "galaxies.fits"            # .txt/.csv, .npy, .npz, .fits, .hdf5
columns = ["X", "Y", "Z"]          # names, or integer indices (ascii, npy)
weight = "WEIGHT"                  # a column, or a constant as a float (1.0)

[randoms]
file = "randoms.fits"
columns = ["X", "Y", "Z"]
nbar = "NZ"                        # data density at each random (for edge_cut)

[geometry]
observer = [0.0, 0.0, 0.0]         # or: box = 1000.0 for a periodic box

[measurement]
estimator = "point"                # "point", "volume" or "fixed-mass"
radii = {min = 5.0, max = 127.0, n = 15}
q = {min = -4.0, max = 6.0, step = 0.5}
r_min = [10.0]                     # added to a generated radii grid
wedges = 3
los = "midpoint"
edge_cut = 0.8
jackknife = 64

[output]
file = "dq_survey.npz"
```

Unknown keys are reported with a suggestion, so typos cannot pass silently.
For the fixed-mass estimator, replace `radii`, `wedges` and `los` with
`k = {min = 1, max = 128, n = 8}`.

## Validation

The test suite (`pytest`, about two minutes) checks that:

* in a periodic box, the estimator reproduces that of Sabiu (2026) to $10^{-10}$
  (spheres, shells, wedges, sandbox);
* every option (box and survey; unit weights, marks and weights; weighted
  randoms; the three corrections; with and without the random-count
  correction; the three lines of sight; spheres, shells and wedges; both
  fixed-radius estimators) agrees to $10^{-10}$ with a brute-force
  implementation;
* Poisson processes give $D_q = 3$:
  * the fixed-radius estimators, for integer $q \ge 2$ at any occupancy and for
    every $q$ once spheres hold more than ~40 points;
  * the fixed-mass estimator, for every $q$ and $k$;
  * both, in a survey octant with a radial selection and an edge cut;
* random multiplicative cascades recover their exact $D_q$ in a box and in a
  survey;
* thinning by a completeness and weighting by its inverse leaves integer $q$
  unbiased.

## Performance

Indicative times on 64 threads after compilation:

| sample | grid | time |
|---|---|---|
| 600,000 points in a box | 14 radii, 6 orders | 4 s |
| 30,000 points in a box | 15 radii, 21 orders, 3 minimum separations, 3 wedges | 2 s |
| survey: 82,000 points, 1.9 million randoms | 14 radii, 6 orders, edge cut, jackknife | 15 s |
| fixed mass, 600,000 points in a box | 8 values of $k$ | 4 s |

The cost grows with the number of centres times the number of points (data and
randoms) within the largest radius. `n_centres` subsamples the centres.

## Citing

If you use PyRenyi, please cite

* C. G. Sabiu (2026), *Cosmological Information in the Rényi Generalised
  Dimensions of the Large-Scale Distribution of Haloes and Galaxies*, Journal
  of the Korean Astronomical Society, submitted;

and, for the methods,

* C. G. Sabiu, L. F. A. Teodoro & M. A. Hendry (2007), *Resolving the universe
  with multifractals*, Nuovo Cimento B 122, 1289 (arXiv:0803.3212), for the
  point-centred (sandbox) estimator;
* P. Grassberger (1988), Phys. Lett. A 128, 369 and (2003),
  arXiv:physics/0307138, for the finite-sample correction;
* W. van de Water & P. Schram (1988), Phys. Rev. A 37, 3118, for the
  fixed-mass estimator.

## License

MIT; see `LICENSE`.
