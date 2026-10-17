# Changelog

## 0.1.0 (unreleased)

First release.

* Fixed-radius estimators of the Renyi generalised dimensions: point-centred
  and volume-weighted, in spheres, in shells beyond a minimum separation, and
  in |mu| wedges about a fixed or observer-based line of sight.
* Periodic boxes (analytic expectation) and surveys (expectation from a
  weighted random catalogue), with weights and marks, an edge cut, and
  delete-one jackknife errors.
* Finite-sample corrections: Grassberger's for unit weights, and its weighted
  generalisation (elementary symmetric polynomials), both exact for integer
  q; Poisson-unbiased normalisation by the random counts.
* Fixed-mass (nearest-neighbour) estimator, free of discreteness bias at
  every q.
* File readers (ASCII, npy, npz, FITS, HDF5) and a command-line interface
  driven by TOML configurations.
* Test point sets: Poisson, deterministic and random multiplicative cascades
  with exact D_q.
