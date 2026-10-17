"""Command-line interface: pyrenyi run / template / show."""
import argparse
import difflib
import os
import sys
import time

import numpy as np

from . import __version__
from .io import SPEC_KEYS, read_catalogue

__all__ = ['main', 'load_config', 'run_config', 'TEMPLATE_BOX', 'TEMPLATE_SURVEY']

GEOMETRY_KEYS = {'box', 'observer'}
MEASUREMENT_KEYS = {'estimator', 'radii', 'q', 'r_min', 'wedges', 'los', 'correction', 'edge_cut',
                    'random_correction', 'min_randoms', 'jackknife', 'n_centres', 'seed', 'threads',
                    'ref_volume', 'k'}
FIXED_RADIUS_ONLY = {'radii', 'wedges', 'los', 'correction', 'ref_volume'}
FIXED_MASS_ONLY = {'k'}
OUTPUT_KEYS = {'file'}
SECTIONS = {'data': SPEC_KEYS, 'randoms': SPEC_KEYS, 'geometry': GEOMETRY_KEYS,
            'measurement': MEASUREMENT_KEYS, 'output': OUTPUT_KEYS}
ESTIMATORS = ('point', 'volume', 'fixed-mass')

TEMPLATE_BOX = '''\
# PyRenyi configuration: run with  pyrenyi run this_file.toml
# Paths are relative to this file. Positions are Cartesian.

[data]
file = "points.npy"            # .txt/.dat/.csv (ascii), .npy, .npz, .fits, .hdf5/.h5
columns = [0, 1, 2]            # x, y, z: integer indices (ascii, npy) or names (npz, fits, hdf5)
# weight = 3                   # a column, or a constant written as a float (1.0)
# mark = 4                     # marks define the measure (e.g. halo mass)

[geometry]
box = 1000.0                   # periodic box side, or [Lx, Ly, Lz]; positions in [0, L)

[measurement]
estimator = "point"            # "point" (centres on the points), "volume" or "fixed-mass"
radii = {min = 5.0, max = 127.0, n = 15}   # log-spaced; or an explicit list
q = {min = -4.0, max = 6.0, step = 0.5}     # or an explicit list
r_min = [10.0]                 # minimum separations (inserted into a generated radii grid)
wedges = 3                     # equal-width |mu| wedges (omit for isotropic only)
los = "z"                      # line of sight in a box: "x", "y", "z" or [lx, ly, lz]
correction = "grassberger"     # "grassberger", "effective" or "none"
jackknife = 27                 # sub-boxes (a cube number)
seed = 0
# threads = 16
# k = {min = 1, max = 128, n = 8}   # fixed-mass: neighbour numbers (log-spaced integers)

[output]
file = "pyrenyi_result.npz"
'''

TEMPLATE_SURVEY = '''\
# PyRenyi configuration for a survey: run with  pyrenyi run this_file.toml
# Paths are relative to this file. Positions are comoving Cartesian
# coordinates (convert RA, Dec, z yourself); the observer sets the line of sight.

[data]
file = "galaxies.fits"
columns = ["X", "Y", "Z"]
weight = "WEIGHT"              # a column, or a constant written as a float (1.0)
# hdu = 1

[randoms]
file = "randoms.fits"
columns = ["X", "Y", "Z"]
weight = "WEIGHT"
nbar = "NZ"                    # data density at each random: needed by edge_cut

[geometry]
observer = [0.0, 0.0, 0.0]

[measurement]
estimator = "point"            # "point", "volume" or "fixed-mass"
radii = {min = 5.0, max = 127.0, n = 15}
q = {min = -4.0, max = 6.0, step = 0.5}
r_min = [10.0]
wedges = 3
los = "midpoint"               # "midpoint", "centre" or a fixed direction
edge_cut = 0.8                 # drop centres with < 80% of the sphere inside the survey
random_correction = true       # Poisson-unbiased normalisation by the random counts
min_randoms = 30
jackknife = 64                 # k-means regions of the centres
seed = 0
# threads = 16
# k = {min = 1, max = 128, n = 8}   # fixed-mass: neighbour numbers

[output]
file = "pyrenyi_survey.npz"
'''


def _toml():
    try:
        import tomllib
        return tomllib
    except ImportError:
        try:
            import tomli
            return tomli
        except ImportError:
            raise ImportError('reading the configuration needs tomli on Python < 3.11: '
                              'pip install tomli') from None


def _check_keys(where, given, allowed):
    for key in given:
        if key not in allowed:
            close = difflib.get_close_matches(key, sorted(allowed), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ''
            raise ValueError(f"unknown key '{key}' in {where}{hint}; allowed: {sorted(allowed)}")


def load_config(path):
    """Read and check a TOML configuration; returns (config dict, its directory)."""
    with open(path, 'rb') as f:
        cfg = _toml().load(f)
    validate_config(cfg)
    return cfg, os.path.dirname(os.path.abspath(path))


def validate_config(cfg):
    """Raise ValueError for unknown sections or keys (typos) and missing ones."""
    _check_keys('the configuration', cfg, SECTIONS)
    for sec, keys in SECTIONS.items():
        if sec in cfg:
            if not isinstance(cfg[sec], dict):
                raise ValueError(f'[{sec}] must be a table')
            _check_keys(f'[{sec}]', cfg[sec], keys)
    if 'data' not in cfg:
        raise ValueError('the configuration needs a [data] section')
    m = cfg.get('measurement', {})
    est = m.get('estimator', 'point')
    if est not in ESTIMATORS:
        raise ValueError(f'estimator must be one of {ESTIMATORS}, got {est!r}')
    wrong = (FIXED_MASS_ONLY if est != 'fixed-mass' else FIXED_RADIUS_ONLY) & set(m)
    if wrong:
        raise ValueError(f'key(s) {sorted(wrong)} do not apply to estimator {est!r}')
    if est == 'fixed-mass' and 'k' not in m:
        raise ValueError("the fixed-mass estimator needs k (neighbour numbers)")
    if est != 'fixed-mass' and 'radii' not in m:
        raise ValueError("the fixed-radius estimators need radii")


def _grid_radii(spec, rmins):
    if isinstance(spec, dict):
        _check_keys('radii', spec, {'min', 'max', 'n', 'spacing'})
        lo, hi, n = float(spec['min']), float(spec['max']), int(spec['n'])
        sp = spec.get('spacing', 'log')
        if sp not in ('log', 'linear'):
            raise ValueError("radii spacing must be 'log' or 'linear'")
        r = np.geomspace(lo, hi, n) if sp == 'log' else np.linspace(lo, hi, n)
        # minimum separations must be on the grid
        r = np.unique(np.concatenate([r, [x for x in rmins if not np.any(np.isclose(r, x))]]))
        return r
    return np.asarray(spec, dtype=np.float64)


def _grid_q(spec):
    if isinstance(spec, dict):
        _check_keys('q', spec, {'min', 'max', 'step'})
        lo, hi, st = float(spec['min']), float(spec['max']), float(spec['step'])
        return np.round(np.arange(lo, hi + 0.5 * st, st), 10)
    return np.asarray(spec, dtype=np.float64)


def _grid_k(spec):
    if isinstance(spec, dict):
        _check_keys('k', spec, {'min', 'max', 'n'})
        return np.unique(np.round(np.geomspace(int(spec['min']), int(spec['max']),
                                               int(spec['n'])))).astype(np.int64)
    return np.asarray(spec, dtype=np.int64)


def _arguments(cfg):
    """Keyword arguments of pyrenyi.measure / pyrenyi.fixed_mass from the config."""
    m = dict(cfg.get('measurement', {}))
    g = cfg.get('geometry', {})
    est = m.pop('estimator', 'point')
    kw = {}
    if 'box' in g:
        kw['box'] = g['box']
    if 'q' in m:
        kw['q'] = _grid_q(m.pop('q'))
    if 'threads' in m:
        kw['nthreads'] = int(m.pop('threads'))
    rmins = m.pop('r_min', None)
    rl = [] if rmins is None else list(np.atleast_1d(np.asarray(rmins, dtype=np.float64)))
    if est == 'fixed-mass':
        kw['k'] = _grid_k(m.pop('k'))
        if len(rl) > 1:
            raise ValueError('the fixed-mass estimator takes a single r_min')
        if rl:
            kw['r_min'] = rl[0]
    else:
        kw['estimator'] = est
        kw['radii'] = _grid_radii(m.pop('radii'), rl)
        if rl:
            kw['r_min'] = rl
        if 'observer' in g:
            kw['observer'] = g['observer']
        if 'los' in m and not isinstance(m['los'], str):
            m['los'] = np.asarray(m['los'], dtype=np.float64)
    kw.update(m)
    return est, kw


def run_config(cfg, base='.', out=None):
    """Run the measurement described by a config dict; returns (result, output path)."""
    import pyrenyi
    out = sys.stdout if out is None else out
    validate_config(cfg)
    est, kw = _arguments(cfg)
    t0 = time.time()
    data = read_catalogue(cfg['data'], base)
    randoms = read_catalogue(cfg['randoms'], base) if 'randoms' in cfg else None
    t1 = time.time()
    print(f'data: {len(data)} points' + (f', randoms: {len(randoms)}' if randoms is not None else '')
          + f' (read in {t1 - t0:.1f} s)', file=out)
    if est == 'fixed-mass':
        res = pyrenyi.fixed_mass(data, randoms, **kw)
    else:
        res = pyrenyi.measure(data, randoms, **kw)
    t2 = time.time()
    path = cfg.get('output', {}).get('file', 'pyrenyi_result.npz')
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    res.save(path)
    print(f'{res}: {t2 - t1:.1f} s; saved to {path}', file=out)
    print(table(res), file=out)
    return res, path


def table(res, r_min=None, wedge=None, errors=False, max_columns=8):
    """A text table of D_q: q down, radii (or neighbour-number pairs) across."""
    from .fixed_mass import FixedMassResult
    if isinstance(res, FixedMassResult):
        D, r = res.dq()
        E = res.errors() if errors else None
        heads = [f'{a}-{b}' for a, b in res.pairs]
        title = 'D_q, fixed mass: neighbour numbers k_a-k_b (typical radius below)'
    else:
        D, r = res.dq(r_min=r_min, wedge=wedge)
        E = res.errors(r_min=r_min, wedge=wedge) if errors else None
        heads = [f'{x:.3g}' for x in r]
        wl = 'isotropic' if wedge is None else f'wedge {wedge}'
        title = f'D_q (fixed radius, {wl}' + (f', r_min = {r_min}' if r_min else '') + '): r across'
    cols = np.flatnonzero(np.any(np.isfinite(D), axis=0))
    if max_columns and len(cols) > max_columns:
        cols = cols[np.unique(np.round(np.linspace(0, len(cols) - 1, max_columns)).astype(int))]
    w = 15 if E is not None else 8
    lines = [title, '      q' + ''.join(f'{heads[j]:>{w}s}' for j in cols)]
    if isinstance(res, FixedMassResult):
        lines.append('      r' + ''.join(f'{r[j]:>{w}.3g}' for j in cols))
    for i, q in enumerate(res.q):
        row = f'{q:7.2f}'
        for j in cols:
            v = D[i, j]
            if E is not None:
                row += f'{v:>8.4f}±{E[i, j]:<6.4f}' if np.isfinite(v) else f'{"nan":>15s}'
            else:
                row += f'{v:8.4f}' if np.isfinite(v) else f'{"nan":>8s}'
        lines.append(row)
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='pyrenyi',
                                 description='Renyi generalised dimensions D_q of point sets.')
    ap.add_argument('--version', action='version', version=f'pyrenyi {__version__}')
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('run', help='run the measurement described by a TOML configuration')
    p.add_argument('config')
    p = sub.add_parser('template', help='print an example configuration')
    p.add_argument('--survey', action='store_true', help='survey version (randoms, weights, edge cut)')
    p = sub.add_parser('show', help='print D_q from a saved result')
    p.add_argument('result')
    p.add_argument('--r-min', type=float, default=None, help='minimum separation (fixed radius)')
    p.add_argument('--wedge', type=int, default=None, help='wedge index (fixed radius)')
    p.add_argument('--errors', action='store_true', help='jackknife errors')
    p.add_argument('--all', action='store_true', help='all radii / pairs')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'run':
            cfg, base = load_config(a.config)
            run_config(cfg, base)
        elif a.cmd == 'template':
            sys.stdout.write(TEMPLATE_SURVEY if a.survey else TEMPLATE_BOX)
        elif a.cmd == 'show':
            import pyrenyi
            from .fixed_mass import FixedMassResult
            res = pyrenyi.load(a.result)
            if isinstance(res, FixedMassResult) and (a.r_min is not None or a.wedge is not None):
                raise ValueError('--r-min and --wedge apply to fixed-radius results')
            if a.errors and (getattr(res, 'Y_jk', None) is None if not isinstance(res, FixedMassResult)
                             else res.n_regions < 2):
                raise ValueError('the result has no jackknife regions')
            print(res)
            print(table(res, a.r_min, a.wedge, a.errors, max_columns=0 if a.all else 8))
        else:
            ap.print_help()
            return 1
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f'pyrenyi: error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
