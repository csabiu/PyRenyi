"""Catalogue readers: ASCII, .npy, .npz, FITS and HDF5 tables."""
import os
import re

import numpy as np

from .catalogue import Catalogue

__all__ = ['read_catalogue', 'SPEC_KEYS', 'FORMATS']

FORMATS = {'.txt': 'ascii', '.dat': 'ascii', '.csv': 'ascii', '.asc': 'ascii', '.ascii': 'ascii',
           '.npy': 'npy', '.npz': 'npz', '.fits': 'fits', '.fit': 'fits', '.fits.gz': 'fits',
           '.fit.gz': 'fits', '.hdf5': 'hdf5', '.h5': 'hdf5', '.hdf': 'hdf5'}
SPEC_KEYS = {'file', 'format', 'columns', 'weight', 'mark', 'nbar', 'comments', 'delimiter',
             'skiprows', 'group', 'hdu'}
_INDEXED = re.compile(r'^(.+)\[(\d+)\]$')       # 'pos[0]': column 0 of a 2-d array


def _format(path, fmt):
    if fmt not in (None, 'auto'):
        if fmt not in set(FORMATS.values()):
            raise ValueError(f"format must be 'auto' or one of {sorted(set(FORMATS.values()))}, got {fmt!r}")
        return fmt
    low = path.lower()
    for ext in sorted(FORMATS, key=len, reverse=True):
        if low.endswith(ext):
            return FORMATS[ext]
    raise ValueError(f'cannot tell the format of {path!r} from its extension; set format = ...')


def _missing(name, available):
    return ValueError(f'column {name!r} not found; available: {", ".join(map(str, available))}')


class _Table:
    """Column access to one file: get(name or index) -> 1-d float array."""

    def __init__(self, path, fmt, spec):
        self.fmt = fmt
        self._close = None
        if fmt == 'ascii':
            arr = np.loadtxt(path, comments=spec.get('comments', '#'), delimiter=spec.get('delimiter'),
                             skiprows=int(spec.get('skiprows', 0)), ndmin=2)
            self._arr, self.available = arr, list(range(arr.shape[1]))
        elif fmt == 'npy':
            arr = np.load(path, allow_pickle=False)
            if arr.dtype.names:
                self._rec, self.available = arr, list(arr.dtype.names)
            else:
                arr = arr.reshape(len(arr), -1)
                self._arr, self.available = arr, list(range(arr.shape[1]))
        elif fmt == 'npz':
            self._npz = np.load(path, allow_pickle=False)
            self.available = list(self._npz.files)
        elif fmt == 'fits':
            try:
                from astropy.io import fits
            except ImportError:
                raise ImportError('reading FITS files needs astropy: pip install pyrenyi[fits]') from None
            hd = fits.open(path, memmap=True)
            self._close = hd.close
            hdu = spec.get('hdu', 1)
            try:
                self._rec = hd[hdu].data
            except (IndexError, KeyError):
                raise ValueError(f'{path}: no HDU {hdu!r}') from None
            if self._rec is None or not hasattr(self._rec, 'columns'):
                raise ValueError(f'{path}: HDU {hdu!r} is not a table')
            self.available = list(self._rec.columns.names)
        elif fmt == 'hdf5':
            try:
                import h5py
            except ImportError:
                raise ImportError('reading HDF5 files needs h5py: pip install pyrenyi[hdf5]') from None
            f = h5py.File(path, 'r')
            self._close = f.close
            grp = spec.get('group')
            if grp is not None and grp not in f:
                raise ValueError(f'{path}: no group {grp!r}')
            self._h5 = f[grp] if grp is not None else f
            names = []
            self._h5.visit(lambda n: names.append(n) if isinstance(self._h5[n], h5py.Dataset) else None)
            self.available = names

    def close(self):
        if self._close is not None:
            self._close()

    def get(self, name):
        fmt = self.fmt
        if fmt == 'ascii' or (fmt == 'npy' and not hasattr(self, '_rec')):
            if isinstance(name, bool) or not isinstance(name, (int, np.integer)):
                raise ValueError(f'{fmt} columns must be integer indices, got {name!r}')
            if not 0 <= name < self._arr.shape[1]:
                raise _missing(name, self.available)
            return self._arr[:, name]
        if isinstance(name, (int, np.integer)) and fmt in ('npy', 'fits'):
            if not 0 <= name < len(self.available):
                raise _missing(name, self.available)
            name = self.available[name]
        if not isinstance(name, str):
            raise ValueError(f'{fmt} columns must be names, got {name!r} '
                             '(write a constant weight, mark or nbar as a float, e.g. 1.0)')
        m = _INDEXED.match(name)
        key, idx = (m.group(1), int(m.group(2))) if m else (name, None)
        if fmt == 'npz':
            if key not in self.available:
                raise _missing(name, self.available)
            a = self._npz[key]
        elif fmt == 'hdf5':
            if key not in self.available:
                raise _missing(name, self.available)
            a = self._h5[key][()]
        else:                                             # fits or structured npy
            names = self.available
            match = [n for n in names if n == key] or [n for n in names if n.lower() == key.lower()]
            if not match:
                raise _missing(name, names)
            a = np.asarray(self._rec[match[0]])
        a = np.asarray(a)
        if idx is not None:
            if a.ndim != 2 or idx >= a.shape[1]:
                raise ValueError(f'column {name!r}: {key!r} has shape {a.shape}')
            a = a[:, idx]
        elif a.ndim != 1:
            raise ValueError(f'column {name!r} has shape {a.shape}; select a column with {key}[i]')
        return np.asarray(a, dtype=np.float64)


def read_catalogue(spec, base=None):
    """Read a Catalogue from a file described by a dict.

    Parameters
    ----------
    spec : dict
        file : path (relative paths are taken from `base`)
        format : 'auto' (from the extension), 'ascii', 'npy', 'npz', 'fits' or 'hdf5'
        columns : the x, y, z columns: names (npz keys, FITS columns, HDF5
            dataset paths, fields of a structured .npy) or integer indices
            (ASCII, plain .npy, FITS); 'name[i]' selects column i of a 2-d
            array (npz, HDF5)
        weight, mark, nbar : optional; a column, or a float for a constant
        comments, delimiter, skiprows : ASCII options (defaults '#', any
            whitespace, 0)
        group : HDF5 group holding the datasets
        hdu : FITS HDU (index or name, default 1)
    base : str, optional
        Directory of relative paths.

    Returns
    -------
    Catalogue
    """
    if not isinstance(spec, dict):
        raise ValueError('a catalogue is described by a table of keys (file, columns, ...)')
    bad = set(spec) - SPEC_KEYS
    if bad:
        raise ValueError(f'unknown catalogue key(s) {sorted(bad)}; allowed: {sorted(SPEC_KEYS)}')
    if 'file' not in spec:
        raise ValueError("a catalogue needs 'file'")
    path = os.path.expanduser(str(spec['file']))
    if base is not None and not os.path.isabs(path):
        path = os.path.join(base, path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fmt = _format(path, spec.get('format'))
    cols = spec.get('columns', [0, 1, 2] if fmt in ('ascii', 'npy') else None)
    if cols is None or len(cols) != 3:
        raise ValueError("'columns' must list the three columns x, y, z")
    t = _Table(path, fmt, spec)
    try:
        pos = np.stack([t.get(c) for c in cols], axis=1)
        extra = {}
        for key in ('weight', 'mark', 'nbar'):
            v = spec.get(key)
            if v is None:
                extra[key] = None
            elif isinstance(v, (float, np.floating)):            # a constant
                extra[key] = np.full(len(pos), float(v))
            else:
                extra[key] = t.get(v)
    finally:
        t.close()
    return Catalogue(pos, weights=extra['weight'], marks=extra['mark'], nbar=extra['nbar'])
