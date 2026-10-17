"""File readers and the command-line interface."""
import numpy as np
import pytest

import pyrenyi
from pyrenyi import cli
from pyrenyi.io import read_catalogue

try:
    import tomllib as toml
except ImportError:
    import tomli as toml

N = 500


@pytest.fixture(scope='module')
def table():
    rng = np.random.default_rng(0)
    return dict(x=rng.uniform(0, 100, N), y=rng.uniform(0, 100, N), z=rng.uniform(0, 100, N),
                w=rng.uniform(0.5, 2, N), nz=rng.uniform(1e-3, 2e-3, N))


def check(cat, t):
    assert np.allclose(cat.positions, np.stack([t['x'], t['y'], t['z']], 1))
    assert np.allclose(cat.weights, t['w'])
    assert np.allclose(cat.nbar, t['nz'])


def test_ascii(tmp_path, table):
    t = table
    f = tmp_path / 'c.txt'
    np.savetxt(f, np.stack([t['x'], t['y'], t['z'], t['w'], t['nz']], 1), header='x y z w nz')
    check(read_catalogue({'file': str(f), 'columns': [0, 1, 2], 'weight': 3, 'nbar': 4}), t)
    g = tmp_path / 'c.csv'
    np.savetxt(g, np.stack([t['w'], t['x'], t['y'], t['z'], t['nz']], 1), delimiter=',')
    c = read_catalogue({'file': 'c.csv', 'columns': [1, 2, 3], 'delimiter': ',', 'nbar': 4,
                        'weight': 2.0}, base=str(tmp_path))
    assert np.allclose(c.weights, 2.0)
    assert np.allclose(c.positions, np.stack([t['x'], t['y'], t['z']], 1))
    assert np.allclose(c.nbar, t['nz'])
    with pytest.raises(ValueError, match='integer'):
        read_catalogue({'file': str(f), 'columns': ['x', 'y', 'z']})
    with pytest.raises(ValueError, match='not found'):
        read_catalogue({'file': str(f), 'columns': [0, 1, 7]})


def test_npy(tmp_path, table):
    t = table
    f = tmp_path / 'c.npy'
    np.save(f, np.stack([t['x'], t['y'], t['z'], t['w'], t['nz']], 1))
    check(read_catalogue({'file': str(f), 'columns': [0, 1, 2], 'weight': 3, 'nbar': 4}), t)
    rec = np.zeros(N, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('w', 'f4'), ('nz', 'f8')])
    for k in rec.dtype.names:
        rec[k] = t[k]
    g = tmp_path / 'r.npy'
    np.save(g, rec)
    c = read_catalogue({'file': str(g), 'columns': ['x', 'y', 'z'], 'weight': 'w', 'nbar': 'nz'})
    assert np.allclose(c.weights, t['w'], rtol=1e-6)
    with pytest.raises(ValueError, match="'ww' not found"):
        read_catalogue({'file': str(g), 'columns': ['x', 'y', 'z'], 'weight': 'ww'})


def test_npz(tmp_path, table):
    t = table
    f = tmp_path / 'c.npz'
    np.savez(f, pos=np.stack([t['x'], t['y'], t['z']], 1), w=t['w'], nz=t['nz'])
    check(read_catalogue({'file': str(f), 'columns': ['pos[0]', 'pos[1]', 'pos[2]'], 'weight': 'w',
                          'nbar': 'nz'}), t)
    with pytest.raises(ValueError, match='shape'):
        read_catalogue({'file': str(f), 'columns': ['pos', 'pos[1]', 'pos[2]']})
    with pytest.raises(ValueError, match='float'):
        read_catalogue({'file': str(f), 'columns': ['pos[0]', 'pos[1]', 'pos[2]'], 'weight': 1})


def test_hdf5(tmp_path, table):
    h5py = pytest.importorskip('h5py')
    t = table
    f = tmp_path / 'c.hdf5'
    with h5py.File(f, 'w') as h:
        g = h.create_group('galaxies')
        for k in ('x', 'y', 'z', 'w', 'nz'):
            g[k] = t[k]
    check(read_catalogue({'file': str(f), 'columns': ['galaxies/x', 'galaxies/y', 'galaxies/z'],
                          'weight': 'galaxies/w', 'nbar': 'galaxies/nz'}), t)
    check(read_catalogue({'file': str(f), 'group': 'galaxies', 'columns': ['x', 'y', 'z'],
                          'weight': 'w', 'nbar': 'nz'}), t)


def test_fits(tmp_path, table):
    fits = pytest.importorskip('astropy.io.fits')
    t = table
    f = tmp_path / 'c.fits'
    cols = [fits.Column(name=k.upper(), format='D', array=t[k]) for k in ('x', 'y', 'z', 'w', 'nz')]
    fits.BinTableHDU.from_columns(cols).writeto(f)
    check(read_catalogue({'file': str(f), 'columns': ['X', 'Y', 'Z'], 'weight': 'W', 'nbar': 'NZ'}), t)
    check(read_catalogue({'file': str(f), 'columns': ['x', 'y', 'z'], 'weight': 'w', 'nbar': 'nz'}), t)
    with pytest.raises(ValueError, match="available: X, Y, Z, W, NZ"):
        read_catalogue({'file': str(f), 'columns': ['X', 'Y', 'Q']})


def test_bad_spec(tmp_path):
    with pytest.raises(ValueError, match='unknown catalogue key'):
        read_catalogue({'file': 'a.npy', 'colums': [0, 1, 2]})
    with pytest.raises(FileNotFoundError):
        read_catalogue({'file': str(tmp_path / 'missing.npy')})
    (tmp_path / 'a.xyz').write_text('1 2 3\n')
    with pytest.raises(ValueError, match='format'):
        read_catalogue({'file': str(tmp_path / 'a.xyz')})


def test_templates_valid():
    for text in (cli.TEMPLATE_BOX, cli.TEMPLATE_SURVEY):
        cli.validate_config(toml.loads(text))


def test_template_box_end_to_end(tmp_path, capsys):
    """The box template, pointed at a small Poisson box, runs and shows."""
    np.save(tmp_path / 'points.npy', pyrenyi.mocks.poisson(3000, 100.0, seed=1))
    cfg = toml.loads(cli.TEMPLATE_BOX)
    cfg['geometry']['box'] = 100.0
    cfg['measurement'].update(radii={'min': 4.0, 'max': 20.0, 'n': 5}, jackknife=8)
    res, path = cli.run_config(cfg, str(tmp_path))
    assert path == str(tmp_path / 'pyrenyi_result.npz')
    assert 10.0 in res.radii and list(res.r_min) == [10.0] and res.n_wedges == 3
    out = capsys.readouterr().out
    assert 'data: 3000 points' in out and 'D_q' in out
    assert cli.main(['show', path, '--errors']) == 0
    assert cli.main(['show', path, '--r-min', '10', '--wedge', '2', '--all']) == 0
    assert '±' in capsys.readouterr().out


def write_survey(tmp_path):
    rng = np.random.default_rng(3)
    d = rng.uniform(0, 100, (3000, 3))
    r = rng.uniform(0, 100, (30000, 3))
    np.savetxt(tmp_path / 'gal.txt', d)
    np.savez(tmp_path / 'ran.npz', pos=r, nz=np.full(len(r), 3000 / 1e6))


SURVEY = '''
[data]
file = "gal.txt"
columns = [0, 1, 2]

[randoms]
file = "ran.npz"
columns = ["pos[0]", "pos[1]", "pos[2]"]
nbar = "nz"

[geometry]
observer = [-500.0, 50.0, 50.0]

[measurement]
estimator = "{est}"
{grid}
q = [-1.0, 1.0, 2.0, 3.0]
edge_cut = 0.6
jackknife = 4

[output]
file = "out/{est}.npz"
'''


@pytest.mark.parametrize('est, grid', [
    ('point', 'radii = [4.0, 6.0, 9.0, 13.0]\nwedges = 2\nlos = "midpoint"'),
    ('volume', 'radii = {min = 4.0, max = 13.0, n = 4}'),
    ('fixed-mass', 'k = {min = 1, max = 16, n = 5}')])
def test_survey_run(tmp_path, est, grid, capsys):
    write_survey(tmp_path)
    (tmp_path / 'out').mkdir()
    f = tmp_path / 'survey.toml'
    f.write_text(SURVEY.format(est=est, grid=grid))
    assert cli.main(['run', str(f)]) == 0
    out = capsys.readouterr().out
    assert 'randoms: 30000' in out
    res = pyrenyi.load(str(tmp_path / 'out' / f'{est}.npz'))
    D, _ = res.dq()
    i2 = int(np.flatnonzero(res.q == 2.0)[0])
    assert np.nanmax(np.abs(D[i2] - 3.0)) < 0.5        # Poisson: D_2 = 3 within noise
    assert cli.main(['show', str(tmp_path / 'out' / f'{est}.npz'), '--errors']) == 0


def test_config_errors(tmp_path, capsys):
    write_survey(tmp_path)
    base = SURVEY.format(est='point', grid='radii = [4.0, 6.0]')
    for bad, msg in ((base.replace('edge_cut', 'edgecut'), "did you mean 'edge_cut'"),
                     (base.replace('[geometry]', '[geometri]'), "unknown key 'geometri'"),
                     (base.replace('columns = [0, 1, 2]', 'columns = [0, 1, 2]\nweigth = 1.0'), "'weigth'"),
                     (base.replace('estimator = "point"', 'estimator = "fixed-mass"'), 'do not apply'),
                     (base.replace('estimator = "point"', 'estimator = "points"'), 'estimator must be')):
        f = tmp_path / 'bad.toml'
        f.write_text(bad)
        assert cli.main(['run', str(f)]) == 2
        assert msg in capsys.readouterr().err
    with pytest.raises(ValueError, match="did you mean 'edge_cut'"):
        cli.load_config(_write(tmp_path, base.replace('edge_cut', 'edgecut')))


def _write(tmp_path, text):
    f = tmp_path / 'bad2.toml'
    f.write_text(text)
    return str(f)


def test_version_and_template(capsys):
    with pytest.raises(SystemExit):
        cli.main(['--version'])
    assert 'pyrenyi' in capsys.readouterr().out
    assert cli.main(['template', '--survey']) == 0
    assert 'edge_cut' in capsys.readouterr().out
