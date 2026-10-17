"""Cell lists for the neighbour search, periodic or not."""
import numpy as np

MAX_CELLS = 2 ** 24


class Grid:
    """A regular grid of cells covering a periodic box or the bounding box of
    a set of point arrays, with cells of side >= r_max / 2.

    Attributes
    ----------
    lo : (3,) lower corner; size : (3,) extent; n : (3,) cells per axis;
    cell : (3,) cell side; reach : (3,) cells to visit on each side of a
    centre's cell to cover r_max; periodic : bool.
    """

    def __init__(self, rmax, arrays, box=None):
        if box is not None:
            self.periodic = True
            self.lo = np.zeros(3)
            self.size = np.asarray(box, dtype=np.float64) * np.ones(3)
        else:
            self.periodic = False
            pts = [a for a in arrays if a is not None and len(a)]
            lo = np.min([a.min(0) for a in pts], 0)
            hi = np.max([a.max(0) for a in pts], 0)
            pad = 1e-9 * max(1.0, float(np.max(hi - lo)))
            self.lo = lo - pad
            self.size = np.maximum(hi - lo + 2 * pad, 1e-6)
        n = np.maximum(np.floor(self.size / (rmax / 2.0)), 1).astype(np.int64)
        while np.prod(n) > MAX_CELLS:
            n = np.maximum(n // 2, 1)
        self.n = n
        self.cell = self.size / n
        self.reach = np.ceil(rmax / self.cell).astype(np.int64)

    def cell_index(self, pos):
        """Flat cell index of each position."""
        c = np.floor((pos - self.lo) / self.cell).astype(np.int64)
        c = np.clip(c, 0, self.n - 1)
        return (c[:, 0] * self.n[1] + c[:, 1]) * self.n[2] + c[:, 2]

    def build(self, pos):
        """(order, start): points of cell c are order[start[c]:start[c + 1]]."""
        if pos is None or len(pos) == 0:
            return np.zeros(0, np.int64), np.zeros(int(np.prod(self.n)) + 1, np.int64)
        cid = self.cell_index(pos)
        order = np.argsort(cid, kind='stable').astype(np.int64)
        start = np.searchsorted(cid[order], np.arange(int(np.prod(self.n)) + 1)).astype(np.int64)
        return order, start
