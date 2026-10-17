"""Brute-force reference implementation of the fixed-radius estimators, for
small catalogues: every distance is computed directly and every formula is
written out, without cell lists, chunks or cumulative shells."""
import math

import numpy as np
from scipy.special import digamma, gammaln


def _sep(a, x, box):
    d = a - x
    if box is not None:
        d -= box * np.round(d / box)
    return d


def _falling(n, s):
    a = n + 1.0 - s
    return np.exp(gammaln(n + 1.0) - gammaln(a)) if a > 0 else 0.0


def _esym(u, s):
    """Elementary symmetric polynomial e_s of the values u."""
    e = np.zeros(s + 1)
    e[0] = 1.0
    for x in u:
        e[1:] = e[1:] + x * e[:-1]
    return e[s]


def renyi_reference(data, u, radii, q, box=None, randoms=None, rw=None, r_min=(), wedges=1,
                    los=(0.0, 0.0, 1.0), observer=(0.0, 0.0, 0.0), estimator='point',
                    correction='grassberger', random_correction=True, centre_weights=None):
    """Y[case, q, r, w] as pyrenyi.measure (min_randoms=0, no edge cut).
    For estimator='volume' the centres are the randoms (centre_weights)."""
    data = np.asarray(data, float)
    radii = np.asarray(radii, float)
    q = np.asarray(q, float)
    box = None if box is None else float(box)
    unit = bool(np.all(u == u[0]))
    du = np.ones(len(data)) if unit else np.asarray(u, float)
    U, u0 = du.sum(), du.mean()
    have_r = randoms is not None
    if have_r:
        rw = np.ones(len(randoms)) if rw is None else np.asarray(rw, float)
        WR, unit_rw, rw_mean = rw.sum(), bool(np.all(rw == rw[0])), rw.mean()
    nmu = wedges
    nW = nmu + 1 if nmu > 1 else 1
    los_mode = los if isinstance(los, str) else None
    obs = np.asarray(observer, float)
    cases = [0.0] + list(r_min)
    vsph = 4 / 3 * np.pi * radii ** 3
    V = box ** 3 if box is not None else 1.0
    frac = np.ones(nW)
    if nmu > 1:
        frac[:nmu] = 1.0 / nmu
    volume = estimator == 'volume'
    if volume:
        cen, cself = randoms, np.arange(len(randoms))
        cw = np.ones(len(randoms)) if centre_weights is None else centre_weights
    else:
        cen, cself, cw = data, np.arange(len(data)), du
    num = np.zeros((len(cases), len(q), len(radii), nW))
    den = np.zeros_like(num)

    def wedge_of(sep, d, mid):
        if nmu == 1:
            return np.full(len(d), -1)
        if los_mode == 'midpoint':
            l = mid - obs
        elif los_mode == 'centre':
            l = np.broadcast_to(x - obs, sep.shape)
        else:
            l = np.broadcast_to(np.asarray(los, float), sep.shape)
        mu = np.abs((sep * l).sum(1)) / np.where(d > 0, d * np.linalg.norm(l, axis=1), 1.0)
        w = np.minimum((mu * nmu).astype(int), nmu - 1)
        return np.where(d > 0, w, -2)

    for i, x in enumerate(cen):
        sd = _sep(data, x, box)
        dd = np.linalg.norm(sd, axis=1)
        keep_d = np.ones(len(data), bool)
        if not volume:
            keep_d[cself[i]] = False
        wd = wedge_of(sd, dd, x + 0.5 * sd)
        if have_r:
            sr = _sep(randoms, x, box)
            dr = np.linalg.norm(sr, axis=1)
            keep_r = np.ones(len(randoms), bool)
            if volume:
                keep_r[cself[i]] = False
            wr = wedge_of(sr, dr, x + 0.5 * sr)
        ui = 0.0 if volume else du[cself[i]]
        for c, lo in enumerate(cases):
            for ir, r in enumerate(radii):
                if c > 0 and r <= lo:
                    continue
                for w in range(nW):
                    def inreg(d, wi, keep):
                        m = keep & (d <= r) & ((d > lo) if c > 0 else True)
                        if nmu > 1 and w < nmu:
                            m &= wi == w
                        return m
                    md = inreg(dd, wd, keep_d)
                    uN = du[md]
                    vol = (vsph[ir] - (4 / 3 * np.pi * lo ** 3 if c > 0 else 0.0)) * frac[w]
                    if volume:
                        norm = U / (WR - rw[cself[i]])
                    else:
                        norm = U - ui
                    if have_r:
                        mr = inreg(dr, wr, keep_r)
                        Rsum, m = rw[mr].sum(), mr.sum()
                        mu = (norm if volume else norm / WR) * Rsum
                        if unit_rw:
                            meff, lal = float(m), np.log((norm if volume else norm / WR) * rw_mean)
                        elif m > 0:
                            meff = Rsum ** 2 / (rw[mr] ** 2).sum()
                            lal = np.log((norm if volume else norm / WR) * (rw[mr] ** 2).sum() / Rsum)
                        else:
                            meff, lal = 0.0, np.log((norm if volume else norm / WR) * rw_mean)
                    else:
                        mu = norm * vol / V
                        meff, lal = 0.0, None
                    corr_r = have_r and random_correction
                    if mu <= 0 and not corr_r:
                        continue
                    lmu = np.log(mu) if mu > 0 else lal
                    lnmu = lal + digamma(meff + 1.0) if corr_r else lmu
                    if unit:
                        tot, neff, lc = float(len(uN)), float(len(uN)), 0.0
                    else:
                        tot = uN.sum()
                        if tot > 0:
                            neff, lc = tot ** 2 / (uN ** 2).sum(), np.log((uN ** 2).sum() / tot)
                        else:
                            neff, lc = 0.0, np.log(u0)
                    occ = tot > 0
                    for iq, qq in enumerate(q):
                        s = qq if volume else qq - 1.0
                        if corr_r:
                            lms = (-s * lal + gammaln(meff + 1) - gammaln(meff + 1 + s)
                                   if meff + 1 + s > 0 else -np.inf)
                        else:
                            lms = -s * lmu
                        isint = abs(s - round(s)) < 1e-12 and round(s) >= 1
                        if not unit and correction == 'grassberger' and isint:
                            T = math.factorial(int(round(s))) * _esym(uN, int(round(s)))
                        else:
                            T = np.exp(s * lc) * _falling(neff, s)
                        val, dv = 0.0, 1.0
                        if volume:
                            if qq < 0:
                                continue
                            if abs(qq - 1) < 1e-12:
                                if not occ:
                                    continue
                                ratio = tot * (np.exp(-lal + gammaln(meff + 1) - gammaln(meff + 2))
                                               if corr_r else 1.0 / mu)
                                lg = np.log(tot) if correction == 'none' else digamma(neff) + lc
                                val, dv = ratio * (lg - lnmu), ratio
                            elif qq == 0:
                                val = 1.0 if occ else 0.0
                            elif occ:
                                val = (np.exp(s * np.log(tot) + lms) if correction == 'none'
                                       else T * np.exp(lms))
                        else:
                            if correction == 'none':
                                tsb = tot + ui
                                if tsb <= 0 or mu <= 0:
                                    continue
                                lsb = np.log(mu * U / (U - ui))
                                val = (np.log(tsb) - lsb if abs(s) < 1e-12
                                       else np.exp(s * (np.log(tsb) - lsb)))
                            elif abs(s) < 1e-12:
                                val = digamma(neff + 1.0) + lc - lnmu
                            else:
                                val = T * np.exp(lms)
                        num[c, iq, ir, w] += cw[i] * val
                        den[c, iq, ir, w] += cw[i] * dv
    Y = np.full(num.shape, np.nan)
    for iq, qq in enumerate(q):
        if volume and qq < 0:
            continue
        ref = np.log(vsph[None, :, None] * frac[None, None, :] / V)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_ = num[:, iq] / den[:, iq]
            if abs(qq - 1) < 1e-12:
                y = r_ + ref
            else:
                y = np.where(r_ > 0, np.log(np.where(r_ > 0, r_, 1)) / (qq - 1) + ref, np.nan)
        Y[:, iq] = np.where(den[:, iq] > 0, y, np.nan)
    return Y
