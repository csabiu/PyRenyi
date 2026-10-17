"""Numba kernels: neighbour sums in spheres, shells and wedges about each
centre, and the per-centre moments that make up the partition functions.

All kernels parallelise over chunks of centres; each chunk writes only its own
slice of the output, so no atomics are needed.
"""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def digamma(x):
    """psi(x) for x > 0 (recurrence to x >= 10, then the asymptotic series)."""
    r = 0.0
    while x < 10.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0
                                                                      + f * (-1.0 / 132.0)))))
    return r + math.log(x) - 0.5 / x + t


@njit(cache=True)
def _cells(c, reach, n, periodic, buf):
    """Cell indices along one axis within `reach` of cell c; returns the count."""
    m = 0
    if periodic:
        if 2 * reach + 1 >= n:
            for k in range(n):
                buf[k] = k
            return n
        for o in range(-reach, reach + 1):
            buf[m] = (c + o) % n
            m += 1
        return m
    lo = max(0, c - reach)
    hi = min(n - 1, c + reach)
    for k in range(lo, hi + 1):
        buf[m] = k
        m += 1
    return m


@njit(cache=True)
def _cell_of(x, lo, cell, n):
    c = int(math.floor((x - lo) / cell))
    if c < 0:
        return 0
    if c > n - 1:
        return n - 1
    return c


@njit(cache=True)
def _wedge(ex, ey, ez, d2, lx, ly, lz, mu_edges, equal_mu):
    """Index of the |mu| wedge of separation e about the line of sight l."""
    ll = lx * lx + ly * ly + lz * lz
    if d2 <= 0.0 or ll <= 0.0:
        return -1
    mu = abs(ex * lx + ey * ly + ez * lz) / math.sqrt(d2 * ll)
    nmu = mu_edges.shape[0] - 1
    if equal_mu:
        w = int(mu * nmu)
    else:
        w = 0
        while w < nmu - 1 and mu >= mu_edges[w + 1]:
            w += 1
    if w > nmu - 1:
        w = nmu - 1
    return w


@njit(parallel=True, cache=True)
def fixed_radius(cen, cen_self, cen_rself, cen_c, cen_u, mu_norm, sb_fac, cen_los, chunk_start,
                 dpos, du, d_order, d_start, rpos, rw, rdv, r_order, r_start,
                 lo, cell, ncell, reach, periodic, boxl,
                 r2, case_start, vol, vsph, mu_edges, equal_mu, los_mid, observer,
                 s_vals, s_int, special, volume, unit_w, corr, u0, smax, fact,
                 have_rand, rand_corr, unit_rw, rw_mean, min_rand, fmin, rsplit, out_num, out_den):
    """Per-chunk sums of c_i * (moment of the neighbour mass about centre i).

    For every centre, the data (and random) neighbours are binned in shells
    between consecutive radii and in |mu| wedges; the shells are accumulated
    into spheres (case 0) or into shells beyond a minimum separation (case e,
    starting at bin case_start[e]); for each region the expected neighbour
    mass mu = mu_norm[i] * (random sum or analytic volume) normalises the
    finite-sample-corrected moment T_s, and c_i * T_s / mu^s is added to
    out_num (out_den holds the matching normalisation). With rand_corr, the
    power mu^-s of the random-based expectation is replaced by its
    Poisson-unbiased estimate alpha^-s Gamma(m + 1) / Gamma(m + 1 + s) of the
    random count m (an effective count for weighted randoms). A region with
    fewer than min_rand randoms takes its expectation from the smallest sphere
    about the centre with at least min_rand randoms (or the sphere of the same
    radius), scaled by the ratio of the region and sphere volumes. With
    rsplit, the odd-numbered randoms give the footprint fraction (for the edge
    cut) and the even-numbered ones the expected counts, so that the cut does
    not select on the noise of the expected counts.
    """
    nch = chunk_start.shape[0] - 1
    nr = r2.shape[0]
    nq = s_vals.shape[0]
    ncase = case_start.shape[0]
    nmu = mu_edges.shape[0] - 1
    nW = nmu + 1 if nmu > 1 else 1
    iso = nW - 1
    S = smax
    r2max = r2[nr - 1]
    have_dv = rdv.shape[0] > 0
    nbuf = max(ncell.max(), 2 * reach.max() + 1)
    for ch in prange(nch):
        E = np.zeros((nr, nW, S + 1))
        Ncnt = np.zeros((nr, nW), np.int64)
        P1 = np.zeros((nr, nW))
        P2 = np.zeros((nr, nW))
        Rs = np.zeros((nr, nW))
        Rn = np.zeros((nr, nW), np.int64)
        R2 = np.zeros((nr, nW))
        DV = np.zeros(nr)
        Ec = np.zeros((nW, S + 1))
        tmp = np.zeros(S + 1)
        Ncum = np.zeros(nW, np.int64)
        Q1 = np.zeros(nW)
        Q2 = np.zeros(nW)
        Rc = np.zeros(nW)
        Rnc = np.zeros(nW, np.int64)
        R2c = np.zeros(nW)
        sR = np.zeros(nr)
        sR2 = np.zeros(nr)
        sN = np.zeros(nr, np.int64)
        fr = np.ones(nr)
        bx = np.empty(nbuf, np.int64)
        by = np.empty(nbuf, np.int64)
        bz = np.empty(nbuf, np.int64)
        for i in range(chunk_start[ch], chunk_start[ch + 1]):
            E[:, :, :] = 0.0
            E[:, :, 0] = 1.0
            Ncnt[:, :] = 0
            P1[:, :] = 0.0
            P2[:, :] = 0.0
            Rs[:, :] = 0.0
            Rn[:, :] = 0
            R2[:, :] = 0.0
            DV[:] = 0.0
            x = cen[i, 0]
            y = cen[i, 1]
            z = cen[i, 2]
            nx = _cells(_cell_of(x, lo[0], cell[0], ncell[0]), reach[0], ncell[0], periodic, bx)
            ny = _cells(_cell_of(y, lo[1], cell[1], ncell[1]), reach[1], ncell[1], periodic, by)
            nz = _cells(_cell_of(z, lo[2], cell[2], ncell[2]), reach[2], ncell[2], periodic, bz)
            lx = cen_los[i, 0]
            ly = cen_los[i, 1]
            lz = cen_los[i, 2]
            # ---- data neighbours ----
            for a in range(nx):
                for b in range(ny):
                    for c in range(nz):
                        cid = (bx[a] * ncell[1] + by[b]) * ncell[2] + bz[c]
                        for t in range(d_start[cid], d_start[cid + 1]):
                            j = d_order[t]
                            if j == cen_self[i]:
                                continue
                            ex = dpos[j, 0] - x
                            ey = dpos[j, 1] - y
                            ez = dpos[j, 2] - z
                            if periodic:
                                ex -= boxl[0] * np.round(ex / boxl[0])
                                ey -= boxl[1] * np.round(ey / boxl[1])
                                ez -= boxl[2] * np.round(ez / boxl[2])
                            d2 = ex * ex + ey * ey + ez * ez
                            if d2 > r2max:
                                continue
                            kb = np.searchsorted(r2, d2)
                            w = -1
                            if nmu > 1:
                                if los_mid:
                                    w = _wedge(ex, ey, ez, d2, 0.5 * (dpos[j, 0] + x) - observer[0],
                                               0.5 * (dpos[j, 1] + y) - observer[1],
                                               0.5 * (dpos[j, 2] + z) - observer[2], mu_edges, equal_mu)
                                else:
                                    w = _wedge(ex, ey, ez, d2, lx, ly, lz, mu_edges, equal_mu)
                            if unit_w:
                                Ncnt[kb, iso] += 1
                                if w >= 0:
                                    Ncnt[kb, w] += 1
                            else:
                                u = du[j]
                                for k in range(S, 0, -1):
                                    E[kb, iso, k] += u * E[kb, iso, k - 1]
                                P1[kb, iso] += u
                                P2[kb, iso] += u * u
                                if w >= 0:
                                    for k in range(S, 0, -1):
                                        E[kb, w, k] += u * E[kb, w, k - 1]
                                    P1[kb, w] += u
                                    P2[kb, w] += u * u
            # ---- random neighbours ----
            if have_rand:
                for a in range(nx):
                    for b in range(ny):
                        for c in range(nz):
                            cid = (bx[a] * ncell[1] + by[b]) * ncell[2] + bz[c]
                            for t in range(r_start[cid], r_start[cid + 1]):
                                j = r_order[t]
                                if j == cen_rself[i]:
                                    continue
                                ex = rpos[j, 0] - x
                                ey = rpos[j, 1] - y
                                ez = rpos[j, 2] - z
                                if periodic:
                                    ex -= boxl[0] * np.round(ex / boxl[0])
                                    ey -= boxl[1] * np.round(ey / boxl[1])
                                    ez -= boxl[2] * np.round(ez / boxl[2])
                                d2 = ex * ex + ey * ey + ez * ez
                                if d2 > r2max:
                                    continue
                                kb = np.searchsorted(r2, d2)
                                if rsplit:
                                    # odd randoms: footprint fraction only
                                    if j % 2 == 1:
                                        DV[kb] += rdv[j]
                                        continue
                                elif have_dv:
                                    DV[kb] += rdv[j]
                                wj = rw[j]
                                Rs[kb, iso] += wj
                                Rn[kb, iso] += 1
                                R2[kb, iso] += wj * wj
                                if nmu > 1:
                                    if los_mid:
                                        w = _wedge(ex, ey, ez, d2, 0.5 * (rpos[j, 0] + x) - observer[0],
                                                   0.5 * (rpos[j, 1] + y) - observer[1],
                                                   0.5 * (rpos[j, 2] + z) - observer[2], mu_edges, equal_mu)
                                    else:
                                        w = _wedge(ex, ey, ez, d2, lx, ly, lz, mu_edges, equal_mu)
                                    if w >= 0:
                                        Rs[kb, w] += wj
                                        Rn[kb, w] += 1
                                        R2[kb, w] += wj * wj
            # cumulative random sums in spheres: the reference for regions
            # with fewer than min_rand randoms, and the footprint fraction
            bi = 0
            if have_rand:
                a1 = 0.0
                a2 = 0.0
                a3 = 0
                for ir in range(nr):
                    a1 += Rs[ir, iso]
                    a2 += R2[ir, iso]
                    a3 += Rn[ir, iso]
                    sR[ir] = a1
                    sR2[ir] = a2
                    sN[ir] = a3
                if min_rand > 0:
                    bi = nr - 1
                    for ir in range(nr):
                        if sN[ir] >= min_rand:
                            bi = ir
                            break
            if fmin > 0.0 and have_dv:
                acc = 0.0
                for ir in range(nr):
                    acc += DV[ir]
                    fr[ir] = acc / vsph[ir]
            cc = cen_c[i]
            # ---- spheres / shells, per radius and wedge ----
            for cs in range(ncase):
                Ec[:, :] = 0.0
                Ec[:, 0] = 1.0
                Ncum[:] = 0
                Q1[:] = 0.0
                Q2[:] = 0.0
                Rc[:] = 0.0
                Rnc[:] = 0
                R2c[:] = 0.0
                for ir in range(case_start[cs], nr):
                    for w in range(nW):
                        if unit_w:
                            Ncum[w] += Ncnt[ir, w]
                        else:
                            if S > 0:
                                for k in range(S + 1):
                                    acc = 0.0
                                    for a2_ in range(k + 1):
                                        acc += Ec[w, a2_] * E[ir, w, k - a2_]
                                    tmp[k] = acc
                                for k in range(S + 1):
                                    Ec[w, k] = tmp[k]
                            Q1[w] += P1[ir, w]
                            Q2[w] += P2[ir, w]
                        Rc[w] += Rs[ir, w]
                        Rnc[w] += Rn[ir, w]
                        R2c[w] += R2[ir, w]
                    if fmin > 0.0 and fr[max(ir, bi)] < fmin:
                        continue
                    for w in range(nW):
                        # expected neighbour mass mu, and for the random-count
                        # correction the count meff with mu = alpha * meff
                        meff = 0.0
                        if have_rand:
                            if min_rand > 0 and Rnc[w] < min_rand:
                                b = max(ir, bi)
                                if sN[b] == 0:
                                    continue
                                rv = vol[cs, ir, w] / vsph[b]
                                mu = mu_norm[i] * sR[b] * rv
                                if unit_rw:
                                    meff = float(sN[b])
                                    lal = math.log(mu_norm[i] * rw_mean * rv)
                                else:
                                    meff = sR[b] * sR[b] / sR2[b]
                                    lal = math.log(mu_norm[i] * sR2[b] / sR[b] * rv)
                            else:
                                mu = mu_norm[i] * Rc[w]
                                if Rnc[w] > 0:
                                    if unit_rw:
                                        meff = float(Rnc[w])
                                        lal = math.log(mu_norm[i] * rw_mean)
                                    else:
                                        meff = Rc[w] * Rc[w] / R2c[w]
                                        lal = math.log(mu_norm[i] * R2c[w] / Rc[w])
                                else:
                                    lal = math.log(mu_norm[i] * rw_mean)
                        else:
                            mu = mu_norm[i] * vol[cs, ir, w]
                            lal = 0.0
                        corr_r = have_rand and rand_corr
                        if mu <= 0.0 and not corr_r:
                            continue
                        lmu = math.log(mu) if mu > 0.0 else lal
                        if corr_r:
                            lgm = math.lgamma(meff + 1.0)
                            lnmu = lal + digamma(meff + 1.0)
                        else:
                            lal = lmu
                            lgm = 0.0
                            lnmu = lmu
                        if unit_w:
                            tot = float(Ncum[w])
                            neff = tot
                            lc = 0.0
                        else:
                            tot = Q1[w]
                            if tot > 0.0:
                                neff = tot * tot / Q2[w]
                                lc = math.log(Q2[w] / tot)
                            else:
                                neff = 0.0
                                lc = math.log(u0)
                        occ = tot > 0.0
                        lg1 = math.lgamma(neff + 1.0)
                        for iq in range(nq):
                            s = s_vals[iq]
                            val = 0.0
                            dval = 1.0
                            # log of the estimate of mu^-s
                            if corr_r:
                                a4 = meff + 1.0 + s
                                if a4 <= 0.0:
                                    lms = -np.inf
                                else:
                                    lms = -s * lal + lgm - math.lgamma(a4)
                            else:
                                lms = -s * lmu
                            if volume:
                                if s < 0.0:
                                    continue
                                if special[iq] == 1:                    # q = 1
                                    if not occ:
                                        continue
                                    if corr_r:
                                        ratio = tot * math.exp(-lal + lgm - math.lgamma(meff + 2.0))
                                    else:
                                        ratio = tot / mu
                                    if corr == 2:
                                        val = ratio * (math.log(tot) - lnmu)
                                    else:
                                        val = ratio * (digamma(neff) + lc - lnmu)
                                    dval = ratio
                                elif s == 0.0:                          # q = 0
                                    val = 1.0 if occ else 0.0
                                elif occ:
                                    if corr == 2:
                                        val = math.exp(s * math.log(tot) + lms)
                                    elif (not unit_w) and corr == 0 and s_int[iq] >= 1:
                                        val = fact[s_int[iq]] * Ec[w, s_int[iq]] * math.exp(lms)
                                    else:
                                        a3 = neff + 1.0 - s
                                        if a3 > 0.0:
                                            val = math.exp(lg1 - math.lgamma(a3) + s * lc + lms)
                            else:
                                if corr == 2:                           # sandbox: centre included
                                    tsb = tot + cen_u[i]
                                    if tsb <= 0.0 or mu <= 0.0:
                                        continue
                                    lsb = math.log(mu * sb_fac[i])
                                    if special[iq] == 1:
                                        val = math.log(tsb) - lsb
                                    else:
                                        val = math.exp(s * (math.log(tsb) - lsb))
                                elif special[iq] == 1:                  # q = 1
                                    val = digamma(neff + 1.0) + lc - lnmu
                                elif (not unit_w) and corr == 0 and s_int[iq] >= 1:
                                    val = fact[s_int[iq]] * Ec[w, s_int[iq]] * math.exp(lms)
                                else:
                                    a3 = neff + 1.0 - s
                                    if a3 > 0.0:
                                        val = math.exp(lg1 - math.lgamma(a3) + s * lc + lms)
                            out_num[ch, cs, iq, ir, w] += cc * val
                            out_den[ch, cs, iq, ir, w] += cc * dval


@njit(parallel=True, cache=True)
def random_sums_at_radii(cen, cen_rself, rad, rmin2, rpos, rw, rdv, r_order, r_start,
                         lo, cell, ncell, periodic, boxl, rsplit, out_r, out_n, out_r2, out_dv):
    """Cumulative random sums within the per-centre radii rad[i, :] (ascending):
    weight (out_r), count (out_n) and squared weight (out_r2) beyond r_min,
    and volume element (out_dv) from d = 0. With rsplit the odd randoms give
    only the volume and the even ones only the sums."""
    nc, nk = rad.shape
    have_dv = rdv.shape[0] > 0
    for i in prange(nc):
        r2i = np.empty(nk)
        for k in range(nk):
            r2i[k] = rad[i, k] * rad[i, k]
        rmax = rad[i, nk - 1]
        sr = np.zeros(nk)
        sn = np.zeros(nk)
        s2 = np.zeros(nk)
        sv = np.zeros(nk)
        x = cen[i, 0]
        y = cen[i, 1]
        z = cen[i, 2]
        rx = int(math.ceil(rmax / cell[0]))
        ry = int(math.ceil(rmax / cell[1]))
        rz = int(math.ceil(rmax / cell[2]))
        bx = np.empty(max(ncell[0], 2 * rx + 1), np.int64)
        by = np.empty(max(ncell[1], 2 * ry + 1), np.int64)
        bz = np.empty(max(ncell[2], 2 * rz + 1), np.int64)
        nx = _cells(_cell_of(x, lo[0], cell[0], ncell[0]), rx, ncell[0], periodic, bx)
        ny = _cells(_cell_of(y, lo[1], cell[1], ncell[1]), ry, ncell[1], periodic, by)
        nz = _cells(_cell_of(z, lo[2], cell[2], ncell[2]), rz, ncell[2], periodic, bz)
        for a in range(nx):
            for b in range(ny):
                for c in range(nz):
                    cid = (bx[a] * ncell[1] + by[b]) * ncell[2] + bz[c]
                    for t in range(r_start[cid], r_start[cid + 1]):
                        j = r_order[t]
                        if j == cen_rself[i]:
                            continue
                        ex = rpos[j, 0] - x
                        ey = rpos[j, 1] - y
                        ez = rpos[j, 2] - z
                        if periodic:
                            ex -= boxl[0] * np.round(ex / boxl[0])
                            ey -= boxl[1] * np.round(ey / boxl[1])
                            ez -= boxl[2] * np.round(ez / boxl[2])
                        d2 = ex * ex + ey * ey + ez * ez
                        if d2 > r2i[nk - 1]:
                            continue
                        kb = np.searchsorted(r2i, d2)
                        if rsplit:
                            if j % 2 == 1:
                                sv[kb] += rdv[j]
                                continue
                        elif have_dv:
                            sv[kb] += rdv[j]
                        if d2 > rmin2:
                            wj = rw[j]
                            sr[kb] += wj
                            sn[kb] += 1.0
                            s2[kb] += wj * wj
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        a4 = 0.0
        for k in range(nk):
            a1 += sr[k]
            a2 += sn[k]
            a3 += s2[k]
            a4 += sv[k]
            out_r[i, k] = a1
            out_n[i, k] = a2
            out_r2[i, k] = a3
            out_dv[i, k] = a4


@njit(parallel=True, cache=True)
def tau_sums(lnA, b, ok, c, chunk_start, taus, out):
    """out[chunk, it, k] = sum_i c_i E[nu_ik^(-t)], t = tau_t / 3, over the
    centres of the chunk with ok[i, k]. The estimate of nu^-t is
    A^-t Gamma(b) / Gamma(b + t) (b > 0: from a Poisson random count m with
    b = m + 1, or the distance to the m-th random with b = m), or A^-t
    exactly (b <= 0)."""
    nch = chunk_start.shape[0] - 1
    nk = lnA.shape[1]
    nt = taus.shape[0]
    for ch in prange(nch):
        for i in range(chunk_start[ch], chunk_start[ch + 1]):
            for k in range(nk):
                if not ok[i, k]:
                    continue
                la = lnA[i, k]
                bb = b[i, k]
                if bb > 0.0:
                    lgb = math.lgamma(bb)
                    for it in range(nt):
                        t = taus[it] / 3.0
                        if bb + t > 0.0:
                            out[ch, it, k] += c[i] * math.exp(-t * la + lgb - math.lgamma(bb + t))
                else:
                    for it in range(nt):
                        out[ch, it, k] += c[i] * math.exp(-taus[it] / 3.0 * la)
