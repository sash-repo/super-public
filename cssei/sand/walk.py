#!/usr/bin/env python3
r"""Exhaustive F_2 walks over a Lie algebra L given by its structure table
(tab[i][j] = [e_i, e_j] as a bitmask), compiled with numba, parallel over chunks.

  nil2(tab)    : all x in L with (ad x)^2 = 0      (Gray code over x, ad x kept as n columns)
  codim1(tab)  : all nonzero phi in L^* with Ker phi a subalgebra
                 (B_phi(e_i,e_j) = phi([e_i,e_j]); Ker phi is a subalgebra iff
                  row_i(B) = B_{i a} phi + phi_i row_a(B) for all i, where phi_a = 1)
Both return numpy arrays of bitmasks (uint64)."""
import numpy as np
import numba as nb

CH = 10          # 2^CH chunks
CAP = 1 << 14    # results per chunk


@nb.njit(cache=True)
def _nil_chunk(T, n, lo, top, out, cnt, c):
    # T[i, j] = [e_i, e_j]; walk x = (top << lo) + gray(k), k < 2^lo
    adx = np.zeros(n, dtype=np.uint64)
    for i in range(lo, n):
        if (top >> (i - lo)) & 1:
            for j in range(n):
                adx[j] ^= T[i, j]
    x = np.uint64(top) << np.uint64(lo)
    m = 0
    N = 1 << lo
    for k in range(N):
        if k > 0:
            b = 0
            kk = k
            while (kk & 1) == 0:
                kk >>= 1
                b += 1
            x ^= np.uint64(1) << np.uint64(b)
            for j in range(n):
                adx[j] ^= T[b, j]
        if x == 0:
            continue
        ok = True
        for j in range(n):
            v = adx[j]
            w = np.uint64(0)
            while v:
                t = v & (~v + np.uint64(1))
                bb = 0
                tt = t
                while tt > 1:
                    tt >>= np.uint64(1)
                    bb += 1
                w ^= adx[bb]
                v ^= t
            if w != 0:
                ok = False
                break
        if ok:
            if m < out.shape[1]:
                out[c, m] = x
            m += 1
    cnt[c] = m


@nb.njit(parallel=True, cache=True)
def _nil_all(T, n, ch, out, cnt):
    lo = n - ch
    for c in nb.prange(1 << ch):
        _nil_chunk(T, n, lo, c, out, cnt, c)


def _table(tab):
    n = len(tab)
    T = np.zeros((n, n), dtype=np.uint64)
    for i in range(n):
        for j in range(n):
            T[i, j] = tab[i][j]
    return T


def _collect(out, cnt):
    if cnt.max() > out.shape[1]:
        raise RuntimeError('chunk capacity exceeded: %d' % cnt.max())
    res = []
    for c in range(len(cnt)):
        res.extend(int(v) for v in out[c, :cnt[c]])
    return res


def nil2(tab, ch=CH, cap=CAP):
    n = len(tab)
    ch = min(ch, n - 1)
    T = _table(tab)
    out = np.zeros((1 << ch, cap), dtype=np.uint64)
    cnt = np.zeros(1 << ch, dtype=np.int64)
    _nil_all(T, n, ch, out, cnt)
    return _collect(out, cnt)


@nb.njit(cache=True)
def _par(v):
    v ^= v >> np.uint64(32); v ^= v >> np.uint64(16); v ^= v >> np.uint64(8)
    v ^= v >> np.uint64(4); v ^= v >> np.uint64(2); v ^= v >> np.uint64(1)
    return v & np.uint64(1)


@nb.njit(cache=True)
def _c1_chunk(M, n, lo, top, out, cnt, c):
    # M[b, i] = bitmask over j of bit b of [e_i, e_j]; row_i(B_phi) = XOR_b phi_b M[b, i]
    row = np.zeros(n, dtype=np.uint64)
    for b in range(lo, n):
        if (top >> (b - lo)) & 1:
            for i in range(n):
                row[i] ^= M[b, i]
    phi = np.uint64(top) << np.uint64(lo)
    m = 0
    for k in range(1 << lo):
        if k > 0:
            b = 0
            kk = k
            while (kk & 1) == 0:
                kk >>= 1
                b += 1
            phi ^= np.uint64(1) << np.uint64(b)
            for i in range(n):
                row[i] ^= M[b, i]
        if phi == 0:
            continue
        a = 0
        while ((phi >> np.uint64(a)) & np.uint64(1)) == 0:
            a += 1
        ra = row[a]
        ok = True
        for i in range(n):
            exp = np.uint64(0)
            if (row[i] >> np.uint64(a)) & np.uint64(1):
                exp ^= phi
            if (phi >> np.uint64(i)) & np.uint64(1):
                exp ^= ra
            if row[i] != exp:
                ok = False
                break
        if ok:
            if m < out.shape[1]:
                out[c, m] = phi
            m += 1
    cnt[c] = m


@nb.njit(parallel=True, cache=True)
def _c1_all(M, n, ch, out, cnt):
    lo = n - ch
    for c in nb.prange(1 << ch):
        _c1_chunk(M, n, lo, c, out, cnt, c)


def codim1(tab, ch=CH, cap=CAP):
    n = len(tab)
    ch = min(ch, n - 1)
    M = np.zeros((n, n), dtype=np.uint64)
    for b in range(n):
        for i in range(n):
            v = 0
            for j in range(n):
                if (tab[i][j] >> b) & 1:
                    v |= 1 << j
            M[b, i] = v
    out = np.zeros((1 << ch, cap), dtype=np.uint64)
    cnt = np.zeros(1 << ch, dtype=np.int64)
    _c1_all(M, n, ch, out, cnt)
    return _collect(out, cnt)


@nb.njit(cache=True)
def _lr_chunk(M, n, lo, top, rmax, out, cnt, c):
    row = np.zeros(n, dtype=np.uint64)
    piv = np.zeros(n + 1, dtype=np.uint64)
    for b in range(lo, n):
        if (top >> (b - lo)) & 1:
            for i in range(n):
                row[i] ^= M[b, i]
    phi = np.uint64(top) << np.uint64(lo)
    m = 0
    for k in range(1 << lo):
        if k > 0:
            b = 0
            kk = k
            while (kk & 1) == 0:
                kk >>= 1
                b += 1
            phi ^= np.uint64(1) << np.uint64(b)
            for i in range(n):
                row[i] ^= M[b, i]
        if phi == 0:
            continue
        r = 0
        ok = True
        for i in range(n):
            v = row[i]
            for t in range(r):
                if v ^ piv[t] < v:      # piv[t]'s top bit is set in v
                    v ^= piv[t]
            if v != 0:
                piv[r] = v
                r += 1
                if r > rmax:
                    ok = False
                    break
                # keep pivots sorted by top bit, descending, insertion
                t = r - 1
                while t > 0 and piv[t] > piv[t - 1]:
                    tmp = piv[t]; piv[t] = piv[t - 1]; piv[t - 1] = tmp
                    t -= 1
        if ok:
            if m < out.shape[1]:
                out[c, m] = phi
            m += 1
    cnt[c] = m


@nb.njit(parallel=True, cache=True)
def _lr_all(M, n, ch, rmax, out, cnt):
    lo = n - ch
    for c in nb.prange(1 << ch):
        _lr_chunk(M, n, lo, c, rmax, out, cnt, c)


def lowrank(tab, rmax, ch=CH, cap=CAP):
    """all nonzero f with rank(B_f) <= rmax on the whole of L, B_f(x,y) = f([x,y])."""
    n = len(tab)
    ch = min(ch, n - 1)
    M = np.zeros((n, n), dtype=np.uint64)
    for b in range(n):
        for i in range(n):
            v = 0
            for j in range(n):
                if (tab[i][j] >> b) & 1:
                    v |= 1 << j
            M[b, i] = v
    out = np.zeros((1 << ch, cap), dtype=np.uint64)
    cnt = np.zeros(1 << ch, dtype=np.int64)
    _lr_all(M, n, ch, rmax, out, cnt)
    return _collect(out, cnt)


@nb.njit(cache=True)
def _nilr_chunk(T, idx, n, lo, top, out, cnt, c):
    r = idx.shape[0]
    adx = np.zeros(n, dtype=np.uint64)
    for i in range(lo, r):
        if (top >> (i - lo)) & 1:
            for j in range(n):
                adx[j] ^= T[idx[i], j]
    x = np.uint64(0)
    for i in range(lo, r):
        if (top >> (i - lo)) & 1:
            x ^= np.uint64(1) << np.uint64(idx[i])
    m = 0
    for k in range(1 << lo):
        if k > 0:
            b = 0
            kk = k
            while (kk & 1) == 0:
                kk >>= 1
                b += 1
            x ^= np.uint64(1) << np.uint64(idx[b])
            for j in range(n):
                adx[j] ^= T[idx[b], j]
        if x == 0:
            continue
        ok = True
        for j in range(n):
            v = adx[j]
            w = np.uint64(0)
            while v:
                t = v & (~v + np.uint64(1))
                bb = 0
                tt = t
                while tt > 1:
                    tt >>= np.uint64(1)
                    bb += 1
                w ^= adx[bb]
                v ^= t
            if w != 0:
                ok = False
                break
        if ok:
            if m < out.shape[1]:
                out[c, m] = x
            m += 1
    cnt[c] = m


@nb.njit(parallel=True, cache=True)
def _nilr_all(T, idx, n, ch, out, cnt):
    lo = idx.shape[0] - ch
    for c in nb.prange(1 << ch):
        _nilr_chunk(T, idx, n, lo, c, out, cnt, c)


def nil2_rows(tab, idx, ch=6, cap=CAP):
    """x in Span{e_i | i in idx} with (ad x)^2 = 0, x != 0."""
    n = len(tab)
    r = len(idx)
    if r == 0:
        return []
    ch = min(ch, r - 1) if r > 1 else 0
    T = _table(tab)
    I = np.array(idx, dtype=np.int64)
    out = np.zeros((1 << ch, cap), dtype=np.uint64)
    cnt = np.zeros(1 << ch, dtype=np.int64)
    _nilr_all(T, I, n, ch, out, cnt)
    return _collect(out, cnt)
