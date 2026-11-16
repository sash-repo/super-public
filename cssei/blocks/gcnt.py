#!/usr/bin/env python3
"""Block-by-block CSS invariants of a Z-graded Lie algebra over F_2 (dim <= 64), numba version
of gcount.py.  For the lowest nonzero component x_c (enumerated over the cone of L_c), each later
component x_m solves a LINEAR system: the part of degree m+c of (ad x)^2 (+ ad x, toral case).
For fixed x_c that system's matrix is fixed, so it is eliminated once; per node only the right-hand
side is evaluated.  Level 0 of each subtree is split into SS residue classes when the cone is small.
usage: gcnt.py TABLE.npz nil|tor [workers]"""
import sys, os, time
import numpy as np, numba as nb
from multiprocessing import Pool

U = np.uint64

@nb.njit(cache=True)
def apply(A, v):
    r = U(0); k = 0
    while v:
        if v & U(1):
            r ^= A[k]
        v >>= U(1); k += 1
    return r

@nb.njit(cache=True)
def adv(adm, x, d):
    A = np.zeros(d, np.uint64)
    for i in range(d):
        if (x >> U(i)) & U(1):
            for j in range(d):
                A[j] ^= adm[i, j]
    return A

@nb.njit(cache=True)
def Pof(A, d, tor):
    P = np.zeros(d, np.uint64)
    for j in range(d):
        P[j] = apply(A, A[j])
        if tor:
            P[j] ^= A[j]
    return P

@nb.njit(cache=True)
def prep_level(adm, d, Ac, idxm, nt, rm, tor, c0):
    """rm[j] = row mask of degree-k entries in column j.  Returns:
    piv_s(j,r) arrays of pivot slots, lead unknown per pivot, comb masks for particular solution,
    consistency slots with comb masks, kernel vectors (as unknown masks)."""
    # T_t columns
    T = np.zeros((nt, d), np.uint64)
    for t in range(nt):
        e = adm[idxm[t]]
        for j in range(d):
            v = apply(Ac, e[j]) ^ apply(e, Ac[j])
            if c0:
                v ^= e[j]
            T[t, j] = v & rm[j]
    # echelon over slots
    bv = np.zeros(nt, np.uint64)     # coefficient masks (over unknowns) of basis rows
    bc = np.zeros(nt, np.uint64)     # comb masks over pivot-slot numbers
    blead = np.zeros(nt, np.int64)
    pj = np.zeros(nt, np.int64); pr = np.zeros(nt, np.int64)
    nb_ = 0
    cj = np.zeros(d * 64, np.int64); cr = np.zeros(d * 64, np.int64); cc = np.zeros(d * 64, np.uint64)
    nc = 0
    for j in range(d):
        m = rm[j]
        r = 0
        while m:
            if m & U(1):
                v = U(0)
                for t in range(nt):
                    if (T[t, j] >> U(r)) & U(1):
                        v |= U(1) << U(t)
                comb = U(0)
                for b in range(nb_):
                    if (v >> U(blead[b])) & U(1):
                        v ^= bv[b]; comb ^= bc[b]
                if v:
                    lead = 0
                    while not ((v >> U(lead)) & U(1)):
                        lead += 1
                    # reduce existing rows on this lead
                    for b in range(nb_):
                        if (bv[b] >> U(lead)) & U(1):
                            bv[b] ^= v; bc[b] ^= comb ^ (U(1) << U(nb_))
                    bv[nb_] = v; bc[nb_] = comb ^ (U(1) << U(nb_)); blead[nb_] = lead
                    pj[nb_] = j; pr[nb_] = r
                    nb_ += 1
                elif comb != U(0) or True:
                    cj[nc] = j; cr[nc] = r; cc[nc] = comb; nc += 1
            m >>= U(1); r += 1
    # kernel: free unknowns
    free = np.ones(nt, np.bool_)
    for b in range(nb_):
        free[blead[b]] = False
    K = np.zeros(nt, np.uint64); nk = 0
    for f in range(nt):
        if free[f]:
            v = U(1) << U(f)
            for b in range(nb_):
                if (bv[b] >> U(f)) & U(1):
                    v |= U(1) << U(blead[b])
            K[nk] = v; nk += 1
    return nb_, bv, bc, blead, pj, pr, nc, cj, cr, cc, nk, K

@nb.njit(cache=True)
def tomask(y, idxm, nt):
    x = U(0)
    for t in range(nt):
        if (y >> U(t)) & U(1):
            x |= U(1) << U(idxm[t])
    return x

@nb.njit(cache=True)
def count_from(adm, d, xc, levels_idx, levels_nt, rms, tor, c0, rr, SS):
    """levels_idx[L, maxnt], rms[L, d] for the levels after c."""
    L = levels_nt.shape[0]
    Ac = adv(adm, xc, d)
    if L == 0:
        if rr != 0:
            return 0
        P = Pof(Ac, d, tor)
        for j in range(d):
            if P[j]:
                return 0
        return 1
    # prepare all levels
    maxnt = levels_idx.shape[1]
    NB = np.zeros(L, np.int64); BV = np.zeros((L, maxnt), np.uint64); BC = np.zeros((L, maxnt), np.uint64)
    BL = np.zeros((L, maxnt), np.int64); PJ = np.zeros((L, maxnt), np.int64); PR = np.zeros((L, maxnt), np.int64)
    NC = np.zeros(L, np.int64); CJ = np.zeros((L, d * 64), np.int64); CR = np.zeros((L, d * 64), np.int64)
    CC = np.zeros((L, d * 64), np.uint64); NK = np.zeros(L, np.int64); KK = np.zeros((L, maxnt), np.uint64)
    KA = np.zeros((L, maxnt, d), np.uint64)
    for l in range(L):
        nt = levels_nt[l]
        nb_, bv, bc, bl, pj, pr, nc, cj, cr, cc, nk, K = prep_level(adm, d, Ac, levels_idx[l], nt, rms[l], tor, c0)
        NB[l] = nb_; NC[l] = nc; NK[l] = nk
        BV[l, :nt] = bv; BC[l, :nt] = bc; BL[l, :nt] = bl; PJ[l, :nt] = pj; PR[l, :nt] = pr
        CJ[l, :nc] = cj[:nc]; CR[l, :nc] = cr[:nc]; CC[l, :nc] = cc[:nc]
        KK[l, :nt] = K
        for t in range(nk):
            KA[l, t] = adv(adm, tomask(K[t], levels_idx[l], nt), d)
    # DFS with explicit stack
    Astk = np.zeros((L + 1, d), np.uint64)
    cnt = np.zeros(L, np.int64)     # gray counter per level
    Astk[0] = Ac
    base0 = np.zeros(d, np.uint64)
    total = 0
    l = 0
    entered = np.zeros(L, np.bool_)
    while l >= 0:
        if not entered[l]:
            # compute rhs, consistency, particular solution
            A = Astk[l]
            P = Pof(A, d, tor)
            ok = True
            for s in range(NC[l]):
                comb = CC[l, s]
                val = (P[CJ[l, s]] >> U(CR[l, s])) & U(1)
                for b in range(NB[l]):
                    if (comb >> U(b)) & U(1):
                        val ^= (P[PJ[l, b]] >> U(PR[l, b])) & U(1)
                if val:
                    ok = False; break
            if not ok:
                l -= 1
                continue
            y = U(0)
            for b in range(NB[l]):
                comb = BC[l, b]; val = U(0)
                for b2 in range(NB[l]):
                    if (comb >> U(b2)) & U(1):
                        val ^= (P[PJ[l, b2]] >> U(PR[l, b2])) & U(1)
                if val:
                    y |= U(1) << U(BL[l, b])
            Ay = adv(adm, tomask(y, levels_idx[l], levels_nt[l]), d)
            for j in range(d):
                Astk[l + 1, j] = A[j] ^ Ay[j]
            entered[l] = True; cnt[l] = 0
            if l == 0:
                for j in range(d):
                    base0[j] = Astk[1, j]
                if rr >= (1 << NK[0]):
                    entered[0] = False
                    l -= 1
                    continue
                cnt[0] = rr
                for b in range(NK[0]):
                    if (rr >> b) & 1:
                        for j in range(d):
                            Astk[1, j] ^= KA[0, b, j]
        elif l == 0:
            cnt[0] += SS
            if cnt[0] >= (1 << NK[0]):
                entered[0] = False
                l -= 1
                continue
            for j in range(d):
                Astk[1, j] = base0[j]
            for b in range(NK[0]):
                if (cnt[0] >> b) & 1:
                    for j in range(d):
                        Astk[1, j] ^= KA[0, b, j]
        else:
            # next gray step
            cnt[l] += 1
            if cnt[l] >= (1 << NK[l]):
                entered[l] = False
                l -= 1
                continue
            g = cnt[l]; b = 0
            while not ((g >> b) & 1):
                b += 1
            for j in range(d):
                Astk[l + 1, j] ^= KA[l, b, j]
        if l == L - 1:
            P = Pof(Astk[l + 1], d, tor)
            good = True
            for j in range(d):
                if P[j]:
                    good = False; break
            if good:
                total += 1
        else:
            l += 1
    return total

@nb.njit(cache=True)
def cone(adm, d, idx, n, tor, low, mask0):
    """members x of L_c (idx) with the lowest-component condition."""
    out = np.zeros(1 << 22, np.uint64); no = 0
    for s in range(1, 1 << n):
        x = U(0)
        for t in range(n):
            if (s >> t) & 1:
                x |= U(1) << U(idx[t])
        A = adv(adm, x, d)
        good = True
        for j in range(d):
            v = apply(A, A[j])
            if tor and not low:
                v ^= A[j]
                v &= mask0[j]
            if v:
                good = False; break
        if good:
            out[no] = x; no += 1
    return out[:no]

G = {}
def setup(path, kind):
    z = np.load(path); ad = z['ad'].astype(np.uint8); deg = z['deg']
    d = len(deg)
    adm = np.zeros((d, d), np.uint64)
    for i in range(d):
        for j in range(d):
            col = ad[i][:, j]
            adm[i, j] = int(sum(int(col[k]) << k for k in range(d)))
    D = sorted(set(deg.tolist()))
    idx = {m: np.nonzero(deg == m)[0] for m in D}
    def rowmask(k):
        rm = np.zeros(d, np.uint64)
        for j in range(d):
            rm[j] = int(sum(1 << r for r in range(d) if deg[r] == deg[j] + k))
        return rm
    G.update(adm=adm, d=d, deg=deg, D=D, idx=idx, rowmask=rowmask, tor=(kind == 'tor'))

def levels_for(c):
    D = G['D']; later = [m for m in D if m > c]
    maxnt = max([len(G['idx'][m]) for m in later] + [1])
    li = np.zeros((len(later), maxnt), np.int64); ln = np.zeros(len(later), np.int64)
    rms = np.zeros((len(later), G['d']), np.uint64)
    for l, m in enumerate(later):
        li[l, :len(G['idx'][m])] = G['idx'][m]; ln[l] = len(G['idx'][m]); rms[l] = G['rowmask'](m + c)
    return li, ln, rms

def work(args):
    path, kind, c, xs, rr, SS = args
    if not G: setup(path, kind)
    li, ln, rms = levels_for(c)
    return sum(count_from(G['adm'], G['d'], U(int(x)), li, ln, rms, G['tor'], G['tor'] and c == 0, rr, SS) for x in xs)

if __name__ == '__main__':
    path, kind = sys.argv[1], sys.argv[2]; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    setup(path, kind)
    print(path, kind, 'dim', G['d'], {m: len(G['idx'][m]) for m in G['D']}, flush=True)
    total = 1; t0 = time.time()
    with Pool(nw, initializer=setup, initargs=(path, kind)) as P:
        for c in G['D']:
            if G['tor'] and c > 0: break
            I = G['idx'][c].astype(np.int64)
            mask0 = G['rowmask'](0)
            xs = cone(G['adm'], G['d'], I, len(I), G['tor'], c < 0, mask0)
            nch = max(1, 16 * nw); SS = max(1, -(-nch // max(1, len(xs))))   # split level 0 when the cone is small
            ncx = max(1, nch // SS)
            chunks = [(path, kind, c, xs[i::ncx], r, SS) for i in range(ncx) for r in range(SS)]
            sub = sum(P.map(work, chunks))
            total += sub
            print('  lowest degree %d: cone %d, contributes %d, total %d, %.0fs' % (c, len(xs), sub, total, time.time() - t0), flush=True)
    print(kind, total)
