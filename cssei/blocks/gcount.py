#!/usr/bin/env python3
"""CSS invariants  #{x | (ad x)^2 = 0}  and  #{x | (ad x)^2 = ad x}  (zero included) of a
Z-graded Lie algebra over F_2, block by block with respect to the grading.
x = sum x_m; with c the lowest degree where x_c != 0, the part of degree m + c of the
condition is the first one that involves x_m, and it is LINEAR in x_m:
   ad x_c ad x_m + ad x_m ad x_c (+ ad x_m if c = 0, toral case) = terms in x_i, i < m.
So x_m runs over an affine space; the lowest component runs over the cone of L_c, enumerated.
usage: gcount.py TABLE.npz [nil|tor|both]"""
import sys, time
import numpy as np

def f2solve(A, b):
    """Solutions of A y = b over F_2: (particular, kernel basis rows) or None."""
    A = A.astype(np.uint8) & 1; b = b.astype(np.uint8) & 1
    m, n = A.shape
    M = np.hstack([A, b[:, None]])
    piv = []; r = 0
    for c in range(n):
        if r == m: break
        rows = np.nonzero(M[r:, c])[0]
        if len(rows) == 0: continue
        p = r + rows[0]
        if p != r: M[[r, p]] = M[[p, r]]
        nz = np.nonzero(M[:, c])[0]; nz = nz[nz != r]
        M[nz] ^= M[r]
        piv.append(c); r += 1
    if M[r:, n].any(): return None
    part = np.zeros(n, np.uint8)
    for i, c in enumerate(piv): part[c] = M[i, n]
    free = [c for c in range(n) if c not in set(piv)]
    K = np.zeros((len(free), n), np.uint8)
    for t, f in enumerate(free):
        K[t, f] = 1
        for i, c in enumerate(piv): K[t, c] = M[i, f]
    return part, K

class G:
    def __init__(self, ad, deg):
        self.ad = ad.astype(np.int64); self.deg = np.asarray(deg); self.d = len(deg)
        self.D = sorted(set(self.deg.tolist()))
        self.idx = {m: np.nonzero(self.deg == m)[0] for m in self.D}
        diff = self.deg[:, None] - self.deg[None, :]     # entry (r, c): degree of the operator
        self.mask = {k: (diff == k) for k in range(2 * self.D[0] - 1, 2 * self.D[-1] + 2)}
    def adv(self, x):   # x: full vector
        nz = np.nonzero(x)[0]
        return self.ad[nz].sum(axis=0) & 1 if len(nz) else np.zeros((self.d, self.d), np.int64)

def run(g, kind):
    D = g.D; top = D[-1]
    total = [1]   # x = 0
    nodes = [0]
    def rest_ok(A, xfull, c):
        P = (A @ A) & 1
        if kind == 'tor': P ^= A
        return not P.any()
    def rec(pos, c, x, A):
        # x: current full vector (components < D[pos] set), A = ad x; c = lowest nonzero degree
        nodes[0] += 1
        if pos == len(D):
            if rest_ok(A, x, c): total[0] += 1
            return
        m = D[pos]; I = g.idx[m]
        k = m + c
        mk = g.mask.get(k)
        P = (A @ A) & 1
        if kind == 'tor': P ^= A
        rhs = P[mk]                     # current value of the degree-k part, must be cancelled
        Ac = g.adv(np.where(g.deg == c, x, 0))
        cols = []
        for e in I:
            E = g.ad[e]
            T = (Ac @ E + E @ Ac) & 1
            if kind == 'tor' and c == 0: T ^= E
            cols.append(T[mk])
        if not len(rhs):
            # no entries of this degree: x_m free
            sol = (np.zeros(len(I), np.uint8), np.eye(len(I), dtype=np.uint8))
        else:
            sol = f2solve(np.array(cols).T, rhs)
            if sol is None: return
        part, K = sol
        # checks of degree < k which now involve x_m?  none: every other product with x_m has degree > k.
        if pos == len(D) - 1:
            # last component: enumerate the affine space vectorized, check everything
            kdim = len(K)
            for s in range(0, 1 << kdim, 1 << 12):
                ss = np.arange(s, min(1 << kdim, s + (1 << 12)))
                bits = ((ss[:, None] >> np.arange(kdim)) & 1).astype(np.int64)
                Y = (part[None, :] + bits @ K) & 1
                Ym = np.zeros((len(ss), g.d), np.int64); Ym[:, I] = Y
                AA = (A[None] + np.einsum('ni,ijk->njk', Ym, g.ad)) & 1
                PP = np.einsum('nij,njk->nik', AA, AA) & 1
                if kind == 'tor': PP ^= AA
                total[0] += int((~PP.reshape(len(ss), -1).any(axis=1)).sum())
                nodes[0] += len(ss)
            return
        kdim = len(K)
        for s in range(1 << kdim):
            y = part.copy()
            for t in range(kdim):
                if s >> t & 1: y ^= K[t]
            x2 = x.copy(); x2[I] = y
            A2 = (A + g.adv(np.where(g.deg == m, x2, 0))) & 1
            rec(pos + 1, c, x2, A2)
    # the lowest component
    for pos, m in enumerate(D):
        if kind == 'tor' and m > 0: break
        I = g.idx[m]; n = len(I)
        t0 = time.time(); ncone = 0
        B = 1 << min(n, 12)
        for s in range(1, 1 << n, B):
            ss = np.arange(s, min(1 << n, s + B))
            bits = ((ss[:, None] >> np.arange(n)) & 1).astype(np.int64)
            AA = np.einsum('ni,ijk->njk', bits, g.ad[I]) & 1
            PP = np.einsum('nij,njk->nik', AA, AA) & 1
            if kind == 'tor':
                if m < 0: ok = ~PP.reshape(len(ss), -1).any(axis=1)        # (ad x_c)^2 = ad x_{2c} = 0
                else: ok = ~(PP ^ AA).reshape(len(ss), -1).any(axis=1)     # c = 0: (ad x_0)^2 = ad x_0 on degree 0
                if m == 0: ok = ~((PP ^ AA)[:, g.mask[0]]).any(axis=1)
            else:
                ok = ~(PP[:, g.mask[2 * m]]).any(axis=1)
            for b in bits[ok]:
                ncone += 1
                x = np.zeros(g.d, np.int64); x[I] = b
                rec(pos + 1, m, x, g.adv(x))
        print('  lowest degree %d: cone %d, running total %d, nodes %d, %.0fs' % (m, ncone, total[0], nodes[0], time.time() - t0), flush=True)
    return total[0]

if __name__ == '__main__':
    z = np.load(sys.argv[1]); kinds = sys.argv[2] if len(sys.argv) > 2 else 'both'
    g = G(z['ad'], z['deg'])
    print(sys.argv[1], 'dim', g.d, 'degrees', {m: len(g.idx[m]) for m in g.D}, flush=True)
    for kind in (['nil', 'tor'] if kinds == 'both' else [kinds]):
        t0 = time.time()
        print(kind, run(g, kind), '%.0fs' % (time.time() - t0), flush=True)
