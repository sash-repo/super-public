#!/usr/bin/env python3
"""Derived series of the finite prolongs as Lie SUPERalgebras: D(S) = [S,S] + squares of odd elements of S.
In the text: Theorem 4.6, 2)-3)."""
import numpy as np, sys
from f2lin import solve, rank, rref, nullspace
from lalg import *
from buildalg import Graded, assoc_dim
from runprolong import parity, conj, Din, Dout, mods
from supercore import square
from analyze import sub_table
M = mods[(1, 0)]; n = 7; Id = np.eye(n, dtype=np.int64)
def sder(G, S, P):
    """S rows (basis of a subalgebra), P parity operator matrix (column convention, P v = v for odd v)"""
    Ci = G.C.astype(np.int64)
    br = [np.einsum('i,j,ijk->k', a, b, Ci) & 1 for a in S for b in S]
    # odd part of span S
    img = np.array([(P @ v) & 1 for v in S])
    coords = np.array([solve(S.T.astype(np.uint8), (r ^ v).astype(np.uint8)) for r, v in zip(img, S)]).astype(np.int64)
    K = nullspace(coords.T.astype(np.uint8), len(S)) if False else nullspace(coords.T.astype(np.uint8).T.T, len(S))
    return br, coords
def series(G, P):
    S = np.eye(G.d, dtype=np.int64); out = []
    while True:
        Ci = G.C.astype(np.int64)
        br = [np.einsum('i,j,ijk->k', a, b, Ci) & 1 for a in S for b in S]
        # odd vectors in span S: v = cS with (P+1)v = 0
        Q = ((S @ P.T) & 1) ^ S                     # rows (P+1) s_i
        c = nullspace(Q.T.astype(np.uint8), len(S))   # combos with sum c_i (P+1)s_i = 0
        odd = (c.astype(np.int64) @ S) & 1 if len(c) else np.zeros((0, G.d), dtype=np.int64)
        sq = [square(G, v).astype(np.int64) for v in odd]
        S2 = rref(np.vstack(br + sq).astype(np.uint8))[0].astype(np.int64)
        out.append(len(S2))
        if len(S2) == len(S): return out, S2
        S = S2
cases = []
for nm, g0 in [('Pi(V7)+gs(2)', [M[k] for k in range(NG)]), ('Pi(V7)+c gs(2)', [M[k] for k in range(NG)] + [Id]),
               ('Pi(V7)+L', list(M)), ('Pi(V7)+c L', list(M) + [Id])]:
    G = Graded(g0, n, tuple(range(n)))
    P = np.diag([(G.deg[i] % 2) for i in range(G.d)]).astype(np.int64)
    cases.append((nm, G, P))
for sname, Dv, sub in [('s_a', Din, [k for k in range(N) if NAMES[k] != '(u2)^2']), ('s_b', Dout, list(range(N)))]:
    P_, odd = parity(M, Dv); Mc = conj(M, P_)
    g0 = [Mc[k] for k in sub]; Dm = sum(Mc[k] for k in np.nonzero(Dv)[0]) & 1
    for extra in ([], [Id]):
        G = Graded(g0 + extra, n, tuple(odd))
        A = np.array([m.flatten() for m in G.levels[0]], dtype=np.uint8).T
        c = solve(A, Dm.flatten() & 1); x = np.zeros(G.d, dtype=np.int64); x[G.off[1]:G.off[2]] = c
        adD = np.einsum('i,ijk->kj', x, G.C.astype(np.int64)) & 1
        cases.append((('c ' if extra else '') + sname + ' + Pi(V7)(3|4)', G, adD))
for nm, G, P in cases:
    ser, S = series(G, P)
    k = len(S)
    Q = ((S @ P.T) & 1)
    co = np.array([solve(S.T.astype(np.uint8), r.astype(np.uint8)) for r in Q]).astype(np.int64)
    od = k - rank((co ^ np.eye(k, dtype=np.int64)).astype(np.uint8))
    T = sub_table(G.C, S)
    print('%-26s total %s  super-derived series %s  core sdim (%d|%d)  assoc(ad core) %d (k^2 = %d)' % (
        nm, G.dims, ser, k - od, od, assoc_dim(T), k * k), flush=True)
