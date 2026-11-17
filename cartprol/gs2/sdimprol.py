#!/usr/bin/env python3
"""Superdimensions of the finite super-prolongs: parity = eigenvalue of ad D, D in g_0 the grading
element (D = u1d1 for s_a, the outer toral D for s_b), in the prolong built by buildalg.Graded.
In the text: Table (3), superdimensions of the finite super-prolongs."""
import numpy as np, pickle, sys
from f2lin import solve, rank, rref
from lalg import *
from buildalg import Graded
from runprolong import parity, conj, Din, Dout, mods
M = mods[(1, 0)]; n = 7; Id = np.eye(n, dtype=np.int64)
def sd(G, Dmat):
    # coordinates of Dmat in g_0 (rows of G.levels[0] are the g_0 basis matrices)
    A = np.array([m.flatten() for m in G.levels[0]], dtype=np.uint8).T
    c = solve(A, Dmat.flatten() & 1); assert c is not None
    x = np.zeros(G.d, dtype=np.int64); x[G.off[1]:G.off[2]] = c
    adD = np.einsum('i,ijk->kj', x, G.C.astype(np.int64)) & 1       # column j = [D, e_j]
    assert (((adD @ adD) & 1) == adD).all()
    out = []
    for k in range(len(G.dims)):
        idx = range(G.off[k], G.off[k + 1])
        B = adD[np.ix_(idx, idx)]
        # blocks: ad D preserves degree
        e = len(idx) - rank((B).astype(np.uint8)); o = len(idx) - rank((B ^ np.eye(len(idx), dtype=np.int64)).astype(np.uint8))
        out.append((e, o))
    return out
for sname, D, sub in [('s_a', Din, [k for k in range(N) if NAMES[k] != '(u2)^2']), ('s_b', Dout, list(range(N)))]:
    P, odd = parity(M, D); Mc = conj(M, P)
    g0 = [Mc[k] for k in sub]
    Dm = sum(Mc[k] for k in np.nonzero(D)[0]) & 1
    for extra in ([], [Id]):
        G = Graded(g0 + extra, n, tuple(odd))
        comps = sd(G, Dm)
        tot = (sum(a for a, b in comps), sum(b for a, b in comps))
        print(sname + ('+K1' if extra else ''), 'V7: components (g_-1, g_0, ...) sdims', comps, ' total', tot)
    # the Lie algebra on Pi(V7) with g0 = L: parity = degree (g_{-1} odd)
