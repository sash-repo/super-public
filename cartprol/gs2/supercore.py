#!/usr/bin/env python3
"""Super-derived algebra (brackets + squares of odd elements) of the finite super-prolongs.
In the text: Theorem 4.6, 2)-3) (super cores of sdim (14|14), (14|16), (15|16))."""
import numpy as np, pickle, sys
from f2lin import solve, rank, rref, nullspace
from lalg import *
from buildalg import Graded, assoc_dim
from runprolong import parity, conj, Din, Dout, mods
from analyze import derived
M = mods[(1, 0)]; n = 7; Id = np.eye(n, dtype=np.int64)
def square(G, x):
    """s(x) with ad s(x) = (ad x)^2 (G centerless)"""
    A = np.einsum('i,ijk->kj', x, G.C.astype(np.int64)) & 1
    A2 = (A @ A) & 1
    adb = np.array([G.C[i].T.flatten() for i in range(G.d)], dtype=np.uint8).T
    s = solve(adb, A2.flatten()); return s
def analyse(name, G, parvec):
    D = derived(G.C).astype(np.int64)
    sq = []
    for i in range(G.d):
        if parvec[i]:
            e = np.zeros(G.d, dtype=np.int64); e[i] = 1
            s = square(G, e)
            if s is None: print('   square of basis', i, 'not in G!'); continue
            sq.append(s.astype(np.int64))
    S = rref(np.vstack([D] + sq).astype(np.uint8))[0]
    comp = [rank(S[:, G.off[k]:G.off[k + 1]]) for k in range(len(G.dims))]
    print('%-22s bracket-derived %d, with squares %d, by degree %s' % (name, len(D), len(S), comp))
    return S
# Pi(V7) with g0 = gs(2), c(gs2), L, c(L): parity = odd degree
for nm, g0 in [('Pi(V7)+gs(2)', [M[k] for k in range(NG)]), ('Pi(V7)+c gs(2)', [M[k] for k in range(NG)] + [Id]),
               ('Pi(V7)+L', list(M)), ('Pi(V7)+c L', list(M) + [Id])]:
    G = Graded(g0, n, tuple(range(n)))
    par = np.array([(G.deg[i] % 2) for i in range(G.d)])
    analyse(nm, G, par)
for sname, Dv, sub in [('s_a', Din, [k for k in range(N) if NAMES[k] != '(u2)^2']), ('s_b', Dout, list(range(N)))]:
    P, odd = parity(M, Dv); Mc = conj(M, P)
    g0 = [Mc[k] for k in sub]; Dm = sum(Mc[k] for k in np.nonzero(Dv)[0]) & 1
    G = Graded(g0, n, tuple(odd))
    A = np.array([m.flatten() for m in G.levels[0]], dtype=np.uint8).T
    c = solve(A, Dm.flatten() & 1); x = np.zeros(G.d, dtype=np.int64); x[G.off[1]:G.off[2]] = c
    adD = np.einsum('i,ijk->kj', x, G.C.astype(np.int64)) & 1
    # parity-homogeneous basis: eigenvectors; odd part basis
    od = nullspace((adD ^ np.eye(G.d, dtype=np.int64)).T.astype(np.uint8), G.d)   # rows v with v adD = v ? use columns
    # odd vectors: adD v = v  -> (adD + 1) v = 0
    od = nullspace((adD ^ np.eye(G.d, dtype=np.int64)).astype(np.uint8), G.d)
    Dd = derived(G.C).astype(np.int64)
    sq = []
    for v in od:
        s = square(G, v.astype(np.int64))
        if s is not None: sq.append(s.astype(np.int64))
    # also s(u+v) = s(u)+s(v)+[u,v], brackets already in Dd
    S = rref(np.vstack([Dd] + sq).astype(np.uint8))[0]
    comp = [rank(S[:, G.off[k]:G.off[k + 1]]) for k in range(len(G.dims))]
    print('%-22s bracket-derived %d, with squares %d, by degree %s' % ('V7(3|4)+' + sname, len(Dd), len(S), comp))

# ---- simplicity and sdim of the super cores
from analyze import sub_table
def core_info(G, S, adD):
    T = sub_table(G.C, S.astype(np.int64))
    a = assoc_dim(T)
    # sdim of the core: restrict adD to span S (S invariant)
    k = len(S)
    Sm = S.astype(np.int64)
    img = (Sm @ adD.T) & 1          # rows: adD applied to basis rows (adD column convention: adD @ v)
    coords = np.array([solve(Sm.T.astype(np.uint8), r.astype(np.uint8)) for r in img]).astype(np.int64)
    ev = k - rank(coords.astype(np.uint8)); od = k - rank((coords ^ np.eye(k, dtype=np.int64)).astype(np.uint8))
    return a, k, (ev, od)
print('--- cores')
for nm, g0 in [('Pi(V7)+gs(2)', [M[k] for k in range(NG)])]:
    G = Graded(g0, n, tuple(range(n)))
    S = derived(G.C)
    par = np.diag([(G.deg[i] % 2) for i in range(G.d)]).astype(np.int64)
    a, k, sd = core_info(G, S, par)
    print(nm, 'core dim', k, 'assoc', a, 'simple' if a == k * k else 'NOT abs. irreducible adjoint', 'sdim', sd)
for sname, Dv, sub in [('s_a', Din, [k for k in range(N) if NAMES[k] != '(u2)^2']), ('s_b', Dout, list(range(N)))]:
    P, odd = parity(M, Dv); Mc = conj(M, P)
    g0 = [Mc[k] for k in sub]; Dm = sum(Mc[k] for k in np.nonzero(Dv)[0]) & 1
    G = Graded(g0, n, tuple(odd))
    A = np.array([m.flatten() for m in G.levels[0]], dtype=np.uint8).T
    c = solve(A, Dm.flatten() & 1); x = np.zeros(G.d, dtype=np.int64); x[G.off[1]:G.off[2]] = c
    adD = np.einsum('i,ijk->kj', x, G.C.astype(np.int64)) & 1
    od = nullspace((adD ^ np.eye(G.d, dtype=np.int64)).astype(np.uint8), G.d)
    Dd = derived(G.C).astype(np.int64)
    sq = [square(G, v.astype(np.int64)).astype(np.int64) for v in od]
    S = rref(np.vstack([Dd] + sq).astype(np.uint8))[0]
    a, k, sd = core_info(G, S, adD)
    # is the core perfect as a superalgebra? derived of core + squares of its odd elements
    T = sub_table(G.C, S.astype(np.int64))
    D2 = derived(T)
    print('V7(3|4)+' + sname, 'core dim', k, 'sdim', sd, 'assoc', a, '=k^2' if a == k * k else '<k^2',
          ' bracket-derived of the core', len(D2))
print('--- do the squares of the odd elements of the 28-dim core generate the super core?')
for sname, Dv, sub in [('s_a', Din, [k for k in range(N) if NAMES[k] != '(u2)^2']), ('s_b', Dout, list(range(N)))]:
    P, odd = parity(M, Dv); Mc = conj(M, P)
    g0 = [Mc[k] for k in sub]; Dm = sum(Mc[k] for k in np.nonzero(Dv)[0]) & 1
    G = Graded(g0, n, tuple(odd))
    A = np.array([m.flatten() for m in G.levels[0]], dtype=np.uint8).T
    c = solve(A, Dm.flatten() & 1); x = np.zeros(G.d, dtype=np.int64); x[G.off[1]:G.off[2]] = c
    adD = np.einsum('i,ijk->kj', x, G.C.astype(np.int64)) & 1
    Dd = derived(G.C).astype(np.int64)
    # odd part of the core Dd: vectors v in span Dd with adD v = v
    Ev = ((Dd @ adD.T) & 1) ^ Dd          # rows: (adD+1) applied to basis vectors of Dd
    K = nullspace(Ev.T.astype(np.uint8) if False else np.array([solve(Dd.T.astype(np.uint8), r.astype(np.uint8)) for r in Ev]).T.astype(np.uint8), len(Dd))
    oddcore = (K.astype(np.int64) @ Dd) & 1
    sq = [square(G, v).astype(np.int64) for v in oddcore]
    S = rref(np.vstack([Dd] + sq).astype(np.uint8))[0]
    print(sname, 'odd part of the 28-core:', len(oddcore), ' core + squares of its odd elements:', len(S))
