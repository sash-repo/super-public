#!/usr/bin/env python3
"""grade24.py -- (2,4)-structures on the Hamiltonian algebras of ham7.py.

A Z/2-grading g = g_+ (+) g_- of a Lie algebra over F_2 is the same as a
derivation D with D^2 = D (D = 0 on g_+, 1 on g_-).  A (2,4)-structure on a
centerless g exists for this grading iff
   (ad x)^2 in ad g for every x in g_+ (a subspace condition, Jacobson),
   (ad y)^4 in ad g for y in a basis of g_- (the rest follows from the
   (2,4)-additivity, the cross term being (ad[x,y])^2 with [x,y] in g_+).
We enumerate ALL idempotent derivations (Der computed as a nullspace).
Called from ham7.py; prints nothing itself.
"""
import itertools
import numpy as np
import ham7 as H


def nullspace(rows, nvars):
    """rows: list of int bitmasks (equations); returns a basis of solutions (ints)"""
    piv = {}            # pivot col -> row (reduced)
    for r in rows:
        for c, pr in piv.items():
            if (r >> c) & 1:
                r ^= pr
        if r:
            c = r.bit_length() - 1
            for c2 in list(piv):
                if (piv[c2] >> c) & 1:
                    piv[c2] ^= r
            piv[c] = r
    free = [c for c in range(nvars) if c not in piv]
    basis = []
    for f in free:
        v = 1 << f
        for c, pr in piv.items():
            if (pr >> f) & 1:
                v |= 1 << c
        basis.append(v)
    return basis


def derivations(G):
    d, br = G["d"], G["br"]
    var = lambda r, c: c * d + r          # coefficient of x_r in D(x_c)
    eqs = []
    for a in range(d):
        for b in range(a + 1, d):
            for r in range(d):
                e = 0
                for k in H.bits(br[a][b]):
                    e ^= 1 << var(r, k)
                for s in range(d):
                    if (br[s][b] >> r) & 1:
                        e ^= 1 << var(s, a)
                    if (br[a][s] >> r) & 1:
                        e ^= 1 << var(s, b)
                if e:
                    eqs.append(e)
    return nullspace(eqs, d * d)


def to_mat(v, d):
    M = np.zeros((d, d), dtype=np.uint8)      # M[r,c]
    for c in range(d):
        col = (v >> (c * d)) & ((1 << d) - 1)
        for r in H.bits(col):
            M[r, c] = 1
    return M


def idempotent_derivations(G, chunk=1 << 15):
    d = G["d"]
    Dr = derivations(G)
    B = np.array([to_mat(v, d) for v in Dr], dtype=np.int64)   # k x d x d
    k = len(Dr)
    found = []
    total = 1 << k
    for start in range(1, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        C = ((idx[:, None] >> np.arange(k)) & 1).astype(np.int64)  # n x k
        D = np.einsum("nk,kij->nij", C, B) % 2
        D2 = np.einsum("nij,njl->nil", D, D) % 2
        ok = np.all((D2 == D).reshape(len(idx), -1), axis=1)
        for t in np.nonzero(ok)[0]:
            found.append(D[t].copy())
    return k, found


def subspaces(D):
    """g_- = image of D, g_+ = kernel of D, as lists of int vectors"""
    d = D.shape[0]
    img = [sum(int(D[r, c]) << r for r in range(d)) for c in range(d)]
    imb = H.Span()
    gm = []
    for v in img:
        if v and imb.add(v):
            gm.append(v)
    # kernel
    I = np.eye(d, dtype=np.int64)
    K = (I + D) % 2                         # projection onto ker D (D idempotent)
    kim = [sum(int(K[r, c]) << r for r in range(d)) for c in range(d)]
    kb = H.Span()
    gp = []
    for v in kim:
        if v and kb.add(v):
            gp.append(v)
    return gp, gm


def power_inner_vec(G, v, k, A):
    d = G["d"]
    M = H.vad_cols(G, v)
    P = M
    for _ in range(k):
        P = H.compose(P, P)
    return A.express(H.flat(P, d))


def structures24(G):
    """list of (g_+, g_-, pmap2 on g_+ basis, pmap4 on g_- basis) for every
    idempotent derivation admitting a (2,4)-structure"""
    A = H.ad_span(G)
    k, Ds = idempotent_derivations(G)
    res = []
    stats = dict(dimDer=k, n_idem=len(Ds), n_ok=0)
    for D in Ds:
        gp, gm = subspaces(D)
        if not gm or not gp:
            continue
        p2 = [power_inner_vec(G, v, 1, A) for v in gp]
        p4 = [power_inner_vec(G, v, 2, A) for v in gm]
        if all(z is not None for z in p2) and all(z is not None for z in p4):
            res.append((gp, gm, p2, p4))
    stats["n_ok"] = len(res)
    return stats, res
