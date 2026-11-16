#!/usr/bin/env python3
"""Identification of the deform of k^(1)(3; 1) with er^(1)(3; 1), and then
H^2(g_-; g) for g = er^(1)(3; 1) -- all without consulting [GL3].

The route: the deform g_lam along the weight-(-3;3) cocycle carries a Weisfeiler
subalgebra L_0 of codimension 3 with depth 1 (see deform2.py).  We

  1) build the whole Weisfeiler filtration L_{-1} = g_lam > L_0 > L_1 > ...,
  2) form gr g_lam = (+) L_i/L_{i+1} and read off gr_{-1} and gr_0,
  3) identify gr_0 as an abstract Lie algebra and gr_{-1} as a gr_0-module,
  4) realize the pair (gr_{-1}, gr_0) inside vect(3; 1) and take the generalized
     Cartan prolong -- which must give back gr g_lam.

The list of algebras in the text says er^(1)(3; un) = (O(1; 1), sl(2) (+) hei(2))^(1)_{*,un};
steps 3) and 4) are exactly the check of that description, and they produce the
g_0-action on g_{-1} = O(1; 1) that is needed below.
"""
import numpy as np
from itertools import product

from k3 import P, rref, rank, nullspace, LieAlg
from deform import cocycles, deform, weisfeiler, unit, span_add, contains, g


# --------------------------------------------------------------- linear algebra
def red(rows, n):
    return rref([list(map(int, r)) for r in rows], n)[0] if rows else []


def quotient_basis(big, small, n):
    """indices of a basis of big/small: rows of `big` completing `small`."""
    cur = [r[:] for r in small]
    out = []
    for r in big:
        if span_add(cur, r[:]):
            out.append(r[:])
    return out


def wfiltration(alg, L0):
    """L_1 = {x in L_0 : [x, g] subset L_0}, and so on upward."""
    n = alg.n
    cur = red(L0, n)
    out = [cur]
    while cur:
        nxt = []
        # x in cur with [x, e_i] in cur for all i
        pivots = [next(i for i, a in enumerate(r) if a) for r in cur]

        def inspan(v):
            v = v[:]
            for i, pc in enumerate(pivots):
                if v[pc]:
                    f = v[pc]
                    v = [(a - f * b) % P for a, b in zip(v, cur[i])]
            return not any(v)

        M = []
        comp = [i for i in range(n) if i not in pivots]
        for r in cur:
            row = []
            for i in range(n):
                br = alg.bracket(r, unit(n, i))
                # reduce br modulo cur, keep the free coordinates
                v = br[:]
                for t, pc in enumerate(pivots):
                    if v[pc]:
                        f = v[pc]
                        v = [(a - f * b) % P for a, b in zip(v, cur[t])]
                row.extend(v[c] for c in comp)
            M.append(row)
        NS = nullspace([list(col) for col in zip(*M)], len(cur)) if M and M[0] else \
            [[1 if i == j else 0 for i in range(len(cur))] for j in range(len(cur))]
        nxt = []
        for s in NS:
            v = [0] * n
            for co, r in zip(s, cur):
                if co:
                    v = [(a + co * b) % P for a, b in zip(v, r)]
            if any(v):
                span_add(nxt, v)
        nxt = red(nxt, n)
        if len(nxt) == len(cur):
            raise RuntimeError("the filtration does not descend")
        out.append(nxt)
        cur = nxt
    return out


def structure_on_quotient(alg, top, bot, n):
    """the Lie algebra top/bot (bot an ideal of top in the graded sense)."""
    reps = quotient_basis(top, bot, n)
    basis = red(bot, n)
    piv = [next(i for i, a in enumerate(r) if a) for r in basis] if basis else []

    def modbot(v):
        v = v[:]
        for i, pc in enumerate(piv):
            if v[pc]:
                f = v[pc]
                v = [(a - f * b) % P for a, b in zip(v, basis[i])]
        return v

    full = [r[:] for r in reps] + [r[:] for r in basis]
    R = rref([r[:] for r in full], n)[0]
    pv = [next(i for i, a in enumerate(r) if a) for r in R]

    # coordinates of a vector of `top` in reps + basis
    aug = [r[:] + [1 if i == j else 0 for i in range(len(full))]
           for j, r in enumerate(full)]
    Ra = rref(aug, n + len(full))[0]
    T = [r[n:] for r in Ra]

    def coords(v):
        w = v[:]
        co = [0] * len(Ra)
        for i, pc in enumerate(pv):
            if w[pc]:
                co[i] = w[pc]
                w = [(a - co[i] * b) % P for a, b in zip(w, Ra[i])]
        assert not any(w), "vector outside the subalgebra"
        out = [0] * len(full)
        for i, c in enumerate(co):
            if c:
                for j in range(len(full)):
                    out[j] = (out[j] + c * T[i][j]) % P
        return out[:len(reps)]

    def brk(i, j):
        v = alg.bracket(reps[i], reps[j])
        return {a: c for a, c in enumerate(coords(v)) if c}

    return reps, brk


def describe(nn, brk, names=None):
    """derived series / lower central series of a small Lie algebra."""
    def sub(basis):
        rows = []
        for i in range(len(basis)):
            for j in range(len(basis)):
                v = [0] * nn
                for a, c in brk_vec(basis[i], basis[j]).items():
                    v[a] = c
                if any(v):
                    span_add(rows, v)
        return red(rows, nn)

    def brk_vec(u, v):
        out = {}
        for i, a in enumerate(u):
            if not a:
                continue
            for j, b in enumerate(v):
                if not b:
                    continue
                for k, c in brk(i, j).items():
                    out[k] = (out.get(k, 0) + a * b * c) % P
        return {k: v for k, v in out.items() if v}

    full = [[1 if i == j else 0 for i in range(nn)] for j in range(nn)]
    d1 = sub(full)
    d2 = sub(d1) if d1 else []
    # lower central series
    lcs, cur = [nn], full
    while True:
        rows = []
        for u in full:
            for v in cur:
                w = [0] * nn
                for a, c in brk_vec(u, v).items():
                    w[a] = c
                if any(w):
                    span_add(rows, w)
        rows = red(rows, nn)
        if len(rows) == len(cur):
            break
        lcs.append(len(rows))
        cur = rows
        if not rows:
            break
    # centre
    M = []
    for j in range(nn):
        col = []
        for i in range(nn):
            v = [0] * nn
            for k, c in brk(i, j).items():
                v[k] = c
            col.extend(v)
        M.append(col)
    Z = nullspace([list(r) for r in zip(*M)], nn)
    return dict(dim=nn, derived=len(d1), derived2=len(d2), lcs=lcs, centre=len(Z),
                brk_vec=brk_vec)


def main():
    print("=" * 70)
    print("1. the deform of k^(1)(3;1) along the weight-(-3;3) cocycle")
    print("=" * 70)
    idx = [i for i, (w, k, c) in enumerate(cocycles) if w == (-3, 3) and k == 2][0]
    coeffs = [0] * len(cocycles)
    coeffs[idx] = 1
    gl = deform(coeffs)
    n = gl.n
    print("   dim g_lam = %d" % n)

    subs = [(cod, L0) for cod, L0 in weisfeiler(gl, 3) if cod == 3]
    print("   torus-graded subalgebras of codimension 3: %d" % len(subs))
    assert len(subs) == 1
    L0 = subs[0][1]

    filt = wfiltration(gl, L0)
    dims = [n] + [len(f) for f in filt]
    print("   Weisfeiler filtration  L_{-1} > L_0 > L_1 > ... :", dims)
    grdims = [dims[i] - dims[i + 1] for i in range(len(dims) - 1)]
    print("   gr dimensions, from degree -1 upward:", grdims)

    print()
    print("=" * 70)
    print("2. gr_0 and its action on gr_{-1}")
    print("=" * 70)
    reps0, brk0 = structure_on_quotient(gl, filt[0], filt[1], n)
    d0 = describe(len(reps0), brk0)
    print("   dim gr_0 = %d" % len(reps0))
    print("   derived: dim [gr_0, gr_0] = %d, dim [[gr_0,gr_0],[gr_0,gr_0]] = %d"
          % (d0["derived"], d0["derived2"]))
    print("   lower central series:", d0["lcs"])
    print("   dim centre = %d" % d0["centre"])
    print("   sl(2) (+) hei(2) has dim 6, derived = 6, centre = 1, "
          "and its nilradical hei(2) is 3-dimensional with 1-dimensional centre")
    return gl, filt, dims


if __name__ == "__main__":
    main()
