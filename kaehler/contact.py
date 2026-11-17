#!/usr/bin/env python3
r"""Prolongs and Spencer cohomology of Subsecs. 6.4 and 6.5.

Exact rational arithmetic throughout (Fraction / python-flint via ../lib/exactlin.py).

  Sec. 4  g_{-1} = K^{2n} abelian, g_0 = the stabilizer of the pair (a symmetric
          form and a 2-form) in its normal shape: the prolong (Lemma 6.4.1) and
          H^2(g_{-1}; g_*) graded by the weight (Theorem 6.4.3, table (Hcase1)).

  Sec. 5  g_- = hei(2m) spanned by  D_{u_i} = d_{u_i} - v_i d_t,
          D_{v_i} = d_{v_i} + u_i d_t  (m = n-1), g_0 = u(m) spanned by the
          fields A_ij, B_ij: the prolong g_* (Lemma 6.5.1a), computed inside
          vect(2m+1) degree by degree in the contact grading deg t = 2,
          deg u_i = deg v_i = 1, and H^2(g_-; g_*) (Theorem 6.5.1b, table (Hcase2)).

Everything is over Q; the algebras u(m), o(2m) etc. are meant complexified.

Run (from this folder):  ../venv/bin/python contact.py
"""

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
import os
from fractions import Fraction
from itertools import combinations

from exactlin import rank, nullspace, backend            # noqa: E402

F = Fraction


# ---------------------------------------------------------------- polynomials
# A vector field is a dict {(monomial, var) : coeff}, monomial a tuple of
# exponents, var the index of the partial derivative.

def vf_add(X, Y, s=1):
    Z = dict(X)
    for k, c in Y.items():
        Z[k] = Z.get(k, F(0)) + s * c
        if Z[k] == 0:
            del Z[k]
    return Z


def vf_scale(X, s):
    return {k: s * c for k, c in X.items() if s * c != 0}


def mono_mul(a, b):
    return tuple(x + y for x, y in zip(a, b))


def mono_diff(a, i):
    """d/dx_i of the monomial a: returns (coeff, monomial) or None."""
    if a[i] == 0:
        return None
    b = list(a)
    b[i] -= 1
    return F(a[i]), tuple(b)


def bracket(X, Y, N):
    """[X, Y] for polynomial vector fields in N variables."""
    Z = {}
    for (fm, a), fc in X.items():
        for (gm, b), gc in Y.items():
            # f d_a (g d_b) = f * d_a(g) d_b
            d = mono_diff(gm, a)
            if d is not None:
                c, mm = d
                key = (mono_mul(fm, mm), b)
                Z[key] = Z.get(key, F(0)) + fc * gc * c
            # - g d_b (f d_a)
            d = mono_diff(fm, b)
            if d is not None:
                c, mm = d
                key = (mono_mul(gm, mm), a)
                Z[key] = Z.get(key, F(0)) - fc * gc * c
    return {k: c for k, c in Z.items() if c != 0}


def vf_degree(key, w):
    mono, var = key
    return sum(w[i] * e for i, e in enumerate(mono)) - w[var]


def basis_of_degree(k, w, N):
    """All (monomial, var) of the given weighted degree."""
    out = []
    maxdeg = k + max(w)

    def rec(i, mono, d):
        if i == N:
            for var in range(N):
                if d - w[var] == k:
                    out.append((tuple(mono), var))
            return
        e = 0
        while d + e * w[i] <= maxdeg:
            rec(i + 1, mono + [e], d + e * w[i])
            e += 1

    rec(0, [], 0)
    return out


# ---------------------------------------------------------------- linear algebra on vf's

def coords(X, basis_index):
    v = [F(0)] * len(basis_index)
    for k, c in X.items():
        v[basis_index[k]] = c
    return v


def span_solve(vectors, target):
    """Is target in the span of vectors?  Returns True/False."""
    if not vectors:
        return all(x == 0 for x in target)
    r0 = rank(vectors)
    return rank(vectors + [target]) == r0


class Echelon:
    """Incremental echelon form, for repeated membership tests."""

    def __init__(self, dim):
        self.rows = []        # (pivot, row) with row[pivot] == 1
        self.dim = dim

    def reduce(self, v):
        v = list(v)
        for p, r in self.rows:
            if v[p] != 0:
                c = v[p]
                v = [a - c * b for a, b in zip(v, r)]
        return v

    def add(self, v):
        v = self.reduce(v)
        for i, x in enumerate(v):
            if x != 0:
                r = [a / x for a in v]
                self.rows.append((i, r))
                self.rows.sort(key=lambda t: t[0])
                return True
        return False

    def contains(self, v):
        return all(x == 0 for x in self.reduce(v))


# ---------------------------------------------------------------- Sec. 5: the contact case

def contact_setup(m):
    """Returns N, weights, g_{-2}, g_{-1}, g_0 for the PT^*M model of Subsec. 6.5.

    Variables: 0 = t (= v_n), 1..m = u_i, m+1..2m = v_i.
    """
    N = 2 * m + 1
    w = [2] + [1] * (2 * m)

    def U(i):
        return 1 + i

    def V(i):
        return 1 + m + i

    def e(i):
        return tuple(1 if j == i else 0 for j in range(N))

    zero = tuple(0 for _ in range(N))

    gm2 = [{(zero, 0): F(1)}]                       # d_t
    gm1 = []
    for i in range(m):
        gm1.append({(zero, U(i)): F(1), (e(V(i)), 0): F(-1)})     # D_{u_i}
    for i in range(m):
        gm1.append({(zero, V(i)): F(1), (e(U(i)), 0): F(1)})      # D_{v_i}

    g0 = []
    for i in range(m):
        for j in range(m):
            if i < j:                                              # A_ij
                X = {(e(V(i)), V(j)): F(1), (e(V(j)), V(i)): F(-1),
                     (e(U(i)), U(j)): F(1), (e(U(j)), U(i)): F(-1)}
                g0.append(X)
    for i in range(m):
        for j in range(m):
            if i <= j:                                             # B_ij
                X = {}
                for (a, b, s) in ((V(i), U(j), F(1)), (V(j), U(i), F(1)),
                                  (U(i), V(j), F(-1)), (U(j), V(i), F(-1))):
                    X[(e(a), b)] = X.get((e(a), b), F(0)) + s
                X = {k: c for k, c in X.items() if c != 0}
                g0.append(X)
    return N, w, gm2, gm1, g0


def check_setup(m, N, w, gm2, gm1, g0, log):
    """The model is the one described in the text."""
    ok = True
    # degrees
    for X, d in ((gm2[0], -2),) + tuple((X, -1) for X in gm1) \
            + tuple((X, 0) for X in g0):
        for k in X:
            if vf_degree(k, w) != d:
                ok = False
                log(f"   !! degree mismatch: {k} is {vf_degree(k, w)}, want {d}")
    # hei(2m): [g_{-1}, g_{-1}] = g_{-2}, [g_{-1}, g_{-2}] = 0
    for i in range(len(gm1)):
        for j in range(i + 1, len(gm1)):
            b = bracket(gm1[i], gm1[j], N)
            if b and not (len(b) == 1 and list(b)[0] == (tuple(0 for _ in range(N)), 0)):
                ok = False
                log(f"   !! [g_-1,g_-1] leaves g_-2: {b}")
    nz = 0
    for i in range(m):
        b = bracket(gm1[i], gm1[m + i], N)
        if b:
            nz += 1
    if nz != m:
        ok = False
        log(f"   !! Heisenberg centre not reached: {nz} of {m}")
    # g_0 normalises g_{-1}
    idx1 = {}
    for X in gm1:
        for k in X:
            idx1.setdefault(k, len(idx1))
    for X in gm1:
        for k in X:
            pass
    allkeys = {}
    for X in gm1 + g0 + gm2:
        for k in X:
            allkeys.setdefault(k, len(allkeys))
    for A in g0:
        for D in gm1:
            b = bracket(A, D, N)
            for k in b:
                allkeys.setdefault(k, len(allkeys))
    base1 = [coords(X, allkeys) for X in gm1]
    for A in g0:
        for D in gm1:
            b = bracket(A, D, N)
            if not span_solve(base1, coords(b, allkeys)):
                ok = False
                log(f"   !! [g_0, g_-1] leaves g_-1")
    log(f"   model check for m = {m}: "
        f"dim g_-2 = {len(gm2)}, dim g_-1 = {len(gm1)}, dim g_0 = {len(g0)}"
        f"  ({'consistent' if ok else 'INCONSISTENT'})")
    return ok


def prolong_contact(m, kmax, log):
    """g_k = {X in vect_k | [X, g_{-1}] in g_{k-1}}, k >= 1, in the contact grading."""
    N, w, gm2, gm1, g0 = contact_setup(m)
    check_setup(m, N, w, gm2, gm1, g0, log)

    layers = {-2: gm2, -1: gm1, 0: g0}
    dims = {-2: 1, -1: 2 * m, 0: len(g0)}

    for k in range(1, kmax + 1):
        prev = layers[k - 1]
        if not prev:
            layers[k] = []
            dims[k] = 0
            continue
        cand = basis_of_degree(k, w, N)
        if not cand:
            layers[k] = []
            dims[k] = 0
            continue
        cidx = {c: i for i, c in enumerate(cand)}

        # target space: span of layers[k-1], in a basis of degree k-1 fields
        tgt = basis_of_degree(k - 1, w, N)
        tidx = {c: i for i, c in enumerate(tgt)}
        ech = Echelon(len(tgt))
        for X in prev:
            ech.add(coords(X, tidx))
        # a complement: rows of the quotient.  We need the conditions
        # "coords(bracket) lies in span(prev)".  Build the quotient projection.
        # Simplest: for each candidate basis vector, compute the bracket with
        # each D in g_{-1}, reduce modulo span(prev), and require 0.
        eqs_rows = []          # one row per unknown coefficient
        for c in cand:
            X = {c: F(1)}
            row = []
            for D in gm1:
                b = bracket(X, D, N)
                v = [F(0)] * len(tgt)
                bad = False
                for kk, cc in b.items():
                    if kk not in tidx:
                        bad = True
                        break
                    v[tidx[kk]] = cc
                if bad:
                    # bracket left the degree-(k-1) space: impossible, but guard
                    v = [F(1)] * len(tgt)
                row += ech.reduce(v)
            eqs_rows.append(row)
        # unknowns are the coefficients; equations are the columns of eqs_rows
        ncols = len(eqs_rows[0])
        eqs = [[eqs_rows[r][c] for r in range(len(eqs_rows))] for c in range(ncols)]
        sols = nullspace(eqs) if eqs else []
        gk = []
        for v in sols:
            X = {}
            for i, cc in enumerate(v):
                if cc != 0:
                    X[cand[i]] = cc
            if X:
                gk.append(X)
        layers[k] = gk
        dims[k] = len(gk)
        if not gk:
            for kk in range(k + 1, kmax + 1):
                layers[kk] = []
                dims[kk] = 0
            break
    return dims, layers, (N, w, gm1, g0)


def sec5(log):
    log("")
    log("== Sec. 5: the prolong of (hei(2m), u(m)), m = n-1 ==")
    log("   g_-1 = span(D_{u_i}, D_{v_i}), g_0 = span(A_ij, B_ij);")
    log("   contact grading deg t = 2, deg u = deg v = 1.")
    log("")
    log(f"{'m':>3} {'n=m+1':>6} {'g_-2':>5} {'g_-1':>5} {'g_0':>5} {'g_1':>5}"
        f" {'g_2':>5} {'g_3':>5} {'g_4':>5} {'dim g_*':>8}  identification")
    for m in range(1, 5):
        dims, layers, _ = prolong_contact(m, 4, log)
        tot = sum(dims.get(k, 0) for k in range(-2, 5))
        # su(m+1,1) complexified is sl(m+2), of dim (m+2)^2 - 1
        sl = (m + 2) ** 2 - 1
        ident = ""
        if tot == sl:
            ident = f"sl({m+2}) = su({m+1},1)_C, dim {sl}"
        else:
            ident = f"dim {tot}; sl({m+2}) would be {sl}"
        log(f"{m:>3} {m+1:>6} {dims.get(-2,0):>5} {dims.get(-1,0):>5}"
            f" {dims.get(0,0):>5} {dims.get(1,0):>5} {dims.get(2,0):>5}"
            f" {dims.get(3,0):>5} {dims.get(4,0):>5} {tot:>8}  {ident}")
    log("")
    log("   NB  g_0 = u(m) does NOT contain the grading (Euler)")
    log("   operator; the full contact prolong takes g_0 = cu(m) = u(m) (+) K.")
    for m in range(1, 5):
        d2 = prolong_with_euler(m, 4)
        tot = sum(d2.get(k, 0) for k in range(-2, 5))
        sl = (m + 2) ** 2 - 1
        log(f"   m = {m}: with the Euler operator adjoined, dims"
            f" {[d2.get(k,0) for k in range(-2,5)]}, total {tot}"
            f"{'  = dim sl(' + str(m+2) + ')' if tot == sl else ''}")


def prolong_with_euler(m, kmax):
    """The same prolong, but g_0 enlarged by the grading operator."""
    N, w, gm2, gm1, g0 = contact_setup(m)
    zero = tuple(0 for _ in range(N))

    def e(i):
        return tuple(1 if j == i else 0 for j in range(N))

    E = {(e(0), 0): F(2)}
    for i in range(1, N):
        E[(e(i), i)] = F(1)
    g0 = g0 + [E]

    layers = {-2: gm2, -1: gm1, 0: g0}
    dims = {-2: 1, -1: 2 * m, 0: len(g0)}
    for k in range(1, kmax + 1):
        prev = layers[k - 1]
        cand = basis_of_degree(k, w, N)
        if not prev or not cand:
            layers[k], dims[k] = [], 0
            continue
        tgt = basis_of_degree(k - 1, w, N)
        tidx = {c: i for i, c in enumerate(tgt)}
        ech = Echelon(len(tgt))
        for X in prev:
            ech.add(coords(X, tidx))
        rows = []
        for c in cand:
            X = {c: F(1)}
            row = []
            for D in gm1:
                b = bracket(X, D, N)
                v = [F(0)] * len(tgt)
                for kk, cc in b.items():
                    v[tidx[kk]] = cc
                row += ech.reduce(v)
            rows.append(row)
        ncols = len(rows[0])
        eqs = [[rows[r][c] for r in range(len(rows))] for c in range(ncols)]
        sols = nullspace(eqs) if eqs else []
        gk = []
        for v in sols:
            X = {cand[i]: cc for i, cc in enumerate(v) if cc != 0}
            if X:
                gk.append(X)
        layers[k], dims[k] = gk, len(gk)
        if not gk:
            for kk in range(k + 1, kmax + 1):
                layers[kk], dims[kk] = [], 0
            break
    return dims


# ---------------------------------------------------------------- Sec. 4: Spencer H^2

def matmul(A, B):
    n, k, m2 = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m2)]
            for i in range(n)]


def zeros(r, c):
    return [[F(0)] * c for _ in range(r)]


def eye(k):
    return [[F(1) if i == j else F(0) for j in range(k)] for i in range(k)]


def gl_stab(d, forms):
    """A basis of {X | X^t B + B X = 0 for every B in forms}, as d x d matrices."""
    rows = []
    for i in range(d):
        for j in range(d):
            r = []
            for B in forms:
                # (E_ij^t B + B E_ij)_{ab} = delta_{ja} B_{ib} + B_{ai} delta_{jb}
                M = zeros(d, d)
                for a in range(d):
                    for b in range(d):
                        v = F(0)
                        if j == a:
                            v += B[i][b]
                        if j == b:
                            v += B[a][i]
                        M[a][b] = v
                r += [x for row in M for x in row]
            rows.append(r)
    ncols = len(rows[0])
    eqs = [[rows[r][c] for r in range(d * d)] for c in range(ncols)]
    out = []
    for v in nullspace(eqs):
        X = zeros(d, d)
        for t in range(d * d):
            X[t // d][t % d] += v[t]
        out.append(X)
    return out


def mixed_pair(J):
    """The classical normal shape (B_1 symmetric, B_2 a 2-form), eq. (symmANDanti)."""
    n = len(J)
    d = 2 * n
    B1 = zeros(d, d)
    B2 = zeros(d, d)
    for i in range(n):
        B1[i][n + i] = F(1)
        B1[n + i][i] = F(1)
    for i in range(n):
        for j in range(n):
            B2[i][n + j] = J[i][j]
            B2[n + j][i] = -J[i][j]
    return B1, B2


def spencer_h2(g0, d, log, name):
    """H^2(g_{-1}; g_*) for g_{-1} = K^d abelian, g_* = g_{-1} (+) g_0, g_1 = 0.

    The module is g_* with the g_{-1}-action  v . w = 0 (w in g_{-1}),
    v . A = -A(v) (A in g_0).  Chevalley-Eilenberg complex C^p = Lambda^p V* (x) g_*.
    We compute the whole H^2, then split it by the weight (deg V = -1, deg g_0 = 0).
    """
    k0 = len(g0)
    dim_gs = d + k0

    # basis of g_*: 0..d-1 are the e_a of g_{-1}; d..d+k0-1 are g_0.
    def act(a, x):
        """e_a . x for x a coordinate vector of g_*; returns a coordinate vector."""
        out = [F(0)] * dim_gs
        for s in range(k0):
            c = x[d + s]
            if c != 0:
                # [e_a, g0[s]] = -g0[s](e_a) = -(column a of g0[s])
                for b in range(d):
                    out[b] -= c * g0[s][b][a]
        return out

    def cochains(p):
        return list(combinations(range(d), p))

    def dmap(p):
        """Matrix of d: C^p -> C^{p+1} in the monomial bases."""
        src = [(S, t) for S in cochains(p) for t in range(dim_gs)]
        dst = [(S, t) for S in cochains(p + 1) for t in range(dim_gs)]
        didx = {k: i for i, k in enumerate(dst)}
        M = zeros(len(dst), len(src))
        for c, (S, t) in enumerate(src):
            x = [F(0)] * dim_gs
            x[t] = F(1)
            for a in range(d):
                if a in S:
                    continue
                T = tuple(sorted(S + (a,)))
                sign = F((-1) ** T.index(a))
                y = act(a, x)
                for u in range(dim_gs):
                    if y[u] != 0:
                        M[didx[(T, u)]][c] += sign * y[u]
        return M, src, dst

    d1, src1, _ = dmap(1)
    d2, src2, _ = dmap(2)
    r1 = rank(d1) if d1 and d1[0] else 0
    r2 = rank(d2) if d2 and d2[0] else 0
    dim_c2 = len(src2)
    h2 = dim_c2 - r2 - r1
    # The differential preserves the weight (V* weight +1, V weight -1, g_0
    # weight 0), so H^2 = H^2_{wt 1} (+) H^2_{wt 2}.  In the Spencer indexing
    # H^{j,p} with j = wt - p these are H^{-1,2}, the intrinsic TORSION, and
    # H^{0,2}, the CURVATURE.  Since d kills Lambda^1 V* (x) V and lands in
    # Lambda^2 V* (x) V, and g_1 = 0, both are read off directly.
    nl2 = d * (d - 1) // 2
    tors = nl2 * d - r1                     # Lambda^2 V* (x) V  modulo im d_1
    curv = nl2 * k0 - r2                    # ker of d on Lambda^2 V* (x) g_0
    log(f"   {name}: dim g_0 = {k0}, dim C^2 = {dim_c2}, rk d_1 = {r1},"
        f" rk d_2 = {r2}  =>  dim H^2(g_-1; g_*) = {h2}"
        f"   [torsion {tors} + curvature {curv}]")
    assert tors + curv == h2
    return h2


def sec4(log):
    log("")
    log("== Sec. 4: the prolong and H^2 for the classical mixed pair ==")
    log("   g_-1 = K^{2n}, g_0 = aut(B_1, B_2) for the normal shape of Thm 6.3.1.")
    log("")
    cases = [("n=1, J=(3)", [[F(3)]]),
             ("n=2, J=diag(3,5)   generic", [[F(3), F(0)], [F(0), F(5)]]),
             ("n=2, J=3*1_2       Kaehler", [[F(3), F(0)], [F(0), F(3)]]),
             ("n=2, J = one cell", [[F(3), F(1)], [F(0), F(3)]]),
             ("n=3, J=diag(2,5,11) generic",
              [[F(2), F(0), F(0)], [F(0), F(5), F(0)], [F(0), F(0), F(11)]]),
             ("n=3, J=3*1_3       Kaehler",
              [[F(3), F(0), F(0)], [F(0), F(3), F(0)], [F(0), F(0), F(3)]])]
    for (name, J) in cases:
        n = len(J)
        d = 2 * n
        B1, B2 = mixed_pair(J)
        g0 = gl_stab(d, [B1, B2])
        # the first prolong
        g1 = prolong_dim_mat(g0, d)
        log(f"   {name:>28}: dim g_0 = {len(g0):>2}, dim g_1 = {g1}")
        if g1 != 0:
            log("      !! g_1 nonzero -- the H^2 below assumes g_1 = 0")
        spencer_h2(g0, d, log, f"{name:>28}")
    log("")
    log("   Hence g_* is NOT vect(n) (+) vect(n), and")
    log("   H^2(g_-1; g_*) =/= 0: g_1 = 0 (so g_* is")
    log("   finite-dimensional, Lemma 6.4.1) and H^2 =/= 0 (Theorem 6.4.3).")


def prolong_dim_mat(g0, d):
    k = len(g0)
    if k == 0:
        return 0
    idx = {(a, s): a * k + s for a in range(d) for s in range(k)}
    eqs = []
    for a in range(d):
        for b in range(a + 1, d):
            for c in range(d):
                r = [F(0)] * (d * k)
                for s in range(k):
                    r[idx[(a, s)]] += g0[s][c][b]
                    r[idx[(b, s)]] -= g0[s][c][a]
                if any(x != 0 for x in r):
                    eqs.append(r)
    return len(nullspace(eqs)) if eqs else d * k


# ------------------------------------------ Sec. 5: H^2(g_-; g_*), g_- = hei(2m)

def ce_h2(gm, gs, wts, N, log, name):
    """H^2(g_-; g_*) for a finite-dimensional g_- and module g_*, both given as
    vector fields; the module structure is the bracket.  Graded by the weight.

    Returns {weight: dim H^2_weight}.
    """
    dm, ds = len(gm), len(gs)
    # coordinates on g_*
    keys = {}
    for X in gs:
        for k in X:
            keys.setdefault(k, len(keys))
    for X in gm:
        for Y in gs:
            for k in bracket(X, Y, N):
                keys.setdefault(k, len(keys))
    for i in range(dm):
        for j in range(dm):
            for k in bracket(gm[i], gm[j], N):
                keys.setdefault(k, len(keys))
    ech = Echelon(len(keys))
    basis_rows = []
    for X in gs:
        v = coords(X, keys)
        basis_rows.append(v)
    # express a vector field in the basis gs
    import copy  # noqa: F401

    def in_gs(X):
        v = coords(X, keys)
        # solve basis_rows^T c = v
        A = [[basis_rows[r][c] for r in range(ds)] for c in range(len(keys))]
        sol = solve_linear(A, v)
        if sol is None:
            raise RuntimeError(f"{name}: field leaves g_*")
        return sol

    # structure constants of g_-  (g_- is a subalgebra of g_*)
    cc = {}
    for i in range(dm):
        for j in range(i + 1, dm):
            cc[(i, j)] = in_gs_of_gm(bracket(gm[i], gm[j], N), gm, keys, name)
    act = {}
    for i in range(dm):
        for s in range(ds):
            act[(i, s)] = in_gs(bracket(gm[i], gs[s], N))

    def cochains(p):
        return [(S, t) for S in combinations(range(dm), p) for t in range(ds)]

    def weight(S, t):
        return sum(-wts[i] for i in S) * -1 + wts_gs[t] if False else 0

    # weights: a cochain S (x) e_t has weight  sum_{i in S} (-deg gm[i]) + deg gs[t]
    def wt(S, t):
        return sum(-degm[i] for i in S) + degs[t]

    degm = [vf_degree(next(iter(X)), wts) for X in gm]
    degs = [vf_degree(next(iter(X)), wts) for X in gs]

    def dmat(p):
        """Matrix of d: C^p -> C^{p+1}, assembled by TARGET.

        (dc)(x_0,..,x_p) = sum_i (-1)^i x_i . c(T minus x_i)
                         + sum_{i<j} (-1)^{i+j} c([x_i,x_j], T minus x_i,x_j).
        """
        src, dst = cochains(p), cochains(p + 1)
        sidx = {k: i for i, k in enumerate(src)}
        didx = {k: i for i, k in enumerate(dst)}
        M = zeros(len(dst), len(src))
        for T in combinations(range(dm), p + 1):
            for i, xi in enumerate(T):
                S = tuple(x for x in T if x != xi)
                sgn = F((-1) ** i)
                for t in range(ds):
                    y = act[(xi, t)]
                    for u in range(ds):
                        if y[u] != 0:
                            M[didx[(T, u)]][sidx[(S, t)]] += sgn * y[u]
            for i in range(p + 1):
                for j in range(i + 1, p + 1):
                    a, b = T[i], T[j]
                    rest = tuple(x for x in T if x not in (a, b))
                    br = cc[(a, b)]
                    for e in range(dm):
                        if br[e] == 0 or e in rest:
                            continue
                        S = tuple(sorted(rest + (e,)))
                        sgn = F((-1) ** (i + j)) * F((-1) ** S.index(e)) * br[e]
                        for t in range(ds):
                            M[didx[(T, t)]][sidx[(S, t)]] += sgn
        return M, src, dst

    d0, _, _ = dmat(0)
    d1, src1, _ = dmat(1)
    d2, src2, _ = dmat(2)
    # d o d = 0 -- the check that the signs and the bracket term are right
    for (A, B, nm) in ((d1, d0, "d1 d0"), (d2, d1, "d2 d1")):
        P = matmul(A, B)
        if any(x != 0 for row in P for x in row):
            raise RuntimeError(f"{name}: {nm} != 0 -- the complex is wrong")
    out = {}
    allw = sorted({wt(S, t) for (S, t) in src2})
    for w in allw:
        c2 = [i for i, (S, t) in enumerate(src2) if wt(S, t) == w]
        c1 = [i for i, (S, t) in enumerate(src1) if wt(S, t) == w]
        A1 = [[d1[r][c] for c in c1] for r in range(len(d1))] if c1 else []
        A2 = [[d2[r][c] for c in c2] for r in range(len(d2))] if c2 else []
        r1 = rank(A1) if A1 and A1[0] else 0
        r2 = rank(A2) if A2 and A2[0] else 0
        h = len(c2) - r2 - r1
        if h:
            out[w] = h
    tot = sum(out.values())
    log(f"   {name}: dim g_- = {dm}, dim g_* = {ds},"
        f" dim H^2(g_-; g_*) = {tot}   by weight: "
        + (", ".join(f"{w}: {d}" for w, d in sorted(out.items())) or "--"))
    return out


def in_gs_of_gm(X, gm, keys, name):
    dm = len(gm)
    rows = [coords(Y, keys) for Y in gm]
    A = [[rows[r][c] for r in range(dm)] for c in range(len(keys))]
    sol = solve_linear(A, coords(X, keys))
    if sol is None:
        raise RuntimeError(f"{name}: [g_-, g_-] leaves g_-")
    return sol


def solve_linear(A, b):
    """Solve A x = b exactly; A a list of rows.  Returns x or None."""
    rows = [list(r) + [bb] for r, bb in zip(A, b)]
    ncols = len(A[0]) if A else 0
    piv = []
    r = 0
    for c in range(ncols):
        p = None
        for i in range(r, len(rows)):
            if rows[i][c] != 0:
                p = i
                break
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
        if r == len(rows):
            break
    for i in range(r, len(rows)):
        if all(x == 0 for x in rows[i][:ncols]) and rows[i][ncols] != 0:
            return None
    x = [F(0)] * ncols
    for i, c in enumerate(piv):
        x[c] = rows[i][ncols]
    return x


def sec5_h2(log):
    log("")
    log("== Sec. 5: the local invariants H^2(g_-; g_*) of the contact pair ==")
    for m in range(1, 4):
        dims, layers, (N, w, gm1, g0) = prolong_contact(m, 3, log)
        gm = layers[-1] + layers[-2]                 # hei(2m)
        gs = layers[-1] + layers[-2] + layers[0]     # g_* = g_- (+) g_0
        try:
            ce_h2(gm, gs, w, N, log, f"m = {m} (n = {m+1})")
        except RuntimeError as e:
            log(f"   m = {m}: {e}")


# ---------------------------------------------------------------- main

def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run4.log")
    fh = open(path, "w")

    def log(s=""):
        print(s)
        fh.write(s + "\n")
        fh.flush()

    log("prolongs and Spencer cohomology of Subsecs. 6.4, 6.5 -- exact arithmetic")
    log("linear algebra backend: " + backend())
    sec4(log)
    sec5(log)
    sec5_h2(log)
    log("")
    log("done.")
    fh.close()


if __name__ == "__main__":
    main()
