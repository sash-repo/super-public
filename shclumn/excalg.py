#!/usr/bin/env python3
r"""The exceptional simple vectorial Lie superalgebras as graded vector fields.

Built from the data of the table of W-gradings of the exceptional algebras: g_{-2}, g_{-1}, g_0, and
the Cartan prolong.  Everything over Q, exact.

    vle(4|3)  = (Pi(Lambda(3)/C.1), c vect(0|3))_*      depth 1
    kas(1|6;3eta) = (Vol_0(0|3), c vect(0|3))_*         depth 1
    vas(4|4)  = (spin, as)_*                            depth 1
    mb(4|5)   = (ba(4), c vect(0|3))^m_*                depth 2  (inside m(4|5))
"""
import sys
from fractions import Fraction
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rref, nullspace                                # noqa: E402

from vfcore import Space, vec_of                                    # noqa: E402


# ---------------------------------------------------------------- utilities
def span_index(fields):
    """A dict monomial-key -> column, covering all the given fields."""
    idx = {}
    for D in fields:
        for k in D:
            if k not in idx:
                idx[k] = len(idx)
    return idx


def echelon(rows):
    """Row-reduce; return the list of non-zero rows (a basis of the row space)."""
    if not rows:
        return []
    R, piv = rref([[Fraction(x) for x in r] for r in rows])
    return [r for r in R if any(r)]


def in_span(basis_rows, pivots, v):
    """Reduce v against an rref basis; return the remainder."""
    v = list(v)
    for r, p in zip(basis_rows, pivots):
        if v[p]:
            f = v[p] / r[p]
            v = [a - f * b for a, b in zip(v, r)]
    return v


def pivots_of(rows):
    ps = []
    for r in rows:
        for i, x in enumerate(r):
            if x:
                ps.append(i)
                break
    return ps


# ------------------------------------------------------------- the prolong
def key_wt(S, k):
    g, m = k
    return tuple(x - y for x, y in zip(S.wt_mono(m), S.wt[g]))


def group_keys(S, deg):
    """Monomial vector fields of the given degree, grouped by torus weight."""
    out = {}
    for g in range(S.n):
        for m in S.monomials(deg + S.deg[g]):
            k = (g, m)
            out.setdefault(key_wt(S, k), []).append(k)
    return out


def to_blocks(S, fields):
    """Weight-homogeneous fields -> {weight: (keys, rows)}."""
    byw = {}
    for D in fields:
        w = None
        for k in D:
            w = key_wt(S, k)
            break
        byw.setdefault(w, []).append(D)
    out = {}
    for w, fs in byw.items():
        keys = sorted({k for D in fs for k in D})
        pos = {k: i for i, k in enumerate(keys)}
        rows = []
        for D in fs:
            v = [Fraction(0)] * len(keys)
            for k, c in D.items():
                v[pos[k]] = Fraction(c)
            rows.append(v)
        out[w] = (keys, echelon(rows))
    return out


def annihilator(keys_all, keys_sub, rows):
    """Left annihilator of the span of `rows` inside the coordinate space keys_all."""
    pos = {k: i for i, k in enumerate(keys_all)}
    n = len(keys_all)
    M = []
    for r in rows:
        v = [Fraction(0)] * n
        for k, c in zip(keys_sub, r):
            if c:
                v[pos[k]] = c
        M.append(v)
    if not M:
        return [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
    return nullspace(M)          # right kernel of M = vectors killed by every row


def prolong(S, gm, g0, top, verbose=False):
    """The Cartan prolong of (g_-, g_0) inside vect(S), up to degree `top`."""
    comp = dict(gm)
    comp[0] = split_weight(S, list(g0))
    for k in range(1, top + 1):
        prevblocks = to_blocks(S, comp[k - 1])
        prevkeys = group_keys(S, k - 1)
        # annihilators, one per weight class of degree k-1
        ann = {}
        for w, keys in prevkeys.items():
            sub = prevblocks.get(w)
            if sub is None:
                ann[w] = ('all', keys)
            else:
                ann[w] = ('mat', keys, annihilator(keys, sub[0], sub[1]))
        newc = []
        for w, keys in group_keys(S, k).items():
            pos = {kk: i for i, kk in enumerate(keys)}
            rows = []
            for a in range(S.n):
                wa = tuple(x - y for x, y in zip(w, S.wt[a]))
                tgt = prevkeys.get(wa, [])
                tpos = {kk: i for i, kk in enumerate(tgt)}
                # T_a : coefficients on `keys` -> coefficients on `tgt`
                T = [[Fraction(0)] * len(keys) for _ in range(len(tgt))]
                for kk in keys:
                    g, m = kk
                    r = S.deriv(a, m)
                    if r is None:
                        continue
                    co, m2 = r
                    T[tpos[(g, m2)]][pos[kk]] += co
                A = ann[wa] if wa in ann else ('all', [])
                if A[0] == 'all':
                    rows.extend(T)
                else:
                    for q in A[2]:
                        rows.append([sum(q[i] * T[i][j] for i in range(len(tgt)))
                                     for j in range(len(keys))])
            ker = nullspace(rows) if rows else [[Fraction(1) if i == j else Fraction(0)
                                                 for j in range(len(keys))]
                                                for i in range(len(keys))]
            for sol in ker:
                D = {kk: c for kk, c in zip(keys, sol) if c}
                if D:
                    newc.append(D)
        comp[k] = newc
        if verbose:
            print('   g_%d: dim %d' % (k, len(newc)))
    return comp


def split_weight(S, fields):
    """Replace a basis by a weight-homogeneous one (the span is weight-graded)."""
    if not fields:
        return []
    idx = span_index(fields)
    inv = {v: k for k, v in idx.items()}
    # group columns by weight
    groups = {}
    for k, j in idx.items():
        g, m = k
        w = tuple(x - y for x, y in zip(S.wt_mono(m), S.wt[g]))
        groups.setdefault(w, []).append(j)
    rows = [vec_of(D, idx) for D in fields]
    out = []
    for w, cols in groups.items():
        sub = [[r[j] for j in cols] for r in rows]
        E = echelon(sub)
        for r in E:
            D = {}
            for jj, c in zip(cols, r):
                if c:
                    D[inv[jj]] = c
            out.append(D)
    # sanity: the weight-split basis must span the same space
    return out


def lin_field(S, A, basis_par):
    """The linear vector field D with ad(D)|_{g_{-1}} = A.

    A is given as a dict (c, b) -> coefficient with  A(e_c) = sum_b A[c,b] e_b,
    where e_a = d/d(coordinate a).  Then D = sum c_{ab} y_a d/d y_b with
    c_{ab} = -(-1)^{(p_a+p_b) p_a} A[a,b].
    """
    D = {}
    for (a, b), co in A.items():
        if not co:
            continue
        pa, pb = basis_par[a], basis_par[b]
        sgn = -1 if ((pa + pb) * pa) % 2 == 0 else 1
        m = ([0] * S.nev, [])
        if a < S.nev:
            e = [0] * S.nev
            e[a] = 1
            m = (tuple(e), ())
        else:
            m = ((0,) * S.nev, (a - S.nev,))
        k = (b, m)
        D[k] = D.get(k, 0) + sgn * co
    return {k: v for k, v in D.items() if v}


# ------------------------------------------------------------------ vle(4|3)
def lam_basis(n):
    """Subsets of {0..n-1}, sorted by size then lexicographically."""
    out = []
    for k in range(n + 1):
        for S in combinations(range(n), k):
            out.append(S)
    return out


def lam_mul(S1, S2):
    if set(S1) & set(S2):
        return None
    merged = list(S1) + list(S2)
    sign = 1
    arr = merged[:]
    for i in range(len(arr)):
        for j in range(len(arr) - 1, i, -1):
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                sign = -sign
    return sign, tuple(arr)


def lam_der(j, S):
    if j not in S:
        return None
    pos = S.index(j)
    return (-1 if pos % 2 else 1), tuple(k for k in S if k != j)


def vect0n_basis(n):
    """Basis of vect(0|n): pairs (T, i) standing for xi^T d/d xi_i."""
    return [(T, i) for T in lam_basis(n) for i in range(n)]


def vect0n_act(n, T, i, S):
    """(xi^T d_i) applied to xi^S, in Lambda(n); returns a dict S' -> coeff."""
    r = lam_der(i, S)
    if r is None:
        return {}
    c, S1 = r
    pr = lam_mul(T, S1)
    if pr is None:
        return {}
    s, S2 = pr
    return {S2: c * s}


def vect0n_div(n, T, i):
    """Div(xi^T d_i) = (-1)^{p(xi^T)} d_i(xi^T)."""
    r = lam_der(i, T)
    if r is None:
        return {}
    c, T1 = r
    sgn = -1 if len(T) % 2 else 1
    return {T1: sgn * c}


def build_prolong_depth1(module_basis, module_par, act, extra_scalar=True,
                         top=2, name=''):
    """Prolong of (V, g_0) where g_0 = image(act) (+ scalars), V of depth 1.

    module_basis: list of labels of a basis of V, ordered even first.
    module_par:   dict label -> parity.
    act: list of (label_of_g0_element, parity, matrix) where matrix is a dict
         (c_index, b_index) -> coeff  with  A(e_c) = sum_b A[c,b] e_b.
    """
    N = len(module_basis)
    nev = sum(1 for b in module_basis if module_par[b] == 0)
    nod = N - nev
    return nev, nod


# ------------------------------------------------------------------ helpers
def check_closed(S, comp, kmax):
    """Verify that the constructed components close under the bracket."""
    bad = []
    for i in sorted(comp):
        for j in sorted(comp):
            if i > j or i + j > kmax or (i + j) not in comp:
                continue
            tgt = comp[i + j]
            idx = span_index(tgt + [S.bracket(X, Y) for X in comp[i] for Y in comp[j]])
            B = echelon([vec_of(D, idx) for D in tgt])
            pv = pivots_of(B)
            for X in comp[i]:
                for Y in comp[j]:
                    br = S.bracket(X, Y)
                    if not br:
                        continue
                    r = in_span(B, pv, vec_of(br, idx))
                    if any(r):
                        bad.append((i, j))
                        break
                else:
                    continue
                break
    return bad
