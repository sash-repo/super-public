#!/usr/bin/env python3
"""Audit of Table 1.5 of Chapter 1 of "Seminar on Supersymmetries", vol. 2 ("Regular odd elements.  Dimensions
of their centralizers").

The table gives, for each simple Lie superalgebra g, a *characteristic* -- a
Dynkin diagram of g some of whose nodes carry the label 1 -- and a number called
sdim Z_h.  Reading the characteristic as Dynkin does (the marks define a
Z-grading of g; the marked simple roots have degree 1, the unmarked ones degree
0), the centralizer of the grading element h is

    Z_h = g_0 = h + sum of g_alpha over the roots alpha of degree 0,

and a positive root has degree 0 exactly when its support avoids every marked
node.  So sdim Z_h is computable from (simple system, set of marks) alone:

    sdim Z_h = ( rank g + #{even roots of degree 0} | #{odd roots of degree 0} ).

This script enumerates *every* simple system (= every shuffle of the eps's and
the delta's) and *every* set of marks, and reports which of them produce the
number the table prints.  Nothing is assumed about which diagram was drawn: the
machine finds the diagrams that fit, and the shape it finds is then compared
with the picture by hand.

Exact integer arithmetic; no floating point anywhere.
"""
import itertools
import sys
from fractions import Fraction


# ------------------------------------------------------------------ roots
def roots_sl(m, n):
    """gl-weights: index i<m is an eps, i>=m is a delta.  Roots x_i - x_j."""
    N = m + n
    par = [0] * m + [1] * n
    out = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = [0] * N
            v[i], v[j] = 1, -1
            out.append((tuple(v), par[i] ^ par[j]))
    return out, N, par


def simple_sl(seq):
    """seq: a permutation of the index set; simple roots x_{s_i} - x_{s_{i+1}}."""
    N = len(seq)
    out = []
    for i in range(N - 1):
        v = [0] * N
        v[seq[i]], v[seq[i + 1]] = 1, -1
        out.append(tuple(v))
    return out


def roots_osp(M, twoN, odd_dim_even_part=None):
    """osp(P|2N): P = 2M or 2M+1 even directions eps_1..eps_M (+ the zero weight
    when P is odd), 2N odd directions delta_1..delta_N.
    Coordinates: first M are eps (even space), last N are delta (odd space)."""
    N = twoN // 2
    P = odd_dim_even_part            # P = 2M or 2M+1
    dim = M + N
    out = []

    def vec(pairs):
        v = [0] * dim
        for k, c in pairs:
            v[k] += c
        return tuple(v)

    # even roots: o(P) part
    for i in range(M):
        for j in range(i + 1, M):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    out.append((vec([(i, s1), (j, s2)]), 0))
    if P % 2:                        # +- eps_i are roots of o(2M+1)
        for i in range(M):
            for s in (1, -1):
                out.append((vec([(i, s)]), 0))
    # even roots: sp(2N) part
    for i in range(N):
        for j in range(i + 1, N):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    out.append((vec([(M + i, s1), (M + j, s2)]), 0))
    for i in range(N):
        for s in (1, -1):
            out.append((vec([(M + i, 2 * s)]), 0))
    # odd roots
    for i in range(M):
        for j in range(N):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    out.append((vec([(i, s1), (M + j, s2)]), 1))
    if P % 2:
        for j in range(N):
            for s in (1, -1):
                out.append((vec([(M + j, s)]), 1))
    par = [0] * M + [1] * N
    return out, dim, par


def simple_osp(seq, M, N, P):
    """seq: an ordering of the M+N coordinates (a shuffle of eps's and delta's).
    Simple roots a_1-a_2, ..., a_{r-1}-a_r and a last one depending on the type."""
    r = M + N
    out = []
    for i in range(r - 1):
        v = [0] * r
        v[seq[i]], v[seq[i + 1]] = 1, -1
        out.append(tuple(v))
    last = seq[-1]
    v = [0] * r
    if P % 2:                        # B(M,N): the last simple root is a_r
        v[last] = 1
    else:                            # D(M,N)
        if last >= M:                # ... = 2 delta   (long even root)
            v[last] = 2
        else:                        # ... = a_{r-1} + a_r  (the fork)
            v[seq[-2]] += 1
            v[last] += 1
    out.append(tuple(v))
    return out


# ------------------------------------------------------------------ expansion
def expand(roots, simple, dim, extra_constraint=False):
    """Express every root in the basis of simple roots; return coefficient lists."""
    r = len(simple)
    cols = [list(s) for s in simple]
    out = []
    for (v, p) in roots:
        # solve  sum_k c_k simple_k = v
        A = [[Fraction(cols[k][i]) for k in range(r)] + [Fraction(v[i])]
             for i in range(dim)]
        c = _solve(A, r)
        if c is None:
            return None
        out.append((tuple(c), p))
    return out


def _solve(A, r):
    rows = [row[:] for row in A]
    piv = []
    i = 0
    for col in range(r):
        sel = None
        for k in range(i, len(rows)):
            if rows[k][col]:
                sel = k
                break
        if sel is None:
            continue
        rows[i], rows[sel] = rows[sel], rows[i]
        f = rows[i][col]
        rows[i] = [x / f for x in rows[i]]
        for k in range(len(rows)):
            if k != i and rows[k][col]:
                g = rows[k][col]
                rows[k] = [a - g * b for a, b in zip(rows[k], rows[i])]
        piv.append(col)
        i += 1
    for k in range(i, len(rows)):
        if rows[k][r] != 0:
            return None              # inconsistent
    c = [Fraction(0)] * r
    for k, col in enumerate(piv):
        c[col] = rows[k][r]
    return [int(x) if x.denominator == 1 else x for x in c]


# ------------------------------------------------------------------ the audit
def achievable(name, roots, dim, simple, rank, printed, note=""):
    """For every set of marks, sdim g_0; report the sets giving `printed`."""
    exp = expand(roots, simple, dim)
    if exp is None:
        return None
    r = len(simple)
    hits = []
    for markbits in range(1 << r):
        marks = [(markbits >> k) & 1 for k in range(r)]
        ev = od = 0
        for (c, p) in exp:
            if any(marks[k] and c[k] for k in range(r)):
                continue
            if p:
                od += 1
            else:
                ev += 1
        if (rank + ev, od) == printed:
            hits.append(tuple(marks))
    return hits
