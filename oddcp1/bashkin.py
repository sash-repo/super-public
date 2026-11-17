"""Cech cohomology of the graded tangent sheaf of a split (CP^1, Lambda E),
E = + L_{-k_j}: an independent engine for Sections 14 and 16 (Bashkin's papers).

Nothing here uses Bunegina--Onishchik's Prop. 18 (the decomposition of ST(E)_q
into line bundles): the whole point is to recompute H^0 and H^1 from the honest
transition functions, so that the numbers in the papers are checked against
something that shares no assumption with them.

Conventions, those of the papers:

    y = x^{-1},   eta_i = x^{-k_i} xi_i,

so that, on the overlap,

    d/dx = -y^2 d/dy - y sum_i k_i eta_i d/deta_i,
    d/dy = -x^2 d/dx - x sum_i k_i  xi_i d/dxi_i,
    d/deta_t = x^{k_t} d/dxi_t.

A basis of T_q over F is
    u_I   = xi_I d/dx          (|I| = q),
    w_{J,t} = xi_J d/dxi_t     (|J| = q+1, t arbitrary),
and Bashkin writes v_{I,t} = w_{I+{t}, t} = xi_I xi_t d/dxi_t for t not in I.
An element of T_q(U_0 cap U_1) is a Laurent polynomial in x times these.

The two subspaces whose sum is the coboundaries B^1 are

    T_q(U_0): spanned by  x^r u_I  and  x^r w_{J,t},           r >= 0,
    T_q(U_1): spanned by  y^s eta_I d/dy  and  y^s eta_J d/deta_t,  s >= 0,

which in the U_0 frame are (this is all the engine needs to know)

    y^s eta_I d/dy   = -x^{2-s-k_I} u_I
                       - x^{1-s-k_I} sum_{i not in I} k_i w_{I+{i}, i},
    y^s eta_J d/deta_t = x^{k_t-s-k_J} w_{J,t}.
"""

from fractions import Fraction
from itertools import combinations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank                                        # noqa: E402


# --------------------------------------------------------------- the generators

def gens(m, q):
    """Basis of T_q over F: ('u', I) and ('w', J, t)."""
    out = [("u", I) for I in combinations(range(1, m + 1), q)] if q >= 0 else []
    if q + 1 >= 0:
        for J in combinations(range(1, m + 1), q + 1):
            for t in range(1, m + 1):
                out.append(("w", J, t))
    return out


def sign_sort(seq):
    """Sort a tuple of distinct indices, returning (sorted, sign)."""
    seq = list(seq)
    sgn = 1
    for i in range(len(seq)):
        for j in range(len(seq) - 1 - i):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                sgn = -sgn
    return tuple(seq), sgn


# ------------------------------------------------------------------- the frames

def U0_sections(m, q, ks, rmax):
    """x^r g for every generator g and 0 <= r <= rmax."""
    for g in gens(m, q):
        for r in range(rmax + 1):
            yield {(g, r): Fraction(1)}


def U1_sections(m, q, ks, smax):
    """The U_1 frame, written in the U_0 one."""
    kof = lambda S: sum(ks[i - 1] for i in S)
    # y^s eta_I d/dy
    for I in (combinations(range(1, m + 1), q) if q >= 0 else []):
        kI = kof(I)
        for s in range(smax + 1):
            el = {(("u", I), 2 - s - kI): Fraction(-1)}
            for i in range(1, m + 1):
                if i in I:
                    continue
                J, sgn = sign_sort(tuple(I) + (i,))
                key = (("w", J, i), 1 - s - kI)
                el[key] = el.get(key, Fraction(0)) - sgn * ks[i - 1]
            yield {k: v for k, v in el.items() if v}
    # y^s eta_J d/deta_t
    for J in (combinations(range(1, m + 1), q + 1) if q + 1 >= 0 else []):
        kJ = kof(J)
        for t in range(1, m + 1):
            for s in range(smax + 1):
                yield {(("w", J, t), ks[t - 1] - s - kJ): Fraction(1)}


# ---------------------------------------------------------------- linear algebra

def _rows(elems, window, m, q):
    """Coordinate rows of elems in the window of exponents, dropping anything
    that pokes out of it (used only for the coboundary generators, whose
    out-of-window part is handled by taking the window wide enough)."""
    cols = {}
    for g in gens(m, q):
        for e in range(-window, window + 1):
            cols[(g, e)] = len(cols)
    rows = []
    for el in elems:
        if any(abs(e) > window for (_, e) in el):
            continue
        row = [Fraction(0)] * len(cols)
        for (g, e), c in el.items():
            row[cols[(g, e)]] = c
        rows.append(row)
    return rows, cols


def h0_h1(ks, q, window=None):
    """(dim H^0, dim H^1) of T_q, computed in a truncated Laurent window.

    H^0 = T_q(U_0) cap T_q(U_1); H^1 = T_q(U_0 cap U_1) / (T_q(U_0)+T_q(U_1)).
    The window is taken so wide that both answers are stable in it; the caller
    can raise it and see nothing change.
    """
    m = len(ks)
    W = window or (2 * max(ks) * (m + 2) + 8)
    g = gens(m, q)
    if not g:
        return 0, 0
    A = list(U0_sections(m, q, ks, W))
    B = list(U1_sections(m, q, ks, W + 2 * sum(ks) + 4))
    rA, cols = _rows(A, W, m, q)
    rB, _ = _rows(B, W, m, q)
    ra, rb = rank(rA), rank(rB)
    rab = rank(rA + rB)
    # H^0 = intersection: dim = ra + rb - dim(sum)
    h0 = ra + rb - rab
    # H^1: the whole overlap module, truncated to the window, modulo the sum.
    # Only the exponents that the window holds entirely are counted, so we
    # restrict to a strictly smaller window for the numerator.
    W2 = W - (2 * sum(ks) + 6)
    whole = []
    for gg in g:
        for e in range(-W2, W2 + 1):
            whole.append({(gg, e): Fraction(1)})
    rw, _ = _rows(whole, W, m, q)
    h1 = rank(rw + rA + rB) - rab
    return h0, h1


def in_H1_span(ks, q, cocycles, window=None):
    """(independent?, spans?) for a list of cocycles, each a dict as above."""
    m = len(ks)
    W = window or (2 * max(ks) * (m + 2) + 8)
    A = list(U0_sections(m, q, ks, W))
    B = list(U1_sections(m, q, ks, W + 2 * sum(ks) + 4))
    rA, _ = _rows(A, W, m, q)
    rB, _ = _rows(B, W, m, q)
    base = rA + rB
    rb = rank(base)
    rc, _ = _rows(cocycles, W, m, q)
    indep = rank(base + rc) - rb == len(cocycles)
    _, h1 = h0_h1(ks, q, window)
    spans = rank(base + rc) - rb == h1
    return indep, spans, rank(base + rc) - rb, h1


# ----------------------------------------------------------------- cocycle sugar

def u(I, r, c=1):
    I, sgn = sign_sort(I)
    return {(("u", I), r): Fraction(c) * sgn}


def w(J, t, r, c=1):
    J, sgn = sign_sort(J)
    return {(("w", J, t), r): Fraction(c) * sgn}


def add(*els):
    out = {}
    for el in els:
        for k, v in el.items():
            out[k] = out.get(k, Fraction(0)) + v
    return {k: v for k, v in out.items() if v}
