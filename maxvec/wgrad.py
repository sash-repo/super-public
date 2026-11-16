#!/usr/bin/env python3
"""
wgrad.py -- which Z-grading is each row of table (96) graded in,
            and is that grading a W-grading of the ambient?

Exact arithmetic over Q, no dependencies beyond tools/exactlin.py.

A vector field on the superspace with even indeterminates u_0..u_{ne-1} and odd
indeterminates u_ne..u_{ne+no-1} is a dict

    { (mono, k) : Fraction }        mono = (a, S),  a = even exponents, S = odd indices,

standing for sum coeff * u^mono * d/du_k.  The bracket is

    [f d_i, g d_j] = f d_i(g) d_j - (-1)^{p(f d_i) p(g d_j)} g d_j(f) d_i.

Given a weight vector w on the indeterminates, the monomial field u^mono d_k has
weight w.mono - w_k; a subalgebra spanned by fields F is *graded* for w iff every
F splits inside its own span, i.e. iff every F is already w-homogeneous (all its
terms of one weight) -- which is what is tested here, plus the weaker span test.

Validation: every algebra built here is checked to be closed under the bracket in
the truncated range, which is what pins down the sign conventions.
"""

import sys
from fractions import Fraction as F
from itertools import combinations

import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank  # noqa: E402


# ------------------------------------------------------------------ monomials

def mono_par(m, ne):
    return len(m[1]) % 2


def mono_mul(m1, m2):
    """(sign, mono) or (0, None) if the product vanishes."""
    a1, S1 = m1
    a2, S2 = m2
    if set(S1) & set(S2):
        return 0, None
    a = tuple(x + y for x, y in zip(a1, a2))
    # sign of sorting S1 ++ S2
    sign = 1
    merged = list(S1) + list(S2)
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            if merged[i] > merged[j]:
                sign = -sign
    return sign, (a, tuple(sorted(merged)))


def mono_deriv(k, m, ne):
    """d/du_k of the monomial m; returns (coeff, mono) or None."""
    a, S = m
    if k < ne:
        if a[k] == 0:
            return None
        b = list(a)
        b[k] -= 1
        return F(a[k]), (tuple(b), S)
    j = k - ne
    if j not in S:
        return None
    pos = S.index(j)
    sign = (-1) ** pos
    T = tuple(x for x in S if x != j)
    return F(sign), (a, T)


# -------------------------------------------------------------- vector fields

class VF:
    """A polynomial vector field: {(mono, k): coeff}."""

    def __init__(self, terms=None, ne=0, no=0):
        self.ne, self.no = ne, no
        self.t = {}
        if terms:
            for key, c in terms.items():
                if c:
                    self.t[key] = self.t.get(key, F(0)) + c
            self.t = {k: v for k, v in self.t.items() if v}

    def par(self, key):
        """parity of the term u^mono d_k."""
        mono, k = key
        return (len(mono[1]) + (1 if k >= self.ne else 0)) % 2

    def add(self, other, c=F(1)):
        r = dict(self.t)
        for k, v in other.t.items():
            r[k] = r.get(k, F(0)) + c * v
        return VF({k: v for k, v in r.items() if v}, self.ne, self.no)

    def __bool__(self):
        return bool(self.t)

    def weight(self, w):
        """The set of weights occurring; w is a list over all indeterminates."""
        out = set()
        for (mono, k), c in self.t.items():
            a, S = mono
            d = sum(w[i] * a[i] for i in range(self.ne))
            d += sum(w[self.ne + j] for j in S)
            out.add(d - w[k])
        return out

    def homog_parts(self, w):
        parts = {}
        for (mono, k), c in self.t.items():
            a, S = mono
            d = sum(w[i] * a[i] for i in range(self.ne))
            d += sum(w[self.ne + j] for j in S)
            d -= w[k]
            parts.setdefault(d, {})[(mono, k)] = c
        return {d: VF(t, self.ne, self.no) for d, t in parts.items()}


def bracket(X, Y):
    ne, no = X.ne, X.no
    out = {}
    for (m1, i), c1 in X.t.items():
        p1 = X.par((m1, i))
        for (m2, j), c2 in Y.t.items():
            p2 = Y.par((m2, j))
            # f d_i (g) d_j
            d = mono_deriv(i, m2, ne)
            if d is not None:
                coef, m = d
                s, mm = mono_mul(m1, m)
                if s:
                    out[(mm, j)] = out.get((mm, j), F(0)) + c1 * c2 * coef * s
            # -(-1)^{p1 p2} g d_j(f) d_i
            d = mono_deriv(j, m1, ne)
            if d is not None:
                coef, m = d
                s, mm = mono_mul(m2, m)
                if s:
                    sign = -1 if (p1 * p2) % 2 == 0 else 1   # -(-1)^{p1p2}
                    out[(mm, i)] = out.get((mm, i), F(0)) + sign * c1 * c2 * coef * s
    return VF({k: v for k, v in out.items() if v}, ne, no)


# ------------------------------------------------------------------ span/rank

def monomials(ne, no, maxdeg):
    """all monomials of total degree <= maxdeg (each odd variable degree 1)."""
    out = []

    def even_exps(k, left):
        if k == ne:
            yield ()
            return
        for e in range(left + 1):
            for rest in even_exps(k + 1, left - e):
                yield (e,) + rest

    for total in range(maxdeg + 1):
        for r in range(min(no, total) + 1):
            for S in combinations(range(no), r):
                for a in even_exps(0, total - r):
                    if sum(a) + r == total:
                        out.append((a, S))
    return out


def to_vec(X, index):
    v = [F(0)] * len(index)
    for key, c in X.t.items():
        if key in index:
            v[index[key]] = c
        else:
            return None          # outside the truncation
    return v


class Span:
    """Echelon over Q for repeated membership tests (one writer, built once)."""

    def __init__(self, index):
        self.index = index
        self.rows = []

    def _reduce(self, v):
        v = list(v)
        for piv, row in self.rows:
            if v[piv]:
                c = v[piv]
                for i in range(len(v)):
                    v[i] -= c * row[i]
        return v

    def add(self, v):
        v = self._reduce(v)
        for i, x in enumerate(v):
            if x:
                v = [y / x for y in v]
                self.rows.append((i, v))
                self.rows.sort(key=lambda r: r[0])
                return True
        return False

    def contains(self, v):
        return not any(self._reduce(v))

    def dim(self):
        return len(self.rows)


# ------------------------------------------------------------------- algebras

def vect_basis(ne, no, maxdeg):
    """all monomial fields u^mono d_k with deg(mono) <= maxdeg."""
    out = []
    for m in monomials(ne, no, maxdeg):
        for k in range(ne + no):
            out.append(VF({(m, k): F(1)}, ne, no))
    return out


def divergence(X):
    """Div(sum f_k d_k) = sum (-1)^{p(f_k) p(u_k)} d_k(f_k), as a polynomial dict."""
    ne = X.ne
    out = {}
    for (m, k), c in X.t.items():
        d = mono_deriv(k, m, ne)
        if d is None:
            continue
        coef, mm = d
        sign = 1
        if k >= ne and len(m[1]) % 2 == 1:
            sign = -1
        out[mm] = out.get(mm, F(0)) + sign * c * coef
    return {k: v for k, v in out.items() if v}


def svect_basis(ne, no, maxdeg):
    """kernel of Div inside vect, computed degree by degree (exact)."""
    B = vect_basis(ne, no, maxdeg)
    polys = [divergence(X) for X in B]
    keys = sorted({k for p in polys for k in p}, key=str)
    kidx = {k: i for i, k in enumerate(keys)}
    # nullspace of the (len(keys) x len(B)) matrix, columns = basis fields
    from exactlin import nullspace
    M = [[F(0)] * len(B) for _ in keys]
    for j, p in enumerate(polys):
        for k, v in p.items():
            M[kidx[k]][j] = v
    if not keys:
        return B
    ns = nullspace(M)          # rows = basis vectors of the kernel
    out = []
    for vec in ns:
        X = VF({}, ne, no)
        for j, c in enumerate(vec):
            if c:
                X = X.add(B[j], F(c))
        if X:
            out.append(X)
    return out


def euler(ne, no, skip):
    """the Euler operator on all indeterminates except those in `skip`."""
    t = {}
    for k in range(ne + no):
        if k in skip:
            continue
        a = [0] * ne
        S = ()
        if k < ne:
            a[k] = 1
        else:
            S = (k - ne,)
        t[((tuple(a), S), k)] = F(1)
    return VF(t, ne, no)


def poly_field(poly, k, ne, no):
    return VF({(m, k): c for m, c in poly.items()}, ne, no)


def contact_K(m, ne, no, npq, nxi, sgn=1):
    """K_f for f = the monomial m, in k(2n+1|nxi):
       indeterminates  t=0, p_1..p_n = 1..n, q_1..q_n = n+1..2n, xi_1..xi_nxi.
       K_f = (2-E)(f) d_t - H_f + d_t(f) E,  E = Euler on p,q,xi."""
    n = npq
    E = euler(ne, no, skip={0})
    # (2 - E)(f): E(f) = (deg in p,q,xi) * f
    degE = sum(m[0][1:]) + len(m[1])
    c0 = F(2 - degE)
    out = VF({}, ne, no)
    if c0:
        out = out.add(VF({(m, 0): c0}, ne, no))
    # H_f = sum_i ( d_{p_i} f d_{q_i} - d_{q_i} f d_{p_i} )
    #       + sgn * (-1)^{p(f)} sum_j d_{xi_j} f d_{xi_j}
    H = {}
    for i in range(n):
        kp, kq = 1 + i, 1 + n + i
        d = mono_deriv(kp, m, ne)
        if d is not None:
            coef, mm = d
            H[(mm, kq)] = H.get((mm, kq), F(0)) + coef
        d = mono_deriv(kq, m, ne)
        if d is not None:
            coef, mm = d
            H[(mm, kp)] = H.get((mm, kp), F(0)) - coef
    pf = len(m[1]) % 2
    for j in range(nxi):
        kx = ne + j
        d = mono_deriv(kx, m, ne)
        if d is not None:
            coef, mm = d
            s = -sgn * (-1 if pf else 1)     # survey (2.3.2'): -(-1)^{p(f)}
            H[(mm, kx)] = H.get((mm, kx), F(0)) + s * coef
    out = out.add(VF({k: -v for k, v in H.items()}, ne, no))
    # d_t(f) * E
    d = mono_deriv(0, m, ne)
    if d is not None:
        coef, mm = d
        for key, c in E.t.items():
            (em, ek) = key
            s, prod = mono_mul(mm, em)
            if s:
                out = out.add(VF({(prod, ek): coef * c * s}, ne, no))
    return out


def pericontact_M(m, ne, no, n, sgn=1):
    """M_f for f = the monomial m, in m(n) = m(n|n+1).

    Indeterminates: q_1..q_n = 0..n-1 (even);  tau = odd index 0, xi_1..xi_n =
    odd indices 1..n.  Formula (2.3.3)-(2.3.4) of the survey [LS1]:

        M_f  = (2-E)(f) d_tau - Le_f - (-1)^{p(f)} d_tau(f) E,
        Le_f = sum_i ( d_{q_i}(f) d_{xi_i} + (-1)^{p(f)} d_{xi_i}(f) d_{q_i} ),

    with E the Euler operator in all the indeterminates but tau.
    """
    ktau = ne
    E = euler(ne, no, skip={ktau})
    degE = sum(m[0]) + len([j for j in m[1] if j != 0])
    out = VF({}, ne, no)
    c0 = F(2 - degE)
    if c0:
        out = out.add(VF({(m, ktau): c0}, ne, no))
    pf = len(m[1]) % 2
    eps = F(-1 if pf else 1)                     # (-1)^{p(f)}
    for i in range(n):
        kq, kx = i, ne + 1 + i
        d = mono_deriv(kq, m, ne)                # - d_{q_i}(f) d_{xi_i}
        if d is not None:
            coef, mm = d
            out = out.add(VF({(mm, kx): -coef}, ne, no))
        d = mono_deriv(kx, m, ne)                # - (-1)^{p(f)} d_{xi_i}(f) d_{q_i}
        if d is not None:
            coef, mm = d
            out = out.add(VF({(mm, kq): -sgn * eps * coef}, ne, no))
    d = mono_deriv(ktau, m, ne)                  # - (-1)^{p(f)} d_tau(f) E
    if d is not None:
        coef, mm = d
        for key, c in E.t.items():
            (em, ek) = key
            sg, prod = mono_mul(mm, em)
            if sg:
                out = out.add(VF({(prod, ek): -eps * coef * c * sg}, ne, no))
    return out


def ham_H(m, ne, no, n, nxi, sgn=1):
    """H_f in h(2n|nxi): p_1..p_n = 0..n-1, q_1..q_n = n..2n-1, xi_j = ne+j."""
    out = {}
    for i in range(n):
        kp, kq = i, n + i
        d = mono_deriv(kp, m, ne)
        if d is not None:
            coef, mm = d
            out[(mm, kq)] = out.get((mm, kq), F(0)) + coef
        d = mono_deriv(kq, m, ne)
        if d is not None:
            coef, mm = d
            out[(mm, kp)] = out.get((mm, kp), F(0)) - coef
    pf = len(m[1]) % 2
    for j in range(nxi):
        kx = ne + j
        d = mono_deriv(kx, m, ne)
        if d is not None:
            coef, mm = d
            out[(mm, kx)] = out.get((mm, kx), F(0)) - sgn * (-1 if pf else 1) * coef
    return VF({k: v for k, v in out.items() if v}, ne, no)


def le_D(m, ne, no, n, sgn=1):
    """D_f in le(n): x_1..x_n = 0..n-1 even, xi_1..xi_n odd."""
    out = {}
    pf = len(m[1]) % 2
    for i in range(n):
        kx, kxi = i, ne + i
        d = mono_deriv(kx, m, ne)
        if d is not None:
            coef, mm = d
            out[(mm, kxi)] = out.get((mm, kxi), F(0)) + coef
        d = mono_deriv(kxi, m, ne)
        if d is not None:
            coef, mm = d
            out[(mm, kx)] = out.get((mm, kx), F(0)) + sgn * (-1 if pf else 1) * coef
    return VF({k: v for k, v in out.items() if v}, ne, no)


# --------------------------------------------------------------------- checks

def field_deg(X):
    return max(sum(m[0]) + len(m[1]) for (m, k) in X.t)


def closed(fields, ne, no, maxdeg, label):
    """Is Span(fields) closed under the bracket?

    The list is truncated, so a bracket may legitimately land above the top
    degree present in it; such brackets are not evidence of anything and are
    skipped.  The bound is read off the fields themselves, not guessed.
    """
    top = max(field_deg(X) for X in fields)
    idx_keys = [(m, k) for m in monomials(ne, no, top) for k in range(ne + no)]
    index = {k: i for i, k in enumerate(idx_keys)}
    S = Span(index)
    for X in fields:
        v = to_vec(X, index)
        if v is None:
            return None, f"{label}: truncation too small"
        S.add(v)
    bad, tested = 0, 0
    for i, X in enumerate(fields):
        for Y in fields[i:]:
            C = bracket(X, Y)
            if not C or field_deg(C) > top:
                continue
            tested += 1
            v = to_vec(C, index)
            if v is None or not S.contains(v):
                bad += 1
    return bad == 0, (f"{label}: {len(fields)} fields (deg <= {top}), "
                      f"{tested} brackets tested, {bad} outside the span")


def graded_for(fields, w, ne, no):
    """is every field w-homogeneous?  returns (True/False, first witness)."""
    for X in fields:
        ws = X.weight(w)
        if len(ws) > 1:
            return False, X
    return True, None


def irreducible_on_minus_one(fields, w, ne, no, index_deg):
    """Is g_0 irreducible on g_{-1}?   (A necessary condition for a W-grading.)

    Returns (irreducible?, dim g_{-1}, dim of the g_0-submodule generated by the
    'smallest' vector) -- we test every basis vector as a generator.
    """
    gm1 = [X for X in fields if X.weight(w) == {-1}]
    g0 = [X for X in fields if X.weight(w) == {0}]
    if not gm1:
        return None, 0, 0
    keys = sorted({k for X in gm1 for k in X.t}, key=str)
    index = {k: i for i, k in enumerate(keys)}
    full = len(gm1)
    worst = full
    for seed in gm1:
        S = Span(index)
        v = to_vec(seed, index)
        S.add(v)
        frontier = [seed]
        while frontier:
            nxt = []
            for X in frontier:
                for A in g0:
                    C = bracket(A, X)
                    if not C:
                        continue
                    v = to_vec(C, index)
                    if v is None:
                        continue
                    if S.add(v):
                        nxt.append(C)
            frontier = nxt
        worst = min(worst, S.dim())
    # dim of g_{-1} as a space
    Sfull = Span(index)
    for X in gm1:
        Sfull.add(to_vec(X, index))
    return worst == Sfull.dim(), Sfull.dim(), worst
