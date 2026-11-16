#!/usr/bin/env python3
"""
sv.py -- exact arithmetic in O^{p|q} = k[x_1..x_p] (x) Lambda(xi_1..xi_q),
truncated in the x-degree, for the Serganova--Vaintrob singularity theory
(Serganova, Vaintrob, Math. Scand. 64 (1989) 251-284).

What it computes for a germ f (even, polynomial):

  I_f       the gradient ideal (d f/d x_i, d f/d xi_j)
  O_f       the local superalgebra O/I_f, its sdim m_0|m_1 (the Milnor number)
  L_f       the "odd layer"  (O / m I_f)_odd  -- by the first-layer theorem this
            is the space in which the extra datum g of a germ f + theta*g with
            one odd parameter theta lives, before the stabiliser of f acts.

Everything is exact (Fraction) linear algebra over Q in the truncated ring.
Truncation is validated by recomputing at D and D+1 and comparing.

Monomial = (exponent tuple in x, frozenset of xi indices).
"""

from fractions import Fraction
from itertools import combinations
import sys


# ---------------------------------------------------------------- the ring


class Ring:
    def __init__(self, p, q, D):
        self.p, self.q, self.D = p, q, D           # D = max total x-degree kept

    def xmons(self):
        """all x-monomials of total degree <= D"""
        p, D = self.p, self.D
        if p == 0:
            yield ()
            return
        def rec(i, left, acc):
            if i == p - 1:
                for e in range(left + 1):
                    yield acc + (e,)
                return
            for e in range(left + 1):
                yield from rec(i + 1, left - e, acc + (e,))
        yield from rec(0, D, ())

    def ximons(self):
        for k in range(self.q + 1):
            for S in combinations(range(self.q), k):
                yield frozenset(S)

    def basis(self):
        return [(a, S) for a in self.xmons() for S in self.ximons()]

    def parity(self, mon):
        return len(mon[1]) % 2


def sign_shuffle(S, T):
    """sign of xi_S * xi_T -> xi_{S u T}; 0 if they meet"""
    if S & T:
        return 0
    n = sum(1 for s in S for t in T if s > t)
    return -1 if n % 2 else 1


def mul(R, f, g):
    """f, g : dict mon -> Fraction"""
    out = {}
    for (a, S), ca in f.items():
        for (b, T), cb in g.items():
            sgn = sign_shuffle(S, T)
            if sgn == 0:
                continue
            e = tuple(ai + bi for ai, bi in zip(a, b)) if R.p else ()
            if sum(e) > R.D:
                continue                       # truncated away
            key = (e, S | T)
            out[key] = out.get(key, Fraction(0)) + sgn * ca * cb
    return {k: v for k, v in out.items() if v}


def d_dx(R, f, i):
    out = {}
    for (a, S), c in f.items():
        if a[i] == 0:
            continue
        e = list(a)
        n = e[i]
        e[i] -= 1
        key = (tuple(e), S)
        out[key] = out.get(key, Fraction(0)) + c * n
    return {k: v for k, v in out.items() if v}


def d_dxi(R, f, j):
    """left derivative d/d xi_j"""
    out = {}
    for (a, S), c in f.items():
        if j not in S:
            continue
        sgn = -1 if sum(1 for s in S if s < j) % 2 else 1
        key = (a, S - {j})
        out[key] = out.get(key, Fraction(0)) + sgn * c
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------------ linear algebra


def rref_dims(R, vectors):
    """
    vectors: list of dicts.  Returns (dim_even, dim_odd) of their span,
    the span being homogeneous (each input vector is parity-homogeneous).
    Also returns the set of pivot monomials, per parity.
    """
    res = {}
    for par in (0, 1):
        rows = []
        for v in vectors:
            pars = {R.parity(m) for m in v}
            if not pars:
                continue
            assert len(pars) == 1, "inhomogeneous vector"
            if pars.pop() == par:
                rows.append(dict(v))
        pivots = {}                      # pivot monomial -> reduced row
        order = {}
        for r in rows:
            r = dict(r)
            while r:
                m = max(r, key=lambda k: (sum(k[0]), k[0], sorted(k[1])))
                if m in pivots:
                    c = r[m] / pivots[m][m]
                    for k, v in pivots[m].items():
                        r[k] = r.get(k, Fraction(0)) - c * v
                        if r[k] == 0:
                            del r[k]
                else:
                    pivots[m] = r
                    break
        res[par] = set(pivots)
    return res


def ideal_span(R, gens, multipliers):
    """span of {m * g : g in gens, m in multipliers} as a list of vectors"""
    out = []
    for g in gens:
        for m in multipliers:
            v = mul(R, {m: Fraction(1)}, g)
            if v:
                out.append(v)
    return out


def homog_parts(R, f):
    """split a vector into (even, odd) parts"""
    ev = {m: c for m, c in f.items() if R.parity(m) == 0}
    od = {m: c for m, c in f.items() if R.parity(m) == 1}
    return ev, od


# ---------------------------------------------------------------- the invariants


def analyse(p, q, f_terms, D, verbose=False):
    """
    f_terms: list of (coefficient, x-exponent tuple, tuple of xi indices)
    Returns dict with sdim O_f = (m0, m1) and dim (O/mI_f)_odd.
    """
    R = Ring(p, q, D)
    f = {}
    for c, a, S in f_terms:
        key = (tuple(a), frozenset(S))
        f[key] = f.get(key, Fraction(0)) + Fraction(c)
    f = {k: v for k, v in f.items() if v}

    gens = []
    for i in range(p):
        g = d_dx(R, f, i)
        if g:
            gens.append(g)
    for j in range(q):
        g = d_dxi(R, f, j)
        if g:
            gens.append(g)

    allmon = R.basis()
    mmon = [m for m in allmon if not (sum(m[0]) == 0 and len(m[1]) == 0)]

    piv_I = rref_dims(R, ideal_span(R, gens, allmon))
    piv_mI = rref_dims(R, ideal_span(R, gens, mmon))

    tot = {0: 0, 1: 0}
    for m in allmon:
        tot[R.parity(m)] += 1

    m0 = tot[0] - len(piv_I[0])
    m1 = tot[1] - len(piv_I[1])
    layer = tot[1] - len(piv_mI[1])
    # the layer lives in m_odd = O_odd (odd elements have no constant term),
    # so no correction is needed for parity 1.
    return {"sdim": (m0, m1), "layer": layer, "D": D}


def stable(p, q, f_terms, D0=None, verbose=False):
    """recompute at D and D+1; report only if the answer is stable"""
    n_est = D0 or 8
    a = analyse(p, q, f_terms, n_est)
    b = analyse(p, q, f_terms, n_est + 1)
    if a["sdim"] != b["sdim"] or a["layer"] != b["layer"]:
        return {"sdim": None, "layer": None, "unstable": (a, b)}
    return a


# ---------------------------------------------------------------- germ parser


def quotient_basis(p, q, f_terms, D, min_mdeg=2):
    """
    A monomial basis of  N_f = (m^{min_mdeg})_odd / (m I_f)_odd,
    i.e. the space in which the extra datum g of the germ f + theta*g lives.
    Returned as a list of monomials (x-exponents, xi-set).
    """
    R = Ring(p, q, D)
    f = {}
    for c, a, S in f_terms:
        key = (tuple(a), frozenset(S))
        f[key] = f.get(key, Fraction(0)) + Fraction(c)
    f = {k: v for k, v in f.items() if v}

    gens = []
    for i in range(p):
        g = d_dx(R, f, i)
        if g:
            gens.append(g)
    for j in range(q):
        g = d_dxi(R, f, j)
        if g:
            gens.append(g)

    allmon = R.basis()
    mmon = [m for m in allmon if not (sum(m[0]) == 0 and len(m[1]) == 0)]
    piv = rref_dims(R, ideal_span(R, gens, mmon))[1]

    def mdeg(m):
        return sum(m[0]) + len(m[1])

    out = [m for m in allmon
           if R.parity(m) == 1 and mdeg(m) >= min_mdeg and m not in piv]
    return sorted(out, key=lambda m: (mdeg(m), sum(m[0]), sorted(m[1])))


def show_mon(m, xnames="xyz", xiname="\\xi"):
    a, S = m
    parts = []
    for i, e in enumerate(a):
        if e == 1:
            parts.append(xnames[i])
        elif e > 1:
            parts.append(f"{xnames[i]}^{e}")
    for j in sorted(S):
        parts.append(f"{xiname}_{j+1}")
    return "".join(parts) if parts else "1"


def germ(p, q, terms):
    """terms: list of (coef, x-exponents tuple, xi index tuple)"""
    return [(c, a, S) for c, a, S in terms]


def X(p, i, k=1):
    e = [0] * p
    e[i] = k
    return tuple(e)


def O(p):
    return tuple([0] * p)
