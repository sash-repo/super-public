#!/usr/bin/env python3
"""Superfunction algebra + contact/pericontact superalgebras on 1|2-dimensional
superstrings, in the conventions of [BLS1] = Bouarroudj-Leites-Shchepochkina,
"Analogs of Bol operators on superstrings", IJAC 32 (2022) 807-835,
arXiv:2110.10504.

Two realizations are implemented:

  K:  k(1|2) in the xi-eta realization, alpha_1 = dt + xi d(eta) + eta d(xi),
      K_f = (2-E)(f) d_t - H_f + d_t(f) E,   E = xi d_xi + eta d_eta,
      H_f = -(-1)^p(f) ( d_xi(f) d_eta + d_eta(f) d_xi ),        [BLS1 (24),(25)]
      D_xi = eta d_t + d_xi,  D_eta = xi d_t + d_eta,
      K_xi = xi d_t - d_eta,  K_eta = eta d_t - d_xi.            [BLS1 (39)]
      Weighted densities  F_{a,b} = F alpha_1^a beta^b  with     [BLS1 (33),(35)]
      div_2 = tilde K_eta tilde K_xi - d_t.

  M:  m(1) in the realization alpha_0 = d(tau) + xi dq + q d(xi),
      M_f = (2-E)(f) d_tau - Le_f - (-1)^p(f) d_tau(f) E,  E = q d_q + xi d_xi,
      Le_f = d_q(f) d_xi + (-1)^p(f) d_xi(f) d_q,               [BLS1 (26),(27)]
      D_xi = xi d_tau - d_q,  D_q = q d_tau + d_xi,             [BLS1 (41)]
      M_xi = xi d_tau + d_q,  M_q = q d_tau - d_xi.
      Weighted densities  F_{a,b} = F alpha_0^a vol^b  with      [BLS1 (37),(38)]
      Delta_m = (1-E) d_tau - d^2/(d_xi d_q).

Everything is exact (sympy rationals / rational functions in the weights).

A superfunction is a dict {(a, e1, e2): coeff}; the monomial is
   K:  t^a xi^e1 eta^e2        (order: t, xi, eta)
   M:  q^a tau^e1 xi^e2        (order: q, tau, xi)
An operator is a python callable on superfunctions (a linear map), so that
composition needs no extra Koszul signs.
"""
from sympy import Rational, expand, simplify, symbols, Poly, S

# ------------------------------------------------------------------ algebra


def clean(u):
    return {k: expand(v) for k, v in u.items() if expand(v) != 0}


def add(*us):
    out = {}
    for u in us:
        for k, v in u.items():
            out[k] = out.get(k, 0) + v
    return clean(out)


def smul(c, u):
    return clean({k: c * v for k, v in u.items()})


def par_mon(m):
    return (m[1] + m[2]) % 2


def parity(u):
    ps = {par_mon(m) for m in u}
    assert len(ps) <= 1, "inhomogeneous parity"
    return ps.pop() if ps else 0


def mul(u, v):
    """product in C[x] (x) Lambda[y1,y2] with the monomial order (x, y1, y2)."""
    out = {}
    for (a1, e1, f1), c1 in u.items():
        for (a2, e2, f2), c2 in v.items():
            if (e1 and e2) or (f1 and f2):
                continue
            # move y1^e2 past y2^f1
            sign = -1 if (f1 and e2) else 1
            k = (a1 + a2, e1 + e2, f1 + f2)
            out[k] = out.get(k, 0) + sign * c1 * c2
    return clean(out)


def d_even(u):
    """d/dx  (x = t resp. q, the even indeterminate)."""
    out = {}
    for (a, e, f), c in u.items():
        if a:
            out[(a - 1, e, f)] = out.get((a - 1, e, f), 0) + a * c
    return clean(out)


def d_odd1(u):
    """left derivative d/dy1 (y1 = xi resp. tau)."""
    out = {}
    for (a, e, f), c in u.items():
        if e:
            out[(a, 0, f)] = out.get((a, 0, f), 0) + c
    return clean(out)


def d_odd2(u):
    """left derivative d/dy2 (y2 = eta resp. xi)."""
    out = {}
    for (a, e, f), c in u.items():
        if f:
            out[(a, e, 0)] = out.get((a, e, 0), 0) + (-1 if e else 1) * c
    return clean(out)


def E_odd(u):
    """y1 d_y1 + y2 d_y2 (counts the odd indeterminates)."""
    return clean({k: (k[1] + k[2]) * v for k, v in u.items()})


def mono(a=0, e=0, f=0, c=1):
    return {(a, e, f): S(c)}


ONE = mono()


def op_mul(g):
    """multiplication by the superfunction g."""
    return lambda u: mul(g, u)


def compose(*ops):
    def r(u):
        for o in reversed(ops):
            u = o(u)
        return u
    return r


def lin(*terms):
    """linear combination: lin((c1,op1),(c2,op2),...)"""
    def r(u):
        return add(*[smul(c, o(u)) for c, o in terms])
    return r


def apply_on_basis(op, monos):
    return {m: op(mono(*m)) for m in monos}
