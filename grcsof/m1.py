#!/usr/bin/env python3
"""m(1) (pericontact, odd time) in the conventions of [BLS1]:

    alpha_0 = d(tau) + xi dq + q d(xi),
    M_f = (2-E)(f) d_tau - Le_f - (-1)^{p(f)} d_tau(f) E,   E = q d_q + xi d_xi,
    Le_f = d_q(f) d_xi + (-1)^{p(f)} d_xi(f) d_q,                  [BLS1 (26),(27)]
    {f,g}_m.b. = (2-E)(f) d_tau(g) + (-1)^{p(f)} d_tau(f) (2-E)(g)
                 - {f,g}_B.b.,
    {f,g}_B.b. = d_q(f) d_xi(g) + (-1)^{p(f)} d_xi(f) d_q(g).      [BLS1 (30),(32)]
    D_1 = d_tau (odd),  D_xi = xi d_tau - d_q,  D_q = q d_tau + d_xi (both even),
    M_xi = xi d_tau + d_q, M_q = q d_tau - d_xi.                   [BLS1 (41)]
    F_{a,b} = F alpha_0^a vol^b,
    L_{M_f}(phi) = M_f(phi) + 2(-1)^{p(f)} (a d_tau(f) + b Delta_m(f)) phi,
    Delta_m = (1-E) d_tau - d^2/(d_xi d_q).                        [BLS1 (37),(38)]

Monomial (a,e1,e2)  <->  q^a tau^e1 xi^e2.
"""
from sympy import S, expand, symbols
from super12 import (clean, add, smul, mul, mono, parity, par_mon,
                     d_even, d_odd1, d_odd2)

d_q, d_tau, d_xi = d_even, d_odd1, d_odd2


def E_m(u):
    """q d_q + xi d_xi  (all coordinates except tau)."""
    return clean({k: (k[0] + k[2]) * v for k, v in u.items()})


def M(f):
    p = parity(f)
    sgn = -1 if p else 1
    two_minus_E = add(smul(2, f), smul(-1, E_m(f)))
    dqf, dxif, dtauf = d_q(f), d_xi(f), d_tau(f)

    def r(u):
        out = mul(two_minus_E, d_tau(u))
        # -Le_f
        out = add(out, smul(-1, add(mul(dqf, d_xi(u)), smul(sgn, mul(dxif, d_q(u))))))
        # -(-1)^p(f) d_tau(f) E
        out = add(out, smul(-sgn, mul(dtauf, E_m(u))))
        return out
    return r


def mbb(f, g):
    p = parity(f)
    sgn = -1 if p else 1
    a = mul(add(smul(2, f), smul(-1, E_m(f))), d_tau(g))
    b = smul(sgn, mul(d_tau(f), add(smul(2, g), smul(-1, E_m(g)))))
    bb = add(mul(d_q(f), d_xi(g)), smul(sgn, mul(d_xi(f), d_q(g))))
    return add(a, b, smul(-1, bb))


def D_1(u):
    return d_tau(u)


def D_xi(u):                              # even
    return add(mul(mono(0, 0, 1), d_tau(u)), smul(-1, d_q(u)))


def D_q(u):                               # even
    return add(mul(mono(1, 0, 0), d_tau(u)), d_xi(u))


def Delta_m(f):
    """(1-E) d_tau (f) - d_xi d_q (f)."""
    a = d_tau(f)
    return add(a, smul(-1, E_m(a)), smul(-1, d_xi(d_q(f))))


def L(f, a, b):
    """Lie derivative on F_{a,b} = F alpha_0^a vol^b, [BLS1 (38)]."""
    s = -1 if parity(f) else 1
    c = smul(2 * s, add(smul(a, d_tau(f)), smul(b, Delta_m(f))))
    Mf = M(f)

    def r(u):
        return add(Mf(u), mul(c, u))
    return r


# pe(2) subset m(1):  generating functions of degree <= 1
PE = {'1': mono(), 'q': mono(1), 'xi': mono(0, 0, 1),
      'q2': mono(2), 'qxi': mono(1, 0, 1), 'tau': mono(0, 1, 0),
      'qtau': mono(1, 1, 0), 'tauxi': mono(0, 1, 1)}
NEG = ['1', 'q', 'xi']
ZERO = ['q2', 'qxi', 'tau']
POS = ['qtau', 'tauxi']

TESTM = [(a, e, f) for a in range(0, 7) for e in (0, 1) for f in (0, 1)]
