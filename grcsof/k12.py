#!/usr/bin/env python3
"""k(1|2) in the xi-eta realization: contact fields, weighted densities
F_{a,b} = F alpha^a beta^b, and a classifier of i(osp(2|2))-invariant
unary and binary differential operators.  Conventions: see super12.py.
"""
from sympy import S, expand, symbols, Matrix, simplify, together, factor, Rational
from super12 import (clean, add, smul, mul, mono, ONE, parity, par_mon,
                     d_even, d_odd1, d_odd2, E_odd, op_mul, compose, lin)

# coordinates: monomial (a,e,f)  <->  t^a xi^e eta^f
d_t, d_xi, d_eta = d_even, d_odd1, d_odd2

# ------------------------------------------------------------- contact fields


def K(f):
    """the contact vector field K_f as a linear operator, [BLS1 (24),(25)]."""
    p = parity(f)
    two_minus_E = add(smul(2, f), smul(-1, E_odd(f)))
    dtf = d_t(f)
    dxif, detaf = d_xi(f), d_eta(f)
    sgn = -1 if p else 1                      # -(-1)^p(f)

    def r(u):
        out = mul(two_minus_E, d_t(u))
        # -H_f = +(-1)^p(f) ( d_xi f . d_eta + d_eta f . d_xi )
        out = add(out, smul(sgn, add(mul(dxif, d_eta(u)), mul(detaf, d_xi(u)))))
        out = add(out, mul(dtf, E_odd(u)))
        return out
    return r


def kb(f, g):
    """contact bracket {f,g}_{k.b.}, [BLS1 (31),(29)]:
       (2-E)(f) d_t g - d_t f (2-E)(g) - {f,g}_P.b.,
       {f,g}_P.b. = -(-1)^p(f) ( d_xi f d_eta g + d_eta f d_xi g )."""
    p = parity(f)
    sgn = -1 if p else 1
    a = mul(add(smul(2, f), smul(-1, E_odd(f))), d_t(g))
    b = mul(d_t(f), add(smul(2, g), smul(-1, E_odd(g))))
    pb = smul(-sgn, add(mul(d_xi(f), d_eta(g)), mul(d_eta(f), d_xi(g))))
    return add(a, smul(-1, b), smul(-1, pb))


# the invariant derivatives D_f, [BLS1 (39)]
def D_xi(u):
    return add(mul(mono(0, 0, 1), d_t(u)), d_xi(u))          # eta d_t + d_xi


def D_eta(u):
    return add(mul(mono(0, 1, 0), d_t(u)), d_eta(u))         # xi d_t + d_eta


# ------------------------------------------------------------ second divergence
def Kt_xi(f):
    """tilde K_xi (f) = (-1)^p(f) (d_eta - xi d_t)(f)."""
    s = -1 if parity(f) else 1
    return smul(s, add(d_eta(f), smul(-1, mul(mono(0, 1, 0), d_t(f)))))


def Kt_eta(f):
    """tilde K_eta(f) = (-1)^p(f) (d_xi - eta d_t)(f)."""
    s = -1 if parity(f) else 1
    return smul(s, add(d_xi(f), smul(-1, mul(mono(0, 0, 1), d_t(f)))))


def div2(f):
    """the second divergence, [BLS1 (33)]: div_2 = tK_eta tK_xi - d_t."""
    return add(Kt_eta(Kt_xi(f)), smul(-1, d_t(f)))


def L(f, a, b, sgn2=1):
    """Lie derivative on F_{a,b} = F alpha^{a/2} beta^b, [BLS1 (35), 2.2.1a]:
       L_{K_f}(phi) = K_f(phi) + (a d_t(f) + sgn * b div_2(f)) phi,
       sgn = sgn2 * (-1)^p(f).  With this normalisation a shifts by (-degree)."""
    s = sgn2 * (-1 if parity(f) else 1)
    c = add(smul(a, d_t(f)), smul(s * b, div2(f)))
    Kf = K(f)

    def r(u):
        return add(Kf(u), mul(c, u))
    return r


# ------------------------------------------------------------------ osp(2|2)
OSP = {'1': mono(), 'xi': mono(0, 1, 0), 'eta': mono(0, 0, 1),
       't': mono(1), 'xieta': mono(0, 1, 1),
       'txi': mono(1, 1, 0), 'teta': mono(1, 0, 1), 't2': mono(2)}
NEG = ['1', 'xi', 'eta']
POS = ['txi', 'teta', 't2']
ZERO = ['t', 'xieta']

TEST = [(a, e, f) for a in range(0, 7) for e in (0, 1) for f in (0, 1)]
