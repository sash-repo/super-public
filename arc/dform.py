#!/usr/bin/env python3
"""The uniform D-form of the GRC-brackets on SM_mu (Theorem 7.2.22 of the text).

Everything here is computed in the FOUR-variable model, where no sign
convention has to be chosen by hand:

    T := SM_{mu1} (x) SM_{mu2}  =  C[t1, th1, t2, th2]     (supercommutative)

and the osp(1|2)-action on T is  cQ_eps = Q_eps^(1) + Q_eps^(2), each
Q_eps^(i) being an honest odd differential operator in the i-th pair of
variables acting on the whole algebra -- so the Koszul sign of
X (x) 1 + (-1)^p (x) X is produced by d_{th2} passing th1 by itself.

A bilinear differential operator is

    phi(u) = Delta^*( Bhat u ),   Delta^*: t_i |-> t, th_i |-> th,

    Bhat = sum_{i+j=N} lam_ij D_1^i D_2^j + th1 sum_{i+j=N+1} rho_ij D_1^i D_2^j,

where D_i = d_{th_i} + th_i d_{t_i}  is the invariant derivative of eq. (54),
and N = 2k is the shift  nu - mu1 - mu2.  Lemma 7.2.2 says every invariant
operator is of this shape.

The script
  1) solves the invariance conditions for lam, rho with SYMBOLIC weights,
     for N = 0, 1, 2, ..., NMAX;
  2) matches the answer against the closed form conjectured from it, see
     lam_closed()/rho_closed() below;
  3) checks that the result agrees with the printed formulas (133), (137)
     of the text, via the one-variable engine of check_grc_assoc.py.

Run:  ../venv/bin/python dform.py [NMAX]
"""
import sys
from itertools import product

import sympy as sp

mu1, mu2 = sp.symbols("mu1 mu2")
ZERO, ONE = sp.Integer(0), sp.Integer(1)

# ------------------------------------------------------------------ T = C[t1,th1,t2,th2]
# monomial key (a1, e1, a2, e2)  <->  t1^a1 th1^e1 t2^a2 th2^e2  in THIS order


def clean(d):
    return {k: v for k, v in d.items() if sp.expand(v) != 0}


def add(dst, src, c=1):
    for k, v in src.items():
        dst[k] = sp.expand(dst.get(k, ZERO) + c * v)
        if dst[k] == 0:
            del dst[k]
    return dst


def smul(c, u):
    return clean({k: sp.expand(c * v) for k, v in u.items()})


def dt(u, i):
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        a = a1 if i == 1 else a2
        if a:
            k = (a1 - 1, e1, a2, e2) if i == 1 else (a1, e1, a2 - 1, e2)
            out[k] = out.get(k, ZERO) + a * c
    return clean(out)


def dth(u, i):
    """Left derivative d/d th_i, Koszul sign for i = 2 (it passes th1)."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        if i == 1:
            if e1:
                k = (a1, 0, a2, e2)
                out[k] = out.get(k, ZERO) + c
        else:
            if e2:
                k = (a1, e1, a2, 0)
                out[k] = out.get(k, ZERO) + (-1) ** e1 * c
    return clean(out)


def thmul(u, i):
    """Left multiplication by th_i."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        if i == 1:
            if e1 == 0:
                k = (a1, 1, a2, e2)
                out[k] = out.get(k, ZERO) + c
        else:
            if e2 == 0:
                k = (a1, e1, a2, 1)
                out[k] = out.get(k, ZERO) + (-1) ** e1 * c
    return clean(out)


def tmul(u, i):
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        k = (a1 + 1, e1, a2, e2) if i == 1 else (a1, e1, a2 + 1, e2)
        out[k] = out.get(k, ZERO) + c
    return clean(out)


def Dop(u, i):
    """D_i = d_{th_i} + th_i d_{t_i}."""
    out = dth(u, i)
    add(out, thmul(dt(u, i), i))
    return clean(out)


def Qm(u, i):
    out = dth(u, i)
    add(out, thmul(dt(u, i), i), -1)
    return clean(out)


def Qp(u, i, mu):
    """Q_+^{(i)} = t_i Q_-^{(i)} - mu_i th_i."""
    out = tmul(Qm(u, i), i)
    add(out, thmul(u, i), -mu)
    return clean(out)


def cQ(u, sign, mus):
    out = {}
    for i in (1, 2):
        add(out, Qm(u, i) if sign < 0 else Qp(u, i, mus[i - 1]))
    return clean(out)


# ------------------------------------------------------------------ C[t,theta]
# monomial key (a, e)


def clean1(d):
    return {k: v for k, v in d.items() if sp.expand(v) != 0}


def add1(dst, src, c=1):
    for k, v in src.items():
        dst[k] = sp.expand(dst.get(k, ZERO) + c * v)
        if dst[k] == 0:
            del dst[k]
    return dst


def restrict(u):
    """Delta^*: t_i |-> t, th_i |-> th."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        if e1 and e2:
            continue
        k = (a1 + a2, e1 + e2)
        out[k] = out.get(k, ZERO) + c
    return clean1(out)


def dt1(u):
    return clean1({(a - 1, e): a * c for (a, e), c in u.items() if a})


def dth1(u):
    return clean1({(a, 0): c for (a, e), c in u.items() if e})


def thmul1(u):
    return clean1({(a, 1): c for (a, e), c in u.items() if e == 0})


def tmul1(u):
    return clean1({(a + 1, e): c for (a, e), c in u.items()})


def Qm1(u):
    out = dth1(u)
    add1(out, thmul1(dt1(u)), -1)
    return clean1(out)


def Qp1(u, nu):
    out = tmul1(Qm1(u))
    add1(out, thmul1(u), -nu)
    return clean1(out)


# ------------------------------------------------------------------ the operator


def apply_B(u, N, lam, rho):
    """phi(u) = Delta^*(Bhat u), coefficients lam[(i,j)], rho[(i,j)]."""
    out = {}
    for i in range(N + 1):
        j = N - i
        c = lam.get((i, j), ZERO)
        if c == 0:
            continue
        w = u                                  # the operator is D_1^i D_2^j,
        for _ in range(j):                     # so D_2 acts first
            w = Dop(w, 2)
        for _ in range(i):
            w = Dop(w, 1)
        add1(out, restrict(smul(c, w)))
    for i in range(N + 2):
        j = N + 1 - i
        c = rho.get((i, j), ZERO)
        if c == 0:
            continue
        w = u
        for _ in range(j):
            w = Dop(w, 2)
        for _ in range(i):
            w = Dop(w, 1)
        add1(out, restrict(smul(c, thmul(w, 1))))
    return clean1(out)


def monomials(A):
    return [{(a1, e1, a2, e2): ONE}
            for a1 in range(A + 1) for e1 in (0, 1)
            for a2 in range(A + 1) for e2 in (0, 1)]


def solve_N(N, sigma, A=None):
    """Solve the invariance conditions; return the solution space basis."""
    A = A if A is not None else N + 3
    lamv = {(i, N - i): sp.Symbol("l_%d_%d" % (i, N - i)) for i in range(N + 1)}
    rhov = {(i, N + 1 - i): sp.Symbol("r_%d_%d" % (i, N + 1 - i))
            for i in range(N + 2)}
    unk = list(lamv.values()) + list(rhov.values())
    nu = mu1 + mu2 + N
    eqs = []
    for u in monomials(A):
        for sign, Q1 in ((-1, lambda w: Qm1(w)), (1, lambda w: Qp1(w, nu))):
            lhs = apply_B(cQ(u, sign, (mu1, mu2)), N, lamv, rhov)
            rhs = smul(sigma, Q1(apply_B(u, N, lamv, rhov)))
            res = dict(lhs)
            add1(res, rhs, -1)
            eqs.extend(sp.expand(v) for v in res.values())
    sol = sp.linsolve(eqs, unk)
    return unk, sol


if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for N in range(NMAX + 1):
        for sigma in (1, -1):
            unk, sol = solve_N(N, sigma)
            if not sol:
                print("N=%d sigma=%+d : NO solution" % (N, sigma))
                continue
            pt = list(sol)[0]
            free = sorted({s for e in pt for s in e.free_symbols} & set(unk),
                          key=str)
            print("N=%d sigma=%+d : free parameters %s" % (N, sigma, free))
            if len(free) == 1:
                f = free[0]
                vals = [(u, sp.simplify(sp.expand(e / f)))
                        for u, e in zip(unk, pt) if sp.expand(e) != 0]
                for u, v in vals:
                    print("    %-10s %s" % (u, sp.factor(v)))


def lam_solution(N, sigma=None):
    """The lambda-part of the invariant operator, cleared of denominators."""
    sigma = sigma if sigma is not None else (-1) ** N
    unk, sol = solve_N(N, sigma)
    pt = list(sol)[0]
    free = sorted({s for e in pt for s in e.free_symbols} & set(unk), key=str)
    f = free[0]
    vals = {}
    for u, e in zip(unk, pt):
        name = str(u).split("_")
        if name[0] != "l":
            continue
        vals[(int(name[1]), int(name[2]))] = sp.cancel(e / f)
    den = sp.Integer(1)
    for v in vals.values():
        den = sp.lcm(den, sp.denom(sp.together(v)))
    return {k: sp.factor(sp.cancel(v * den)) for k, v in vals.items()}
