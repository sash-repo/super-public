#!/usr/bin/env python3
"""The closed formula for the ODD GRC-brackets (eq. (137) of the text).

Reading the symbolic solutions of the invariance equations for m = 0, 1 in
the notation of the text gives the following candidate, in which nothing but the printed
gamma_{a,m} occurs.  Write, for k = m + 1/2,

    gamma_{a,m} = (-1)^{a+1} C(mu1+m, m-a) C(mu2+m, a),   gamma_{-1,m} = 0,
                                                          gamma_{m+1,m} = 0,
    C_a = gamma_{a,m} (mu1+a),          a = 0..m,
    E_a = -gamma_{a,m} (mu2+m-a),       a = 0..m.

Then the invariant operator of order k = m + 1/2 is

  [f,g]_k = sum_{0<=a<=m} ( C_a (-1)^{p(f)} f^{(a)} D(g^{(m-a)})
                          + E_a D(f^{(a)}) g^{(m-a)} )
          + theta * sum_{0<=a<=m+1} (-2) (C_a + E_{a-1}) f^{(a)} g^{(m+1-a)}
          + theta * 2 (mu1+mu2+m) sum_{0<=a<=m} gamma_{a,m}
                                       D(f^{(a)}) D(g^{(m-a)}),        (ODD)

the first line being the printed formula (134) and the two theta-lines the
missing part.  For m = 0 this is exactly eq. (135) of the text.

This script CHECKS (ODD) directly against the definition of invariance,

    Q(B(f,g)) = B(Qf,g) + (-1)^{p(f)} B(f,Qg),  Q = Q_-, Q_+,

with SYMBOLIC weights mu1, mu2 -- so a pass is a proof for that m, valid for
all weights at once, not a sample.

Run:  ../venv/bin/python oddclosed.py [MMAX]
"""
import sys

import sympy as sp

from check_grc_assoc import add, mul, dth, thmul, dt, dtn, Qminus, Qplus

mu1, mu2 = sp.symbols("mu1 mu2")
ONE, ZERO = sp.Integer(1), sp.Integer(0)


def smul(c, u):
    return {k: c * v for k, v in u.items() if c * v != 0}


def D(u):                                  # D_theta of eq. (54)
    out = dth(u)
    return add(out, thmul(dt(u)), 1)


def binom(x, n):
    r = ONE
    for i in range(n):
        r *= (x - i)
    return sp.expand(r / sp.factorial(n))


def gamma(a, m):
    if a < 0 or a > m:
        return ZERO
    return (-1) ** (a + 1) * binom(mu1 + m, m - a) * binom(mu2 + m, a)


def Ccoef(a, m):
    return sp.expand(gamma(a, m) * (mu1 + a)) if 0 <= a <= m else ZERO


def Ecoef(a, m):
    return sp.expand(-gamma(a, m) * (mu2 + m - a)) if 0 <= a <= m else ZERO


def bracket_odd(m, f, g, pf):
    """The candidate (ODD) for k = m + 1/2."""
    sgn = -ONE if pf % 2 else ONE
    out = {}
    for a in range(m + 1):
        add(out, smul(sgn * Ccoef(a, m), mul(dtn(f, a), D(dtn(g, m - a)))))
        add(out, smul(Ecoef(a, m), mul(D(dtn(f, a)), dtn(g, m - a))))
    th = {}
    for a in range(m + 2):
        c = -2 * (Ccoef(a, m) + Ecoef(a - 1, m))
        if c != 0:
            add(th, smul(sp.expand(c), mul(dtn(f, a), dtn(g, m + 1 - a))))
    for a in range(m + 1):
        c = 2 * (mu1 + mu2 + m) * gamma(a, m)
        if c != 0:
            add(th, smul(sp.expand(c),
                         mul(D(dtn(f, a)), D(dtn(g, m - a)))))
    add(out, thmul(th))
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}


def monos(A):
    return [({(a, e): ONE}, e) for a in range(A + 1) for e in (0, 1)]


def check(m, A=None):
    """Direct invariance check with symbolic weights."""
    A = A if A is not None else m + 4
    nu = mu1 + mu2 + 2 * m + 1
    bad = []
    for Q in (Qminus, Qplus):
        for f, pf in monos(A):
            for g, pg in monos(A):
                sg = -ONE if pf % 2 else ONE
                lhs = Q(bracket_odd(m, f, g, pf), nu)
                res = dict(lhs)
                add(res, bracket_odd(m, Q(f, mu1), g, (pf + 1) % 2), -1)
                add(res, bracket_odd(m, f, Q(g, mu2), pf), -sg)
                res = {k: sp.simplify(sp.expand(v)) for k, v in res.items()}
                res = {k: v for k, v in res.items() if v != 0}
                if res:
                    bad.append((Q.__name__, f, g, res))
    return bad


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("Direct invariance check of the closed formula (ODD), "
          "symbolic mu1, mu2:")
    allok = True
    for m in range(M + 1):
        bad = check(m)
        k = sp.Rational(2 * m + 1, 2)
        print(f"   k = {str(k):>5}  (m = {m}):  "
              f"{'INVARIANT' if not bad else 'FAILS: ' + str(bad[0][:2])}")
        allok = allok and not bad
    print(f"\nAll orders checked pass: {allok}")
    if allok:
        print("\nThe coefficients, for reference:")
        for m in range(min(M, 3) + 1):
            print(f"  m = {m}:")
            for a in range(m + 1):
                print(f"     C_{a} = {sp.factor(Ccoef(a, m))}"
                      f"     E_{a} = {sp.factor(Ecoef(a, m))}")
            for a in range(m + 2):
                c = sp.factor(sp.expand(-2 * (Ccoef(a, m) + Ecoef(a - 1, m))))
                print(f"     P_{a} = {c}")
            for a in range(m + 1):
                print(f"     Q_{a} = "
                      f"{sp.factor(sp.expand(2*(mu1+mu2+m)*gamma(a, m)))}")
