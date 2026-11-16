#!/usr/bin/env python3
"""Uniqueness of the GRC-brackets (Theorem 6.1.5 of the text), second ingredient:
the WHOLE space of bilinear
differential operators, not only the D_theta-class.

Theorem 6.1.5 is easily verified inside the class of
operators assembled from D_theta-derivatives of the arguments and the
coefficient theta.  Here the ansatz is the whole space:

    B(f,g) = sum  c_{a,e1,b,e2,d,r,s} * (-1)^{s p(f)} t^r theta^d
                       (d_theta^{e1} d_t^a f) (d_theta^{e2} d_t^b g)

with a,b >= 0, e1,e2,d in {0,1}, r >= 0 the degree of the coefficient in t,
and s in {0,1} the (-1)^{p(f)}-dressing (which is how a map into Pi(SM_nu) is
written as an even one; including both dressings makes the answer independent
of that convention, so a dimension 1 here means uniqueness in either).

The grading is NOT imposed: every term with a+b <= N is offered to the solver,
so the output is the full solution space of

    Q(B(f,g)) = B(Qf,g) + (-1)^{p(f)} B(f,Qg)  for Q = Q_-, Q_+          (INV)

on SM_{mu1} (x) SM_{mu2} --> SM_{nu},  nu = mu1+mu2+s,  s in (1/2)Zee.

Since Q_-^2 = -d_t, invariance forces r = 0;
the run with COEFDEG > 0 verifies that no t-dependent coefficient survives.

Usage:
    python3 fullops.py              dimension scan, s = -2 .. 2N+2
    python3 fullops.py sym          the k <= 9/2 operators with symbolic mu's
"""
import sys
from fractions import Fraction as F

from check_grc_assoc import (Z, add, mul, dt, dth, thmul, tmul, smul, dtn,
                             Qminus, Qplus)

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import nullspace as _ns, rref as _rref, backend
except Exception:                                        # pragma: no cover
    _ns = _rref = None
    backend = lambda: "none"

# ------------------------------------------------------------------ the basis


def basis(N, coefdeg=0):
    """All terms with a+b <= N and coefficient t^r, r <= coefdeg."""
    out = []
    for a in range(N + 1):
        for b in range(N + 1 - a):
            for e1 in (0, 1):
                for e2 in (0, 1):
                    for d in (0, 1):
                        for r in range(coefdeg + 1):
                            for s in (0, 1):
                                out.append((a, e1, b, e2, d, r, s))
    return out


def label(T):
    a, e1, b, e2, d, r, s = T
    p = []
    if s:
        p.append("(-1)^p(f)")
    if r:
        p.append(f"t^{r}" if r > 1 else "t")
    if d:
        p.append("th")
    p.append(("D_th " if e1 else "") + (f"f^({a})" if a else "f"))
    p.append(("D_th " if e2 else "") + (f"g^({b})" if b else "g"))
    return " ".join(p)


def degree(T):
    """Degree in the grading d_t = 1, d_theta = 1/2, theta = -1/2, t = -1;
    an operator of degree k shifts the weight by 2k."""
    a, e1, b, e2, d, r, s = T
    return F(2 * (a + b) + e1 + e2 - d - 2 * r, 2)


def apply_term(T, f, g, pf, ONE=F(1)):
    a, e1, b, e2, d, r, s = T
    u = dtn(f, a)
    if e1:
        u = dth(u)
    v = dtn(g, b)
    if e2:
        v = dth(v)
    out = mul(u, v)
    if d:
        out = thmul(out)
    for _ in range(r):
        out = tmul(out)
    if s and pf % 2:
        out = smul(-ONE, out)
    return out


def monos(A, ONE=F(1)):
    return [({(a, e): ONE}, e) for a in range(A + 1) for e in (0, 1)]


# ------------------------------------------------------- the invariance system
def rows_for(bas, mu1, mu2, nu, A, ONE=F(1), zero=Z):
    n = len(bas)
    rows = []
    for Q in (Qminus, Qplus):
        for f, pf in monos(A, ONE):
            for g, pg in monos(A, ONE):
                sg = -ONE if pf % 2 else ONE
                Qf, Qg = Q(f, mu1), Q(g, mu2)
                cols = {}
                for i, T in enumerate(bas):
                    res = Q(apply_term(T, f, g, pf, ONE), nu)
                    add(res, apply_term(T, Qf, g, (pf + 1) % 2, ONE), -1)
                    add(res, apply_term(T, f, Qg, pf, ONE), -sg)
                    for m, val in res.items():
                        cols.setdefault(m, [zero] * n)[i] += val
                rows.extend(cols.values())
    return rows


def eval_rows(bas, A, ONE=F(1), zero=Z):
    """The evaluation matrix: a null vector of it is a combination of terms
    that is the ZERO map (e.g. theta f g = (-1)^{p(f)} theta f g, both sides
    vanishing on odd f).  Such degeneracies must be divided out, or they are
    miscounted as invariant operators."""
    n = len(bas)
    rows = []
    for f, pf in monos(A, ONE):
        for g, pg in monos(A, ONE):
            cols = {}
            for i, T in enumerate(bas):
                for m, val in apply_term(T, f, g, pf, ONE).items():
                    cols.setdefault(m, [zero] * n)[i] += val
            rows.extend(cols.values())
    return rows


_INDEP = {}


def indep_basis(N, coefdeg=0, A=None):
    """A sublist of basis(N, coefdeg) that is linearly independent AS MAPS —
    the pivot columns of the evaluation matrix."""
    key = (N, coefdeg)
    if key in _INDEP:
        return _INDEP[key]
    bas = basis(N, coefdeg)
    if A is None:
        A = N + 4
    _, piv = _rref(eval_rows(bas, A))
    out = [bas[c] for c in piv]
    _INDEP[key] = out
    return out


def solve(N, mu1, mu2, s, A=None, coefdeg=0):
    """dim and a basis of the space of invariant operators of order <= N
    with nu = mu1 + mu2 + s."""
    bas = indep_basis(N, coefdeg, A)
    if A is None:
        A = N + 4
    rows = rows_for(bas, mu1, mu2, mu1 + mu2 + s, A)
    if _ns is None:
        raise RuntimeError("exactlin missing")
    return bas, _ns(rows)                 # exactlin.nullspace(M[, p]) — no n!


def show(bas, v):
    return " + ".join(f"({c}) {label(T)}" for T, c in zip(bas, v) if c)


# ---------------------------------------------------------------------- main
def scan(N=5, coefdeg=0, weights=None):
    weights = weights or [(F(1, 31), F(5, 31)), (F(2, 7), F(3, 5)),
                          (F(7, 3), F(11, 13))]
    print(f"# whole space of bilinear differential operators, a+b <= {N}, "
          f"coefficient degree in t <= {coefdeg}")
    print(f"# exactlin backend: {backend()};  terms {len(basis(N, coefdeg))}, "
          f"independent as maps: {len(indep_basis(N, coefdeg))}")
    print("#  s = nu - mu1 - mu2 (the shift);  an operator of order k has "
          "s = 2k")
    print(f"{'2s':>4}  {'s':>6} {'order':>7}  dim (3 generic weight pairs)   operator")
    bad = 0
    for s2 in range(-4, 4 * N + 3):
        s = F(s2, 2)
        dims, bases = [], []
        for (m1, m2) in weights:
            bas, B = solve(N, m1, m2, s, coefdeg=coefdeg)
            dims.append(len(B))
            bases.append((bas, B))
        # an operator of order k needs terms with a+b up to ceil(k); orders
        # k > N are therefore only partially represented in the ansatz.
        tested = s2 <= 4 * N
        expect = 1 if (s2 % 2 == 0 and s2 >= 0 and tested) else 0
        ok = all(d == expect for d in dims)
        mark = "" if ok else ("   (truncated: order k > N)" if not tested
                              else "   <-- UNEXPECTED")
        if not ok and tested:
            bad += 1
        txt = ""
        if dims[0] == 1:
            bas, B = bases[0]
            txt = show(bas, B[0])
            txt = (txt[:86] + " ...") if len(txt) > 90 else txt
        k = f"k={s/2}" if s2 % 2 == 0 else ""
        print(f"{s2:>4}  {str(s):>6} {k:>7}  {dims}   {txt}{mark}")
    print(f"\n# unexpected dimensions: {bad}")
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tdep":
        # the Lemma: no t-dependent coefficient survives
        print("# with t-dependent coefficients allowed (t^r, r <= 2):")
        scan(N=int(sys.argv[2]) if len(sys.argv) > 2 else 2, coefdeg=2)
    else:
        scan(N=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
