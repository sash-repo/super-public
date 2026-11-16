#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sv2coc.py -- central extensions of  svect^L_lambda(1|2) = the algebra that
Ivanov-Krivonos call k^A(1|2).

The structure constants are those verified in sv2.py (every relation there is
checked over a window of modes against vector fields built from scratch); here
they are re-entered in closed form, CROSS-CHECKED against sv2.py's vector
fields, and then used to compute the space of degree-zero 2-cocycles.

Exact arithmetic throughout (Fraction; exactlin/FLINT for the ranks).

Basis:  L_n, J^+_n, J^-_n, J^0_n  (n in Z, even)
        G_1^r, G_2^r, Gb^{r1}, Gb^{r2}  (r in Z+1/2, odd)

Run: ../venv/bin/python sv2coc.py
"""

import sys
import os
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace, backend           # noqa: E402

L_, JP, JM, J0, G1, G2, B1, B2 = range(8)
NAMES = ['L', 'J+', 'J-', 'J0', 'G1', 'G2', 'Gb1', 'Gb2']
EVEN = (L_, JP, JM, J0)
ODD = (G1, G2, B1, B2)
PAR = [0, 0, 0, 0, 1, 1, 1, 1]

FAIL = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(msg)
        print("  FAIL: " + msg)


# --------------------------------------------------------- structure constants

SL2 = {(JP, JM): {J0: 1}, (JM, JP): {J0: -1},
       (J0, JP): {JP: 2}, (JP, J0): {JP: -2},
       (J0, JM): {JM: -2}, (JM, J0): {JM: 2}}

GACT = {(JP, G1): (G2, -1), (JM, G2): (G1, -1),
        (J0, G1): (G1, -1), (J0, G2): (G2, 1)}
GBACT = {(JP, B2): (B1, 1), (JM, B1): (B2, 1),
         (J0, B1): (B1, 1), (J0, B2): (B2, -1)}

# {G_al^r, Gb^{s be}} = -delta L_{r+s} + (r-s-lambda) * SIG[(al,be)]
SIG = {(G1, B1): {J0: F(1, 2)}, (G2, B2): {J0: F(-1, 2)},
       (G1, B2): {JM: 1}, (G2, B1): {JP: 1}}


def br(u, v, lam):
    """[u,v] as a dict {(type, mode): coeff}; u=(type, mode)."""
    (i, m), (j, n) = u, v
    out = {}

    def add(t, k, c):
        if c:
            out[(t, k)] = out.get((t, k), 0) + c

    # normalize: handle the pairs in a fixed order, then use (anti)symmetry
    if _order(i) > _order(j):
        sgn = -(-1) ** (PAR[i] * PAR[j])
        for k, c in br(v, u, lam).items():
            add(k[0], k[1], sgn * c)
        return {k: c for k, c in out.items() if c}

    if i == L_ and j == L_:
        add(L_, m + n, m - n)
    elif i == L_ and j in (JP, JM, J0):
        add(j, m + n, -n)
    elif i == L_ and j in (G1, G2):
        add(j, m + n, F(m, 2) - n + lam / 2)
    elif i == L_ and j in (B1, B2):
        add(j, m + n, F(m, 2) - n - lam / 2)
    elif i in (JP, JM, J0) and j in (JP, JM, J0):
        for t, c in SL2.get((i, j), {}).items():
            add(t, m + n, c)
    elif i in (JP, JM, J0) and j in (G1, G2):
        a = GACT.get((i, j))
        if a:
            add(a[0], m + n, a[1])
    elif i in (JP, JM, J0) and j in (B1, B2):
        a = GBACT.get((i, j))
        if a:
            add(a[0], m + n, a[1])
    elif i in (G1, G2) and j in (G1, G2):
        pass                                    # both doublets are abelian
    elif i in (B1, B2) and j in (B1, B2):
        pass
    elif i in (G1, G2) and j in (B1, B2):
        if (i, j) in ((G1, B1), (G2, B2)):
            add(L_, m + n, -1)
        for t, c in SIG[(i, j)].items():
            add(t, m + n, (m - n - lam) * c)
    else:
        raise ValueError((i, j))
    return {k: c for k, c in out.items() if c}


def _order(i):
    return [0, 1, 1, 1, 2, 2, 3, 3][i]


# ------------------------------------------------------- 0. cross-check vs sv2

def cross_check(lam):
    """Compare every structure constant with sv2.py's vector fields."""
    import sv2
    from sympy import Rational, simplify
    sv2.GOOD_SIGN[0] = 'parodd'
    lam_s = Rational(lam.numerator, lam.denominator)

    def vf(t, m):
        m = Rational(m.numerator, m.denominator) if isinstance(m, F) else Rational(m)
        if t == L_:
            X = sv2.L(int(m))
        elif t in (JP, JM, J0):
            X = sv2.J({JP: '+', JM: '-', J0: '0'}[t], int(m))
        elif t in (G1, G2):
            X = sv2.G(1 if t == G1 else 2, m)
        else:
            X = sv2.Gbar(1 if t == B1 else 2, m)
        return VFsub(X, lam_s)

    def VFsub(X, l):
        return sv2.VF({a: {k: v.subs(sv2.lam, l) for k, v in X.c[a].items()}
                       for a in sv2.COMPS}, X.p)

    modes = {0: [F(k) for k in (-2, -1, 0, 1, 2)],
             1: [F(2 * k + 1, 2) for k in (-2, -1, 0, 1)]}
    n = 0
    for i in range(8):
        for j in range(8):
            for m in modes[PAR[i]]:
                for p in modes[PAR[j]]:
                    got = sv2.bracket(vf(i, m), vf(j, p))
                    want = None
                    for (t, k), c in br((i, m), (j, p), lam).items():
                        Y = vf(t, k).scal(Rational(F(c).numerator,
                                                   F(c).denominator))
                        want = Y if want is None else want + Y
                    if want is None:
                        ok = got.is_zero()
                    else:
                        ok = (got - want).is_zero()
                    check(ok, "structure constant [%s_%s,%s_%s] wrong"
                          % (NAMES[i], m, NAMES[j], p))
                    n += 1
    print("   %d structure constants agree with the vector fields (lambda=%s)"
          % (n, lam))


def jacobi(lam, M=3):
    """super Jacobi for the encoded structure constants."""
    modes = {0: [F(k) for k in range(-M, M + 1)],
             1: [F(2 * k + 1, 2) for k in range(-M, M)]}
    bad = 0
    n = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in modes[PAR[i]][:3]:
                    for p in modes[PAR[j]][:3]:
                        for q in modes[PAR[k]][:3]:
                            acc = {}
                            for (a, b, c, s) in (
                                    ((i, m), (j, p), (k, q),
                                     (-1) ** (PAR[i] * PAR[k])),
                                    ((j, p), (k, q), (i, m),
                                     (-1) ** (PAR[j] * PAR[i])),
                                    ((k, q), (i, m), (j, p),
                                     (-1) ** (PAR[k] * PAR[j]))):
                                for (t, u), co in br(a, b, lam).items():
                                    for (t2, u2), co2 in br((t, u), c, lam).items():
                                        acc[(t2, u2)] = acc.get((t2, u2), 0) + s * co * co2
                            if any(v for v in acc.values()):
                                bad += 1
                            n += 1
    check(bad == 0, "Jacobi fails %d times" % bad)
    print("   graded Jacobi identity for the encoded constants: %d triples, "
          "%d failures" % (n, bad))


# ------------------------------------------------- 1. degree-zero 2-cocycles

def cocycles(lam, M, gauge=False):
    """Space of degree-0 2-cochains satisfying the cocycle identity on all
    triples inside the window |mode| <= M.  Returns (dim, basis, index).

    gauge=True adds the normalisation c(L_m, J^a_{-m}) = 0, which kills the
    coboundaries and leaves the canonical representative."""
    # unknowns: c((i,m),(j,-m)); canonical representative (i<=j, and m>=0 when i==j)
    idx = {}

    def key(i, m, j, n):
        """canonical unknown and sign for c((i,m),(j,n)) with m+n=0."""
        if (i, m) <= (j, n):
            return (i, j, m), 1
        # c(v,u) = -(-1)^{|u||v|} c(u,v)
        return (j, i, n), -(-1) ** (PAR[i] * PAR[j])

    def modes(p):
        if p == 0:
            return [F(k) for k in range(-M, M + 1)]
        return [F(2 * k + 1, 2) for k in range(-M - 1, M + 1)]

    for i in range(8):
        for j in range(8):
            for m in modes(PAR[i]):
                n = -m
                if n not in modes(PAR[j]):
                    continue
                k, _ = key(i, m, j, n)
                if k not in idx:
                    idx[k] = len(idx)
    # a super-skew cochain has c(u,u)=0 forced for odd u? no: for odd u,v
    # c is symmetric, so c(u,u) is unconstrained.  For even u, c(u,u) = 0.
    rows = []
    forced = []
    for (i, j, m), col in idx.items():
        if i == j and PAR[i] == 0 and m == -m:
            forced.append(col)

    eqs = []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in modes(PAR[i]):
                    for n in modes(PAR[j]):
                        q = -m - n
                        if q not in modes(PAR[k]):
                            continue
                        row = [0] * len(idx)
                        nz = False
                        for (a, b, c, s) in (
                                ((i, m), (j, n), (k, q),
                                 (-1) ** (PAR[i] * PAR[k])),
                                ((j, n), (k, q), (i, m),
                                 (-1) ** (PAR[j] * PAR[i])),
                                ((k, q), (i, m), (j, n),
                                 (-1) ** (PAR[k] * PAR[j]))):
                            for (t, u), co in br(a, b, lam).items():
                                if u != -c[1]:
                                    continue
                                kk, sg = key(t, u, c[0], c[1])
                                if kk in idx:
                                    row[idx[kk]] += s * co * sg
                                    nz = True
                        if nz and any(row):
                            eqs.append(row)
    for col in forced:
        row = [0] * len(idx)
        row[col] = 1
        eqs.append(row)
    if gauge:
        # c(L_m, J^a_{-m}) = 0 kills the three coboundaries of f(J^a_0);
        # c(L_1, L_{-1}) = 0 (the cocycle vanishes on the sl(2) of L_0,L_+-1)
        # kills the remaining one, of f(L_0).
        for (i, j, m), col in idx.items():
            if ((i == L_ and j in (JP, JM, J0))
                    or (j == L_ and i in (JP, JM, J0))
                    or (i == L_ and j == L_ and abs(m) == 1)):
                row = [0] * len(idx)
                row[col] = 1
                eqs.append(row)
    sol = nullspace(eqs) if eqs else []
    return sol, idx


def coboundaries(lam, M, idx):
    """d f for f a degree-0 1-cochain, i.e. f supported on the mode-0 part."""
    def modes(p):
        if p == 0:
            return [F(k) for k in range(-M, M + 1)]
        return [F(2 * k + 1, 2) for k in range(-M - 1, M + 1)]

    def key(i, m, j, n):
        if (i, m) <= (j, n):
            return (i, j, m), 1
        return (j, i, n), -(-1) ** (PAR[i] * PAR[j])

    out = []
    for t in EVEN:                       # f(L_0), f(J^a_0)
        row = [0] * len(idx)
        for i in range(8):
            for m in modes(PAR[i]):
                j_list = range(8)
                for j in j_list:
                    n = -m
                    if n not in modes(PAR[j]):
                        continue
                    # (df)(u,v) = -f([u,v])
                    co = br((i, m), (j, n), lam).get((t, F(0)), 0)
                    if co:
                        kk, sg = key(i, m, j, n)
                        if kk in idx:
                            row[idx[kk]] = -co * sg if sg == 1 else -co * sg
        out.append(row)
    return out


def report(lam, M):
    sol, idx = cocycles(lam, M)
    cob = coboundaries(lam, M, idx)
    cob = [r for r in cob if any(r)]
    d_cocyc = len(sol)
    if cob:
        r_all = rank(list(sol) + cob) if sol else rank(cob)
        d_cob_in = r_all - d_cocyc          # coboundaries not in the solution space
        d_cob = rank(cob)
        # coboundaries ARE cocycles, so d_cob_in should be 0
    else:
        d_cob = 0
    print("   lambda = %-6s window |mode| <= %d :  cocycles %d, coboundaries %d,"
          "  H^2 = %d" % (lam, M, d_cocyc, d_cob, d_cocyc - d_cob))
    return sol, idx, d_cocyc - d_cob


def describe(sol, idx, lam):
    """Print a solution in readable form."""
    inv = {v: k for k, v in idx.items()}
    for s in sol:
        terms = []
        for c, col in zip(s, range(len(s))):
            if c:
                i, j, m = inv[col]
                terms.append("c(%s_%s,%s_%s)=%s"
                             % (NAMES[i], m, NAMES[j], -m, c))
        print("      " + ";  ".join(terms))


# ------------------------------------------- 2. the closed form, verified

def cclosed(u, v, lam):
    """The candidate cocycle, in the geometric labels:

        c(L_m, L_-m)      = (m^3 - m)/2
        c(J^+_m, J^-_-m)  = m,     c(J^0_m, J^0_-m) = 2m
        c(G_al^r, Gb^{-r be}) = -delta_al^be ((r - lambda/2)^2 - 1/4)
        everything else zero.

    Note (r - lambda/2) is the mode of G with respect to L_0, i.e. in the
    shifted labels the formula is the lambda-free  (r'^2 - 1/4)."""
    (i, m), (j, n) = u, v
    if m + n != 0:
        return 0
    half = F(1, 2)
    if (i, j) == (L_, L_):
        return (m ** 3 - m) / 2
    if (i, j) == (JP, JM):
        return m
    if (i, j) == (JM, JP):
        return m          # = -c(J^+_-m, J^-_m) = -(-m)
    if (i, j) == (J0, J0):
        return 2 * m
    if (i, j) in ((G1, B1), (G2, B2)):
        return -((m - lam / 2) ** 2 - F(1, 4))
    if (i, j) in ((B1, G1), (B2, G2)):
        return -((n - lam / 2) ** 2 - F(1, 4))
    return 0


def verify_closed(lam, M=6, expand=None):
    """Check the cocycle identity for cclosed on every triple in the window."""
    modes = {0: [F(k) for k in range(-M, M + 1)],
             1: [F(2 * k + 1, 2) for k in range(-M - 1, M + 1)]}
    bad = 0
    n = 0
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in modes[PAR[i]]:
                    for p in modes[PAR[j]]:
                        q = -m - p
                        if q not in modes[PAR[k]]:
                            continue
                        tot = 0
                        for (a, b, c, s) in (
                                ((i, m), (j, p), (k, q),
                                 (-1) ** (PAR[i] * PAR[k])),
                                ((j, p), (k, q), (i, m),
                                 (-1) ** (PAR[j] * PAR[i])),
                                ((k, q), (i, m), (j, p),
                                 (-1) ** (PAR[k] * PAR[j]))):
                            for (t, u), co in br(a, b, lam).items():
                                tot += s * co * cclosed((t, u), c, lam)
                        if expand is not None:
                            tot = expand(tot)
                        if tot != 0:
                            bad += 1
                            if bad == 1:
                                print("      first failure: %s_%s, %s_%s, %s_%s -> %s"
                                      % (NAMES[i], m, NAMES[j], p,
                                         NAMES[k], q, tot))
                        n += 1
    check(bad == 0, "the closed-form cocycle fails %d times (lambda=%s)"
          % (bad, lam))
    print("   lambda = %-9s |mode| <= %d : %6d triples, %d failures"
          % (lam, M, n, bad))


def main():
    print("=" * 78)
    print("central extensions of svect^L_lambda(1|2)   [exactlin: %s]" % backend())
    print("=" * 78)
    print("\n0. The structure constants")
    cross_check(F(1, 3))
    cross_check(F(0))
    for lam in (F(0), F(1, 3)):
        jacobi(lam)

    print("\n1. Degree-zero 2-cocycles (the anomalies)")
    for lam in (F(0), F(1), F(1, 2), F(1, 3), F(2, 7)):
        for M in (2, 3):
            sol, idx, h2 = report(lam, M)

    print("\n2. The cocycle itself, gauge-fixed by c(L_m, J^a_{-m}) = 0")
    for lam in (F(0), F(1, 3)):
        sol, idx = cocycles(lam, 3, gauge=True)
        print("   lambda = %s: the gauge leaves a %d-dimensional space" %
              (lam, len(sol)))
        describe(sol, idx, lam)

    print("\n3. The same cocycle in closed form, verified beyond the window")
    print("   c(L_m,L_-m) = (m^3-m)/2,  c(J^+_m,J^-_-m) = m,  c(J^0_m,J^0_-m) = 2m,")
    print("   c(G_al^r,Gb^{-r be}) = -delta_al^be ((r-lambda/2)^2 - 1/4),  rest 0")
    for lam in (F(0), F(1), F(1, 3), F(-2, 5)):
        verify_closed(lam, 6)
    from sympy import symbols, expand as sxp
    verify_closed(symbols('lambda'), 5, expand=lambda z: sxp(z))
    print("   the L-L part is (m^3-m)/2, which no coboundary can produce,")
    print("   so the class is non-trivial; with the truncated count of item 1,")
    print("   dim H^2(svect^L_lambda(1|2); C) = 1 for every lambda.")

    print("\n" + "=" * 78)
    print("checks: %d, failures: %d" % (CHECKS[0], len(FAIL)))
    for f in FAIL[:20]:
        print("  ! " + f)
    print("=" * 78)


if __name__ == '__main__':
    main()
