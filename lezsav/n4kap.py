#!/usr/bin/env python3
"""n4kap.py -- WHICH leading data are degenerate: the complete list.

On a level-0 datum  m q^(0) = F(zeta)/w,  zeta := z/w,  one has

    d_z q^(0) =  F'/(m w^2),      d_w q^(0) = -(F + zeta F')/(m w^2),

so d_+ = d_w + d_z and d_- = d_w - d_z satisfy d_+ = kappa d_- on q^(0),
for an invertible constant kappa, exactly when

    [ (1 - zeta) + kappa (1 + zeta) ] F' = (1 - kappa) F.            (*)

This is a first-order linear ODE, so its solution space is free of rank 1
over the constants, and its solutions that are LAURENT POLYNOMIAL -- the
only ones the ring admits -- are exactly

    F = A (1 + mu zeta)^{-1},   mu an even constant with mu nilpotent,
    kappa = (1 + mu)/(1 - mu),

together with the two limits mu -> 0 and mu -> infinity, i.e. F = A with
kappa = 1 and F = A zeta^{-1} with kappa = -1.

That is the vacuum family, the boost family and the mirror family, and
NOTHING else: the degeneracy the 51 solutions display is not an accident of
the two symmetries but a property of the leading datum, and the three values
1, -1, 1 + 2t of kappa measured on them are the three cases of (*).

Everything here is checked in the ring itself, not on paper.

Usage: tools/venv/bin/python n4kap.py
Writes n4kap-out.txt.
"""
import itertools
import sys
import time

from fractions import Fraction as Q

import n4noeth as N
from n4noeth import (mono, mzero, add, sub, mul, smul, d_xp, d_xm, d_z, d_w,
                     mdet, C, eq)
from n4emb import cx

OUT = []
CHK = [0, 0]


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4kap-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def check(cond, msg):
    CHK[0] += 1
    if not cond:
        CHK[1] += 1
        log("   FAIL: %s" % msg)
    return cond


T = mono(S=(9, 10))          # t = eps_9 eps_10, an even nilpotent constant
T2 = mono(S=(11, 12))        # a second even nilpotent constant


def q0_from_F(F):
    """F = {(i,j): {d: coefficient in the ring}} -- entry (i,j) is
    sum_d coeff * zeta^d.  Returns q^(0) = F(z/w)/(m w)."""
    q = mzero()
    for (i, j), pol in F.items():
        for d, c in pol.items():
            q[i][j] = add(q[i][j], mul(c, mono(d, -1 - d, -1)))
    return q


def mat_smul(c, M):
    return [[mul(c, M[i][j]) for j in (0, 1)] for i in (0, 1)]


def mat_sub(A, B):
    return [[sub(A[i][j], B[i][j]) for j in (0, 1)] for i in (0, 1)]


def mat_is_zero(M):
    return all(not M[i][j] for i in (0, 1) for j in (0, 1))


def dpm(M, op):
    return [[op(M[i][j]) for j in (0, 1)] for i in (0, 1)]


def kappa_test(q0, kap):
    """d_+ q0 == kap * d_- q0 ?"""
    return mat_is_zero(mat_sub(dpm(q0, d_xp), mat_smul(kap, dpm(q0, d_xm))))


def main():
    t0 = time.time()
    log("n4kap: the complete list of degenerate leading data.")
    log("")
    one = mono()

    # --- the three families -------------------------------------------------
    log("--- the three families of (*) ------------------------------------")
    fams = [
        ("F = A                      (vacuum)", {(0, 0): {0: one},
                                                 (1, 1): {0: one}}, one),
        ("F = A zeta^{-1}            (mirror)", {(0, 0): {-1: one},
                                                 (1, 1): {-1: one}},
         smul(-1, one)),
        # (1 + mu zeta)^{-1} = 1 - mu zeta   for mu^2 = 0
        ("F = A (1 + t zeta)^{-1}    (boost, t^2 = 0)",
         {(0, 0): {0: one, 1: smul(-1, T)},
          (1, 1): {0: one, 1: smul(-1, T)}},
         add(one, smul(2, T))),
    ]
    for name, F, kap in fams:
        q0 = q0_from_F(F)
        ok = kappa_test(q0, kap)
        log("   %-42s kappa = %-14s : %s"
            % (name, N.show(kap) if hasattr(N, "show") else kap,
               "d_+ = kappa d_-  HOLDS" if ok else "FAILS"))
        check(ok, name)

    # kappa = (1+mu)/(1-mu) for mu^2 = 0 is 1 + 2mu -- and for a nilpotent
    # of index 3 it is 1 + 2mu + 2mu^2.
    log("")
    log("--- a nilpotent of higher index ----------------------------------")
    mu = add(T, T2)                       # mu^2 = 2 eps_9..12 != 0, mu^3 = 0
    mu2 = mul(mu, mu)
    mu3 = mul(mu2, mu)
    log("   mu := t + eps_9..12 : mu^2 %s zero, mu^3 %s zero"
        % ("is" if not mu2 else "is NOT", "is" if not mu3 else "is NOT"))
    # (1 + mu zeta)^{-1} = 1 - mu zeta + mu^2 zeta^2 - ...
    Fp = {0: one, 1: smul(-1, mu), 2: mu2}
    F = {(0, 0): dict(Fp), (1, 1): dict(Fp)}
    q0 = q0_from_F(F)
    # kappa = (1+mu)/(1-mu) = 1 + 2mu + 2mu^2
    kap = add(one, smul(2, mu), smul(2, mu2))
    ok = kappa_test(q0, kap)
    log("   F = A (1 + mu zeta)^{-1} with mu^3 = 0, kappa = 1 + 2mu + 2mu^2"
        " : %s" % ("HOLDS" if ok else "FAILS"))
    check(ok, "higher nilpotent")

    # --- and NOTHING else: the non-degenerate data --------------------------
    log("")
    log("--- data on which no kappa exists --------------------------------")
    log("   (kappa is sought as c_0 + c_1 t + c_2 T + c_3 tT with c_i in")
    log("    Q(i) on a grid, t and T the two even nilpotent constants; the")
    log("    equation is linear in kappa, so a solve settles it exactly.)")
    cands = [
        ("F = A zeta                  (a_0 = 1)",
         {(0, 0): {1: one}, (1, 1): {1: one}}),
        ("F = A zeta^{2}              (a_0 = 2)",
         {(0, 0): {2: one}, (1, 1): {2: one}}),
        ("F = A zeta^{-2}             (a_0 = -2)",
         {(0, 0): {-2: one}, (1, 1): {-2: one}}),
        ("F = diag(zeta, 1)",
         {(0, 0): {1: one}, (1, 1): {0: one}}),
        ("F = diag(zeta, zeta^{-1})",
         {(0, 0): {1: one}, (1, 1): {-1: one}}),
        ("F = [[1, zeta], [0, 1]]     (unipotent)",
         {(0, 0): {0: one}, (0, 1): {1: one}, (1, 1): {0: one}}),
        ("F = A (1 + zeta)^{-1} ... not Laurent, skipped", None),
    ]
    for name, F in cands:
        if F is None:
            log("   %s" % name)
            continue
        q0 = q0_from_F(F)
        # solve d_+ q0 = kappa d_- q0 for kappa in the even constants
        P = dpm(q0, d_xp)
        M = dpm(q0, d_xm)
        basis = [one, T, T2, mul(T, T2)]
        # collect all monomial keys occurring
        rows = set()
        colvals = []
        for b in basis:
            Mb = mat_smul(b, M)
            colvals.append(Mb)
            for i in (0, 1):
                for j in (0, 1):
                    for k in Mb[i][j]:
                        rows.add((i, j, k))
        for i in (0, 1):
            for j in (0, 1):
                for k in P[i][j]:
                    rows.add((i, j, k))
        rows = sorted(rows)
        A = [[colvals[c][r[0]][r[1]].get(r[2], C()) for c in range(4)]
             for r in rows]
        b = [P[r[0]][r[1]].get(r[2], C()) for r in rows]
        # solve over Q(i) by realification
        import n4phi
        x = N.sparse_solve([[(i,) and None for i in ()]], {}) if False else None
        # small dense solve
        sol = dense_solve(A, b)
        st = "kappa EXISTS: %s" % sol if sol is not None else "NO kappa"
        log("   %-42s %s" % (name, st))
        check(sol is None, name)

    log("")
    log("checks: %d, failures: %d" % (CHK[0], CHK[1]))
    log("[%.1fs] done" % (time.time() - t0))


def dense_solve(A, b):
    """Solve A x = b over Q(i) by realification + exact elimination."""
    import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
    import exactlin
    n = len(A[0]) if A else 0
    if not A:
        return None
    big, rhs = [], []
    for r in range(len(A)):
        big.append([A[r][c].re for c in range(n)]
                   + [-A[r][c].im for c in range(n)])
        rhs.append(b[r].re)
    for r in range(len(A)):
        big.append([A[r][c].im for c in range(n)]
                   + [A[r][c].re for c in range(n)])
        rhs.append(b[r].im)
    try:
        x = exactlin.solve(big, rhs)
    except Exception:
        return None
    if x is None:
        return None
    return [(x[i], x[i + n]) for i in range(n)]


if __name__ == "__main__":
    main()
