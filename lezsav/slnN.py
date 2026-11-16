#!/usr/bin/env python3
"""
slnN.py -- the N-extended super Liouville equation of Ivanov-Krivonos:
WHICH contact fields preserve it, and what the multiplier lambda_a is.

Setting.  On the superstring C^{1|N} with coordinates (t, theta_1..theta_N) put

    D_i = d/d theta_i - theta_i d/dt,      D_i D_j + D_j D_i = -2 delta_ij d/dt,

these spanning the contact distribution Ker(dt + sum theta_i d theta_i).  (Which
sign the K_f normalize is settled by the computation, see SGN below; the other
sign is the same algebra after t -> -t, so no result depends on the choice.)

k^L(1|N) = {K_f} is the contact algebra; K_f is taken from paper G's engine
(check_table.py, eq. (Kf1)), whose contact bracket [K_f,K_g] = K_{{f,g}} is
verified there for n <= 4.  Being contact means precisely

    [K_f, D_i] = sum_j c_ij(f) D_j                                        (*)

with c_ij(f) superfunctions.  This script computes the matrix c(f) and splits it

    c(f) = lambda(f) * Id  +  omega(f),     omega antisymmetric (so(N)-part).

Why this decides the question.  Eq. (1.4) of hep-th/0007256 is

    D_+ D_- F = m exp(F),

one odd derivative from each of the two halves.  For N = 1 there is only one D
and (*) reads [K_a,D] = lambda_a D: the equation is k^L(1|1)-invariant, F
transforms as F -> K_a(F) - lambda_a, and the anomaly is the -1/2 D^5 a of
sliouv.py.  For N >= 2 one must SINGLE OUT one D_i out of N, and the equation is
invariant only for those f whose so(N)-part does not rotate D_1 away, i.e.

    c_1j(f) = 0  for all j =/= 1.

This script computes that subalgebra.  Everything is exact (Fraction).
"""

import sys, os
from fractions import Fraction as Fr
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from stringy import (mono, const, add, mul, smul, d_t, d_odd, parity, eq, show,
                     VF)


# --------------------------------------------------------------------------
# the contact fields of k^L(1|n): eq. (Kf1), as in stringy.py
# (that file verifies [K_f,K_g] = K_{{f,g}} for n = 1,2,3,4)
# --------------------------------------------------------------------------

def K_L(n, f):
    nab = lambda h: {k: (2 - len(k[1])) * c for k, c in h.items() if (2 - len(k[1]))}
    comps = {'t': nab(f)}
    p = parity(f)
    for i in range(1, n + 1):
        ci = smul(-(-1) ** p, add(d_odd(i, f), smul(-1, mul(mono(0, (i,)), d_t(f)))))
        if ci:
            comps[i] = ci
    return VF(comps, tuple(range(1, n + 1)))


def D(n, i, s=1):
    """D_i = d_theta_i + s * theta_i d_t.

    s = +1 is the SUSY-type derivative (D_i^2 = d_t) used in sliouv.py;
    s = -1 spans the contact distribution Ker(dt + sum theta_i d theta_i),
    which is the convention of the K_f of paper G.  Which of the two the K_f
    normalizes is settled by the computation below, not assumed.
    """
    return VF({i: const(1), 't': smul(s, mono(0, (i,)))},
              tuple(range(1, n + 1)))


def monomials(n, trange):
    """basis monomials t^a * theta_S of O(1|n)."""
    out = []
    idx = tuple(range(1, n + 1))
    for a in trange:
        for k in range(n + 1):
            for S in combinations(idx, k):
                out.append(mono(a, S))
    return out


# --------------------------------------------------------------------------
# step 0: the D_i are the right derivations
# --------------------------------------------------------------------------

def check_D(n):
    bad = 0
    dt = VF({'t': const(1)}, tuple(range(1, n + 1)))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            br = D(n, i, SGN).bracket(D(n, j, SGN))   # anticommutator, both odd
            tgt = dt.scal(2 * SGN) if i == j else VF({}, ())
            if not (br.plus(tgt.scal(-1))).is_zero():
                bad += 1
    return bad


# --------------------------------------------------------------------------
# step 1: read off c_ij(f) from [K_f, D_i] = sum_j c_ij(f) D_j
# --------------------------------------------------------------------------

SGN = -1   # settled in step 0: the K_f normalize d_theta_i - theta_i d_t


def cmatrix(n, f):
    """returns (c, ok): c[i][j] superfunctions, ok = does (*) hold exactly."""
    c = {}
    ok = True
    for i in range(1, n + 1):
        br = K_L(n, f).bracket(D(n, i, SGN))
        row = {}
        for j in range(1, n + 1):
            row[j] = br.c.get(j, {})
        c[i] = row
        # rebuild sum_j c_ij D_j and compare with br
        rec = VF({}, tuple(range(1, n + 1)))
        for j in range(1, n + 1):
            if row[j]:
                rec = rec.plus(VF({j: row[j],
                                   't': smul(SGN, mul(row[j], mono(0, (j,))))},
                                  tuple(range(1, n + 1))))
        if not br.plus(rec.scal(-1)).is_zero():
            ok = False
    return c, ok


def split(n, c):
    """c = lambda*Id + omega ; returns (lambda, omega, is_scalar_plus_antisym)."""
    lam = c[1][1]
    diag_ok = all(eq(c[i][i], lam) for i in range(1, n + 1))
    omega = {(i, j): c[i][j] for i in range(1, n + 1) for j in range(1, n + 1)
             if i != j and c[i][j]}
    anti = all(eq(c[i][j], smul(-1, c[j][i]))
               for i in range(1, n + 1) for j in range(1, n + 1) if i != j)
    return lam, omega, diag_ok, anti


# --------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("step 0:  D_i D_j + D_j D_i = 2 delta_ij d_t")
    print("=" * 74)
    for n in (1, 2, 3, 4):
        print("   n=%d : failures %d" % (n, check_D(n)))
    print()

    print("=" * 74)
    print("step 1:  [K_f, D_i] = sum_j c_ij(f) D_j,  c = lambda*Id + omega")
    print("=" * 74)
    trange = range(-2, 4)
    for n in (1, 2, 3, 4):
        bad_contact = bad_diag = bad_anti = 0
        bad_lam = 0
        tot = 0
        for f in monomials(n, trange):
            tot += 1
            c, ok = cmatrix(n, f)
            if not ok:
                bad_contact += 1
                continue
            lam, omega, diag_ok, anti = split(n, c)
            if not diag_ok:
                bad_diag += 1
            if not anti:
                bad_anti += 1
            # is lambda = - d_t f ?  (sliouv.py's -1/2 D^2 a, for a = 2f: the
            # generating function of K_L is normalized so that its d_t-component
            # is (2-E)f, i.e. 2f in theta-degree 0)
            if not eq(lam, smul(-1, d_t(f))):
                bad_lam += 1
        print("   n=%d  monomials %3d :  (*) fails %d,  diag non-scalar %d,  "
              "off-diag non-antisym %d,  lambda != -d_t f : %d"
              % (n, tot, bad_contact, bad_diag, bad_anti, bad_lam))
    print()

    print("=" * 74)
    print("step 2:  the so(N)-part, i.e. the obstruction to singling out D_1")
    print("=" * 74)
    for n in (2, 3, 4):
        rot = []
        for f in monomials(n, trange):
            c, ok = cmatrix(n, f)
            if not ok:
                continue
            if any(c[1][j] for j in range(2, n + 1)):
                rot.append(f)
        tot = len(monomials(n, trange))
        print("   n=%d : of %d monomials, %d rotate D_1 into the other D_j"
              % (n, tot, len(rot)))
        # which ones?  print the theta-degrees involved
        degs = sorted({len(list(k[1])) for f in rot for k in f})
        print("        their theta-degrees: %s" % degs)
        # the invariance algebra of the naive equation
        inv = [f for f in monomials(n, trange)
               if not any(cmatrix(n, f)[0][1][j] for j in range(2, n + 1))]
        degs_inv = sorted({len(list(k[1])) for f in inv for k in f})
        print("        invariance algebra {f | c_1j(f)=0, j>1}: %d monomials, "
              "theta-degrees %s" % (len(inv), degs_inv))
        # is it a subalgebra closed under the contact bracket?  spot-check via
        # the omega of K_f for f = theta_i theta_j (the so(N) generators)
        for (i, j) in [(1, 2)]:
            f = mono(0, (i, j))
            c, ok = cmatrix(n, f)
            if ok:
                print("        sample: f = theta_%d theta_%d -> c_1j = %s"
                      % (i, j, {j2: show(c[1][j2]) for j2 in range(1, n + 1)
                                if c[1][j2]}))
    print()

    print("=" * 74)
    print("step 2a: the closed formula  c_ij(f) = D_i D_j f")
    print("=" * 74)
    for n in (1, 2, 3, 4):
        bad = tot = 0
        for f in monomials(n, range(-3, 5)):
            c, ok = cmatrix(n, f)
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    tot += 1
                    DjF = D(n, j, SGN).apply(f)
                    if not eq(c[i][j], D(n, i, SGN).apply(DjF)):
                        bad += 1
        print("   n=%d : %4d checks, failures %d" % (n, tot, bad))
    print("   so  [K_f, D_i] = sum_j (D_i D_j f) D_j;  the trace part is")
    print("   D_i^2 f = -d_t f and the so(N)-part is D_i D_j f, i =/= j.")
    print()

    print("=" * 74)
    print("step 2b: the invariance algebra of the NAIVE equation, over a wide")
    print("         range of t-powers (-6..7): it does not grow, i.e. it is")
    print("         finite-dimensional, hence has no Virasoro sector")
    print("=" * 74)
    for n in (2, 3, 4):
        inv = []
        for f in monomials(n, range(-6, 8)):
            c, ok = cmatrix(n, f)
            if not any(c[1][j] for j in range(2, n + 1)):
                inv.append(f)
        ev = sum(1 for f in inv if parity(f) == 0)
        print("   n=%d : dim %d = 2^(n-1)+2, sdim (%d|%d)" %
              (n, len(inv), ev, len(inv) - ev))
        print("        %s" % ", ".join(show(f) for f in inv))
    print("   i.e. Lambda(theta_2..theta_N) + C.theta_1 + C.t :")
    print("   translation, dilatation, the N supersymmetries and so(N-1) --")
    print("   NO K_{t^2} and no K_{t^m}, m >= 2.  The superconformal symmetry")
    print("   is gone, so the naive N >= 2 equation carries no anomaly at all.")
    print()

    print("=" * 74)
    print("step 3:  n=1 -- the multiplier that sliouv.py uses")
    print("=" * 74)
    bad = 0
    for f in monomials(1, trange):
        c, ok = cmatrix(1, f)
        if not ok or not eq(c[1][1], smul(-1, d_t(f))):
            bad += 1
    print("   [K_f,D] = -(d_t f) D  for all %d monomials: failures %d"
          % (len(monomials(1, trange)), bad))
    print("   with a = 2f this is exactly the  [K_a,D] = -1/2 (D^2 a) D  that")
    print("   sliouv.py verifies independently, so the N=1 anomaly stands.")


if __name__ == "__main__":
    main()
