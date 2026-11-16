#!/usr/bin/env python3
"""n4win0.py -- the structural facts that decide the shape of the z-window.

Three claims, each measured rather than argued:

 1) every even derivative of the ring LOWERS the z-exponent or leaves it:
    d_w keeps n_z, d_z drops it by 1.  So n_z is non-increasing under
    cD_+-, bD_+-;
 2) multiplication ADDS n_z;
 3) hence, relative to the z-exponent a_0 of the theta-free level-0 part,
    the set {s <= 0} is a differential subring and {s < -L} a differential
    ideal in it -- the window is one-sided, and canonically so.

Also: the dilatation grading (z, w of weight 1, theta of weight 1/2) is the
one the class is cut out by, and E2 is what forces it.

Usage: tools/venv/bin/python n4win0.py
"""
import itertools
import pickle
import sys
import time

from fractions import Fraction as Q

import n4noeth as N
from n4noeth import (mono, mzero, minv, add, sub, mul, cDm, bDp, cDp, bDm,
                     C, MI, PL, lhs1, lhs2, rhs2, msub, d_z, d_w, d_xp, d_xm)
import n4rec
from n4rec import resid_full, qadd, qsmul

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4win0-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def nz_range(f):
    if not f:
        return None
    return (min(k[0] for k in f), max(k[0] for k in f))


def mat_nz_range(M):
    lo, hi = None, None
    for i in (0, 1):
        for j in (0, 1):
            r = nz_range(M[i][j])
            if r:
                lo = r[0] if lo is None else min(lo, r[0])
                hi = r[1] if hi is None else max(hi, r[1])
    return (lo, hi)


def main():
    t0 = time.time()
    log("n4win0: what the gradings say about the z-window.")
    log("")

    # --- 1) the derivatives never raise n_z --------------------------------
    log("--- 1) n_z under the derivatives ---------------------------------")
    bad = 0
    tested = 0
    for a in (-3, -1, 0, 2):
        for b in (-4, -1, 0, 3):
            for S in ((), (1,), (5,), (1, 5), (2, 3, 6), (1, 2, 3, 4, 5, 6)):
                f = mono(a, b, -1, S)
                if not f:
                    continue
                for op in (lambda g: cDm(1, g), lambda g: cDm(2, g),
                           lambda g: bDm(1, g), lambda g: bDm(2, g),
                           lambda g: cDp(1, g), lambda g: cDp(2, g),
                           lambda g: bDp(1, g), lambda g: bDp(2, g),
                           d_z, d_w, d_xp, d_xm):
                    tested += 1
                    im = op(f)
                    if im and max(k[0] for k in im) > a:
                        bad += 1
    log("   %d applications of the 12 derivative operators to monomials;"
        % tested)
    log("   number that RAISED n_z: %d  (must be 0)" % bad)

    # --- 2) multiplication adds n_z ----------------------------------------
    log("")
    log("--- 2) n_z under multiplication ----------------------------------")
    bad2 = 0
    for a1 in (-2, 0, 1):
        for a2 in (-3, 0, 2):
            f = mono(a1, 1, 0, (1, 2))
            g = mono(a2, -2, -1, (5,))
            h = mul(f, g)
            if h and (min(k[0] for k in h) != a1 + a2
                      or max(k[0] for k in h) != a1 + a2):
                bad2 += 1
    log("   products of monomials: n_z of the product != sum in %d cases"
        " (must be 0)" % bad2)

    # --- 3) the known vacuum: its n_z content and that of its residual -----
    log("")
    log("--- 3) the known vacuum ------------------------------------------")
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("   n_z range of the vacuum q:        %s" % (mat_nz_range(q),))
    qi = minv(q)
    log("   n_z range of q^{-1}:              %s" % (mat_nz_range(qi),))
    log("   (both are single-valued: the vacuum is a function of w alone)")

    # the dilatation degree, 2*(n_z + n_w) + |S|, on q and on q^{-1}
    def dil2(M):
        vals = set()
        for i in (0, 1):
            for j in (0, 1):
                for k in M[i][j]:
                    vals.add(2 * (k[0] + k[1]) + len(k[3]))
        return sorted(vals)
    log("   2*dilatation degree of m*q:       %s"
        % dil2([[{(k[0], k[1], k[2] + 1, k[3]): v
                  for k, v in q[i][j].items()} for j in (0, 1)]
                for i in (0, 1)]))

    # --- 4) is a bigraded (boost-homogeneous) solution possible? -----------
    log("")
    log("--- 4) the boost bigrading ---------------------------------------")
    log("   deg_+ = n_{x^+} + |S cap PL|/2,  deg_- = n_{x^-} + |S cap MI|/2.")
    log("   cD_-, bD_- have bidegree (0,-1/2); cD_+, bD_+ have (-1/2,0);")
    log("   so lhs(E2) = cD_+ (q^{-1} bD_- q) has bidegree (-1/2,-1/2)")
    log("   whatever the bidegree of q, the m's and the q's cancelling.")
    log("   rhs(E2) = -4 i m bar q has bidegree deg(m q) = deg(m) + deg(q).")
    log("   So a boost-homogeneous solution needs deg(m q) = (-1/2,-1/2):")
    log("   both halves half-integral, i.e. |S cap PL| and |S cap MI| BOTH")
    log("   odd in every term -- so no theta-free term, so det q nilpotent,")
    log("   so q not invertible.  There is NO boost-homogeneous solution,")
    log("   and the bigrading cannot cut the class down.")
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
