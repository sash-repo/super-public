#!/usr/bin/env python3
"""n4cons3.py -- WHY no completion of T = -u is polynomial in the currents.

n4cons2.py refuted a list of 1218 candidates at the vacuum itself, and a
refutation at one point is strong enough to ask for a reason.  Here is the
reason, and it is a residue.

The vacuum and everything built from it are free of z = x^+ - x^-, and on
z-free elements both d_+ and d_- act as d_w, w = x^+ + x^-.  On the Laurent
ring in w the image of d_w is spanned by the w^n with n != -1: the
coefficient of w^{-1} -- the residue -- is the whole obstruction to
integrating in w.  So

    T is completed to a d_-closed current by an element of the ring
    <=>  Res_w d_- T = 0,

and the script measures that residue exactly, theta-monomial by
theta-monomial.  It then adjoins the one thing that can cancel it,
log w (whose w-derivative is 1/w, outside the image), and re-runs the solve
of n4cons2 with the extra columns m^p log w.

Usage: ../venv/bin/python n4cons3.py   ->  n4cons3-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import n4cons2                                               # noqa: E402
from n4noeth import (C, mono, mul, add, smul, eq, minv, dT, csolve, cnull,
                     d_xm, d_xp, is_solution)                # noqa: E402
from n4cons2 import candidates, stack                       # noqa: E402
from n4shapeq import pretty                                  # noqa: E402
from n4rec import qadd, qsmul                                # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons3-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def zfree(f):
    return all(k[0] == 0 for k in f)


def residue(f):
    """the w^{-1} part."""
    return {k: v for k, v in f.items() if k[1] == -1}


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4cons3: the residue that obstructs the completion of T = -u.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))
    log("the vacuum is free of z = x^+ - x^-: %s"
        % all(zfree(q[i][j]) for i in (0, 1) for j in (0, 1)))
    qi = minv(q)

    for dop, nm in ((d_xm, "d_-"), (d_xp, "d_+")):
        R = dT(q, qi, dop)
        res = residue(R)
        log("")
        log("%s T: %d terms, w-exponents %s"
            % (nm, len(R), sorted({k[1] for k in R})))
        log("   residue (the w^{-1} part): %d terms" % len(res))
        for k, v in sorted(res.items(), key=lambda t: str(t[0])):
            log("      %-28s %s" % (pretty(k), v))

    # --- the solve with log w adjoined -----------------------------------
    log("")
    log("=" * 70)
    log("the same solve as before, with the columns d(m^p log w) = m^p/w")
    log("=" * 70)
    cand = candidates(q)
    names = sorted(cand, key=str)
    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        target = dT(q, qi, dop)
        cols = {n_: dop(cand[n_]) for n_ in names}
        lognames = []
        for p in (-1, 0, 1):
            n_ = ("log", p)
            lognames.append(n_)
            cols[n_] = mono(nw=-1, nm=p)      # d_w (m^p log w) = m^p / w
        allnames = names + lognames
        cl, tg = stack([(target, cols)], allnames)
        x = csolve(cl, tg)
        log("")
        log("dop = %s: %d rows, %d unknowns -> %s"
            % (dopname, len(tg), len(cl),
               "SOLVED" if x is not None else "still no solution"))
        if x is not None:
            nz = {allnames[j]: x[j] for j in range(len(allnames)) if x[j]}
            log("   %d nonzero coefficients; the log ones are:" % len(nz))
            for n_ in lognames:
                j = allnames.index(n_)
                if x[j]:
                    log("      m^%d log w : %s" % (n_[1], x[j]))
        log("   [%.1fs]" % (time.time() - t0))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
