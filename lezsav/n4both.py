#!/usr/bin/env python3
"""n4both.py -- is the completed stress tensor conserved in ONE direction or
in both?

n4mirror.py and n4boost.py returned the SAME two coefficients for dop = d_-
and for dop = d_+.  On the vacuum family that means nothing (there d_+ = d_-)
and on the mirror family it means nothing either (there d_+ = -d_-), but on
the boosted solutions the two derivatives are independent, and if bold T + X
were killed by both of them it would be constant on those solutions -- in
which case the family, however large, cannot tell a current of the plus half
from a constant, and the word "completion" would be claiming too much.

So this run asks the question directly, on a boosted solution:

    is d_+ bold T different from d_- bold T at all;
    does X complete bold T for d_-, for d_+, or for both;
    and is bold T + X constant?

Usage: ../venv/bin/python n4both.py   ->  n4both-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q_

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, add, smul, eq, minv, dT, d_xm, d_xp, d_z, d_w,
                     is_solution, show)                          # noqa: E402
from n4cons7 import logcols                                      # noqa: E402
from n4cons8 import composites                                   # noqa: E402
from n4rec import qadd, qsmul                                    # noqa: E402
from n4emb import C as CC, I                                     # noqa: E402
import n4mirror as MIR                                           # noqa: E402
import n4boost as BST                                            # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4both-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


X_COEFFS = {("2", ("1", "bD+2", "cD+2")): Q_(-3, 4),
            ("2", ("s1", "bD+1", "cD+2")): Q_(-1, 4)}


def dX(qq, dop):
    """dop of the completion X = sum_k coeff_k log f_k."""
    fs, _ = composites(qq, "+")
    lc = logcols(fs, dop)
    out = {}
    for k, c in X_COEFFS.items():
        if k not in lc:
            raise KeyError("the composite %s is not invertible here" % (k,))
        out = add(out, smul(c, lc[k]))
    return out


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    with open("n4cons-tan.pkl", "rb") as fh:
        tans = pickle.load(fh)

    log("n4both: is bold T + X conserved in one direction or in both?")
    log("")

    fam = [("vacuum", q)]
    for d in tans:
        for lam in (1, -1, Q_(1, 2), 2, 3, CC(0, 1), CC(1, 1), Q_(-1, 3)):
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq):
                fam.append(("q + (%s) d" % lam, qq))

    kinds = [("vacuum (z-free)", [(nm, qq) for nm, qq in fam]),
             ("mirror (w-free)", [(nm, MIR.sigma(qq)) for nm, qq in fam]),
             ("boosted (both)", [(nm, BST.boost(qq)) for nm, qq in fam])]

    for kname, sols in kinds:
        log("=" * 70)
        log(kname)
        log("=" * 70)
        n_indep, n_m, n_p, n_const = 0, 0, 0, 0
        for nm, qq in sols:
            qqi = minv(qq)
            tm, tp = dT(qq, qqi, d_xm), dT(qq, qqi, d_xp)
            if not eq(tm, tp) and not eq(tm, smul(-1, tp)):
                n_indep += 1
            rm = add(tm, dX(qq, d_xm))
            rp = add(tp, dX(qq, d_xp))
            if eq(rm, {}):
                n_m += 1
            if eq(rp, {}):
                n_p += 1
            if eq(rm, {}) and eq(rp, {}):
                n_const += 1
        n = len(sols)
        log("   %d solutions" % n)
        log("   d_+ T neither equal nor opposite to d_- T:      %d of %d"
            % (n_indep, n))
        log("   d_-(T + X) = 0:                                 %d of %d"
            % (n_m, n))
        log("   d_+(T + X) = 0:                                 %d of %d"
            % (n_p, n))
        log("   both, i.e. T + X constant on the solution:      %d of %d"
            % (n_const, n))
        log("   [%.1fs]" % (time.time() - t0))
        log("")

    # what T itself looks like on a boosted solution, for the record
    nm, qq = kinds[2][1][0]
    qqi = minv(qq)
    log("on the boosted vacuum:")
    log("   d_- T = %s" % show(dT(qq, qqi, d_xm), cap=4))
    log("   d_+ T = %s" % show(dT(qq, qqi, d_xp), cap=4))
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
