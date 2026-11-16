#!/usr/bin/env python3
"""n4cons5.py -- the completion of T = -u, with the candidates GRADED right.

The refutations of n4cons2.py and n4cons4.py have a cause that has nothing
to do with the currents: the list was off by one unit of the grading.

On the class in which the vacuum lives every monomial has
        g := n_w + |S|/2,
and d_- T is homogeneous with g = -1 and m-degree 0 (its level-0 term is
-1/w, its level-2 terms go with w^{-2}, ..., its level-8 term with w^{-5}).
Since d_w lowers g by 1, a completion X must be homogeneous with

        g(X) = 0,   m-degree 0.

Every candidate of n4cons2.py is built from
        s(M, D) = tr(M q^{-1} D q),
and q has g = -1, q^{-1} has g = +1, an odd derivative has g = -1/2 and an
even one g = -1: so an odd s has g = -1/2, an even one g = -1, a product of
two odd ones g = -1, and m^p changes nothing (m is of g = 0).  The whole
list therefore sits in g = -1, one unit below the target, and only the
column log w -- of g = 0 -- could ever contribute.  That is exactly what
n4cons4.py measured: level 0 solvable, level 2 inconsistent.

What supplies the missing unit is the field itself.  The matrix
        Q := m^{-1} q^{-1}
has g = +1 and m-degree 0, so the four scalars tr(M Q), M = 1, sigma_k, are
the g = +1 building blocks the list lacked, and

        tr(M Q) * (any candidate of n4cons2)

is of g = 0 and m-degree 0, as required.  This is not an artifice: at
N <= 2 the stress tensor is Psi^{-1}(...)Psi with Psi = exp(u), and
m^{-1}Psi is precisely a scalar of g = +1 -- the same factor, written with
q rather than with a square root.

So: pool = {the graded-right products} + {log w}, filtered to g = 0 and
m-degree 0, then the same solve, then the family, then the holdout.

Usage: ../venv/bin/python n4cons5.py   ->  n4cons5-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q_

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, mul, add, smul, eq, minv, dT, csolve, cnull,
                     d_xm, d_xp, is_solution, mmul, mtr)      # noqa: E402
from n4cons2 import candidates, SIG, stack                    # noqa: E402
from n4rec import qadd, qsmul                                 # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons5-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def grades(f):
    """the set of (g, n_m) occurring in f, g = n_w + |S|/2 doubled."""
    return {(2 * k[1] + len(k[3]), k[2]) for k in f}


def homog(f, g2, nm):
    return f and grades(f) == {(g2, nm)}


def build_pool(q):
    qi = minv(q)
    Q4 = [(nm, smul(1, mul(mono(nm=-1), mtr(mmul(M, qi)))))
          for nm, M in SIG.items()]
    base = candidates(q)
    pool = {}
    for bn, bf in base.items():
        for mn, P in Q4:
            if not P:
                continue
            f = mul(P, bf)
            if homog(f, 0, 0):
                pool[("Q", mn) + bn] = f
    for bn, bf in base.items():
        if homog(bf, 0, 0):
            pool[("b",) + bn] = bf
    return pool


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4cons5: the completion of T = -u, with the candidates graded right.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))
    qi = minv(q)
    tgt = dT(q, qi, d_xm)
    log("d_- T: grades (2g, n_m) occurring: %s" % sorted(grades(tgt)))

    pool = build_pool(q)
    log("candidates of g = 0 and m-degree 0: %d" % len(pool))
    names = sorted(pool, key=str)

    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        log("")
        log("=" * 70)
        log("dop = %s" % dopname)
        log("=" * 70)
        target = dT(q, qi, dop)
        cols = {n_: dop(pool[n_]) for n_ in names}
        cols[("log",)] = mono(nw=-1)          # d_w log w = 1/w
        allnames = names + [("log",)]
        cl, tg = stack([(target, cols)], allnames)
        log("   at the vacuum: %d rows, %d unknowns" % (len(tg), len(cl)))
        x = csolve(cl, tg)
        if x is None:
            log("   NO solution at the vacuum: refuted for %s." % dopname)
            log("   [%.1fs]" % (time.time() - t0))
            continue
        nz = [(allnames[j], x[j]) for j in range(len(allnames)) if x[j]]
        log("   SOLVED at the vacuum: %d nonzero coefficients" % len(nz))
        for n_, v in sorted(nz, key=lambda t: str(t[0]))[:40]:
            log("      %-46s %s" % (str(n_), v))
        A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
        ker = cnull(A)
        log("   kernel at the vacuum: %d  [%.1fs]" % (len(ker),
                                                      time.time() - t0))

        # --- across the family --------------------------------------------
        with open("n4cons-tan.pkl", "rb") as fh:
            tans = pickle.load(fh)
        fam = [("vacuum", q)]
        for d in tans:
            for lam in (1, -1, Q_(1, 2), 2, C(0, 1)):
                qq = qadd(q, qsmul(lam, d))
                if is_solution(qq):
                    fam.append(("q + (%s) d" % lam, qq))
        log("   family: %d exact solutions" % len(fam))
        blocks = []
        for nmf, qq in fam:
            qqi = minv(qq)
            pf = build_pool(qq)
            cc = {n_: dop(pf[n_]) for n_ in names if n_ in pf}
            cc[("log",)] = mono(nw=-1)
            blocks.append((dT(qq, qqi, dop), cc))
        cl, tg = stack(blocks, allnames)
        log("   across the family: %d rows, %d unknowns  [%.1fs]"
            % (len(tg), len(cl), time.time() - t0))
        xf = csolve(cl, tg)
        if xf is None:
            log("   the law holds at the vacuum but NOT across the family.")
            continue
        nzf = [(allnames[j], xf[j]) for j in range(len(allnames)) if xf[j]]
        log("   SOLVED across the family: %d nonzero coefficients" % len(nzf))
        for n_, v in sorted(nzf, key=lambda t: str(t[0]))[:40]:
            log("      %-46s %s" % (str(n_), v))
        ntr = max(2, len(blocks) - 3)
        cl2, tg2 = stack(blocks[:ntr], allnames)
        xt = csolve(cl2, tg2)
        bad = 0
        for i in range(ntr, len(blocks)):
            tt, bc = blocks[i]
            tot = dict(tt)
            for j, n_ in enumerate(allnames):
                if xt[j]:
                    tot = add(tot, smul(xt[j], bc.get(n_, {})))
            if not eq(tot, {}):
                bad += 1
        log("   holdout: trained on %d, %d of %d held-out members satisfy it"
            % (ntr, len(blocks) - ntr - bad, len(blocks) - ntr))
        log("   [%.1fs]" % (time.time() - t0))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
