#!/usr/bin/env python3
"""n4cons8.py -- is the logarithmic completion of n4cons7.py real?

n4cons7.py found, at the vacuum, a combination of logarithms of the
second-order currents whose d_w-derivative cancels that of T = -u.  With 67
columns against 36 rows that is no evidence at all by itself: what decides
is whether ONE set of coefficients works at every point of the family, and
whether coefficients fitted on part of the family predict the rest.

So this run
  1) measures the freedom at the vacuum (the kernel of the columns);
  2) solves ACROSS the family of exact solutions of n4fam.py;
  3) HOLDS OUT members: fits on the first ones, tests on the rest;
  4) repeats with the candidates restricted to the composites of ONE half,
     which is what "the stress tensor is a current of its half" asks for.

Usage: ../venv/bin/python n4cons8.py   ->  n4cons8-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q_

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, mul, add, smul, eq, minv, dT, csolve, cnull,
                     d_xm, d_xp, is_solution, mmul, mtr,
                     inv_scalar, lead_of)                      # noqa: E402
from n4cons2 import SIG, scalars                               # noqa: E402
from n4cons6 import second_order                               # noqa: E402
from n4cons7 import logcols                                    # noqa: E402
from n4rec import qadd, qsmul                                  # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons8-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def composites(q, half=None):
    """the scalar composites offered to the logarithm; half='+' keeps only
    those built from the derivatives of the plus half."""
    qi = minv(q)
    ev, od = scalars(q)
    so = second_order(q, qi)
    fs = {}
    for mn, M in SIG.items():
        f = mul(mono(nm=-1), mtr(mmul(M, qi)))
        if f and half is None:
            fs[("Q", mn)] = f
    for k, f in ev.items():
        if f and (half is None or half in k[1]):
            fs[("e", k)] = f
    for k, f in so.items():
        if half is None or (half in k[1] and half in k[2]):
            fs[("2", k)] = f
    return fs, qi


def blocks_for(fam, dop, half, names=None):
    out = []
    for _, qq in fam:
        fs, qqi = composites(qq, half)
        cols = logcols(fs, dop)
        cols[("log", "w")] = mono(nw=-1)
        out.append((dT(qq, qqi, dop), cols))
    if names is None:
        names = sorted(set().union(*[set(c) for _, c in out]), key=str)
    return out, names


def stack(blocks, names):
    cols = {nm: [] for nm in names}
    tgt = []
    for target, bc in blocks:
        keys = sorted(set(target) | {k for f in bc.values() for k in f})
        for nm in names:
            f = bc.get(nm, {})
            cols[nm].extend([f.get(k, C()) for k in keys])
        tgt.extend([-target.get(k, C()) for k in keys])
    return [cols[nm] for nm in names], tgt


def report(x, names, tag):
    nz = [(names[j], x[j]) for j in range(len(names)) if x[j]]
    log("      %s: %d nonzero coefficients" % (tag, len(nz)))
    for n_, v in sorted(nz, key=lambda t: str(t[0]))[:30]:
        log("         log %-40s %s" % (str(n_), v))


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    with open("n4cons-tan.pkl", "rb") as fh:
        tans = pickle.load(fh)
    log("n4cons8: is the logarithmic completion real?")
    log("")
    fam = [("vacuum", q)]
    for d in tans:
        for lam in (1, -1, Q_(1, 2), 2, 3, C(0, 1), C(1, 1), Q_(-1, 3)):
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq):
                fam.append(("q + (%s) d" % lam, qq))
    log("family of exact solutions: %d members" % len(fam))

    for half, hname in ((None, "both halves"), ("+", "the plus half only")):
        for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
            log("")
            log("=" * 70)
            log("candidates: %s;  dop = %s" % (hname, dopname))
            log("=" * 70)
            blocks, names = blocks_for(fam, dop, half)
            log("   %d log columns" % len(names))

            cl, tg = stack(blocks[:1], names)
            x = csolve(cl, tg)
            log("   at the vacuum alone (%d rows): %s"
                % (len(tg), "solvable" if x is not None else "INCONSISTENT"))
            if x is not None:
                A = [[cl[j][r] for j in range(len(cl))]
                     for r in range(len(tg))]
                log("      freedom there: kernel of dimension %d"
                    % len(cnull(A)))

            cl, tg = stack(blocks, names)
            xf = csolve(cl, tg)
            log("   across the family (%d rows, %d unknowns): %s  [%.1fs]"
                % (len(tg), len(cl),
                   "SOLVED" if xf is not None else "INCONSISTENT",
                   time.time() - t0))
            if xf is None:
                continue
            report(xf, names, "across the family")
            A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
            log("      freedom across the family: kernel %d" % len(cnull(A)))

            ntr = max(2, len(blocks) - 3)
            cl2, tg2 = stack(blocks[:ntr], names)
            xt = csolve(cl2, tg2)
            bad = 0
            for i in range(ntr, len(blocks)):
                tt, bc = blocks[i]
                tot = dict(tt)
                for j, n_ in enumerate(names):
                    if xt[j]:
                        tot = add(tot, smul(xt[j], bc.get(n_, {})))
                if not eq(tot, {}):
                    bad += 1
            log("      holdout: fitted on %d, %d of %d held-out members obey"
                % (ntr, len(blocks) - ntr - bad, len(blocks) - ntr))
            log("      [%.1fs]" % (time.time() - t0))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
