#!/usr/bin/env python3
"""n4cons6.py -- the last enrichment of the candidate list, and where the
completion of T = -u breaks.

n4cons5.py graded the list right and still found nothing.  Two things were
missing from it, and both are of the right grade:

  * the SECOND-order scalars  tr(M q^{-1} D D' q), D, D' odd -- of g = -1,
    like a product of two first-order odd ones, but not equal to any
    combination of them;
  * the quartic products of the first-order odd scalars, of g = -2, which
    become g = 0 when multiplied by tr(M Q) tr(M' Q).

Both are added here, the second class in the form
tr(M Q) tr(M' Q) * s_i s_j s_k s_l with the four indices running over a
reduced set (else the list is of the order of 10^5).

The system is then solved on the rows of theta-level <= 2r for r = 0, 1, 2,
3, 4 in turn, so that the answer says WHERE the completion fails and not
merely that it does.

Usage: ../venv/bin/python n4cons6.py   ->  n4cons6-out.txt
"""
import itertools
import pickle
import sys
import time

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, mul, add, smul, eq, minv, dT, csolve,
                     d_xm, d_xp, is_solution, mmul, mtr, mapply)  # noqa: E402
from n4cons2 import SIG, ODD_DERIVS, EVEN_DERIVS, scalars     # noqa: E402
from n4cons5 import grades, homog                             # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons6-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def second_order(q, qi):
    """tr(M q^{-1} D D' q) for the eight odd D and D'."""
    out = {}
    for n1, d1 in ODD_DERIVS:
        for n2, d2 in ODD_DERIVS:
            DDq = mapply(d1, mapply(d2, q))
            for mn, M in SIG.items():
                f = mtr(mmul(M, mmul(qi, DDq)))
                if f:
                    out[(mn, n1, n2)] = f
    return out


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4cons6: the enriched, correctly graded candidate list, and where")
    log("the completion of T = -u breaks.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))
    qi = minv(q)
    ev, od = scalars(q)
    so = second_order(q, qi)
    log("first-order scalars: %d even, %d odd; second-order: %d"
        % (sum(1 for f in ev.values() if f), len(od), len(so)))

    Qs = {}
    for mn, M in SIG.items():
        f = mul(mono(nm=-1), mtr(mmul(M, qi)))
        if f:
            Qs[mn] = f
    log("the g = +1 scalars tr(M Q), Q = m^{-1}q^{-1}: %d" % len(Qs))

    pool = {}
    okeys = sorted(od)
    # g = -1 base: pairs of odd, second order, even first order
    base = {}
    for i in range(len(okeys)):
        for j in range(i, len(okeys)):
            f = mul(od[okeys[i]], od[okeys[j]])
            if f:
                base[("p", okeys[i], okeys[j])] = f
    for k, f in so.items():
        base[("2", k)] = f
    for k, f in ev.items():
        if f:
            base[("e", k)] = f
    log("base scalars of g = -1: %d" % len(base))
    for bn, bf in base.items():
        for mn, P in Qs.items():
            f = mul(P, bf)
            if homog(f, 0, 0):
                pool[("Q", mn) + bn] = f
    log("candidates of g = 0 after multiplying by tr(M Q): %d" % len(pool))

    # quartics: (s_i s_j)(s_k s_l) * tr(MQ) tr(M'Q), reduced index set
    red = okeys[:12]
    nq = 0
    for combo in itertools.combinations(range(len(red)), 4):
        f4 = od[red[combo[0]]]
        for c in combo[1:]:
            f4 = mul(f4, od[red[c]])
        if not f4:
            continue
        for mn1 in Qs:
            for mn2 in Qs:
                f = mul(Qs[mn1], mul(Qs[mn2], f4))
                if homog(f, 0, 0):
                    pool[("Q2", mn1, mn2) + combo] = f
                    nq += 1
    log("plus %d quartic candidates: %d in all" % (nq, len(pool)))

    names = sorted(pool, key=str)
    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        target = dT(q, qi, dop)
        cols = {n_: dop(pool[n_]) for n_ in names}
        cols[("log",)] = mono(nw=-1)
        allnames = names + [("log",)]
        keys = sorted(set(target) | {k for f in cols.values() for k in f})
        bylev = {}
        for k in keys:
            bylev.setdefault(len(k[3]), []).append(k)
        log("")
        log("dop = %s: rows by theta-level %s"
            % (dopname, {L: len(ks) for L, ks in sorted(bylev.items())}))
        for cut in sorted(bylev):
            rows = [k for L in sorted(bylev) if L <= cut for k in bylev[L]]
            cl = [[cols[n_].get(k, C()) for k in rows] for n_ in allnames]
            tg = [-target.get(k, C()) for k in rows]
            x = csolve(cl, tg)
            log("   levels <= %d (%4d rows, %d unknowns): %s  [%.1fs]"
                % (cut, len(rows), len(cl),
                   "solvable" if x is not None else "INCONSISTENT",
                   time.time() - t0))
            if x is None:
                break
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
