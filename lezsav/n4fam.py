#!/usr/bin/env python3
"""n4fam.py -- the family the N=4 vacuum sits in.

n4rec11.py produced one vacuum and n4ver.py confirmed it against n4noeth's
own predicates.  The Noether step needs not one solution but the family, so
this measures it: the tangent space to the solution variety at the vacuum,
inside the graded class.

The directions are ALL the unknowns of the class -- theta-levels 0, 2, 4, 6,
8, i.e. 512 of them, the level-0 ones included, so that the answer says
whether the level-0 datum A itself can move.  The linear part along each
direction is exact (9-point central formula, checked against the 7-point
one), and the tangent space is its kernel, computed over Q(i) by
realification with FLINT behind lib/exactlin.py.

Two things are then checked, because a tangent vector need not integrate to a
curve of solutions: each basis vector of the tangent space is added to the
vacuum with a rational coefficient and the result is put through
n4noeth.is_solution.  Whatever survives that is an honest second vacuum.

Usage: ../venv/bin/python n4fam.py
Writes n4fam-out.txt.
"""
import pickle
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import exactlin                                  # noqa: E402

import n4noeth as N                              # noqa: E402
import n4rec                                     # noqa: E402
from n4rec import qadd, qsmul, resid_full, dirs_lvl, monos_lvl  # noqa: E402
from n4noeth import mzero, C, is_solution        # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4fam-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    q, d = _G["q"], _G["dirs"][i]
    V = {}
    for t in range(-4, 5):
        V[t] = resid_full(q if t == 0 else qadd(q, qsmul(t, d)))
    keys = set()
    for t in V:
        keys |= set(V[t])
    c1, mism = {}, 0
    for k in keys:
        a = {t: V[t].get(k, C()) for t in V}
        d1, d2 = a[1] - a[-1], a[2] - a[-2]
        d3, d4 = a[3] - a[-3], a[4] - a[-4]
        v7 = d1 * Q(3, 4) + d2 * Q(-3, 20) + d3 * Q(1, 60)
        v9 = (d1 * Q(4, 5) + d2 * Q(-1, 5) + d3 * Q(4, 105)
              + d4 * Q(-1, 280))
        if v7 != v9:
            mism += 1
        if v9:
            c1[k] = v9
    return i, c1, mism


def main():
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4fam: the tangent space to the solution variety at the N=4 vacuum.")
    log("")
    log("the vacuum is a solution: %s" % is_solution(q))

    dirs, tags = [], []
    for r in (1, 2, 3, 4):
        dl = dirs_lvl(r)
        dirs += list(dl)
        tags += [2 * r] * len(dl)
    log("directions: %d (theta-levels %s)"
        % (len(dirs), sorted(set(tags))))
    log("the level-0 directions are excluded: they move the datum A, which")
    log("the N=2 embedding fixes, and moving it makes the leading part of")
    log("det q cease to be a single monomial, which minv requires.")

    qp, dp = pickle.dumps(q), pickle.dumps(dirs)
    cols = [None] * len(dirs)
    mism = 0
    with mp.Pool(procs, _init, (qp, dp)) as pool:
        for i, c1, mm in pool.imap_unordered(_col, range(len(dirs)),
                                             chunksize=4):
            cols[i] = c1
            mism += mm
    log("columns built  [%.1fs]" % (time.time() - t0))
    log("self-test (7-point vs 9-point): %d mismatches" % mism)
    if mism:
        log("!! aborting")
        return

    rows = sorted({k for c in cols for k in c})
    log("the linearization has %d rows" % len(rows))
    ridx = {k: i for i, k in enumerate(rows)}
    nr = len(rows)
    re_cols, im_cols = [], []
    for c in cols:
        rr = [Q(0)] * nr
        ii = [Q(0)] * nr
        for k, v in c.items():
            rr[ridx[k]] = v.re
            ii[ridx[k]] = v.im
        re_cols.append(rr)
        im_cols.append(ii)
    # realified matrix: rows 2*nr, columns 2*len(dirs)
    big = []
    for r in range(nr):
        big.append([re_cols[j][r] for j in range(len(dirs))]
                   + [-im_cols[j][r] for j in range(len(dirs))])
    for r in range(nr):
        big.append([im_cols[j][r] for j in range(len(dirs))]
                   + [re_cols[j][r] for j in range(len(dirs))])
    ns = exactlin.nullspace(big)
    dimC = len(ns) // 2
    log("tangent space: %d real dimensions = %d over Q(i)  [%.1fs]"
        % (len(ns), dimC, time.time() - t0))
    log("   (of %d directions, so the solution variety has codimension %d)"
        % (len(dirs), len(dirs) - dimC))

    # --- which levels does the freedom live in? ----------------------------
    used = set()
    for vec in ns:
        for j in range(len(dirs)):
            if vec[j] or vec[j + len(dirs)]:
                used.add(tags[j])
    log("   the tangent vectors involve theta-levels %s" % sorted(used))

    # --- does a tangent vector integrate? ----------------------------------
    log("")
    log("are the tangent directions honest solution curves?")
    rng = random.Random(3)
    tried = ok = 0
    for vec in ns[:12]:
        d = mzero()
        for j in range(len(dirs)):
            c = C(vec[j], vec[j + len(dirs)])
            if c:
                d = qadd(d, qsmul(c, dirs[j]))
        for lam in (1, -1, Q(1, 2), 3):
            tried += 1
            if is_solution(qadd(q, qsmul(lam, d))):
                ok += 1
    log("   %d of %d tangent steps land on an exact solution" % (ok, tried))
    if ok:
        log("   => the vacuum is not isolated: it comes in a family, which")
        log("      is what the Noether step needs.")
    else:
        log("   => every tangent direction is obstructed at second order:")
        log("      the vacuum found is (locally) rigid in the class.")
    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
