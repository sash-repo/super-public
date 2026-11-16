#!/usr/bin/env python3
"""n4win.py -- the tangent space in the TWO-VARIABLE class.

The class the whole N=4 search lives in is cut out by ONE grading, the
dilatation (z, w of weight 1, theta of weight 1/2, and m*q of degree -1):

    q^(2r) in  \\bigoplus_{a+b = -1-r}  z^a w^b m^{-1} Lambda^{2r}(theta),

and it is infinite-dimensional because a is free.  n4win0.py shows the
window is nevertheless ONE-SIDED and canonical: the even derivatives never
raise n_z and multiplication adds it, so with a_0 the z-exponent of the
theta-free level-0 part and s := n_z - a_0,

    {s <= 0} is a differential subring,   {s < -L} a differential ideal,

for every L >= 0.  s <= 0 is forced (a window with s > 0 is not closed under
the derivatives, which lower n_z by up to 2 in the residual), and L is the
only thing to choose.  L = 0 is the class of all the work so far.

THIS ENGINE measures the tangent space to the solution variety at the known
vacuum over the enlarged direction set s in [-L, 0], theta-levels 2..8, and
asks: is there a tangent direction
with a nonzero s < 0 part, i.e. a deformation of the vacuum that depends on
x^+ and x^- through TWO independent combinations?

Everything is exact over Q(i); the residual is the TRUE one (no truncation),
so a direction reported as tangent is tangent to the honest variety.

Usage: ../venv/bin/python n4win.py [L]
Writes n4win-out.txt.
"""
import itertools
import pickle
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import exactlin                                    # noqa: E402

import n4noeth as N                                # noqa: E402
import n4rec                                       # noqa: E402
from n4rec import qadd, qsmul, resid_full          # noqa: E402
from n4noeth import mono, mzero, C, MI, PL, is_solution   # noqa: E402

OUT = []
TAG = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4win-out%s.txt" % TAG, "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- the windowed class -----------------------------------------------------

def monos_win(r, s):
    """theta-level 2r, z-exponent s (a_0 = 0), dilatation degree -1."""
    return [mono(s, -1 - r - s, -1, S)
            for S in itertools.combinations(MI + PL, 2 * r)]


def dirs_win(r, s):
    out = []
    for g in (0, 1):
        for b in (0, 1):
            for mo in monos_win(r, s):
                dq = mzero()
                dq[g][b] = mo
                out.append(dq)
    return out


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
    global TAG
    t0 = time.time()
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    TAG = "-L%d" % L
    procs = max(1, mp.cpu_count() - 2)
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4win: the tangent space at the N=4 vacuum in the two-variable")
    log("class, window s in [%d, 0]." % (-L))
    log("")
    log("the base point is a solution: %s" % is_solution(q))

    dirs, tag_s, tag_l = [], [], []
    for s in range(-L, 1):
        for r in (1, 2, 3, 4):
            dl = dirs_win(r, s)
            dirs += dl
            tag_s += [s] * len(dl)
            tag_l += [2 * r] * len(dl)
    log("directions: %d  (z-shifts %s, theta-levels %s)"
        % (len(dirs), sorted(set(tag_s)), sorted(set(tag_l))))
    log("level-0 directions are excluded: they move the theta-free leading")
    log("monomial, which must stay a single monomial for q to be invertible")
    log("in the ring (the inverse of a sum of two monomials is not Laurent).")

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

    # a check of the triangularity claim: a direction at shift s must
    # produce rows only at n_z <= s.
    viol = 0
    for j, c in enumerate(cols):
        for k in c:
            if k[2][0] > tag_s[j]:
                viol += 1
    log("triangularity: rows above the direction's own z-shift: %d (must be 0)"
        % viol)

    rows = sorted({k for c in cols for k in c})
    log("the linearization has %d rows" % len(rows))
    byz = {}
    for k in rows:
        byz[k[2][0]] = byz.get(k[2][0], 0) + 1
    log("   rows by n_z: %s" % sorted(byz.items()))
    ridx = {k: i for i, k in enumerate(rows)}
    nr, nd = len(rows), len(dirs)
    re_cols, im_cols = [], []
    for c in cols:
        rr = [Q(0)] * nr
        ii = [Q(0)] * nr
        for k, v in c.items():
            rr[ridx[k]] = v.re
            ii[ridx[k]] = v.im
        re_cols.append(rr)
        im_cols.append(ii)
    big = []
    for r in range(nr):
        big.append([re_cols[j][r] for j in range(nd)]
                   + [-im_cols[j][r] for j in range(nd)])
    for r in range(nr):
        big.append([im_cols[j][r] for j in range(nd)]
                   + [re_cols[j][r] for j in range(nd)])
    log("solving a %d x %d rational system ..." % (len(big), 2 * nd))
    ns = exactlin.nullspace(big)
    dimC = len(ns) // 2
    log("tangent space: %d real dimensions = %d over Q(i)  [%.1fs]"
        % (len(ns), dimC, time.time() - t0))

    # --- where does the freedom live? --------------------------------------
    log("")
    log("=== does any tangent direction reach s < 0? ======================")
    shifts_used = set()
    per_vec = []
    for vec in ns:
        su = set()
        for j in range(nd):
            if vec[j] or vec[j + nd]:
                su.add(tag_s[j])
        shifts_used |= su
        per_vec.append(su)
    log("z-shifts touched by the tangent space as a whole: %s"
        % sorted(shifts_used))
    nneg = sum(1 for su in per_vec if any(s < 0 for s in su))
    log("real basis vectors with a nonzero s<0 part: %d of %d"
        % (nneg, len(ns)))

    # the s<0 content is what matters: project the kernel onto the s<0 block
    # and take the rank -- that is the dimension of the genuinely
    # two-variable part of the tangent space.
    negidx = [j for j in range(nd) if tag_s[j] < 0]
    if negidx:
        proj = []
        for vec in ns:
            proj.append([vec[j] for j in negidx]
                        + [vec[j + nd] for j in negidx])
        rk = exactlin.rank(proj) if proj else 0
        log("rank of the kernel projected on the s<0 block: %d real = %s"
            % (rk, "%g" % (rk / 2.0)))
        log("   (that is the dimension of the two-variable part of the")
        log("    tangent space; 0 means every tangent direction is a")
        log("    function of w alone, to first order.)")
    log("")
    log("[%.1fs] done" % (time.time() - t0))
    with open("n4win-kernel%s.pkl" % TAG, "wb") as fh:
        pickle.dump({"ns": ns, "tag_s": tag_s, "tag_l": tag_l,
                     "nd": nd, "L": L}, fh)
    log("kernel written to n4win-kernel%s.pkl" % TAG)


if __name__ == "__main__":
    main()
