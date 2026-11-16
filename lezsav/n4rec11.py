#!/usr/bin/env python3
"""n4rec11.py -- the recursion to the top, with the freedom carried along.

The two blockages that level-wise solving reports were both
artifacts of the SAME defect: n4rec.py's `step` solves one level, keeps ONE
particular solution, and marches on with the earlier levels frozen.  When the
next level fails it is not the vacuum that is obstructed, it is that
particular choice.

  - n4rec9.py: allowing the 56 kernel parameters of step 0 to move solved
    step 1 exactly (verified against the true residual);
  - n4rec10.py: allowing the level-4 part to move solved step 2 exactly --
    and the repair needed NO level-2 component at all, only 4 level-4 and 20
    level-6 ones.  That is precisely the freedom `step` had thrown away.

This finishes the march the same way: at each remaining level, hold every
earlier row at zero and repair the current one, over ALL the unknowns of the
graded class at once -- levels 2, 4, 6 and 8, 508 directions.  After the last
level the equations at theta-level 8 are the closing conditions, with no
unknown left to meet them; so the full residual at the end is the answer to
the whole question.

The derivative along a direction is exact: t -> F(q + t d) is a polynomial in
t of degree at most 4 (five level-2 factors would need ten of the eight odd
generators), so the 5-point central formula gives its linear coefficient
exactly.  As a self-test the 7-point formula, exact up to degree 6, is
computed as well and the two must agree on every row of every direction.

Usage: ../venv/bin/python n4rec11.py
Writes n4rec11-out.txt.
"""
import pickle
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4phi
import n4rec
import n4rec2
from n4rec import qadd, qsmul, resid_full, dirs_lvl, rows_of_step
from n4noeth import mzero, C

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec11-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def level_of(k):
    return len(k[2][3])


def step_of(k):
    """Which step of the recursion the row belongs to."""
    lv = level_of(k)
    return (lv - 1) // 2 if k[0] == "C" else lv // 2


_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    """The exact linear coefficient of t -> F(q + t d).

    A term carrying t^k carries at least 2k thetas (every factor of the
    direction has theta-level >= 2), and a row at theta-level L is reached
    from level at most L + 2 (the equations apply two odd derivatives), so
    on the rows used here k <= 5.  The 7- and 9-point central formulas are
    exact up to degree 6 and 8; they are compared, and the 5-point one --
    exact only up to degree 4 -- is compared too, to show where the higher
    terms actually appear."""
    q, d = _G["q"], _G["dirs"][i]
    V = {}
    for t in range(-4, 5):
        V[t] = resid_full(q if t == 0 else qadd(q, qsmul(t, d)))
    keys = set()
    for t in V:
        keys |= set(V[t])
    c1, hi5, hi7 = {}, 0, 0
    for k in keys:
        a = {t: V[t].get(k, C()) for t in V}
        d1, d2 = a[1] - a[-1], a[2] - a[-2]
        d3, d4 = a[3] - a[-3], a[4] - a[-4]
        v5 = (8 * d1 - d2) * Q(1, 12)
        v7 = d1 * Q(3, 4) + d2 * Q(-3, 20) + d3 * Q(1, 60)
        v9 = (d1 * Q(4, 5) + d2 * Q(-1, 5) + d3 * Q(4, 105)
              + d4 * Q(-1, 280))
        if v5 != v9:
            hi5 += 1
        if v7 != v9:
            hi7 += 1
        if v9:
            c1[k] = v9
    return i, c1, hi5, hi7


def main():
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    with open("n4rec10-q3.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4rec11: the theta-recursion to the top, carrying the freedom of")
    log("every earlier level instead of freezing one solution.")
    log("")

    F = resid_full(q)
    by = {}
    for k in F:
        by[step_of(k)] = by.get(step_of(k), 0) + 1
    log("start: q solves steps 0, 1, 2; nonzero rows by step: %s"
        % (sorted(by.items()) or "none"))

    dirs = []
    for r in (1, 2, 3, 4):
        dirs += list(dirs_lvl(r))
    log("directions: %d (levels 2, 4, 6, 8)" % len(dirs))

    for r in (3, 4):
        log("")
        log("--- step %d: equations at theta-level %d, constraints at level"
            " %d ---" % (r, 2 * r, 2 * r + 1))
        F = resid_full(q)
        tgt = {k: -v for k, v in F.items() if step_of(k) == r}
        holdrows = {k for k in F if step_of(k) < r}
        if holdrows:
            log("   !! %d earlier rows are nonzero -- aborting" % len(holdrows))
            return
        log("   %d nonzero rows to repair" % len(tgt))
        if not tgt:
            log("   already satisfied")
            continue
        qp, dp = pickle.dumps(q), pickle.dumps(dirs)
        cols = [None] * len(dirs)
        n5 = n7 = 0
        with mp.Pool(procs, _init, (qp, dp)) as pool:
            for i, c1, h5, h7 in pool.imap_unordered(_col, range(len(dirs)),
                                                     chunksize=4):
                cols[i] = c1
                n5 += h5
                n7 += h7
        log("   columns built  [%.1fs]" % (time.time() - t0))
        log("   self-test: 7-point vs 9-point %d mismatches (must be 0);"
            " 5-point vs 9-point %d (degree > 4 where nonzero)" % (n7, n5))
        if n7:
            log("   !! the derivative is not settled -- aborting")
            return
        # keep every earlier row at zero, repair step r
        keep = {k for k in tgt}
        for c in cols:
            keep |= {k for k in c if step_of(k) <= r}
        sub = [{k: v for k, v in c.items() if step_of(k) <= r} for c in cols]
        x = N.sparse_solve(sub, tgt)
        if x is None:
            bad = n4phi.diagnose(sub, tgt)
            log("   INCONSISTENT at first order: %d sectors, %d rows"
                % (len(bad), sum(len(b[0]) for b in bad)))
            log("   (first order only -- the dependence is polynomial of"
                " degree up to 4)")
            return
        dq = mzero()
        for j, c in enumerate(x):
            if c:
                dq = qadd(dq, qsmul(c, dirs[j]))
        qn = qadd(q, dq)
        Fn = resid_full(qn)
        left = {k: v for k, v in Fn.items() if step_of(k) <= r}
        log("   first-order step found; exact residual on steps 0..%d"
            " afterwards: %d rows" % (r, len(left)))
        if left:
            log("   the step is only first-order -- the higher terms bite;"
                " stopping")
            return
        q = qn
        log("   step %d SOLVED exactly  [%.1fs]" % (r, time.time() - t0))

    # --- the closing check --------------------------------------------------
    F = resid_full(q)
    log("")
    log("=== THE CLOSING CHECK ===========================================")
    log("full residual of (IK4) at the constructed q: %d nonzero rows"
        % len(F))
    if not F:
        log("")
        log("q solves (IK4) EXACTLY: every constraint and both equations,")
        log("at every theta-level.  The N=4 vacuum of the graded class")
        log("EXISTS, and this is one.")
        with open("n4rec11-vacuum.pkl", "wb") as fh:
            pickle.dump(q, fh)
        log("written to n4rec11-vacuum.pkl")
        # a short description of it
        nz = 0
        for i in (0, 1):
            for j in (0, 1):
                nz += len(q[i][j])
        log("the solution has %d nonzero monomial coefficients in its four"
            " entries" % nz)
        lv = {}
        for i in (0, 1):
            for j in (0, 1):
                for k in q[i][j]:
                    lv[len(k[3])] = lv.get(len(k[3]), 0) + 1
        log("by theta-level: %s" % sorted(lv.items()))
    else:
        by = {}
        for k in F:
            by[step_of(k)] = by.get(step_of(k), 0) + 1
        log("remaining by step: %s" % sorted(by.items()))
    log("=================================================================")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
