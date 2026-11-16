#!/usr/bin/env python3
"""n4rec10.py -- does the level 4 -> 6 blockage survive step 1's freedom?

n4rec9.py solved step 1 exactly (verified against the true residual) and then
the recursion died at the next step, level 4 -> 6.  That report is worth
exactly as much as the one it replaces: n4rec.py's `step` tests level 4 with
the level-2 and level-4 parts of q FROZEN at one solution of step 1, varying
only the 112 level-6 unknowns -- which is the same mistake as testing step 1
at the single point y = 0.  Step 1 does not have a solution, it has a FAMILY,
and step 2 may need another member of it.

So this asks the honest question:

   is there a direction that keeps step 1 solved and repairs step 2?

Concretely, one linear system on the rows of BOTH steps at once,

   J1(d) = 0                (step 1 stays solved, to first order)
   J2(d) = -F2              (step 2 is repaired, to first order)

with d ranging over all of level 2 + level 4 + level 6 -- 504 directions.  If
it is solvable, the level 4 -> 6 blockage is removable exactly as the level
2 -> 4 one was, and the recursion has to be re-run with the family carried
along instead of a single member.  If it is not, the blockage is of a
different kind from the first, and that is worth knowing too.

The derivative is taken EXACTLY, not by a one-sided difference: for each
direction the map t -> F(q + t d) is a polynomial in t (degree <= 3 -- three
level-2 factors reach level 6 and two odd derivatives bring them back to
level 4), so it is interpolated at t = -2, -1, 0, 1, 2 and the coefficient of
t is read off.  The coefficient of t^4 is reported as a self-test: it must
vanish.

Usage: ../venv/bin/python n4rec10.py
Writes n4rec10-out.txt.
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
from n4rec import qadd, qsmul, resid_full, dirs_lvl, rows_of_step, step
from n4rec2 import keep_step1
from n4noeth import mzero, C

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec10-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


keep_step2 = rows_of_step(2)


def keep_both(k):
    return keep_step1(k) or keep_step2(k)


_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    """The exact linear part of F along dirs[i], by interpolation in t."""
    q, d = _G["q"], _G["dirs"][i]
    V = {}
    for t in (-2, -1, 1, 2):
        R = resid_full(qadd(q, qsmul(t, d)))
        V[t] = {k: v for k, v in R.items() if keep_both(k)}
    R0 = resid_full(q)
    V[0] = {k: v for k, v in R0.items() if keep_both(k)}
    keys = set()
    for t in V:
        keys |= set(V[t])
    c1, c4 = {}, {}
    for k in keys:
        a = {t: V[t].get(k, C()) for t in (-2, -1, 0, 1, 2)}
        v1 = (8 * (a[1] - a[-1]) - (a[2] - a[-2])) * Q(1, 12)
        if v1:
            c1[k] = v1
        v4 = (a[-2] - 4 * a[-1] + 6 * a[0] - 4 * a[1] + a[2]) * Q(1, 24)
        if v4:
            c4[k] = v4
    return i, c1, len(c4)


def main():
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 2)
    with open("n4rec9-q2.pkl", "rb") as fh:
        D = pickle.load(fh)
    q2 = D["q2"]
    log("n4rec10: is the level 4 -> 6 blockage removable inside step 1's")
    log("solution family?")
    log("")

    F = resid_full(q2)
    F1 = {k: v for k, v in F.items() if keep_step1(k)}
    F2 = {k: v for k, v in F.items() if keep_step2(k)}
    log("at q2 (the step-1 solution of n4rec9):")
    log("   step-1 rows nonzero: %d   (step 1 is solved)" % len(F1))
    log("   step-2 rows nonzero: %d" % len(F2))

    ds2, ds4, ds6 = dirs_lvl(1), dirs_lvl(2), dirs_lvl(3)
    dirs = list(ds2) + list(ds4) + list(ds6)
    log("directions: %d level-2 + %d level-4 + %d level-6 = %d"
        % (len(ds2), len(ds4), len(ds6), len(dirs)))

    qp, dp = pickle.dumps(q2), pickle.dumps(dirs)
    cols = [None] * len(dirs)
    quartic = 0
    with mp.Pool(procs, _init, (qp, dp)) as pool:
        for i, c1, n4 in pool.imap_unordered(_col, range(len(dirs)),
                                             chunksize=4):
            cols[i] = c1
            quartic += n4
    log("columns built  [%.1fs]" % (time.time() - t0))
    log("self-test: %d nonzero quartic coefficients (must be 0)" % quartic)
    if quartic:
        log("!! the map is not of degree <= 3 in a direction -- the")
        log("   interpolation is invalid; aborting")
        return

    # --- the reference test: level-6 alone (what n4rec.py's step does) ------
    only6 = [{k: v for k, v in cols[j].items() if keep_step2(k)}
             for j in range(len(ds2) + len(ds4), len(dirs))]
    tgt2 = {k: -v for k, v in F2.items()}
    x = N.sparse_solve(only6, tgt2)
    log("")
    log("1) the test the recursion made -- level-6 unknowns only:")
    log("   %s" % ("solvable" if x is not None else "INCONSISTENT"
                   " (this is the 'dies at level 4 -> 6')"))

    # --- the honest test: step 1 preserved, step 2 repaired ----------------
    both = []
    for c in cols:
        both.append(dict(c))
    target = {}
    for k, v in F2.items():
        target[k] = -v
    # step-1 rows: target 0 (they are already zero and must stay zero)
    y = N.sparse_solve(both, target)
    log("")
    log("2) the honest test -- all %d directions, step-1 rows held at zero"
        % len(dirs))
    if y is None:
        bad = n4phi.diagnose(both, target)
        rows = sum(len(b[0]) for b in bad)
        log("   INCONSISTENT: %d blocked sectors, %d rows"
            % (len(bad), rows))
        log("   => at FIRST ORDER, no direction keeps step 1 solved and")
        log("      repairs step 2.  Unlike step 1, whose blockage vanished")
        log("      as soon as the kernel parameters were allowed to move,")
        log("      this one survives the whole of level 2 + 4 + 6.")
        log("      (First order only: the level-2 dependence is cubic here,")
        log("      so this is evidence, not proof.)")
    else:
        nz2 = sum(1 for j in range(len(ds2)) if y[j])
        nz4 = sum(1 for j in range(len(ds2), len(ds2) + len(ds4)) if y[j])
        nz6 = sum(1 for j in range(len(ds2) + len(ds4), len(dirs)) if y[j])
        log("   SOLVABLE: a first-order direction exists (%d level-2, %d"
            " level-4, %d level-6 components nonzero)" % (nz2, nz4, nz6))
        log("   => the level 4 -> 6 blockage is removable exactly as the")
        log("      level 2 -> 4 one was; the recursion must be re-run with")
        log("      step 1's family carried along.")
        # is it an honest solution, or only first order?
        dq = mzero()
        for j, c in enumerate(y):
            if c:
                dq = qadd(dq, qsmul(c, dirs[j]))
        Fn = resid_full(qadd(q2, dq))
        n1 = len([k for k in Fn if keep_step1(k)])
        n2 = len([k for k in Fn if keep_step2(k)])
        log("   exact residual after that step: %d step-1 rows, %d step-2"
            " rows" % (n1, n2))
        if n1 == 0 and n2 == 0:
            log("   and it is an EXACT solution of both steps.")
            with open("n4rec10-q3.pkl", "wb") as fh:
                pickle.dump(qadd(q2, dq), fh)
            log("   written to n4rec10-q3.pkl")

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
