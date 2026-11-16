#!/usr/bin/env python3
"""n4rec2.py -- search the kernel of the first recursion step.

n4rec.py showed that the theta-recursion's first step, q^(0) -> q^(2), is
solvable for EVERY level-0 datum A tried, and that it leaves a kernel of
dimension 56: the level-2 part of the vacuum is determined only up to a
56-dimensional affine space.  Taking the particular solution and marching on
dies at the next step -- but that says nothing, because the 56 parameters were
never chosen.  This chooses them.

THE SHAPE OF THE PROBLEM.  Write

    q = A/(mw) + p + sum_i y_i k_i + x,

p + <k_i> the affine solution space of step 0, x the level-4 unknown.  The
rows of step 1 -- the equations at theta-level 2 and the constraints at level
3 -- are AFFINE in x and QUADRATIC in y: two level-2 factors meet at level 4,
and the two odd derivatives of the equations bring that back down to level 2.
(For the same reason nothing of higher degree occurs: three level-2 factors
land at level 6.)  So step 1 is a quadratic system in 56 + 280 unknowns, and
Newton on it is exact and terminates -- there is no truncation to control.

The derivative in y is taken by a SYMMETRIC difference, which is exact for a
quadratic map: [F(q+d) - F(q-d)]/2 = L(d) + 2Q(y,d).  The derivative in x is
the plain difference, x being affine.  Everything over Q(i).

Usage:  ../venv/bin/python n4rec2.py [--iters 6] [--A embedded|identity]
Writes n4rec2-out.txt.
"""
import itertools
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4vac
import n4phi
import n4rec
from n4rec import (monos_lvl, dirs_lvl, qadd, qsmul, resid_full, q0_of, step)
from n4noeth import mzero, minv, C, MI, PL

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec2-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def keep_step1(k):
    """Rows of step 1: equations at theta-level 2, constraints at level 3."""
    lv = len(k[2][3])
    return (lv == 3) if k[0] == "C" else (lv == 2)


_G = {}


def _init(qp, dp):
    import pickle
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _sym(i):
    """Exact derivative of the step-1 rows along direction i."""
    q, d = _G["q"], _G["dirs"][i]
    kind = d[1]
    if kind == "affine":
        Fp = resid_full(qadd(q, d[0]))
        F0 = _G.get("F0")
        out = {}
        for k, v in Fp.items():
            if keep_step1(k):
                out[k] = v
        return i, out, "p"
    Fp = resid_full(qadd(q, d[0]))
    Fm = resid_full(qadd(q, qsmul(-1, d[0])))
    out = {}
    keys = set(Fp) | set(Fm)
    for k in keys:
        if not keep_step1(k):
            continue
        v = (Fp.get(k, C()) - Fm.get(k, C())) * Q(1, 2)
        if v:
            out[k] = v
    return i, out, "s"


def columns(q, dirs, procs, F0):
    import pickle
    qp, dp = pickle.dumps(q), pickle.dumps(dirs)
    cols = [None] * len(dirs)
    with mp.Pool(procs, _init, (qp, dp)) as pool:
        for i, col, how in pool.imap_unordered(_sym, range(len(dirs)),
                                               chunksize=4):
            if how == "p":                    # affine: subtract the base value
                c = {}
                for k in set(col) | set(F0):
                    v = col.get(k, C()) - F0.get(k, C())
                    if v:
                        c[k] = v
                cols[i] = c
            else:
                cols[i] = col
    return cols


if __name__ == "__main__":
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 8)
    ITERS = 6
    if "--iters" in sys.argv:
        ITERS = int(sys.argv[sys.argv.index("--iters") + 1])
    which = "embedded"
    if "--A" in sys.argv:
        which = sys.argv[sys.argv.index("--A") + 1]

    if which == "identity":
        A = [[C(1), C()], [C(), C(1)]]
    else:
        ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
        B = n4vac.constrained_basis(ms0)
        qb, kern = n4vac.build_embedded_base(B)
        A = [[qb[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)]
             for i in (0, 1)]
    log("level-0 datum A = [[%s, %s], [%s, %s]]"
        % (A[0][0], A[0][1], A[1][0], A[1][1]))

    q0 = q0_of(A)
    p, ker0 = step(q0, 0, procs)
    log("step 0 solved: kernel dimension %d  [%.1fs]"
        % (len(ker0), time.time() - t0))

    # the kernel vectors as level-2 matrices
    ds2 = dirs_lvl(1)
    kmats = []
    for v in ker0:
        M = mzero()
        for i, c in enumerate(v):
            if c:
                M = qadd(M, qsmul(c, ds2[i]))
        kmats.append(M)
    ds4 = dirs_lvl(2)
    log("direction space of step 1: %d kernel + %d level-4 = %d unknowns"
        % (len(kmats), len(ds4), len(kmats) + len(ds4)))

    dirs = [(M, "sym") for M in kmats] + [(M, "affine") for M in ds4]

    q = qadd(q0, p)
    for it in range(ITERS):
        F = {k: v for k, v in resid_full(q).items() if keep_step1(k)}
        log("")
        log("iteration %d: %d nonzero step-1 rows  [%.1fs]"
            % (it, len(F), time.time() - t0))
        if not F:
            log("   step 1 SOLVED exactly.")
            break
        _G["F0"] = F
        cols = columns(q, dirs, procs, F)
        target = {k: -v for k, v in F.items()}
        x = N.sparse_solve(cols, target)
        if x is None:
            bad = n4phi.diagnose(cols, target)
            log("   Newton step inconsistent: %d blocked sectors, %d rows"
                % (len(bad), sum(len(b[0]) for b in bad)))
            for b in bad[:6]:
                log("      %d rows: %s" % (len(b[0]), b[1]))
            log("   -- falling back to Gauss-Newton (kills the reachable"
                " part)")
            x, n_gn = n4phi.gn_solve(cols, target)
            log("      %d sectors handled by the normal equations" % n_gn)
            log("      |F| before the step: %.6f" % n4phi.fnorm(F))
        dq = mzero()
        for i, c in enumerate(x):
            if c:
                dq = qadd(dq, qsmul(c, dirs[i][0]))
        q = qadd(q, dq)

    F = {k: v for k, v in resid_full(q).items() if keep_step1(k)}
    log("")
    log("after %d iterations: %d nonzero step-1 rows" % (ITERS, len(F)))
    if not F:
        log("STEP 1 CLOSED.  Continuing the recursion from level 4 ...")
        ok = True
        for r in (2, 3):
            part, ker = step(q, r, procs)
            if part is None:
                log("   the recursion dies at level %d" % (2 * r))
                ok = False
                break
            log("   level %d -> %d solved, kernel %d"
                % (2 * r, 2 * r + 2, len(ker)))
            q = qadd(q, part)
        if ok:
            Fall = resid_full(q)
            log("   closing check: %d nonzero rows in all"
                % len([k for k in Fall]))
    log("[%.1fs] done" % (time.time() - t0))
