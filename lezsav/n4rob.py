#!/usr/bin/env python3
"""n4rob.py -- is the step-1 obstruction real, or is it the linearization?

n4abs.py repairs a step by a FIRST-ORDER solve, and the residual depends on
the level-2 unknowns polynomially (degree up to 4), so an inconsistency
reported there is a statement about the linearization until it is shown to
be one about the system.  That is exactly the trap the remark after
Proposition 7.2.1 records, and it has to be closed rather than assumed away.

It can be closed in one direction exactly.  The level-4 unknowns enter the
rows of step 1 LINEARLY -- that is the content of the grading -- so for a
FIXED level-2 part u the step-1 system is an honest linear system in the
level-4 part v, and its consistency is decided exactly.  What is left is the
dependence on u, and u ranges over the kernel of the step-0 system.  So this
engine samples that kernel: it draws random points u of it, each of which
keeps every step-0 row at zero exactly, and at each one decides the step-1
system in v exactly.

An inconsistency at every sampled u is not a proof; but it is a statement
about the system at many points of the variety of admissible u, not about a
single tangent space, and it is the strongest thing available short of
eliminating 392 unknowns of degree 4.

Usage: ../venv/bin/python n4rob.py [a0] [NLO] [NHI] [SAMPLES] [loop]
Writes n4rob-out-<loop><a0>.txt.
"""
import pickle
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4phi
import n4rec
import n4loop
from n4rec import qadd, qsmul, resid_full
from n4noeth import mzero, C
from n4loop import dirs_abs, q0_of_loop, step_of, E

OUT = []
TAG = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rob-out%s.txt" % TAG, "w") as fh:
        fh.write("\n".join(OUT) + "\n")


_G = {}


def _init(qp, dp, lin):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)
    _G["lin"] = lin


def _col(i):
    """If lin, the dependence is exactly linear and one difference suffices;
    the claim is checked, not assumed: the second difference must vanish."""
    q, d = _G["q"], _G["dirs"][i]
    R0 = resid_full(q)
    Rp = resid_full(qadd(q, d))
    Rm = resid_full(qadd(q, qsmul(-1, d)))
    keys = set(R0) | set(Rp) | set(Rm)
    c1, nonlin = {}, 0
    for k in keys:
        a0_, ap, am = R0.get(k, C()), Rp.get(k, C()), Rm.get(k, C())
        lin = (ap - am) * Q(1, 2)
        if ap + am - a0_ * 2:
            nonlin += 1
        if lin:
            c1[k] = lin
    return i, c1, nonlin


def build(q, ds, procs):
    qp, dp = pickle.dumps(q), pickle.dumps(ds)
    cols = [None] * len(ds)
    nonlin = 0
    with mp.Pool(procs, _init, (qp, dp, True)) as pool:
        for i, c1, nl in pool.imap_unordered(_col, range(len(ds)),
                                             chunksize=4):
            cols[i] = c1
            nonlin += nl
    return cols, nonlin


def main():
    global TAG
    t0 = time.time()
    a0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nlo = int(sys.argv[2]) if len(sys.argv) > 2 else -3
    nhi = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    nsamp = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    which = sys.argv[5] if len(sys.argv) > 5 else "mono"
    procs = max(1, mp.cpu_count() - 2)
    TAG = "-%s%d" % (which, a0)
    LOOPS = {"mono": lambda a: E((0, 0, a, 2), (1, 1, a, 2)),
             "diag": lambda a: E((0, 0, a, 1), (1, 1, 0, 1)),
             "uni": lambda a: E((0, 0, 0, 1), (0, 1, a, 1), (1, 1, 0, 1))}
    q0 = q0_of_loop(LOOPS[which](a0))
    log("n4rob: is the step-1 obstruction a property of the system?")
    log("loop %s, a_0 = %d, absolute window n_z in [%d, %d], %d samples."
        % (which, a0, nlo, nhi, nsamp))
    log("")

    # --- step 0: a particular solution and the kernel, in the level-2 part --
    d2, t2 = dirs_abs(1, nlo, nhi)
    cols2, nl2 = build(q0, d2, procs)
    log("level-2 unknowns: %d; the dependence of the step-0 rows on them is"
        " linear (%d second differences nonzero, must be 0)" % (len(d2), nl2))
    F = resid_full(q0)
    tgt0 = {k: -v for k, v in F.items() if step_of(k) == 0}
    sub0 = [{k: v for k, v in c.items() if step_of(k) == 0} for c in cols2]
    x0 = N.sparse_solve(sub0, tgt0)
    if x0 is None:
        log("!! step 0 itself is inconsistent -- nothing to sample")
        return
    ker0 = N.sparse_null(sub0)
    log("step 0 solvable; kernel of dimension %d  [%.1fs]"
        % (len(ker0), time.time() - t0))

    def qof(coeffs):
        u = mzero()
        for j, c in enumerate(coeffs):
            if c:
                u = qadd(u, qsmul(c, d2[j]))
        return qadd(q0, u)

    d4, t4 = dirs_abs(2, nlo, nhi)
    log("level-4 unknowns: %d" % len(d4))
    log("")
    rng = random.Random(19)
    bad = ok = 0
    for s in range(nsamp):
        coeffs = list(x0)
        if s:
            for v in ker0:
                lam = C(rng.randint(-4, 4), rng.randint(-4, 4))
                if lam:
                    coeffs = [a + lam * b for a, b in zip(coeffs, v)]
        q = qof(coeffs)
        Fq = resid_full(q)
        left0 = [k for k in Fq if step_of(k) == 0]
        if left0:
            log("  sample %d: !! %d step-0 rows nonzero -- skipped"
                % (s, len(left0)))
            continue
        tgt1 = {k: -v for k, v in Fq.items() if step_of(k) == 1}
        cols4, nl4 = build(q, d4, procs)
        sub1 = [{k: v for k, v in c.items() if step_of(k) == 1} for c in cols4]
        x1 = N.sparse_solve(sub1, tgt1)
        st = "step 1 SOLVABLE in v" if x1 is not None else "step 1 blocked"
        log("  sample %d (%s): %d rows of step 1, %d level-4 columns,"
            " %d nonlinear : %s"
            % (s, "the particular solution" if s == 0 else "random point of"
               " the kernel", len(tgt1), len(d4), nl4, st))
        if x1 is None:
            bad += 1
        else:
            ok += 1
        log("     [%.1fs]" % (time.time() - t0))
    log("")
    log("=== %d of %d sampled level-2 parts leave step 1 solvable ==="
        % (ok, ok + bad))
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
