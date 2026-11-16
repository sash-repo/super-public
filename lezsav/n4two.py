#!/usr/bin/env python3
"""n4two.py -- the recursion in the class with a GENUINELY two-variable
leading term.

The class is cut out by the dilatation grading alone,

    q^(2r) in  \\bigoplus_{a+b=-1-r} z^a w^b m^{-1} Lambda^{2r}(theta),

and is infinite-dimensional because a is free.  What decides the answer is
NOT the tail in a but the leading monomial: the theta-free level-0 part must
be a single monomial

    q^(0) = A z^{a_0} w^{-1-a_0} / m

(q must be invertible in the ring, and the inverse of a sum of two monomials
is not Laurent), and on such a q^(0)

    d_z q^(0) / d_w q^(0)  =  (a_0 / z) / ((-1-a_0) / w),

which is a CONSTANT -- so d_+ and d_- proportional, so a function of one
light-cone combination -- exactly when a_0 = 0 (the vacuum family: q^(0) is
w-free) or a_0 = -1 (the mirror family: z-free).  For every other a_0 the
solution depends on x^+ and x^- through two independent combinations by
construction, and the question becomes finite:

    does the theta-recursion close for a_0 not in {0, -1}?

Step 0 answers a piece of it outright, because it has no earlier choice in
it: its solvability is a property of (a_0, A) alone.

Window: the unknowns at theta-level 2r range over z^{a_0+s} w^{-1-a_0-r-s}
for s in [SLO, SHI].  n4win0.py shows the derivatives never raise n_z and
multiplication adds it, so the rows a window can reach are known in advance
and the ones it cannot are reported rather than silently dropped.

Usage: ../venv/bin/python n4two.py [SLO] [SHI]
Writes n4two-out.txt.
"""
import itertools
import pickle
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4phi
import n4rec
from n4rec import qadd, qsmul, resid_full
from n4noeth import mono, mzero, C, MI, PL, is_solution
from n4emb import cx

OUT = []
TAG = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4two-out%s.txt" % TAG, "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- the class --------------------------------------------------------------

def monos(a0, r, s):
    return [mono(a0 + s, -1 - a0 - r - s, -1, S)
            for S in itertools.combinations(MI + PL, 2 * r)]


def dirs(a0, r, slo, shi):
    out, tag = [], []
    for s in range(slo, shi + 1):
        for g in (0, 1):
            for b in (0, 1):
                for mo in monos(a0, r, s):
                    dq = mzero()
                    dq[g][b] = mo
                    out.append(dq)
                    tag.append(s)
    return out, tag


def q0_of(A, a0):
    q = mzero()
    for i in (0, 1):
        for j in (0, 1):
            if A[i][j]:
                q[i][j] = {(a0, -1 - a0, -1, ()): cx(A[i][j])}
    return q


def det2(A):
    return cx(A[0][0]) * cx(A[1][1]) - cx(A[0][1]) * cx(A[1][0])


def level_of(k):
    return len(k[2][3])


def step_of(k):
    lv = level_of(k)
    return (lv - 1) // 2 if k[0] == "C" else lv // 2


_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    """Exact linear coefficient, with a linearity self-test: the residual is
    evaluated at five shifts and the second difference must vanish on the
    rows of the current step (it is the level-2r+2 unknowns that enter
    linearly there)."""
    q, d = _G["q"], _G["dirs"][i]
    V = {t: resid_full(q if t == 0 else qadd(q, qsmul(t, d)))
         for t in (-2, -1, 0, 1, 2)}
    keys = set()
    for t in V:
        keys |= set(V[t])
    c1, nonlin = {}, 0
    for k in keys:
        a = {t: V[t].get(k, C()) for t in V}
        lin = (a[1] - a[-1]) * Q(1, 2)
        quad = a[1] + a[-1] - a[0] * 2
        if quad or (a[2] - a[-2]) != lin * 4:
            nonlin += 1
        if lin:
            c1[k] = lin
    return i, c1, nonlin


def run_step(q, a0, r, slo, shi, procs, verbose=True):
    """Solve the rows of step r for the level-(2r+2) unknowns."""
    F = resid_full(q)
    F0 = {k: v for k, v in F.items() if step_of(k) == r}
    earlier = {k for k in F if step_of(k) < r}
    ds, tg = dirs(a0, r + 1, slo, shi)
    qp, dp = pickle.dumps(q), pickle.dumps(ds)
    cols = [None] * len(ds)
    nonlin = 0
    with mp.Pool(procs, _init, (qp, dp)) as pool:
        for i, c1, nl in pool.imap_unordered(_col, range(len(ds)),
                                             chunksize=4):
            cols[i] = {k: v for k, v in c1.items() if step_of(k) == r}
            nonlin += nl
    reach = {k for c in cols for k in c}
    unreach = {k for k in F0 if k not in reach}
    target = {k: -v for k, v in F0.items()}
    x = N.sparse_solve(cols, target)
    ker = N.sparse_null(cols) if x is not None else None
    if verbose:
        log("      %d unknowns, %d nonzero rows at this step, %d of them"
            " outside the window's reach" % (len(ds), len(F0), len(unreach)))
        if earlier:
            log("      !! %d earlier rows nonzero" % len(earlier))
    return x, ker, ds, tg, F0, unreach, nonlin


def main():
    global TAG
    t0 = time.time()
    slo = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    shi = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    TAG = "-s%d_%d" % (slo, shi)
    procs = max(1, mp.cpu_count() - 2)
    log("n4two: is there a solution whose leading monomial is genuinely")
    log("two-variable?   window s in [%d, %d]." % (slo, shi))
    log("")
    log("q^(0) = A z^{a_0} w^{-1-a_0} / m.  d_+ and d_- are proportional on")
    log("it exactly when a_0 = 0 or a_0 = -1; every other a_0 is a genuinely")
    log("two-variable leading term.")
    log("")

    random.seed(7)
    As = [("identity", [[C(1), C()], [C(), C(1)]]),
          ("diag(2,2)", [[C(2), C()], [C(), C(2)]]),
          ("diag(1,-1)", [[C(1), C()], [C(), C(-1)]]),
          ("diag(1,i)", [[C(1), C()], [C(), C(0, 1)]]),
          ("antidiag", [[C(), C(1)], [C(1), C()]]),
          ("upper", [[C(1), C(1)], [C(), C(1)]])]
    for t in range(4):
        while True:
            A = [[C(random.randint(-3, 3), random.randint(-3, 3))
                  for _ in (0, 1)] for _ in (0, 1)]
            if det2(A):
                break
        As.append(("random %d" % (t + 1), A))

    log("=== step 0, the level-0 equations and the level-1 constraints ====")
    log("(step 0 carries no earlier choice, so its solvability is a")
    log(" property of the pair (a_0, A) alone)")
    log("")
    summary = {}
    for a0 in (0, -1, 1, -2, 2):
        kind = ("w-free" if a0 == 0 else
                "z-free" if a0 == -1 else "TWO-VARIABLE")
        log("--- a_0 = %-3d  (%s) ---------------------------------------"
            % (a0, kind))
        good = []
        for name, A in As:
            q = q0_of(A, a0)
            x, ker, ds, tg, F0, unreach, nonlin = run_step(
                q, a0, 0, slo, shi, procs, verbose=False)
            st = ("solvable, kernel %d" % len(ker)) if x is not None \
                else "INCONSISTENT"
            log("    A = %-11s det = %-12s : %-22s  (%d rows, %d beyond"
                " the window, %d nonlinear)"
                % (name, det2(A), st, len(F0), len(unreach), nonlin))
            if x is not None:
                good.append(name)
        summary[a0] = (len(good), len(As))
        log("    -> solvable for %d of %d data" % (len(good), len(As)))
        log("    [%.1fs]" % (time.time() - t0))
        log("")

    log("=== summary =====================================================")
    for a0 in sorted(summary):
        n, tot = summary[a0]
        log("   a_0 = %-3d : step 0 solvable for %d of %d level-0 data"
            % (a0, n, tot))
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
