#!/usr/bin/env python3
"""n4two2.py -- the theta-recursion to the top for a two-variable leading
monomial.

n4two.py finds step 0 solvable for a_0 = 1, i.e. for

    q^(0) = A z w^{-2} / m,     d_z q^(0) / d_w q^(0) = -w / (2z),

which is NOT a constant: on such a q the two light-cone derivatives are
independent, which is exactly what the completion of bold T needs and what
no solution found so far provides.

This pushes that start through the remaining theta-levels the way n4rec11
does -- repairing the current level over ALL the unknowns of the class at
once, never freezing an earlier choice, since freezing an earlier choice is what produces
spurious blockages.  The step is iterated to convergence rather than
taken once, because moving every level at once makes the dependence
polynomial (degree <= 4) rather than linear.

The last level has no unknown left to meet it: the residual after it is the
answer.

Usage: ../venv/bin/python n4two2.py [a0] [SLO] [SHI]
Writes n4two2-out-a<a0>-s<slo>_<shi>.txt.
"""
import pickle
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4phi
import n4rec
import n4two
from n4rec import qadd, qsmul, resid_full
from n4noeth import mzero, C, is_solution
from n4two import dirs, q0_of, det2, step_of, level_of

OUT = []
TAG = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4two2-out%s.txt" % TAG, "w") as fh:
        fh.write("\n".join(OUT) + "\n")


_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    q, d = _G["q"], _G["dirs"][i]
    V = {t: resid_full(q if t == 0 else qadd(q, qsmul(t, d)))
         for t in range(-4, 5)}
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
    a0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    slo = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    shi = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    TAG = "-a%d-s%d_%d" % (a0, slo, shi)
    procs = max(1, mp.cpu_count() - 2)

    A = [[C(2), C()], [C(), C(2)]]
    q = q0_of(A, a0)
    log("n4two2: the theta-recursion for a_0 = %d, window s in [%d, %d]."
        % (a0, slo, shi))
    log("q^(0) = A z^{%d} w^{%d} / m with A = diag(2,2)." % (a0, -1 - a0))
    if a0 not in (0, -1):
        log("d_z q^(0) / d_w q^(0) = (%d/z)/(%d/w) is NOT constant: the two"
            % (a0, -1 - a0))
        log("light-cone derivatives are independent on this leading term.")
    log("")

    ds, tg = [], []
    for r in (1, 2, 3, 4):
        d_, t_ = dirs(a0, r, slo, shi)
        ds += d_
        tg += t_
    log("unknowns: %d (theta-levels 2,4,6,8; z-shifts %s)"
        % (len(ds), sorted(set(tg))))

    for r in (0, 1, 2, 3):
        log("")
        log("--- step %d: equations at theta-level %d, constraints at %d ---"
            % (r, 2 * r, 2 * r + 1))
        for it in range(6):
            F = resid_full(q)
            earlier = {k for k in F if step_of(k) < r}
            if earlier:
                log("   !! %d earlier rows nonzero -- aborting" % len(earlier))
                return
            tgt = {k: -v for k, v in F.items() if step_of(k) == r}
            if not tgt:
                log("   satisfied after %d iteration(s)  [%.1fs]"
                    % (it, time.time() - t0))
                break
            qp, dp = pickle.dumps(q), pickle.dumps(ds)
            cols = [None] * len(ds)
            mism = 0
            with mp.Pool(procs, _init, (qp, dp)) as pool:
                for i, c1, mm in pool.imap_unordered(_col, range(len(ds)),
                                                    chunksize=4):
                    cols[i] = c1
                    mism += mm
            if mism:
                log("   !! derivative not settled (%d mismatches)" % mism)
                return
            sub = [{k: v for k, v in c.items() if step_of(k) <= r}
                   for c in cols]
            reach = {k for c in sub for k in c}
            unreach = [k for k in tgt if k not in reach]
            log("   iteration %d: %d rows to repair, %d beyond the window's"
                " reach" % (it, len(tgt), len(unreach)))
            if unreach:
                byz = {}
                for k in unreach:
                    byz[k[2][0]] = byz.get(k[2][0], 0) + 1
                log("      the unreachable rows sit at n_z = %s"
                    % sorted(byz.items()))
            x = N.sparse_solve(sub, tgt)
            if x is None:
                bad = n4phi.diagnose(sub, tgt)
                log("   INCONSISTENT: %d blocked sectors, %d rows"
                    % (len(bad), sum(len(b[0]) for b in bad)))
                log("   [%.1fs]" % (time.time() - t0))
                return
            dq = mzero()
            for j, c in enumerate(x):
                if c:
                    dq = qadd(dq, qsmul(c, ds[j]))
            q = qadd(q, dq)
            log("      step taken  [%.1fs]" % (time.time() - t0))
        else:
            log("   !! did not converge in 6 iterations")
            return

    F = resid_full(q)
    log("")
    log("=== THE CLOSING CHECK ===========================================")
    log("full residual of (IK4): %d nonzero rows" % len(F))
    if not F:
        log("")
        log("q solves (IK4) EXACTLY.")
        log("is_solution: %s" % is_solution(q))
        with open("n4two2-vacuum%s.pkl" % TAG, "wb") as fh:
            pickle.dump(q, fh)
        log("written to n4two2-vacuum%s.pkl" % TAG)
        nz = sorted({k[0] for i in (0, 1) for j in (0, 1) for k in q[i][j]})
        log("z-exponents present: %s" % nz)
    else:
        by, byz = {}, {}
        for k in F:
            by[step_of(k)] = by.get(step_of(k), 0) + 1
            byz[k[2][0]] = byz.get(k[2][0], 0) + 1
        log("remaining by step: %s" % sorted(by.items()))
        log("remaining by n_z:  %s" % sorted(byz.items()))
    log("=================================================================")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
