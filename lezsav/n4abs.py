#!/usr/bin/env python3
"""n4abs.py -- the theta-recursion for a two-variable leading monomial, with
an ABSOLUTE z-window and only the levels that can matter.

Two lessons are built in.

 1) THE WINDOW MUST BE ABSOLUTE.  n4two3.py runs step 0 over the leading
    monomials z^{a_0} w^{-1-a_0} with the unknowns confined to z^{a_0+s},
    s in [-2,2], and finds it solvable for |a_0| <= 2 and inconsistent
    beyond -- but the inconsistent cases are exactly the ones whose rows
    reach n_z above a_0 + 2, i.e. the window, not the system, is what
    fails.  Widening it moves the boundary.  So the unknowns here range
    over an absolute n_z in [NLO, NHI], and the rows no window can reach
    are counted, not silently dropped.

 2) THE FREEDOM MUST BE CARRIED.  Solving one level and keeping a single
    solution of it reports obstructions that are not there; so at each step
    every earlier row is held at zero and the current one repaired over all
    the unknowns at once, and the step is iterated to convergence.

Only the theta-levels that can enter a step are used: the rows at levels 2r
(equations) and 2r+1 (constraints) involve q up to level 2r+2.

Usage: ../venv/bin/python n4abs.py [a0] [NLO] [NHI] [MAXSTEP]
Writes n4abs-out-a<a0>-n<NLO>_<NHI>.txt.
"""
import pickle
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4phi
import n4rec
import n4loop
from n4rec import qadd, qsmul, resid_full
from n4noeth import mzero, C, is_solution
from n4loop import dirs_abs, q0_of_loop, det_is_monomial, step_of, E

OUT = []
TAG = ""


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4abs-out%s.txt" % TAG, "w") as fh:
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
    nlo = int(sys.argv[2]) if len(sys.argv) > 2 else -3
    nhi = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    maxstep = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    procs = max(1, mp.cpu_count() - 2)
    LOOPS = {
        "mono": lambda a: E((0, 0, a, 2), (1, 1, a, 2)),
        "diag": lambda a: E((0, 0, a, 1), (1, 1, 0, 1)),
        "diag2": lambda a: E((0, 0, a, 1), (1, 1, -a, 1)),
        "uni": lambda a: E((0, 0, 0, 1), (0, 1, a, 1), (1, 1, 0, 1)),
    }
    which = sys.argv[5] if len(sys.argv) > 5 else "mono"
    q = q0_of_loop(LOOPS[which](a0))
    TAG = "-%s%d-n%d_%d" % (which, a0, nlo, nhi)
    ok, tf = det_is_monomial(q)
    log("n4abs: the recursion for the loop F = %s with a_0 = %d,"
        % (which, a0))
    log("i.e. m q^(0) = F(zeta)/w with zeta = z/w.")
    log("absolute z-window n_z in [%d, %d], steps 0..%d."
        % (nlo, nhi, maxstep))
    log("det q^(0) is a single monomial: %s" % ok)
    if not (which == "mono" and a0 in (0, -1)):
        log("the loop is not of the degenerate shape A (1 + mu zeta)^{-1},")
        log("so d_+ and d_- are independent on this leading term.")
    log("")

    for r in range(maxstep + 1):
        # only the levels that can enter step r
        ds, tg = [], []
        for lev in range(1, r + 2):
            d_, t_ = dirs_abs(lev, nlo, nhi)
            ds += d_
            tg += t_
        log("--- step %d: rows at theta-level %d and %d; %d unknowns"
            " (levels %s) ---"
            % (r, 2 * r, 2 * r + 1, len(ds),
               ", ".join(str(2 * l) for l in range(1, r + 2))))
        done = False
        for it in range(6):
            F = resid_full(q)
            earlier = {k for k in F if step_of(k) < r}
            if earlier:
                log("   !! %d earlier rows nonzero -- aborting" % len(earlier))
                return
            tgt = {k: -v for k, v in F.items() if step_of(k) == r}
            if not tgt:
                log("   satisfied  [%.1fs]" % (time.time() - t0))
                done = True
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
            byz = {}
            for k in tgt:
                byz[k[2][0]] = byz.get(k[2][0], 0) + 1
            log("   it %d: %d rows to repair at n_z = %s; %d unreachable"
                % (it, len(tgt), sorted(byz.items()), len(unreach)))
            x = N.sparse_solve(sub, tgt)
            if x is None:
                bad = n4phi.diagnose(sub, tgt)
                log("   INCONSISTENT: %d blocked sectors, %d rows"
                    % (len(bad), sum(len(b[0]) for b in bad)))
                # where does the inconsistency sit?
                log("   [%.1fs]" % (time.time() - t0))
                return
            dq = mzero()
            for j, c in enumerate(x):
                if c:
                    dq = qadd(dq, qsmul(c, ds[j]))
            q = qadd(q, dq)
            log("      step taken  [%.1fs]" % (time.time() - t0))
        if not done:
            log("   !! did not converge")
            return
        log("")

    F = resid_full(q)
    log("=== THE CLOSING CHECK ===========================================")
    log("full residual: %d nonzero rows" % len(F))
    if not F:
        log("q solves (IK4) EXACTLY; is_solution: %s" % is_solution(q))
        with open("n4abs-sol%s.pkl" % TAG, "wb") as fh:
            pickle.dump(q, fh)
    else:
        by, byz = {}, {}
        for k in F:
            by[step_of(k)] = by.get(step_of(k), 0) + 1
            byz[k[2][0]] = byz.get(k[2][0], 0) + 1
        log("by step: %s" % sorted(by.items()))
        log("by n_z:  %s" % sorted(byz.items()))
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
