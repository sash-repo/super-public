#!/usr/bin/env python3
"""n4rec.py -- the N=4 vacuum by the THETA-RECURSION, not by Newton.

Every search so far linearized the whole system at a base and asked Newton to
close the gap; all of them stalled, and n4log.py has now closed the two ring
extensions that might have supplied the missing directions.  This takes the
other route: build the vacuum level by
level in the theta-filtration, which cannot be blocked by the shape of a
window because there is no window to choose -- the class is finite.

WHY IT IS FINITE (n4grade.py).  The residual at the embedded base is exactly
homogeneous: z^0, m^0, and nw + |S|/2 = -1 throughout; the base is
homogeneous of the same grade with m^{-1}.  So a vacuum of the shape the N=2
one has lives in

    q = sum_{0<=r<=4} q^(2r),   q^(2r) in w^{-1-r} m^{-1} Lambda^{2r}(theta),

512 coefficients, cut by the constraints to 140.

WHY IT RECURSES.  The odd derivatives move the theta-level by exactly +-1, so
the constraints at odd level 2r+1 and the equations at even level 2r involve
q only up to level 2r+2, and involve q^(2r+2) LINEARLY (machine-verified in
n4grade.py: every second difference vanishes).  So

    q^(0) = A/(mw)  -->  q^(2)  -->  q^(4)  -->  q^(6)  -->  q^(8),

each arrow one exact linear solve, and the equations at level 8 are the
closing conditions, with no unknowns left to meet them.

WHAT IS CONCLUSIVE.  Step 0 has no earlier choice in it, so its solvability
is a property of A alone: if the level-0 system is inconsistent for every
invertible A, then NO q of the graded class solves (IK4) -- not "Newton is
blocked at a base", but "there is nothing here to find".  Later steps carry
the kernel of the earlier ones, and the script reports those dimensions
rather than pretending a single branch settles them.

Usage:
    ../venv/bin/python n4rec.py            (the embedded A, then a scan)
    ../venv/bin/python n4rec.py --steps 4  (push the recursion further)
Writes n4rec-out.txt.
"""
import itertools
import random
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import n4noeth as N
import n4vac
import n4phi
from n4noeth import (mono, mzero, minv, add, sub, cDm, bDp, C, MI, PL,
                     lhs1, lhs2, rhs2, msub)
from n4emb import cx

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4rec-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- the graded window ------------------------------------------------------

def monos_lvl(r):
    return [mono(0, -1 - r, -1, S)
            for S in itertools.combinations(MI + PL, 2 * r)]


def dirs_lvl(r):
    out = []
    for g in (0, 1):
        for b in (0, 1):
            for mo in monos_lvl(r):
                dq = mzero()
                dq[g][b] = mo
                out.append(dq)
    return out


def qadd(a, b):
    out = mzero()
    for i in (0, 1):
        for j in (0, 1):
            out[i][j] = add(a[i][j], b[i][j])
    return out


def qsmul(c, a):
    out = mzero()
    for i in (0, 1):
        for j in (0, 1):
            out[i][j] = {k: cx(c) * v for k, v in a[i][j].items()}
            out[i][j] = {k: v for k, v in out[i][j].items() if v}
    return out


# --- the full residual: constraints and equations, keyed by level -----------

def constr_list(qq):
    L = []
    for g in (0, 1):
        u = (qq[g][0], qq[g][1])
        L += [cDm(1, u[0]), cDm(2, u[1]), add(cDm(1, u[1]), cDm(2, u[0]))]
    for b in (0, 1):
        v = (qq[0][b], qq[1][b])
        L += [bDp(1, v[0]), bDp(2, v[1]), add(bDp(1, v[1]), bDp(2, v[0]))]
    return L


def resid_full(qq):
    """{('C',slot,key): v} for the constraints, {('E',slot,key): v} for the
    two equations.  Keys carry the theta-level in key[3]."""
    out = {}
    for slot, f in enumerate(constr_list(qq)):
        for k, v in f.items():
            out[("C", slot, k)] = v
    qqi = minv(qq)
    R1 = lhs1(qq, qqi)
    R2 = msub(lhs2(qq, qqi), rhs2(qq))
    slot = 0
    for M in (R1, R2):
        for i in (0, 1):
            for j in (0, 1):
                for k, v in M[i][j].items():
                    out[("E", slot, k)] = v
                slot += 1
    return out


def rows_of_step(r):
    """Which rows the step q^(2r) -> q^(2r+2) must satisfy."""
    def keep(k):
        lv = len(k[2][3])
        return (lv == 2 * r + 1) if k[0] == "C" else (lv == 2 * r)
    return keep


# --- one recursion step -----------------------------------------------------

_G = {}


def _init(qp, dp):
    import pickle
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    return i, resid_full(qadd(_G["q"], _G["dirs"][i]))


def step(q_known, r, procs, verbose=True):
    """Solve the level-2r rows for q^(2r+2).  Returns (particular, kernel)
    or (None, None) if the level is inconsistent."""
    keep = rows_of_step(r)
    F0 = {k: v for k, v in resid_full(q_known).items() if keep(k)}
    ds = dirs_lvl(r + 1)
    import pickle
    qp, dp = pickle.dumps(q_known), pickle.dumps(ds)
    cols = [None] * len(ds)
    with mp.Pool(procs, _init, (qp, dp)) as pool:
        for i, Ri in pool.imap_unordered(_col, range(len(ds)), chunksize=4):
            Ci = {}
            for k, v in Ri.items():
                if keep(k):
                    d = v - F0.get(k, C())
                    if d:
                        Ci[k] = d
            for k in F0:
                if keep(k) and k not in Ri:
                    d = -F0[k]
                    if d:
                        Ci[k] = d
            cols[i] = Ci
    target = {k: -v for k, v in F0.items()}
    x = N.sparse_solve(cols, target)
    if verbose:
        log("   step %d->%d: %d unknowns, %d rows (%d nonzero at q_known)"
            % (2 * r, 2 * r + 2, len(ds), len(set(k for c in cols for k in c)
                                              | set(F0)), len(F0)))
    if x is None:
        if verbose:
            bad = n4phi.diagnose(cols, target)
            log("      INCONSISTENT: %d blocked sectors, %d rows"
                % (len(bad), sum(len(b[0]) for b in bad)))
        return None, None
    part = mzero()
    for i, c in enumerate(x):
        if c:
            part = qadd(part, qsmul(c, ds[i]))
    ker = N.sparse_null(cols)
    if verbose:
        log("      solvable; kernel dimension %d" % len(ker))
    return part, ker


# --- level-0 data -----------------------------------------------------------

def q0_of(A):
    q = mzero()
    for i in (0, 1):
        for j in (0, 1):
            if A[i][j]:
                q[i][j] = {(0, -1, -1, ()): cx(A[i][j])}
    return q


def det2(A):
    return cx(A[0][0]) * cx(A[1][1]) - cx(A[0][1]) * cx(A[1][0])


if __name__ == "__main__":
    t0 = time.time()
    procs = max(1, mp.cpu_count() - 8)
    STEPS = 4
    if "--steps" in sys.argv:
        STEPS = int(sys.argv[sys.argv.index("--steps") + 1])

    # the embedded base, for its level-0 datum
    ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms0)
    qb, kern = n4vac.build_embedded_base(B)
    Aemb = [[qb[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)]
            for i in (0, 1)]
    log("the embedded base has q^(0) = A/(mw) with")
    log("   A = [[%s, %s], [%s, %s]],  det A = %s"
        % (Aemb[0][0], Aemb[0][1], Aemb[1][0], Aemb[1][1], det2(Aemb)))
    log("[%.1fs]" % (time.time() - t0))

    # --- step 0 for the embedded A -----------------------------------------
    log("")
    log("=== the recursion from the embedded A ============================")
    q = q0_of(Aemb)
    ok = True
    kers = []
    for r in range(STEPS):
        part, ker = step(q, r, procs)
        if part is None:
            ok = False
            log("   -> the recursion DIES at level %d" % (2 * r))
            break
        kers.append(len(ker))
        q = qadd(q, part)
    if ok:
        log("   reached level %d with kernel dimensions %s"
            % (2 * STEPS, kers))
        F = resid_full(q)
        top = {k: v for k, v in F.items() if k[0] == "E"}
        log("   closing check: %d nonzero equation keys left" % len(top))
    log("[%.1fs]" % (time.time() - t0))

    # --- is step 0 solvable for ANY A? -------------------------------------
    log("")
    log("=== is step 0 solvable for other level-0 data? ===================")
    log("step 0 has no earlier choice in it, so its solvability is a")
    log("property of A alone.")
    random.seed(11)
    trials = []
    trials.append(("embedded", Aemb))
    trials.append(("identity", [[C(1), C()], [C(), C(1)]]))
    trials.append(("diag(1,-1)", [[C(1), C()], [C(), C(-1)]]))
    trials.append(("diag(1,i)", [[C(1), C()], [C(), C(0, 1)]]))
    trials.append(("antidiag", [[C(), C(1)], [C(1), C()]]))
    trials.append(("upper", [[C(1), C(1)], [C(), C(1)]]))
    for t in range(6):
        while True:
            A = [[C(random.randint(-3, 3), random.randint(-3, 3))
                  for _ in (0, 1)] for _ in (0, 1)]
            if det2(A):
                break
        trials.append(("random %d" % (t + 1), A))
    good = []
    for name, A in trials:
        part, ker = step(q0_of(A), 0, procs, verbose=False)
        st = ("solvable, kernel %d" % len(ker)) if part is not None \
            else "INCONSISTENT"
        log("   A = %-12s det = %-10s : %s" % (name, det2(A), st))
        if part is not None:
            good.append(name)
    log("")
    log("   level-0 step solvable for %d of %d level-0 data tried"
        % (len(good), len(trials)))
    log("[%.1fs] done" % (time.time() - t0))
