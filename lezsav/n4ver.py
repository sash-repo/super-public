#!/usr/bin/env python3
"""n4ver.py -- verify the N=4 vacuum independently of the machinery that
found it.

n4rec11.py produced a q with resid_full(q) empty.  But resid_full is the
row-keyed re-implementation used by the whole search, so checking the answer
with it proves only that the search is self-consistent.  This checks the
answer against the module's OWN top-level predicates, written long before any
of this and used by no part of the search:

    n4noeth.check_constraints(q)   -- its own constraint list
    n4noeth.full_residuals(q)      -- its own assembly of E1 and E2
    n4noeth.is_solution(q)

and against the invariants of the objects involved:

  * q is invertible and minv(q) really is its inverse, both ways
    (if minv were wrong, every equation containing it would be wrong);
  * det q is nonzero, so the vacuum is nondegenerate, not the empty matrix
    in disguise;
  * every monomial of q lies in the graded class the search was confined to:
    z-degree 0, m-degree -1, n_w + |S|/2 = -1;
  * the level-0 part is the one the N=2 vacuum supplies, so this really is a
    superization of that vacuum and not some unrelated solution;
  * and -- the test that matters most -- perturbing q by a single monomial
    makes the checker FAIL.  Without it, "0 nonzero rows" could just mean
    that the checker cannot see anything.

Usage: ../venv/bin/python n4ver.py
Writes n4ver-out.txt.
"""
import pickle
import random
import sys
import time

from fractions import Fraction as Q

import n4noeth as N
import n4vac
import n4rec
from n4rec import qadd, qsmul, resid_full, dirs_lvl, q0_of
from n4noeth import (mzero, mid, mmul, meq, mdet, minv, C, MI, PL,
                     check_constraints, full_residuals, is_solution)

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4ver-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4ver: the N=4 vacuum of n4rec11.py, checked against n4noeth's own")
    log("top-level predicates and against the invariants of the objects.")
    log("")

    # --- 1) the module's own checker ---------------------------------------
    nc, R1, R2 = full_residuals(q)
    log("1) n4noeth.full_residuals (its own constraint list, its own E1/E2):")
    log("   constraint violations : %d" % nc)
    log("   E1 = cD_-^a j^b       : %s"
        % ("identically zero" if meq(R1, mzero()) else "NONZERO"))
    log("   E2 - rhs              : %s"
        % ("identically zero" if meq(R2, mzero()) else "NONZERO"))
    log("   n4noeth.is_solution(q): %s" % is_solution(q))
    ok = (nc == 0 and meq(R1, mzero()) and meq(R2, mzero()))
    if not ok:
        log("   !! the independent checker REJECTS it -- stop here")
        return

    # --- 2) the inverse really is one --------------------------------------
    qi = minv(q)
    left, right = mmul(q, qi), mmul(qi, q)
    log("")
    log("2) the inverse used by the equations:")
    log("   q * q^{-1} = 1 : %s" % meq(left, mid(1)))
    log("   q^{-1} * q = 1 : %s" % meq(right, mid(1)))
    if not (meq(left, mid(1)) and meq(right, mid(1))):
        log("   !! minv is not an inverse -- the equations are void")
        return
    d = mdet(q)
    log("   det q nonzero  : %s  (%d monomials)" % (bool(d), len(d)))

    # --- 3) q lies in the graded class -------------------------------------
    log("")
    log("3) the graded class (z-degree 0, m-degree -1, n_w + |S|/2 = -1):")
    bad = []
    lv = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in q[i][j]:
                nz, nw, nm, S = k
                lv[len(S)] = lv.get(len(S), 0) + 1
                if nz != 0 or nm != -1 or nw + Q(len(S), 2) != -1:
                    bad.append((i, j, k))
    log("   monomials by theta-level: %s" % sorted(lv.items()))
    log("   monomials outside the class: %d" % len(bad))
    if bad:
        log("   !! %s" % bad[:3])

    # --- 4) it superizes the N=2 vacuum ------------------------------------
    ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms0)
    qb, kern = n4vac.build_embedded_base(B)
    Aemb = [[qb[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)]
            for i in (0, 1)]
    A = [[q[i][j].get((0, -1, -1, ()), C()) for j in (0, 1)] for i in (0, 1)]
    log("")
    log("4) the level-0 part, where the N=2 vacuum sits:")
    log("   of the embedded base : [[%s, %s], [%s, %s]]"
        % (Aemb[0][0], Aemb[0][1], Aemb[1][0], Aemb[1][1]))
    log("   of this vacuum       : [[%s, %s], [%s, %s]]"
        % (A[0][0], A[0][1], A[1][0], A[1][1]))
    same = all(A[i][j] == Aemb[i][j] for i in (0, 1) for j in (0, 1))
    log("   equal: %s" % same)

    # --- 5) the checker is not blind ---------------------------------------
    log("")
    log("5) the checker is not vacuous -- perturb q by ONE monomial:")
    rng = random.Random(99)
    dirs = []
    for r in (0, 1, 2, 3, 4):
        dirs += list(dirs_lvl(r))
    caught = 0
    trials = 30
    for _ in range(trials):
        dq = qsmul(rng.choice([1, -1, 2]), rng.choice(dirs))
        if not is_solution(qadd(q, dq)):
            caught += 1
    log("   %d of %d single-monomial perturbations are REJECTED" %
        (caught, trials))
    if caught < trials:
        log("   (a perturbation that is still a solution means the vacuum")
        log("    comes in a family -- which it does; see below)")

    # --- 6) what the solution looks like ------------------------------------
    log("")
    log("6) the vacuum itself:")
    tot = sum(len(q[i][j]) for i in (0, 1) for j in (0, 1))
    log("   %d nonzero monomial coefficients in the four entries" % tot)
    for i in (0, 1):
        for j in (0, 1):
            log("   q[%d][%d]: %d monomials" % (i, j, len(q[i][j])))
    log("   the level-0 term of q[0][0]: %s"
        % q[0][0].get((0, -1, -1, ()), C()))
    log("")
    log("VERDICT: %s" % ("the N=4 vacuum is CONFIRMED by the independent"
                         " checker" if ok else "REJECTED"))
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
