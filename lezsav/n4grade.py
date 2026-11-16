#!/usr/bin/env python3
"""n4grade.py -- the shape of the graded class, and the theta-recursion.

Two facts found on 30-31 August put the vacuum question into a FINITE form.

  (a) The residual at the embedded base is exactly homogeneous: every one of
      its 83 keys has z^0, m^0, and w-exponent + (#thetas)/2 = -1.  The base
      itself is homogeneous of the same grade with m^{-1}.  So the scaling
      grade  gr = nw + |S|/2  and the m-degree are conserved by the whole
      system, and a vacuum of the shape the N=2 one has -- graded,
      m-homogeneous, z-independent -- lives in a FINITE window:

          q = sum_{0<=r<=4} q^(2r),   q^(2r) in  w^{-1-r} m^{-1} Lambda^{2r},

      512 coefficients in all (4 slots x 2^{8-1} theta-monomials).

  (b) The odd derivatives move the theta-level by exactly +-1, so the
      equations are TRIANGULAR in the theta-filtration: the constraints at
      odd level 2r+1 and the equations at even level 2r involve q only up to
      level 2r+2, and involve q^(2r+2) LINEARLY.  Hence the recursion
      q^(0) -> q^(2) -> ... -> q^(8), each step a linear solve.

This script measures that structure before the recursion is run:
  1. the dimension of the graded window and of its constrained subspace,
     level by level;
  2. the level profile of the base and of the residual;
  3. the linearity check the recursion rests on -- that the residual at level
     2r really is affine in q^(2r+2) (second differences must vanish).

Exact over Q(i).  Usage: tools/venv/bin/python n4grade.py
Writes n4grade-out.txt.
"""
import itertools
import sys
import time

from fractions import Fraction as Q

import n4noeth as N
import n4vac
import n4phi
from n4noeth import (mono, mzero, minv, check_constraints, add, sub, smul,
                     C, MI, PL)

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4grade-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def graded_monos_lvl(r):
    """The grade -1, m^{-1}, z^0 monomials with exactly 2r thetas."""
    out = []
    for S in itertools.combinations(MI + PL, 2 * r):
        out.append(mono(0, -1 - r, -1, S))
    return out


def window_dirs(r):
    """Single-entry matrices over level 2r of the graded window."""
    out = []
    for g in (0, 1):
        for b in (0, 1):
            for mo in graded_monos_lvl(r):
                dq = mzero()
                dq[g][b] = mo
                out.append(dq)
    return out


def level_profile(f):
    prof = {}
    for (a, b, c, S) in f:
        prof[len(S)] = prof.get(len(S), 0) + 1
    return prof


if __name__ == "__main__":
    t0 = time.time()

    # --- 1) the graded window and its constrained subspace ------------------
    log("--- the graded window (gr = nw + |S|/2 = -1, m^{-1}, z^0) ---------")
    tot = 0
    for r in range(5):
        n = len(graded_monos_lvl(r))
        tot += 4 * n
        log("  level %d: %3d theta-monomials x 4 slots = %4d coefficients"
            % (2 * r, n, 4 * n))
    log("  total: %d coefficients" % tot)

    allms = []
    for r in range(5):
        allms += graded_monos_lvl(r)
    Bg = n4vac.constrained_basis(allms)
    log("")
    log("constrained subspace of the graded window: %d matrices" % len(Bg))
    prof = {}
    for M in Bg:
        lv = set()
        for i in (0, 1):
            for j in (0, 1):
                for k in M[i][j]:
                    lv.add(len(k[3]))
        prof[tuple(sorted(lv))] = prof.get(tuple(sorted(lv)), 0) + 1
    log("  theta-levels spanned by each basis matrix: %s"
        % sorted(prof.items()))
    log("[%.1fs]" % (time.time() - t0))

    # --- 2) the base and the residual, by level ----------------------------
    ms0 = n4vac.graded_monos(0, 0, -5, 1, mfree=(-1, -1))
    B = n4vac.constrained_basis(ms0)
    q, kern = n4vac.build_embedded_base(B)
    qi = minv(q)
    R = n4phi.resid_vec(q, qi)
    log("")
    log("--- the embedded base --------------------------------------------")
    log("constraint violations: %d" % check_constraints(q))
    for i in (0, 1):
        for j in (0, 1):
            log("  q[%d][%d] by theta-level: %s"
                % (i, j, sorted(level_profile(q[i][j]).items())))
    gr = set()
    for i in (0, 1):
        for j in (0, 1):
            for (a, b, c, S) in q[i][j]:
                gr.add((a, b + len(S) // 2, c))
    log("  (z, grade, m) of the base: %s" % sorted(gr))

    log("")
    log("--- the residual -------------------------------------------------")
    lev = {}
    for (slot, k) in R:
        lev.setdefault(len(k[3]), []).append(slot)
    for L in sorted(lev):
        log("  theta-level %d: %d keys, slots %s"
            % (L, len(lev[L]), sorted(set(lev[L]))))
    grR = set()
    for (slot, (a, b, c, S)) in R:
        grR.add((a, b + len(S) // 2, c))
    log("  (z, grade, m) of the residual: %s" % sorted(grR))
    log("  [slots 0-3 = E1 entries (00,01,10,11), 4-7 = E2 entries]")
    log("[%.1fs]" % (time.time() - t0))

    # --- 3) the linearity the recursion rests on ---------------------------
    log("")
    log("--- is the residual affine in the NEXT level? ---------------------")
    log("test: for directions u, v at level 2r+2, the second difference")
    log("   R(q+u+v) - R(q+u) - R(q+v) + R(q)")
    log("must vanish at theta-levels <= 2r (equations) and <= 2r+1")
    log("(constraints).")

    def resid_all(qq):
        qqi = minv(qq)
        r1 = n4phi.resid_vec(qq, qqi)
        cc = {}
        for slot, f in enumerate(n4vac_constr(qq)):
            for k, v in f.items():
                cc[("C", slot, k)] = v
        out = {("E", k): v for k, v in r1.items()}
        out.update(cc)
        return out

    def n4vac_constr(qq):
        from n4noeth import cDm, bDp
        L = []
        for g in (0, 1):
            u = (qq[g][0], qq[g][1])
            L += [cDm(1, u[0]), cDm(2, u[1]), add(cDm(1, u[1]), cDm(2, u[0]))]
        for b in (0, 1):
            v = (qq[0][b], qq[1][b])
            L += [bDp(1, v[0]), bDp(2, v[1]), add(bDp(1, v[1]), bDp(2, v[0]))]
        return L

    def qadd(a, b):
        out = mzero()
        for i in (0, 1):
            for j in (0, 1):
                out[i][j] = add(a[i][j], b[i][j])
        return out

    def keylevel(k):
        return len(k[-1][3]) if k[0] == "C" else len(k[1][1][3])

    import random
    random.seed(7)
    worst = 0
    for r in range(4):
        ds = window_dirs(r + 1)
        pairs = [(random.choice(ds), random.choice(ds)) for _ in range(3)]
        bad = 0
        for u, v in pairs:
            R0 = resid_all(q)
            Ru = resid_all(qadd(q, u))
            Rv = resid_all(qadd(q, v))
            Ruv = resid_all(qadd(qadd(q, u), v))
            keys = set(R0) | set(Ru) | set(Rv) | set(Ruv)
            for k in keys:
                d = (Ruv.get(k, C()) - Ru.get(k, C())
                     - Rv.get(k, C()) + R0.get(k, C()))
                if not d:
                    continue
                lv = keylevel(k)
                lim = (2 * r + 1) if k[0] == "C" else (2 * r)
                if lv <= lim:
                    bad += 1
        log("  level %d -> %d: %d second-difference violations at or below"
            " the imposed levels" % (2 * r, 2 * r + 2, bad))
        worst += bad
    log("  TOTAL violations: %d  %s"
        % (worst, "(the recursion is linear at each step)" if not worst
           else "(the recursion is NOT linear -- rethink)"))
    log("[%.1fs] done" % (time.time() - t0))
