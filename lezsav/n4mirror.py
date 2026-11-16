#!/usr/bin/env python3
"""n4mirror.py -- the z-dependent solutions, and what they pin.

The text says, at the end of the Subsection on the N=4 vacuum, that the
completion of bold T = -u is not pinned by the vacuum and cannot be: a
vacuum is a function of w = x^+ + x^- alone, so on it d_+ = d_-, and the
conservation law of one half is indistinguishable from that of the other.
Pinning needs solutions that depend on z = x^+ - x^-, which the graded class

    q^(2r) in w^{-1-r} m^{-1} Lambda^{2r}(theta)

excludes by construction.  This engine produces them -- not by deforming,
which would have to leave the class, but by a SYMMETRY of (IK4) that carries
the class onto its z-mirror.

THE MIRROR.  Let sigma act on the ring by

    z <-> w        (i.e. x^+ -> x^+,  x^- -> -x^-),
    theta_- -> i theta_-   (the four odd labels of the minus half),

and let sigma(q) := i . sigma_ring(q).  Then, on every solution:

  * the six constraints hold identically (sigma_ring intertwines cD_- and
    bD_+ with themselves up to the factor -i on the minus half);
  * E1 holds identically -- d_{x^+} is sigma-invariant, so cD_+, bD_+ are;
  * E2 holds after the scaling: c_b = (q^{-1} bD_{-g} q)_b^g is invariant
    under q -> nu q for a constant nu, while the right-hand side -4 i m qbar
    scales by nu; and the mirror sends lhs2 to i . rhs2 exactly, so nu = i
    closes it.

Each of these is verified here rather than assumed, on every member of the
family of n4fam.py -- the mirror image is put through n4noeth.is_solution.

WHAT IT PINS.  The completion found in n4cons8.py was fitted on 17 solutions
on all of which d_+ = d_-.  On the mirror solutions d_+ = -d_-, so the rows
they contribute are new.  Requiring ONE combination of logarithms, with one
set of coefficients, to complete bold T on the doubled family is therefore a
strictly stronger demand, and this run measures three things:

  1) does the completion of n4cons8.py survive on the mirror family;
  2) does a solution exist across the doubled family at all;
  3) by how much the freedom -- 14 complex directions among 18 columns at
     the vacuum -- drops.

Usage: ../venv/bin/python n4mirror.py   ->  n4mirror-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q_

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, mul, add, smul, eq, minv, dT, csolve, cnull,
                     d_xm, d_xp, is_solution, mmul, mtr, mzero, MI,
                     full_residuals, lhs2, rhs2, meq)           # noqa: E402
from n4cons7 import logcols                                     # noqa: E402
from n4cons8 import composites                                  # noqa: E402
from n4rec import qadd, qsmul                                   # noqa: E402
from n4emb import cx, I                                         # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4mirror-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


def cpow(c, n):
    out = cx(1)
    c = cx(c)
    for _ in range(n):
        out = out * c
    return out


def mirror_ring(f, lam=I):
    """z <-> w on the exponents, theta_- -> lam . theta_-."""
    out = {}
    for (a, b, c, S), v in f.items():
        k = (b, a, c, S)
        out[k] = out.get(k, C()) + v * cpow(lam, sum(1 for j in S if j in MI))
    return {k: v for k, v in out.items() if v}


def sigma(q, lam=I, nu=I):
    """The mirror of a solution: nu . (z <-> w, theta_- -> lam theta_-)."""
    out = mzero()
    for g in (0, 1):
        for b in (0, 1):
            out[g][b] = {k: cx(nu) * v
                         for k, v in mirror_ring(q[g][b], lam).items()}
    return out


def zw_support(q):
    zs, ws = set(), set()
    for g in (0, 1):
        for b in (0, 1):
            for (a, b_, c, S) in q[g][b]:
                zs.add(a)
                ws.add(b_)
    return sorted(zs), sorted(ws)


def logw_col(dop):
    """dop(log w) -- d_w log w = 1/w for both d_+ and d_-."""
    return mono(nw=-1)


def logz_col(dop):
    """dop(log z):  d_+ log z = 1/z,  d_- log z = -1/z."""
    return mono(nz=-1) if dop is d_xp else smul(-1, mono(nz=-1))


def blocks_for(fam, dop, half="+"):
    """(target, columns) per solution, with log w and log z adjoined."""
    out = []
    for _, qq in fam:
        fs, qqi = composites(qq, half)
        cols = logcols(fs, dop)
        cols[("log", "w")] = logw_col(dop)
        cols[("log", "z")] = logz_col(dop)
        out.append((dT(qq, qqi, dop), cols))
    return out


def stack(blocks, names):
    cols = {nm: [] for nm in names}
    tgt = []
    for target, bc in blocks:
        keys = sorted(set(target) | {k for f in bc.values() for k in f})
        for nm in names:
            f = bc.get(nm, {})
            cols[nm].extend([f.get(k, C()) for k in keys])
        tgt.extend([-target.get(k, C()) for k in keys])
    return [cols[nm] for nm in names], tgt


def residual_of(x, names, blocks):
    """How many of the blocks the coefficient vector x actually satisfies."""
    ok = 0
    for target, bc in blocks:
        tot = dict(target)
        for j, nm in enumerate(names):
            if x[j]:
                tot = add(tot, smul(x[j], bc.get(nm, {})))
        if eq(tot, {}):
            ok += 1
    return ok


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    with open("n4cons-tan.pkl", "rb") as fh:
        tans = pickle.load(fh)

    log("n4mirror: the z-dependent solutions of (IK4), and what they pin.")
    log("")

    # ---------------------------------------------------------------- (1)
    log("=" * 70)
    log("(1) the mirror is a symmetry of (IK4)")
    log("=" * 70)
    zs, ws = zw_support(q)
    log("   the vacuum:  z-exponents %s,  w-exponents %s" % (zs, ws))
    log("   it is an exact solution: %s" % is_solution(q))

    # the structure, step by step, on the vacuum itself
    m = sigma(q, I, 1)                       # unscaled mirror
    nc, R1, R2 = full_residuals(m)
    log("   the unscaled mirror: constraints violated %d, E1 residual %s"
        % (nc, "0" if meq(R1, mzero()) else "NONZERO"))
    L, R = lhs2(m, minv(m)), rhs2(m)
    mu, uniform = None, True
    for i in (0, 1):
        for j in (0, 1):
            for k in sorted(set(L[i][j]) | set(R[i][j])):
                a, b = L[i][j].get(k, C()), R[i][j].get(k, C())
                if b == C():
                    uniform = uniform and (a == C())
                    continue
                d = b.re * b.re + b.im * b.im
                r = a * C(b.re / d, -b.im / d)
                if mu is None:
                    mu = r
                elif r != mu:
                    uniform = False
    log("   and lhs2 = mu . rhs2 with a CONSTANT mu: %s, mu = %s"
        % (uniform, mu))
    log("   so the scaling q -> nu q, which leaves lhs2 and E1 and the")
    log("   constraints alone and scales rhs2, closes E2 at nu = mu = %s."
        % mu)

    # the family, and its mirror
    fam = [("vacuum", q)]
    for d in tans:
        for lam in (1, -1, Q_(1, 2), 2, 3, C(0, 1), C(1, 1), Q_(-1, 3)):
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq):
                fam.append(("q + (%s) d" % lam, qq))
    log("")
    log("   the family of n4fam.py: %d exact solutions" % len(fam))
    mir = [("sigma(%s)" % nm, sigma(qq)) for nm, qq in fam]
    good = sum(1 for _, mm in mir if is_solution(mm))
    log("   their mirrors: %d of %d are exact solutions of (IK4)"
        % (good, len(mir)))
    zs, ws = zw_support(mir[0][1])
    log("   the mirror vacuum: z-exponents %s,  w-exponents %s" % (zs, ws))
    log("   so on the family d_+ = +d_-, and on the mirror family d_+ = -d_-.")
    if good != len(mir):
        log("   MIRROR IS NOT A SYMMETRY -- stopping.")
        return
    mir = [(nm, mm) for nm, mm in mir]

    # sigma^2
    s2 = sigma(sigma(q))
    log("   sigma^2 = -1 on the vacuum: %s"
        % all(eq(s2[g][b], smul(-1, q[g][b])) for g in (0, 1) for b in (0, 1)))
    log("   [%.1fs]" % (time.time() - t0))

    # ---------------------------------------------------------------- (2)
    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        log("")
        log("=" * 70)
        log("(2) the completion across the DOUBLED family;  dop = %s"
            % dopname)
        log("=" * 70)
        bv = blocks_for(fam, dop)
        bm = blocks_for(mir, dop)
        names = sorted(set().union(*[set(c) for _, c in bv + bm]), key=str)
        log("   %d log columns (the plus half, plus log w and log z)"
            % len(names))

        # (a) does the old answer survive?
        cl, tg = stack(bv, names)
        xv = csolve(cl, tg)
        log("   on the vacuum family alone: %s"
            % ("SOLVED" if xv is not None else "INCONSISTENT"))
        if xv is not None:
            A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
            kv = len(cnull(A))
            log("      freedom: %d real = %d complex directions"
                % (kv, kv // 2))
            nz = [(names[j], xv[j]) for j in range(len(names)) if xv[j]]
            for n_, v in sorted(nz, key=lambda t: str(t[0])):
                log("      log %-40s %s" % (str(n_), v))
            ok = residual_of(xv, names, bm)
            log("      it satisfies %d of the %d MIRROR solutions"
                % (ok, len(bm)))

        # (b) the doubled solve
        cl, tg = stack(bv + bm, names)
        x = csolve(cl, tg)
        log("   across the doubled family (%d rows, %d unknowns): %s  [%.1fs]"
            % (len(tg), len(cl),
               "SOLVED" if x is not None else "INCONSISTENT",
               time.time() - t0))
        if x is None:
            log("      so NO single combination of these logarithms completes")
            log("      bold T on both halves at once.")
            continue
        nz = [(names[j], x[j]) for j in range(len(names)) if x[j]]
        log("      %d nonzero coefficients:" % len(nz))
        for n_, v in sorted(nz, key=lambda t: str(t[0])):
            log("      log %-40s %s" % (str(n_), v))
        A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
        k = len(cnull(A))
        log("      freedom across the doubled family: %d real = %d complex"
            % (k, k // 2))

        # (c) holdout across the doubled family
        blocks = bv + bm
        ntr = max(2, len(blocks) - 6)
        cl2, tg2 = stack(blocks[:ntr], names)
        xt = csolve(cl2, tg2)
        if xt is not None:
            ok = residual_of(xt, names, blocks[ntr:])
            log("      holdout: fitted on %d, %d of %d held-out obey"
                % (ntr, ok, len(blocks) - ntr))
        log("      [%.1fs]" % (time.time() - t0))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
