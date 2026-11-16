#!/usr/bin/env python3
"""n4boost.py -- the vacuum deformed in the z-direction, exactly.

The mirror of n4mirror.py gives z-dependent solutions, but each of them is
still a function of ONE light-cone variable: on the vacuum family d_+ = d_-,
on the mirror family d_+ = -d_-.  This engine produces solutions that depend
on both, by the BOOST

    x^+ -> lam x^+,   x^- -> lam^{-1} x^-,   theta_+ -> lam^{1/2} theta_+,
    theta_- -> lam^{-1/2} theta_-,

which leaves d_{x^+} d_{x^-} and the mass alone.  For a generic lam the
substitution takes w^{-k} out of the Laurent ring; for

    lam = 1 + t,    t = eps_9 eps_10   (an even NILPOTENT constant, t^2 = 0)

it terminates after one term and stays inside it:

    w -> w + t z,    z -> z + t w,    theta_+ -> (1 + t/2) theta_+,
    theta_- -> (1 - t/2) theta_-.

The image of the vacuum then depends on z and w both, to first order in t --
which is the deformation the text calls for at the end of the Subsection on
the vacuum.

Nothing is assumed: the boosted solutions are put through
n4noeth.is_solution, and the completion is then re-solved on the enlarged
family.

Usage: ../venv/bin/python n4boost.py   ->  n4boost-out.txt
"""
import pickle
import sys
import time

from fractions import Fraction as Q_

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import (C, mono, add, smul, eq, minv, dT, csolve, cnull,
                     d_xm, d_xp, is_solution, mzero, MI, PL,
                     full_residuals, sort_odd)                  # noqa: E402
from n4cons7 import logcols                                     # noqa: E402
from n4cons8 import composites                                  # noqa: E402
from n4rec import qadd, qsmul                                   # noqa: E402
from n4emb import cx, I                                         # noqa: E402
import n4mirror as MIR                                          # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4boost-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


T_LABELS = (9, 10)          # t = eps_9 eps_10, an even nilpotent constant


def boost_ring(f, half=Q_(1, 2)):
    """z^a w^b theta_S  ->  (z + t w)^a (w + t z)^b lam^{...} theta_S,
    with t = eps_9 eps_10 and t^2 = 0, so each factor contributes one term:

        d/dt [(z + t w)^a (w + t z)^b] at t = 0
              = a z^{a-1} w^{b+1} + b z^{a+1} w^{b-1},

    and the odd rescaling theta_+ -> (1 + half.t) theta_+,
    theta_- -> (1 - half.t) theta_- adds half.(|S n PL| - |S n MI|)."""
    out = {}
    for (a, b, c, S), v in f.items():
        if any(j in S for j in T_LABELS):
            raise ValueError("the boost parameter already occurs in f")
        out[(a, b, c, S)] = out.get((a, b, c, S), C()) + v
        pieces = []
        if a:
            pieces.append(((a - 1, b + 1, c, S), Q_(a)))
        if b:
            pieces.append(((a + 1, b - 1, c, S), Q_(b)))
        nod = (sum(1 for j in S if j in PL) - sum(1 for j in S if j in MI))
        if nod and half:
            pieces.append(((a, b, c, S), half * nod))
        for k0, cc in pieces:
            S2, sgn = sort_odd(tuple(k0[3]) + T_LABELS)
            if sgn == 0:
                continue
            k = (k0[0], k0[1], k0[2], S2)
            out[k] = out.get(k, C()) + v * cc * sgn
    return {k: v for k, v in out.items() if v}


def boost(q, half=Q_(1, 2), qscale=0):
    """The boosted solution; qscale is the coefficient of t in q -> lam^s q,
    which is the one freedom E2 can still ask for."""
    out = mzero()
    for g in (0, 1):
        for b in (0, 1):
            f = boost_ring(q[g][b], half)
            if qscale:
                for (a, bw, c, S), v in list(q[g][b].items()):
                    S2, sgn = sort_odd(tuple(S) + T_LABELS)
                    if sgn == 0:
                        continue
                    k = (a, bw, c, S2)
                    f[k] = f.get(k, C()) + v * Q_(qscale) * sgn
            out[g][b] = {k: v for k, v in f.items() if v}
    return out


def blocks_for(fam, dop, half="+"):
    out = []
    for _, qq in fam:
        fs, qqi = composites(qq, half)
        cols = logcols(fs, dop)
        cols[("log", "w")] = MIR.logw_col(dop)
        cols[("log", "z")] = MIR.logz_col(dop)
        out.append((dT(qq, qqi, dop), cols))
    return out


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    with open("n4cons-tan.pkl", "rb") as fh:
        tans = pickle.load(fh)

    log("n4boost: the vacuum deformed in the z-direction by a nilpotent boost.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))

    qb = boost(q)
    zs, ws = MIR.zw_support(qb)
    log("the boosted vacuum: z-exponents %s, w-exponents %s" % (zs, ws))
    nc, R1, R2 = full_residuals(qb)
    from n4noeth import meq
    log("   constraints violated %d, E1 %s, E2 %s"
        % (nc, "0" if meq(R1, mzero()) else "NONZERO",
           "0" if meq(R2, mzero()) else "NONZERO"))
    log("   is an exact solution of (IK4): %s" % is_solution(qb))
    if not is_solution(qb):
        log("")
        log("   the plain boost is not a symmetry; stopping here rather than")
        log("   guessing at a compensating scaling.")
        return

    fam = [("vacuum", q)]
    for d in tans:
        for lam in (1, -1, Q_(1, 2), 2, 3, C(0, 1), C(1, 1), Q_(-1, 3)):
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq):
                fam.append(("q + (%s) d" % lam, qq))
    log("")
    log("the family: %d solutions" % len(fam))
    bo = [("boost(%s)" % nm, boost(qq)) for nm, qq in fam]
    good = sum(1 for _, bb in bo if is_solution(bb))
    log("their boosts: %d of %d are exact solutions" % (good, len(bo)))
    mir = [("sigma(%s)" % nm, MIR.sigma(qq)) for nm, qq in fam]

    for dop, dopname in ((d_xm, "d_-"), (d_xp, "d_+")):
        log("")
        log("=" * 70)
        log("the completion on family + mirrors + boosts;  dop = %s" % dopname)
        log("=" * 70)
        allf = fam + mir + bo
        blocks = blocks_for(allf, dop)
        names = sorted(set().union(*[set(c) for _, c in blocks]), key=str)
        log("   %d solutions, %d log columns" % (len(allf), len(names)))
        cl, tg = MIR.stack(blocks, names)
        x = csolve(cl, tg)
        log("   %d rows: %s  [%.1fs]"
            % (len(tg), "SOLVED" if x is not None else "INCONSISTENT",
               time.time() - t0))
        if x is None:
            continue
        nz = [(names[j], x[j]) for j in range(len(names)) if x[j]]
        for n_, v in sorted(nz, key=lambda t: str(t[0])):
            log("      log %-40s %s" % (str(n_), v))
        A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
        k = len(cnull(A))
        log("      freedom: %d real = %d complex directions" % (k, k // 2))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
