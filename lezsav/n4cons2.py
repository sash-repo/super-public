#!/usr/bin/env python3
"""n4cons2.py -- the Noether step at N=4 with a SYSTEMATIC candidate list.

n4cons.py solved

        dop( T + sum_i x_i m^{p} S_i ) = 0

across the family with the fourteen hand-picked composites of
n4noeth.candidate_shapes, and found none.  A negative answer is only as good
as the list, so this rebuilds the list from the shape of the N=2 answer
instead of from guesses.

At N=2 the stress tensor is (item 2) of Theorem 5.3.1)

        T = d_t phi - (i/8) (cD v)(bD v^+),

i.e. the derivative of the o(2)-PARTNER of the dilaton plus a product of two
odd currents of the same half.  So the N=4 list is built from the two
ingredients that formula uses, and from nothing else:

    the first-order scalars   s(M, D) := tr( M q^{-1} D q ),
    M running over 1 and the three Pauli matrices -- i.e. over the dilaton
    direction and the three su(2)-partner directions -- and D over the two
    even derivatives d_+, d_- and the eight odd ones;

    the candidates: the even s(M, d_pm), all products of two odd s's, and
    m^p times each, p = -1, 0, 1.

The Pauli matrices are what makes "the partner directions" mean something:
tr(q^{-1} d q) is 2 dT, and tr(sigma_k q^{-1} d q) are the three d phi^k.

Strategy: solve on ONE member first -- fewer equations, so if there is no
solution there, there is none across the family and the list is refuted at
once.  Only if there is one does the family, and then the holdout, come in.

Usage: ../venv/bin/python n4cons2.py
Writes n4cons2-out.txt.
"""
import itertools
import pickle
import sys
import time

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import n4noeth as N                                          # noqa: E402
from n4noeth import (C, mzero, is_solution, minv, mono, mul, add, smul,
                     eq, d_xm, d_xp, dT, cnull, csolve,
                     cDm, bDm, cDp, bDp, mmul, mtr, mapply)  # noqa: E402
from n4rec import qadd, qsmul                                # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons2-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# --- the four directions in the doublet index ------------------------------
# 1 = the dilaton direction (its scalar is 2 dT), sigma_1,2,3 = the three
# partner directions phi^k of (IK4q).

SIG = {
    "1": [[mono(c=1), {}], [{}, mono(c=1)]],
    "s1": [[{}, mono(c=1)], [mono(c=1), {}]],
    "s2": [[{}, mono(c=C(0, -1))], [mono(c=C(0, 1)), {}]],
    "s3": [[mono(c=1), {}], [{}, mono(c=-1)]],
}

ODD_DERIVS = [
    ("cD-1", lambda f: cDm(1, f)), ("cD-2", lambda f: cDm(2, f)),
    ("bD-1", lambda f: bDm(1, f)), ("bD-2", lambda f: bDm(2, f)),
    ("cD+1", lambda f: cDp(1, f)), ("cD+2", lambda f: cDp(2, f)),
    ("bD+1", lambda f: bDp(1, f)), ("bD+2", lambda f: bDp(2, f)),
]
EVEN_DERIVS = [("d+", d_xp), ("d-", d_xm)]

MPOWS = (-1, 0, 1)


def scalars(q):
    """s(M, D) = tr(M q^{-1} D q) for the four M and the ten D."""
    qi = minv(q)
    ev, od = {}, {}
    for dn, dop in EVEN_DERIVS:
        Dq = mapply(dop, q)
        for mn, M in SIG.items():
            ev[(mn, dn)] = mtr(mmul(M, mmul(qi, Dq)))
    for dn, dop in ODD_DERIVS:
        Dq = mapply(dop, q)
        for mn, M in SIG.items():
            f = mtr(mmul(M, mmul(qi, Dq)))
            if f:
                od[(mn, dn)] = f
    return ev, od


def candidates(q):
    """name -> ring element."""
    ev, od = scalars(q)
    cand = {}
    for k, f in ev.items():
        if f:
            cand[("s", k)] = f
    okeys = sorted(od)
    for i in range(len(okeys)):
        for j in range(i, len(okeys)):
            f = mul(od[okeys[i]], od[okeys[j]])
            if f:
                cand[("p", okeys[i], okeys[j])] = f
    # the fourteen hand-picked composites of n4noeth, kept in the list
    shapes, _ = N.candidate_shapes(q)
    for nm, f in shapes.items():
        if f:
            cand[("h", nm)] = f
    out = {}
    for nm, f in cand.items():
        for p in MPOWS:
            out[nm + (p,)] = mul(mono(nm=p), f) if p else f
    return out


def block(q, dop, names=None):
    qi = minv(q)
    target = dT(q, qi, dop)
    cand = candidates(q)
    if names is None:
        names = sorted(cand, key=str)
    cols = {nm: dop(cand[nm]) for nm in names if nm in cand}
    return target, cols, names


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


def report(x, names, tag):
    nz = {names[j]: x[j] for j in range(len(names)) if x[j]}
    log("   %s: SOLVED, %d nonzero coefficients" % (tag, len(nz)))
    for nm, v in sorted(nz.items(), key=str):
        log("      %-40s %s" % (str(nm), v))


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4cons2: the conserved completion of T = -u at N=4,")
    log("with the candidate list rebuilt from the shape of the N=2 answer.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))

    ev, od = scalars(q)
    log("first-order scalars at the vacuum: %d even, %d odd (nonzero)"
        % (sum(1 for f in ev.values() if f), len(od)))
    log("   even:  %s" % sorted(k for k, f in ev.items() if f))
    log("   odd:   %s" % sorted(od))

    cand = candidates(q)
    log("candidates: %d (with the powers %s of m)" % (len(cand), list(MPOWS)))

    for dop, dopname in ((d_xm, "d/dx^-"), (d_xp, "d/dx^+")):
        log("")
        log("=" * 70)
        log("dop = %s   -- solving at the vacuum alone first" % dopname)
        log("=" * 70)
        target, cols, names = block(q, dop)
        log("   dop(T) has %d terms" % len(target))
        cl, tg = stack([(target, cols)], names)
        log("   system: %d rows, %d unknowns  [%.1fs]"
            % (len(tg), len(cl), time.time() - t0))
        x = csolve(cl, tg)
        if x is None:
            log("   NO solution even at the single point: the list is "
                "refuted for %s." % dopname)
            continue
        report(x, names, "at the vacuum")
        A = [[cl[j][r] for j in range(len(cl))] for r in range(len(tg))]
        ker = cnull(A)
        log("   kernel at the vacuum: %d" % len(ker))
        log("   [%.1fs]" % (time.time() - t0))

        # --- now across the family ----------------------------------------
        log("")
        log("   testing the same list across the family")
        with open("n4cons-tan.pkl", "rb") as fh:
            tans = pickle.load(fh)
        fam = [("vacuum", q)]
        for d in tans:
            for lam in (1, -1, Q(1, 2), 2, C(0, 1)):
                qq = qadd(q, qsmul(lam, d))
                if is_solution(qq):
                    fam.append(("q + (%s) d" % lam, qq))
        log("   family: %d exact solutions" % len(fam))
        blocks = []
        for nm, qq in fam:
            tt, cc, _ = block(qq, dop, names)
            blocks.append((tt, cc))
        cl, tg = stack(blocks, names)
        log("   system: %d rows, %d unknowns  [%.1fs]"
            % (len(tg), len(cl), time.time() - t0))
        x = csolve(cl, tg)
        if x is None:
            log("   the law holds at the vacuum but NOT across the family:")
            log("   no completion in this list is conserved on the whole "
                "solution variety.")
            continue
        report(x, names, "across the family")
        # --- holdout ------------------------------------------------------
        ntr = max(2, len(blocks) - 3)
        cl2, tg2 = stack(blocks[:ntr], names)
        xt = csolve(cl2, tg2)
        bad = 0
        for i in range(ntr, len(blocks)):
            tt, bc = blocks[i]
            tot = dict(tt)
            for j, nm in enumerate(names):
                if xt[j]:
                    tot = add(tot, smul(xt[j], bc.get(nm, {})))
            if not eq(tot, {}):
                bad += 1
        log("   holdout: trained on %d, %d of %d held-out members satisfy it"
            % (ntr, len(blocks) - ntr - bad, len(blocks) - ntr))
        log("   [%.1fs]" % (time.time() - t0))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
