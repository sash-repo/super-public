#!/usr/bin/env python3
"""n4cons.py -- the Noether step at N=4: the conserved completion of T = -u.

The vacuum of the graded class was constructed in n4rec11.py and verified
independently in n4ver.py; n4fam.py showed it is not isolated.  This script
does with it what n2noeth-cons.py does at N=2: solve

        dop( T + sum_i x_i m^{p_i} S_i ) = 0

ACROSS a family of exact solutions, where T = -u = (1/2) log det q (its
derivatives taken as (1/2) tr(q^{-1} dop q), so no logarithm is formed) and
the S_i are the scalar composites of the two currents of (IK4) listed in
n4noeth.candidate_shapes.

Solving across a family, rather than at one solution, is what makes the
answer a statement about the SYSTEM: a combination conserved at a single
point of the solution variety is an accident of that point.

The family is built from the vacuum in two ways:
  1) steps along the tangent line (n4fam: 1-dimensional over Q(i)), each
     member re-checked with n4noeth.is_solution;
  2) odd-parameter members q + eps*d with eps an even nilpotent monomial
     (a product of two odd coordinates), eps^2 = 0 -- the device that
     generated the N=2 family in n2noeth-fam.py.

Every member is verified to be an exact solution before it is used.

Finally the coefficients are found on a TRAINING subset and tested on
members held out of the solve: the holdout is what distinguishes a genuine
conservation law from a linear system that has more unknowns than content.

Usage: ../venv/bin/python n4cons.py
Writes n4cons-out.txt.
"""
import itertools
import pickle
import sys
import time
import multiprocessing as mp

from fractions import Fraction as Q

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import n4noeth as N                                          # noqa: E402
import n4rec                                                 # noqa: E402
from n4noeth import (C, mzero, is_solution, minv, mono, mul, add, smul,
                     eq, d_xm, d_xp, dT, candidate_shapes, cnull, csolve,
                     MI, PL)                                 # noqa: E402
from n4rec import qadd, qsmul, dirs_lvl                      # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4cons-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


# ---------------------------------------------------------------------------
# the tangent direction (cached: n4fam.py computes the same thing)
# ---------------------------------------------------------------------------

_G = {}


def _init(qp, dp):
    _G["q"] = pickle.loads(qp)
    _G["dirs"] = pickle.loads(dp)


def _col(i):
    from n4rec import resid_full
    q, d = _G["q"], _G["dirs"][i]
    V = {}
    for t in range(-4, 5):
        V[t] = resid_full(q if t == 0 else qadd(q, qsmul(t, d)))
    keys = set()
    for t in V:
        keys |= set(V[t])
    c1 = {}
    for k in keys:
        a = {t: V[t].get(k, C()) for t in V}
        d1, d2 = a[1] - a[-1], a[2] - a[-2]
        d3, d4 = a[3] - a[-3], a[4] - a[-4]
        v9 = (d1 * Q(4, 5) + d2 * Q(-1, 5) + d3 * Q(4, 105)
              + d4 * Q(-1, 280))
        if v9:
            c1[k] = v9
    return i, c1


def tangent(q):
    """The tangent space to the solution variety at q, as matrices dq."""
    try:
        with open("n4cons-tan.pkl", "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        pass
    t0 = time.time()
    dirs = []
    for r in (1, 2, 3, 4):
        dirs += list(dirs_lvl(r))
    qp, dp = pickle.dumps(q), pickle.dumps(dirs)
    cols = [None] * len(dirs)
    with mp.Pool(max(1, mp.cpu_count() - 2), _init, (qp, dp)) as pool:
        for i, c1 in pool.imap_unordered(_col, range(len(dirs)), chunksize=4):
            cols[i] = c1
    rows = sorted({k for c in cols for k in c})
    A = [[cols[j].get(k, C()) for j in range(len(dirs))] for k in rows]
    ns = cnull(A)
    out = []
    for vec in ns:
        d = mzero()
        for j, c in enumerate(vec):
            if c:
                d = qadd(d, qsmul(c, dirs[j]))
        out.append(d)
    log("   tangent space: %d dimensions over Q(i)  [%.1fs]"
        % (len(out), time.time() - t0))
    with open("n4cons-tan.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------

def build_family(q, tans, want=9):
    fam = [("vacuum", q)]
    lams = [1, -1, Q(1, 2), 2, 3, Q(-1, 3), C(0, 1), C(1, 1), Q(2, 5)]
    for d in tans:
        for lam in lams:
            if len(fam) >= want:
                break
            qq = qadd(q, qsmul(lam, d))
            if is_solution(qq):
                fam.append(("q + (%s) d" % lam, qq))
            else:
                log("   !! step %s is NOT an exact solution -- dropped" % lam)
    # odd-parameter members: eps even nilpotent, eps^2 = 0
    odds = list(itertools.combinations(MI + PL, 2))
    for d in tans:
        for S in odds:
            if len(fam) >= want + 6:
                break
            eps = mono(S=S)
            de = mzero()
            for i in (0, 1):
                for j in (0, 1):
                    de[i][j] = mul(eps, d[i][j])
            if not any(de[i][j] for i in (0, 1) for j in (0, 1)):
                continue
            qq = qadd(q, de)
            if is_solution(qq):
                fam.append(("q + eps%s d" % (S,), qq))
    return fam


# ---------------------------------------------------------------------------
# the conservation solve
# ---------------------------------------------------------------------------

def block(q, dop):
    """(target, {name: column}) for one solution, as ring elements."""
    shapes, qi = candidate_shapes(q)
    target = dT(q, qi, dop)
    cols = {}
    for nm, s in shapes.items():
        for p in N.CAND_MPOWS:
            sp = mul(mono(nm=p), s) if p else s
            cols[(nm, p)] = dop(sp)
    return target, cols


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


def run(dop, dopname, fam, blocks):
    log("")
    log("=" * 70)
    log("dop = %s" % dopname)
    log("=" * 70)
    names = sorted(blocks[0][1].keys())
    log("   candidates: %d shapes x %s powers of m = %d columns"
        % (len(set(nm for nm, _ in names)), list(N.CAND_MPOWS), len(names)))

    # is the problem non-trivial at all?
    nz = sum(1 for tgt, _ in blocks if not eq(tgt, {}))
    log("   dop(T) is nonzero on %d of %d family members" % (nz, len(blocks)))
    if not nz:
        log("   => T alone is already conserved; nothing to complete.")

    # --- the full solve ----------------------------------------------------
    cols, tgt = stack(blocks, names)
    log("   stacked system: %d rows, %d unknowns" % (len(tgt), len(cols)))
    x = csolve(cols, tgt)
    if x is None:
        log("   NO conserved completion inside this list of candidates.")
        return None, names
    nzc = {nm: x[j] for j, nm in enumerate(names) if x[j]}
    log("   SOLVED.  nonzero coefficients (%d):" % len(nzc))
    for nm, v in sorted(nzc.items(), key=lambda t: str(t[0])):
        log("      %-28s  %s" % ("%s * m^%d" % (nm[0], nm[1]), v))

    # --- how much freedom is there? ---------------------------------------
    A = [[cols[j][r] for j in range(len(cols))] for r in range(len(tgt))]
    ker = cnull(A)
    log("   homogeneous kernel (combinations conserved by themselves): %d"
        % len(ker))
    for v in ker:
        nzk = {names[j]: v[j] for j in range(len(names)) if v[j]}
        log("      %s" % {("%s*m^%d" % (nm[0], nm[1])): str(c)
                          for nm, c in nzk.items()})
    return x, names


def holdout(dop, dopname, fam, blocks, ntrain):
    log("")
    log("   holdout: solve on %d members, test on the other %d"
        % (ntrain, len(blocks) - ntrain))
    names = sorted(blocks[0][1].keys())
    cols, tgt = stack(blocks[:ntrain], names)
    x = csolve(cols, tgt)
    if x is None:
        log("      the training subset alone has no solution -- skipped")
        return
    bad = 0
    for i in range(ntrain, len(blocks)):
        target, bc = blocks[i]
        tot = dict(target)
        for j, nm in enumerate(names):
            if x[j]:
                tot = add(tot, smul(x[j], bc.get(nm, {})))
        if not eq(tot, {}):
            bad += 1
            log("      member %d (%s): residual NONZERO (%d terms)"
                % (i, fam[i][0], len(tot)))
    log("      %d of %d held-out members satisfy the law found on the "
        "training set" % (len(blocks) - ntrain - bad, len(blocks) - ntrain))


def main():
    t0 = time.time()
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)
    log("n4cons: the conserved completion of T = -u at N=4.")
    log("")
    log("the vacuum is an exact solution: %s" % is_solution(q))
    tans = tangent(q)
    fam = build_family(q, tans)
    log("")
    log("family of exact solutions: %d members" % len(fam))
    for nm, _ in fam:
        log("   %s" % nm)

    for dop, dopname in ((d_xm, "d/dx^-"), (d_xp, "d/dx^+")):
        blocks = []
        for nm, qq in fam:
            blocks.append(block(qq, dop))
            log("   [%.1fs] block built: %s" % (time.time() - t0, nm))
        x, names = run(dop, dopname, fam, blocks)
        if x is not None:
            holdout(dop, dopname, fam, blocks, max(2, len(blocks) - 3))

    log("")
    log("[%.1fs] done" % (time.time() - t0))


if __name__ == "__main__":
    main()
