#!/usr/bin/env python3
r"""
hw4d.py -- the definitive computation behind the answer to the question below.

Question:  Statement 7.3.22 gives (c, h; H_1, H_2) = (0, 0; 0, 0) for
Spin(cF_{lambda; mu}) at n = 4.  Does that make the module trivial?  Is it
irreducible?  Is the trivial module its quotient?

What is computed here, level by level and all of it exact:

 0) the grading.  ell_0 = 1/2 hat K_t; the mode t^{a} w^{S} of cF_{-1; 0} has
    LEVEL (1 - 2a - |S|)/2.  Every level-lowering K_f kills the vacuum, so the
    vacuum is a lowest-energy vector and M = U(g).|0> lives in levels >= 0.

 1) g_0 on the vacuum.  Over k^L(1|4) the seven elements K_t and o(4) kill it,
    the eighth -- Z = K_{t^{-1} w_1 w_2 w_3 w_4} -- does not.  Over k^M(1|4)
    all eight kill it.

 2) Z spans k^L(1|4) / k^L'(1|4):  the bracket never produces the monomial
    t^{-1} w_1 w_2 w_3 w_4, so the one element of g_0 that moves the vacuum is
    exactly the one outside the derived algebra.

 3) the tower hat Z^k.|0>, k >= 0: independent, all of level 0, all with the
    same labels; K_t and o(4) kill every one of them.  So
    M_0 = U(g_0).|0> = C[hat Z].|0> is infinite-dimensional over k^L(1|4) and
    is the line C.|0> over k^L'(1|4).

 4) the graded dimensions of M over k^L'(1|4) at levels 0, 1, 2, ...  --
    the module is very far from trivial.

 5) (g.M) at level 0.  It is exactly the span of hat Z^k.|0> with k >= 1, so
    the vacuum is NOT in g.M and M / g.M is the trivial module.

 6) no vector of M is killed by all of g: the trivial module is a quotient of
    M, never a submodule.

Usage:  ../../tools/venv/bin/python hw4d.py [N] [Lmax]
"""

import sys
import time
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import (SUBSETS, mono, act_L, act_M, br_L, br_M, TORUS_L, TORUS_M,   # noqa: E402
                 VAC, hatX, sadd)
from hw4 import build, level_of_mode, level_shift, g_basis, Span              # noqa: E402
from hw4b import g0_keys                                                      # noqa: E402
from exactlin import rank as exrank                                           # noqa: E402

ZKEY = (Q(-1), (1, 2, 3, 4))


def nm(k):
    S = "".join(map(str, k[1]))
    return "K_{t^%s%s}" % (k[0], (" w^" + S) if S else "")


# ------------------------------------------------------ 2) the derived algebra

def derived_g0(amax=8):
    """The span of [g, g] inside g_0, and whether Z is in it."""
    rows, seen = [], set()
    ks = [(a, S) for a in range(-amax, amax + 1) for S in SUBSETS]
    for i, (a1, S1) in enumerate(ks):
        d1 = 2 * a1 + len(S1)
        for (a2, S2) in ks[i:]:
            if d1 + 2 * a2 + len(S2) != 4:
                continue
            b = br_L(mono(a1, S1), mono(a2, S2))
            if b:
                rows.append(b)
                seen.update(b)
    cols = sorted(seen)
    M = [[r.get(c, Q(0)) for c in cols] for r in rows]
    r0 = exrank(M) if rows else 0
    inside = ZKEY in cols
    if inside:
        v = [Q(1) if c == ZKEY else Q(0) for c in cols]
        inside = exrank(M + [v]) == r0
    return len(rows), r0, cols, inside


# ------------------------------------------------------------- the level tools

def levels(W, act, par):
    lvmode = {m: level_of_mode(act, par, m) for m in W.allmodes}

    def lev(word):
        return sum(lvmode[m] for m in word)
    return lvmode, lev


def graded(W, act, par, gens, Lmax, seed=None, verbose=True, tag=""):
    """M = U(g).seed, kept level by level up to Lmax.  Returns
    {level: Span} and the running total."""
    lvmode, lev = levels(W, act, par)
    byL = {}

    def put(st):
        ls = {lev(w) for w in st}
        if len(ls) != 1:
            return False, None
        L = ls.pop()
        if L < 0 or L > Lmax:
            return False, None
        byL.setdefault(L, Span())
        return byL[L].add(dict(st)), L

    frontier = list(seed if seed else [dict(VAC)])
    for st in frontier:
        put(st)
    for r in range(2 * Lmax + 4):
        new = []
        for st in frontier:
            for k in gens:
                img = hatX(W, mono(*k), st)
                if not img:
                    continue
                ok, L = put(img)
                if ok:
                    new.append(img)
        if verbose:
            print("      %s round %d: %s"
                  % (tag, r + 1, {L: byL[L].dim() for L in sorted(byL)}))
        if not new:
            break
        frontier = new
    return byL


# ------------------------------------------------- 5) the level-0 part of g.M

def gM_level0(W, act, par, gens, byL, Lmax):
    """The span of hat K_f . v over all f and all v in M of matching level,
    intersected with level 0 -- and whether the vacuum is in it."""
    sp = Span()
    for L in sorted(byL):
        for row in byL[L].piv.values():
            for k in gens:
                if level_shift(*k) != -L:
                    continue
                img = hatX(W, mono(*k), row)
                if img:
                    sp.add(dict(img))
    return sp, sp.contains(dict(VAC))


# ----------------------------------------------------------------- 6) invariants

def invariants(W, gens, byL):
    """Vectors of M killed by every generator.  A g-invariant vector is an
    ell_0-eigenvector of eigenvalue 0, hence of level 0, so only M_0 matters."""
    if 0 not in byL:
        return 0
    rows = list(byL[0].piv.values())
    cols = {}
    for j, st in enumerate(rows):
        for k in gens:
            for w, c in hatX(W, mono(*k), st).items():
                cols.setdefault((k, w), {})[j] = c
    if not cols:
        return len(rows)
    Mx = [[col.get(j, Q(0)) for j in range(len(rows))] for col in cols.values()]
    return len(rows) - exrank(Mx)


# ---------------------------------------------------------------------- driver

def run(sector="L", N=5, Lmax=2, mu=None, lam=Q(-1), amin=-4, amax=7):
    if mu is None:
        mu = Q(0) if sector == "L" else Q(1, 4)
    act = act_L if sector == "L" else act_M
    t0 = time.time()
    print("=" * 78)
    print("k^%s(1|4), V = cF_{%s; %s}, Spin(V), window |a| <= %d, levels <= %d"
          % (sector, lam, mu, N, Lmax))
    print("=" * 78)
    W, act, tor, K, bad = build(sector, N, lam, mu, 1, Nseed=4)
    par = (lam, mu)
    print("  pairing a + a' = %s, invariance defects = %d, %d modes"
          % (K, bad, len(W.allmodes)))

    # 0) the grading
    lvmode, lev = levels(W, act, par)
    zero = [m for m in W.allmodes if lvmode[m] == 0]
    print("\n  0) the grading: %d modes of level 0, of them bosonic %d, "
          "fermionic %d" % (len(zero), sum(1 for m in zero if W.tp(m) == 0),
                            sum(1 for m in zero if W.tp(m) == 1)))
    lowbad = [k for k in g_basis(amin, amax)
              if level_shift(*k) < 0 and hatX(W, mono(*k), VAC)]
    print("     level-lowering generators that fail to kill the vacuum: %s"
          % (lowbad or "none -- the vacuum is a lowest-energy vector"))

    # 1) g_0 on the vacuum
    g0 = g0_keys(sector)
    movers = [k for k in g0 if hatX(W, mono(*k), VAC)]
    print("\n  1) g_0: dim = %d; the ones that move the vacuum: %s"
          % (len(g0), [nm(k) for k in movers] or "none"))

    # 2) the derived algebra (k^L only)
    if sector == "L":
        nrows, r0, cols, inside = derived_g0()
        print("\n  2) [g, g] cap g_0: rank %d out of dim g_0 = %d, from %d "
              "brackets; Z = %s in [g, g]: %s"
              % (r0, len(g0), nrows, nm(ZKEY), inside))
        print("     => Z spans k^L(1|4) / k^L'(1|4)"
              if not inside else "     => Z is a commutator")

    # 3) the tower
    if movers:
        Zk = movers[0]
        tw, st = [], dict(VAC)
        for _ in range(7):
            tw.append(st)
            st = hatX(W, mono(*Zk), st)
            if not st:
                break
        sp = Span()
        ind = sum(1 for s in tw if sp.add(dict(s)))
        print("\n  3) the tower hat Z^k.|0>: %d terms computed, %d independent"
              % (len(tw), ind))
        rest = [k for k in g0 if k != Zk]
        bad3 = [(nm(k), j) for k in rest for j, s in enumerate(tw)
                if hatX(W, mono(*k), s)]
        print("     K_t and o(4) on the tower: %s"
              % (bad3 or "all zero -- every rung carries the same labels "
                 "(0, 0; 0, 0)"))

    # 4) M level by level, over g and over g' = k^L'(1|4)
    gens_all = g_basis(amin, amax)
    gens_der = [k for k in gens_all if k != ZKEY]
    print("\n  4) M = U(g).|0>, level by level:")
    byL = graded(W, act, par, gens_all, Lmax, tag="g ")
    print("     graded dimensions of M   : %s"
          % {L: byL[L].dim() for L in sorted(byL)})
    if sector == "L":
        byLd = graded(W, act, par, gens_der, Lmax, tag="g'", verbose=False)
        print("     graded dimensions of M'  : %s   (over k^L'(1|4))"
              % {L: byLd[L].dim() for L in sorted(byLd)})

    # 5) the level-0 part of g.M
    sp0, has = gM_level0(W, act, par, gens_all, byL, Lmax)
    print("\n  5) (g.M) at level 0: dim = %d (M_0 has dim %d); "
          "the vacuum is in it: %s" % (sp0.dim(), byL[0].dim(), has))
    print("     => the trivial module is %s quotient of M"
          % ("NOT a" if has else "THE unique irreducible"))
    if sector == "L":
        sp0d, hasd = gM_level0(W, act, par, gens_der, byLd, Lmax)
        print("     over k^L'(1|4): (g'.M') at level 0 has dim %d (M'_0 has "
              "dim %d); the vacuum is in it: %s"
              % (sp0d.dim(), byLd[0].dim(), hasd))

    # 6) invariant vectors
    ninv = invariants(W, gens_all, byL)
    print("\n  6) g-invariant vectors of M: %d "
          "(a g-invariant vector has h = 0, hence level 0)" % ninv)

    print("\n  (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    Lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    sec = sys.argv[3] if len(sys.argv) > 3 else "L"
    run(sec, N=N, Lmax=Lmax)
