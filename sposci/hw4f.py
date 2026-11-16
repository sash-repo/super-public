#!/usr/bin/env python3
r"""
hw4f.py -- the decisive computation for the question below, in the polarization
that makes the vacuum a genuine highest weight vector.

For each of

    k^L(1|4) at lambda = -1, mu = 0   -- and its derived algebra k^L'(1|4),
    k^M(1|4) at lambda = -1, mu = 1/4,

the ENERGY polarization is used (a mode is a creation operator iff its level
-- minus its ell_0-weight -- is positive; the level-0 sigma-pairs of the
untwisted sector are split by the t-degree).  Then:

  1) which elements of g_0 kill the vacuum, and which ell_0-raising ones do;
  2) M = U(g).|0>, level by level, and its level-0 part M_0;
  3) whether the vacuum lies in g.M -- i.e., whether the trivial module is
     a quotient of M;
  4) whether M has a g-invariant vector -- i.e., whether the trivial module is
     a submodule.

Usage:  ../../tools/venv/bin/python hw4f.py [N] [Lmax]
"""

import sys
import time
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import (SUBSETS, SRANK, mono, act_L, act_M, br_L, br_M,              # noqa: E402
                 TORUS_L, TORUS_M, VAC, hatX)
import runns4 as R                                                            # noqa: E402
from hw4 import residue_form, g_basis, Span                                   # noqa: E402
from hw4b import g0_keys                                                      # noqa: E402
from hw4e import make_fock, modes_of                                          # noqa: E402
from exactlin import rank as exrank                                           # noqa: E402

SECT = {"L": (act_L, br_L, TORUS_L), "M": (act_M, br_M, TORUS_M)}
S0 = (Q(-1), (1, 2, 3, 4))                    # the generator of k^L / k^L'


def nm(k):
    S = "".join(map(str, k[1]))
    return "K_{t^%s%s}" % (k[0], (" w^" + S) if S else "")


def setup(sector, N, lam, mu, Nseed=4, pol="energy"):
    act, br, torus = SECT[sector]
    par = (lam, mu)
    form, K, bad = residue_form(act, par, torus, N, Nseed)
    modes = modes_of(K, N)
    lvl = {m: -Q(1, 2) * act(par, mono(1), {m: Q(1)}).get(m, Q(0)) for m in modes}

    if pol == "energy":
        def order(m):
            return (-lvl[m], m[0], SRANK[m[1]])
    else:                                   # the t-degree order of ns4/runns4
        def order(m):
            return (Q(0), m[0], SRANK[m[1]])

    W = make_fock(act, par, form, K, N, 1, order)
    bad2 = [m for m in modes if W.is_creation(m) and lvl[m] < 0]
    return W, act, br, torus, par, lvl, K, bad, bad2


def lev_of(lvl):
    def f(word):
        return sum(lvl[m] for m in word)
    return f


def graded(W, gens, lev, Lmax, seed, rounds=8):
    """M = U(g).seed, level by level.  A vector of level outside [0, Lmax] is
    reported rather than silently dropped."""
    byL, out_of_range = {}, set()

    def put(st):
        ls = {lev(w) for w in st}
        if len(ls) != 1:
            return False
        L = ls.pop()
        if L < 0:
            out_of_range.add(L)
            return False
        if L > Lmax:
            return False
        byL.setdefault(L, Span())
        return byL[L].add(dict(st))

    frontier = [dict(s) for s in seed]
    for st in frontier:
        put(st)
    for r in range(rounds):
        new = []
        for st in frontier:
            for k in gens:
                img = hatX(W, mono(*k), st)
                if img and put(img):
                    new.append(img)
        if not new:
            break
        frontier = new
    return byL, sorted(out_of_range)


def gM_at_0(W, gens, lev, byL, shiftof):
    sp = Span()
    for L in sorted(byL):
        for row in byL[L].piv.values():
            for k in gens:
                if shiftof[k] != -L:
                    continue
                img = hatX(W, mono(*k), row)
                if img:
                    sp.add(dict(img))
    return sp, sp.contains(dict(VAC))


def invariants(W, gens, byL):
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


def run(sector, N, Lmax, lam=Q(-1), mu=None, amin=-4, amax=7, pol="energy"):
    if mu is None:
        mu = Q(0) if sector == "L" else Q(1, 4)
    W, act, br, torus, par, lvl, K, bad, bad2 = setup(sector, N, lam, mu, pol=pol)
    lev = lev_of(lvl)
    print("=" * 78)
    print("k^%s(1|4), V = cF_{%s; %s}, %s polarization, |a| <= %d, "
          "levels <= %d" % (sector, lam, mu, pol, N, Lmax))
    print("=" * 78)
    print("  pairing a + a' = %s, invariance defects = %d, %d modes; "
          "creation modes of negative level: %d"
          % (K, bad, len(W.allmodes), len(bad2)))
    print("  labels (c, h; H) of the vacuum: %s" % (R.labels(W, br, torus),))

    gens = g_basis(amin, amax)
    shiftof = {}
    for k in gens:
        b = br(mono(1), mono(*k))
        d = b.get((Q(k[0]), k[1]), Q(0)) if b else Q(0)
        shiftof[k] = -Q(1, 2) * d

    g0 = [k for k in gens if shiftof[k] == 0]
    movers0 = [k for k in g0
               if set(hatX(W, mono(*k), VAC)) - {()}]
    rais = [k for k in gens if shiftof[k] < 0 and hatX(W, mono(*k), VAC)]
    print("\n  1) g_0 (from the bracket with K_t): dim = %d; those that move "
          "the vacuum: %s" % (len(g0), [nm(k) for k in movers0] or "none"))
    print("     ell_0-raising elements that do not kill the vacuum: %s"
          % ([nm(k) for k in rais] or "none -- |0> IS a highest weight vector"))

    for tag, gg in ([("g = k^L(1|4) ", gens),
                     ("g' = k^L'(1|4)", [k for k in gens if k != S0])]
                    if sector == "L" else [("g = k^M(1|4) ", gens)]):
        print("\n  --- over %s" % tag)
        byL, oor = graded(W, gg, lev, Lmax, [VAC])
        print("     graded dimensions of M: %s%s"
              % ({str(L): byL[L].dim() for L in sorted(byL)},
                 "" if not oor else "   (levels below 0 reached: %s)" % oor))
        sp0, has = gM_at_0(W, gg, lev, byL, shiftof)
        print("     (g.M) at level 0: dim %d, M_0 has dim %d; the vacuum is in "
              "it: %s" % (sp0.dim(), byL[0].dim(), has))
        print("     => the trivial module is %s"
              % ("NOT a quotient of M" if has
                 else "the unique irreducible quotient M / g.M"))
        print("     g-invariant vectors of M: %d" % invariants(W, gg, byL))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    Lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.time()
    for sec, pol in (("L", "tdegree"), ("L", "energy"), ("M", "energy")):
        run(sec, N, Lmax, pol=pol)
        print()
    print("(%.0f s)" % (time.time() - t0))
