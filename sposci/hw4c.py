#!/usr/bin/env python3
r"""
hw4c.py -- the three supplementary checks for the question below,

  A) the tower hat Z^k . |0> is closed under the whole of g_0, and the
     structure constants are the cocycle values gamma(X, Z):
             hat X . hat Z^k |0> = k gamma(X, Z) hat Z^{k-1} |0>,
     so g_0 acts on the level-0 space as C[Z] with the derivation
     gamma(., Z) d/dZ -- never by a scalar, whence the level-0 space is
     an INDECOMPOSABLE, infinite-dimensional g_0-module and the vacuum is not
     a highest weight vector for g_0;

  B) the central charge is 0 on every vector of the tower, not only on the
     vacuum (the labels of the whole tower coincide), on a window large enough
     for the truncation not to bite;

  C) the same happens on the DOUBLED module W = V (+) V^#, which is what the
     two tables of Subsection ssHW use.

Usage:  ../../tools/venv/bin/python hw4c.py [N]
"""

import sys
import time
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import (SUBSETS, mono, act_L, act_M, br_L, br_M, TORUS_L, TORUS_M,   # noqa: E402
                 VAC, hatX, sadd)
import runns4 as R                                                            # noqa: E402
from hw4 import build, residue_form, level_of_mode, Span                      # noqa: E402
from hw4b import g0_keys, name, tower, _eig, charge_on                        # noqa: E402


def report_tower(W, br, torus, Zk, kmax, tag):
    print("\n  --- %s" % tag)
    tw = tower(W, Zk, kmax)
    print("    the tower hat Z^k|0>: %d nonzero terms" % len(tw))
    sp = Span()
    ind = 0
    for st in tw:
        if sp.add(dict(st)):
            ind += 1
    print("    linearly independent: %d" % ind)
    for k, st in enumerate(tw):
        h = _eig(hatX(W, mono(1), st), st)
        Hs = tuple(_eig(hatX(W, mono(*g), st), st) for g in torus)
        c = charge_on(W, br, st)
        print("      k=%d: %3d words, c=%s, h=%s, H=%s"
              % (k, len(st), c, None if h is None else Q(1, 2) * h, Hs))
    return tw


def closure(W, sector, Zk, tw, br):
    """A) g_0 . (the tower) stays in the tower, with the predicted constants."""
    print("\n    closure of the tower under g_0 (and the constants):")
    sp = Span()
    for st in tw:
        sp.add(dict(st))
    for k in g0_keys(sector):
        gam = None
        outside = 0
        rows = []
        for j, st in enumerate(tw):
            img = hatX(W, mono(*k), st)
            if img and not sp.contains(dict(img)):
                outside += 1
            if j >= 1 and img:
                # img should be j * gamma * tw[j-1]
                r = _eig(img, tw[j - 1])
                rows.append(None if r is None else r / j)
            elif j >= 1:
                rows.append(Q(0))
        vals = {r for r in rows if r is not None}
        print("      %-26s leaves the tower in %d of %d cases; "
              "gamma = %s" % (name(k), outside, len(tw),
                              vals if vals else "n/a"))


def doubled(N=3, lam=Q(-1), mu=Q(0), sector="L", kmax=5):
    """C) the same on W = V (+) V^#, the module the tables use."""
    print("\n  --- C) the doubled module W = V (+) V^# over k^%s(1|4), "
          "lambda = %s, mu = %s" % (sector, lam, mu))
    act = act_L if sector == "L" else act_M
    br = br_L if sector == "L" else br_M
    torus = TORUS_L if sector == "L" else TORUS_M
    if sector == "L":
        par_d = (Q(-2) - lam, -mu)
    else:
        par_d = (Q(-2) - lam, Q(1, 2) - mu)
    form, _ = R.dual_form(act, (lam, mu), par_d, N) if hasattr(R, "dual_form") \
        else (None, None)
    if form is None:
        print("    (built by runns4.build_W from the residue pairing)")
    from ns4 import invariant_forms, split_symmetry, keys
    fs, _ = invariant_forms(act, (lam, mu), par_d, N=N,
                            gens=keys(range(-1, 3)), torus=torus)
    if not fs:
        print("    no invariant pairing V x V^# found in this window -- abort")
        return
    Wd = R.build_W(act, (lam, mu), par_d, fs[0], N, 1)
    Zk = (Q(-1), (1, 2, 3, 4))
    st = hatX(Wd, mono(*Zk), VAC)
    print("    hat Z . |0> = %s" % ("0" if not st else "%d words" % len(st)))
    if st:
        report_tower(Wd, br, torus, Zk, kmax, "the tower on W")
    return Wd


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    print("=" * 78)
    print("hw4c: the level-0 tower at n = 4, window |a| <= %d" % N)
    print("=" * 78)
    W, act, tor, K, bad = build("L", N, Q(-1), Q(0), 1, Nseed=4)
    print("  k^L(1|4): pairing a+a' = %s, invariance defects = %d, %d modes"
          % (K, bad, len(W.allmodes)))
    Zk = (Q(-1), (1, 2, 3, 4))
    tw = report_tower(W, br_L, TORUS_L, Zk, 4,
                      "A/B) the tower over k^L(1|4), lambda = -1, mu = 0")
    closure(W, "L", Zk, tw, br_L)
    doubled(N=3, sector="L")
    print("\n  (%.0f s)" % (time.time() - t0))
