#!/usr/bin/env python3
r"""
hw4e.py -- the polarization, and what the vacuum is.

The Fock space Spin(V) is built on a maximal isotropic subspace of the space of
modes; sigma pairs the mode (a, S) with (K - a, S^c), and WHICH member of each
pair is declared a creation operator is a choice.  ns4.py/runns4.py order the
modes by the t-degree a.  That choice is isotropic, hence legitimate, but it is
not the one that makes ell_0 = 1/2 hat K_t bounded on the module: the level

        level(a, S) = -(the ell_0-weight of the mode)

is opposite on the two members of a pair, and the t-degree ordering picks the
member of NEGATIVE level in a few pairs.  Here we compare the two:

  * ENERGY polarization -- creation iff level > 0 (ties at level 0 broken by
    the t-degree, and swept over all 2^z choices).  Then, |0> is annihilated by
    every ell_0-raising element and is a genuine highest weight vector.

  * the t-degree polarization used so far.

and we compute (c, h; H) in each.

Usage:  ../../tools/venv/bin/python hw4e.py [N]
"""

import sys
import time
from itertools import product
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import (SUBSETS, SRANK, mono, act_L, act_M, br_L, br_M,              # noqa: E402
                 TORUS_L, TORUS_M, VAC, hatX, Fock)
import runns4 as R                                                            # noqa: E402
from hw4 import residue_form, level_of_mode                                   # noqa: E402


SECT = {"L": (act_L, br_L, TORUS_L), "M": (act_M, br_M, TORUS_M)}


def modes_of(K, N):
    lo, hi = max(-N, K - N), min(N, K + N)
    return [(Q(a), S) for a in range(int(lo), int(hi) + 1) for S in SUBSETS]


def make_fock(act, par, form, K, N, shift, order):
    modes = modes_of(K, N)

    def Bfun(m, n):
        return form.get((m, n), Q(0))

    def actfun(f, m):
        return act(par, f, {m: Q(1)})

    return Fock(modes, lambda m: len(m[1]) % 2, Bfun, actfun, order, shift)


def labels(W, br, torus):
    c = R.charge(W, br)
    h = R.h_direct(W)
    H = R.H_of(W, torus)
    return c, h, H


def raising_defects(W, act, par, amin=-4, amax=7):
    """Elements that RAISE ell_0 and yet do not kill the vacuum -- the vacuum
    is a highest weight vector exactly when there are none."""
    out = []
    lv = {m: level_of_mode(act, par, m) for m in W.allmodes}
    for a in range(amin, amax + 1):
        for S in SUBSETS:
            st = hatX(W, mono(a, S), VAC)
            if not st:
                continue
            lvs = {sum(lv[m] for m in w) for w in st}
            if min(lvs) < 0:
                out.append(((a, S), sorted(lvs)))
    return out, lv


def run(sector="L", N=5, lam=Q(-1), mu=None, Nseed=4):
    if mu is None:
        mu = Q(0) if sector == "L" else Q(1, 4)
    act, br, torus = SECT[sector]
    par = (lam, mu)
    print("=" * 78)
    print("k^%s(1|4), V = cF_{%s; %s}, window |a| <= %d" % (sector, lam, mu, N))
    print("=" * 78)
    form, K, bad = residue_form(act, par, torus, N, Nseed)
    print("  pairing a + a' = %s, invariance defects = %d" % (K, bad))

    modes = modes_of(K, N)
    lvl = {m: -Q(1, 2) * act(par, mono(1), {m: Q(1)}).get(m, Q(0)) for m in modes}
    sig = {m: (K - m[0], tuple(sorted(set(range(1, 5)) - set(m[1])))) for m in modes}
    zero = [m for m in modes if lvl[m] == 0]
    print("  modes: %d; of level 0: %d; sigma-pairs of level 0: %d"
          % (len(modes), len(zero), len(zero) // 2))
    import collections
    print("  level census: %s"
          % dict(sorted(collections.Counter(lvl.values()).items())))

    # --- the t-degree polarization (the one used so far)
    ord_t = lambda m: (m[0], SRANK[m[1]])                      # noqa: E731
    W = make_fock(act, par, form, K, N, 1, ord_t)
    wrong = [m for m in modes if W.is_creation(m) and lvl[m] < 0]
    print("\n  A) the t-degree polarization:")
    print("     creation modes of NEGATIVE level (i.e. ell_0-raising): %d %s"
          % (len(wrong), wrong[:4]))
    try:
        print("     labels (c, h; H) = %s" % (labels(W, br, torus),))
    except ValueError as e:
        print("     labels FAILED: %s" % e)
    df, _ = raising_defects(W, act, par)
    print("     ell_0-raising elements that do not kill the vacuum: %d %s"
          % (len(df), [(k, v) for k, v in df[:3]]))

    # --- the energy polarization
    print("\n  B) the energy polarization (creation iff level > 0):")
    pairs0 = []
    seen = set()
    for m in zero:
        if m in seen:
            continue
        seen.add(m)
        seen.add(sig[m])
        pairs0.append((m, sig[m]))
    print("     level-0 sigma-pairs to be split by hand: %d -> %d polarizations"
          % (len(pairs0), 2 ** len(pairs0)))
    seenlab = {}
    for choice in product((0, 1), repeat=len(pairs0)):
        pick = {}
        for (m, n), c in zip(pairs0, choice):
            pick[m], pick[n] = (0, 1) if c == 0 else (1, 0)

        def order(m, pick=pick):
            if lvl[m] != 0:
                return (-lvl[m], 0, m[0], SRANK[m[1]])
            return (0, pick[m], m[0], SRANK[m[1]])

        Wp = make_fock(act, par, form, K, N, 1, order)
        bad2 = [m for m in modes if Wp.is_creation(m) and lvl[m] < 0]
        assert not bad2, bad2
        try:
            lab = labels(Wp, br, torus)
        except ValueError as e:
            lab = "FAILED: %s" % str(e)[:60]
        df2, _ = raising_defects(Wp, act, par)
        key = (str(lab), len(df2))
        seenlab.setdefault(key, 0)
        seenlab[key] += 1
    for (lab, nd), cnt in sorted(seenlab.items()):
        print("     %4d polarizations: labels %s ; raising defects %d"
              % (cnt, lab, nd))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    for sec, mu in (("L", Q(0)), ("M", Q(1, 4))):
        run(sec, N=N, mu=mu)
        print()
    print("(%.0f s)" % (time.time() - t0))
