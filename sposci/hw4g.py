#!/usr/bin/env python3
r"""
hw4g.py -- the cheap, targeted confirmation of Proposition 7.3.25 3):
the vacuum is not in g.M.

By the proof, everything reduces to the words of one and of two raising
factors, so instead of building M we test directly

        <One | hat x hat y_1 ... hat y_r One>  = 0     for r = 1, 2,

over all generators x of level -L and all raising y_i of levels summing to L.
That is what a non-zero cocycle would break, and Statement 7.4.4 says
there is none.

Usage:  ../../tools/venv/bin/python hw4g.py [N]
"""

import sys
import time
from fractions import Fraction as Q

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns4 import mono, VAC, hatX                                              # noqa: E402
from hw4 import g_basis                                                       # noqa: E402
from hw4f import setup, SECT, S0, nm                                          # noqa: E402


def shifts(br, gens):
    out = {}
    for k in gens:
        b = br(mono(1), mono(*k))
        d = b.get((Q(k[0]), k[1]), Q(0)) if b else Q(0)
        out[k] = -Q(1, 2) * d
    return out


def run(sector, N, pol, lam=Q(-1), mu=None, amin=-4, amax=7, drop=None):
    if mu is None:
        mu = Q(0) if sector == "L" else Q(1, 4)
    W, act, br, torus, par, lvl, K, bad, bad2 = setup(sector, N, lam, mu, pol=pol)
    gens = [k for k in g_basis(amin, amax) if k != drop]
    sh = shifts(br, gens)
    tag = "k^%s%s(1|4), %s polarization, mu = %s" % (
        sector, "'" if drop else "", pol, mu)
    print("  --- %s (%d modes, %d generators)" % (tag, len(W.allmodes), len(gens)))

    # the raising generators (level shift > 0) and their action on the vacuum
    rais = [k for k in gens if sh[k] > 0]
    lower = [k for k in gens if sh[k] < 0]
    zero = [k for k in gens if sh[k] == 0]
    v0 = {k: hatX(W, mono(*k), VAC) for k in rais}
    v0 = {k: s for k, s in v0.items() if s}
    print("      generators: %d raising (%d move the vacuum), %d of level 0, "
          "%d lowering" % (len(rais), len(v0), len(zero), len(lower)))

    # r = 0: g_0 . One
    z0 = [nm(k) for k in zero if set(hatX(W, mono(*k), VAC)) - {()}]
    print("      r = 0: level-0 generators moving the vacuum off C.One: %s"
          % (z0 or "none"))

    # r = 1
    hits1 = []
    for y, sy in v0.items():
        for x in lower:
            if sh[x] != -sh[y]:
                continue
            c = hatX(W, mono(*x), sy).get((), Q(0))
            if c:
                hits1.append((nm(x), nm(y), c))
    print("      r = 1: pairs (x, y) with <One| x y |One> =/= 0: %d %s"
          % (len(hits1), hits1[:3]))

    # r = 2
    hits2, tested = [], 0
    items = sorted(v0.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    for y1, _ in items:
        for y2, sy2 in items:
            s = hatX(W, mono(*y1), sy2)
            if not s:
                continue
            L = sh[y1] + sh[y2]
            for x in lower:
                if sh[x] != -L:
                    continue
                tested += 1
                c = hatX(W, mono(*x), s).get((), Q(0))
                if c:
                    hits2.append((nm(x), nm(y1), nm(y2), c))
    print("      r = 2: %d triples tested, non-zero: %d %s"
          % (tested, len(hits2), hits2[:3]))
    ok = not hits1 and not hits2 and not z0
    print("      => the vacuum is %s in g.M at this depth"
          % ("NOT" if ok else "possibly"))
    return ok


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    print("=" * 78)
    print("hw4g: <One | g . M> at depth <= 2, window |a| <= %d" % N)
    print("=" * 78)
    run("L", N, "tdegree")
    run("L", N, "tdegree", drop=S0)
    run("L", N, "energy")
    run("M", N, "energy")
    print("\n(%.0f s)" % (time.time() - t0))
