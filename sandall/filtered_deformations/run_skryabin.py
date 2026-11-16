#!/usr/bin/env python3
"""H^2(g_-; g) and (H^2(g_-; g))^{g_0} for the Skryabin algebras over F_3:

    my(6; 1) = (E^2(V), V, gl(V))_{*,1},        smy^(1)(6; 1),
    by(7; 1) = (E^3(V), E^2(V), V, gl(V))_{*,1}, sby^(1)(7; 1),

the Claims on my, smy^(1), by^(1), sby^(1) in the text.  Usage:  run_skryabin.py my by
"""
import sys
import numpy as np
from gvect import P, rref
from prolong import make
from ceh import CE, module_data


def derived(g):
    """the derived algebra [g, g], as a list of indices-spanning rows (numpy)."""
    rows = []
    for i in range(g.n):
        ri = g.row(i)
        for m in range(i + 1, g.n):
            v = np.zeros(g.n, dtype=np.int64)
            for k, c in ri[m].items():
                v[k] = c
            if v.any():
                rows.append(v)
        if len(rows) > 4 * g.n:
            rows = list(rref(np.array(rows, dtype=np.int64))[0])
    return rref(np.array(rows, dtype=np.int64))[0] if rows else np.zeros((0, g.n), np.int64)


def report(kind):
    print("=" * 68)
    print("g = %s(%d; 1)" % (kind, 6 if kind in ("my", "smy") else 7))
    g = make(kind)
    hidx = [i for i in range(g.n) if g.deg[i] < 0]
    g0idx = [i for i in range(g.n) if g.deg[i] == 0]
    print("   dim g = %d,  dim g_- = %d,  dim g_0 = %d" % (g.n, len(hidx), len(g0idx)))

    D = derived(g)
    print("   dim [g, g] = %d" % len(D))

    ce = CE(g, hidx)
    H = ce.cohomology(2)
    tot = sum(v[0] for v in H.values())
    print("   dim H^2(g_-; g) = %d" % tot)
    info = module_data(ce, H, g0idx, 2)
    for d in sorted(info):
        it = info[d]
        wts = sorted(set(map(tuple, it["weights"])))
        print("   degree %3d: dim %2d  g_0-invariants %d  submodule dims %s"
              % (d, it["dim"], it["inv"], sorted(set(it["subdims"]))))
        print("               weights mod 3: %s"
              % sorted({tuple(a % 3 for a in w) for w in it["weights"]}))
    print("   total invariants (H^2(g_-;g))^{g_0} = %d"
          % sum(v["inv"] for v in info.values()))
    return g, ce, H, info


if __name__ == "__main__":
    for kind in (sys.argv[1:] or ["my"]):
        report(kind)
