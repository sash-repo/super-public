#!/usr/bin/env python3
"""The same as run_skryabin.py, but for the *derived* algebras: the Claims of the
text name smy^(1)(6; 1), sby^(1)(7; 1), by^(1)(7; 1), whereas prolong.py builds
the prolongs smy(6; 1), sby(7; 1), by(7; 1) themselves.  Usage:

    run_derived.py smy sby by
"""
import sys
import numpy as np
from gvect import P, rref
from prolong import make, GradedAlg
from ceh import CE, module_data


def derived_basis(g):
    """a (deg, wt)-homogeneous basis of [g, g], as vector fields."""
    blocks = {}
    for a in range(g.n):
        for b in range(a + 1, g.n):
            br = g.V.bracket(g.basis[a], g.basis[b])
            if br:
                key = (g.V.fdeg(br), g.V.fwt(br))
                blocks.setdefault(key, []).append(br)
    out = []
    for key, fields in blocks.items():
        keys = sorted({k for D in fields for k in D})
        M = np.array([[D.get(k, 0) % P for k in keys] for D in fields],
                     dtype=np.int64)
        R, _ = rref(M)
        for r in R:
            D = {k: int(c) for k, c in zip(keys, r) if c % P}
            if D:
                out.append(D)
    return out


def report(kind):
    print("=" * 68)
    g = make(kind)
    print("   the prolong %s(%d; 1) has dim %d" % (kind, 6 if "my" in kind else 7, g.n))
    d = GradedAlg(g.V, derived_basis(g), kind + "^(1)")
    print("g = %s^(1):  dim = %d" % (kind, d.n))
    bydeg = {}
    for i in range(d.n):
        bydeg.setdefault(d.deg[i], []).append(i)
    print("   graded dimensions: %s" % {k: len(v) for k, v in sorted(bydeg.items())})

    hidx = [i for i in range(d.n) if d.deg[i] < 0]
    g0idx = [i for i in range(d.n) if d.deg[i] == 0]
    print("   dim g_- = %d,  dim g_0 = %d" % (len(hidx), len(g0idx)))

    ce = CE(d, hidx)
    H = ce.cohomology(2)
    print("   dim H^2(g_-; g) = %d" % sum(v[0] for v in H.values()))
    info = module_data(ce, H, g0idx, 2)
    for deg in sorted(info):
        it = info[deg]
        print("   degree %3d: dim %2d  g_0-invariants %d  submodule dims %s"
              % (deg, it["dim"], it["inv"], sorted(set(it["subdims"]))))
        print("               weights mod 3: %s"
              % sorted({tuple(a % 3 for a in w) for w in it["weights"]}))
    print("   total invariants (H^2(g_-;g))^{g_0} = %d"
          % sum(v["inv"] for v in info.values()))


if __name__ == "__main__":
    for kind in (sys.argv[1:] or ["smy"]):
        report(kind)
