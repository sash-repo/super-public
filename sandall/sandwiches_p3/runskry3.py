#!/usr/bin/env python3
r"""
runskry3.py --- the 3-structures of my(6;\One), smy(6;\One), by(7;\One),
sby(7;\One) and of their simple derived algebras, over F_3.

For every basis element D of the algebra L the engine computes D^3 (the cube of
the vector field, i.e. the p-structure of the ambient vect(n;\One)) and decides
whether D^3 lies in L.  If every cube does, Jacobson's theorem gives a unique
p-structure on L with these values, so L is restricted; if some cube does not,
the engine asks the weaker question whether (ad_D)^3 is nevertheless an inner
derivation of L.

Output: runs/skry3.log
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402
from gvect import P, rref                                             # noqa: E402
from skry3 import cube, is_derivation, Model                          # noqa: E402

LOG = os.path.join(HERE, "runs", "skry3.log")


def load(kind):
    from prolong import make
    g = make(kind, verbose=False)
    return Model(g.V, g.basis, g.deg, g.wt, kind), g


def derived(model):
    """the derived algebra L' as a Model (basis of brackets, row-reduced)."""
    rows = []
    for i in range(model.n):
        for j in range(i + 1, model.n):
            co = model.coords(model.V.bracket(model.basis[i], model.basis[j]))
            assert co is not None
            if co.any():
                rows.append(co)
    if not rows:
        return None
    R, piv = rref(np.array(rows, dtype=np.int64))
    basis = [model.field(v) for v in R]
    deg, wt = [], []
    for D in basis:
        deg.append(model.V.fdeg(D))
        wt.append(model.V.fwt(D))
    return Model(model.V, basis, deg, wt, model.name + "^(1)")


def analyse(model, say, full_ad=True):
    V = model.V
    cubes, outside = {}, []
    for i, D in enumerate(model.basis):
        C = cube(V, D)
        assert is_derivation(V, C), "the cube is not a derivation"
        co = model.coords(C) if C else np.zeros(model.n, dtype=np.int64)
        if co is None:
            outside.append(i)
        else:
            cubes[i] = (C, co)
    nz = [i for i in cubes if cubes[i][1].any()]
    say("  dim = %d, degrees %d..%d" % (model.n, min(model.deg), max(model.deg)))
    say("  cubes inside L: %d of %d;  non-zero: %d"
        % (len(cubes), model.n, len(nz)))
    if outside:
        say("  cubes OUTSIDE L: %d" % len(outside))
        for i in outside[:6]:
            say("    deg %d  wt %s : (%s)^3 = %s"
                % (model.deg[i], model.wt[i], V.show(model.basis[i]),
                   V.show(cube(V, model.basis[i]))))
    # ad^3 = ad_{D^3} on L, for every basis element whose cube is inside
    if full_ad:
        bad = 0
        for i, (C, co) in cubes.items():
            A = model.admat(model.basis[i])
            B = model.admat(C) if C else np.zeros((model.n, model.n),
                                                  dtype=np.int64)
            if A is None or B is None or not np.array_equal((A @ A @ A) % P,
                                                            B % P):
                bad += 1
        say("  ad^3 = ad_{D^3} on L: %d of %d basis elements"
            % (len(cubes) - bad, len(cubes)))
    # the shape of the p-map
    for i in sorted(nz):
        say("    (%s)^[3] = %s   [deg %d, wt %s]"
            % (V.show(model.basis[i]), V.show(cubes[i][0]),
               model.deg[i], model.wt[i]))
    return cubes, outside, nz


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    say("skry3: the 3-structures of the Skryabin algebras, p = 3, un = One")
    for kind in (sys.argv[1:] or ("my", "smy", "by", "sby")):
        model, g = load(kind)
        say("")
        say("=" * 70)
        say("%s(%d; One)" % (kind, model.V.n))
        cubes, outside, nz = analyse(model, say)
        if not outside:
            say("  ==> RESTRICTED (Jacobson), the 3-map being the cube of the field")
        d = derived(model)
        if d is not None and d.n != model.n:
            say("  derived algebra L^(1): dim %d" % d.n)
            analyse(d, say)
        # second derived, if it moves
        if d is not None:
            d2 = derived(d)
            if d2 is not None and d2.n != d.n:
                say("  L^(2): dim %d" % d2.n)
                analyse(d2, say)
    log.close()


if __name__ == "__main__":
    main()
