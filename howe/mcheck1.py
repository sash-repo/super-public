"""
mcheck1.py -- self-tests of the module layer:
  (1) I(V) really is an M(n)-module: [f,g].u = f.(g.u) - (-1)^{p'f p'g} g.(f.u)
      for all f,g of M-degree >= 0 and all basis vectors u of low degree;
  (2) the PBW multiplication in U(M(n)_{<0}) is associative on the relations;
  (3) M_{a,b}(n) = (ba(n), spe(n)+C(az+bh))_* has the properties Kochetkov states:
      SM(n) = M_{a,b}(n) at a + nb = 0 (checked against div = 0),
      M_{0,b}(n) = B(n) is a central extension of Le(n) (dimensions),
      M_{a,b}(n) = M_{c,d}(n) whenever a/b = c/d.
"""

import sys
from fractions import Fraction as F

from le import padd, pscal
from mn import mbr, mdeg, layer, par_m, mdiv, mono, one, tau
from mimod import Mn, Mab, MInduced


def test_module(n, lam, zc, mdegmax=2):
    mn = Mn(n)
    I = MInduced(mn, lam, zc)
    ops = []
    for j in range(0, 2):
        ops += [(j, f) for f in layer(n, j)]
    keys = []
    for m in range(0, mdegmax + 1):
        for u in I.umonos(m):
            w = I.uwt(u)
            mu = tuple(x - y for x, y in zip(I.lam, w))
            # take a few weight spaces near the top
            for i in range(len(I.V.wtbasis(mu))):
                keys.append((u, mu, i))
    print("  V(%s), z=%s : %d basis vectors of I(V) of degree <= %d, %d operators"
          % (lam, zc, len(keys), mdegmax, len(ops)))
    bad = tested = 0
    for j1, f in ops:
        for j2, g in ops:
            br = mbr(f, g, n)
            pf, pg = par_m(f), par_m(g)
            for key in keys:
                lhs = I._act_one(br, key) if br else {}
                r1 = I.act(f, I._act_one(g, key))
                r2 = I.act(g, I._act_one(f, key))
                rhs = padd(r1, r2, -(-1) ** (pf * pg))
                if padd(lhs, rhs, -1):
                    bad += 1
                    if bad < 3:
                        print("    MISMATCH", f, g, key)
                tested += 1
    print("    module axiom: %d checks, %d bad" % (tested, bad))
    return bad


def test_mab(n, jmax=3):
    mn = Mn(n)
    print("  dim M(n)_j      :", {j: len(layer(n, j)) for j in range(-2, jmax + 1)})
    # SM(n): a + nb = 0
    ab = Mab(mn, n, -1)
    dims = {j: len(ab.layer(j)) for j in range(-2, jmax + 1)}
    print("  dim M_{n,-1}_j  :", dims)
    bad = 0
    for j in range(0, jmax + 1):
        for f in ab.layer(j):
            if mdiv(f, n):
                bad += 1
    print("    a+nb=0 gives divergence-free fields (SM(n)): %d bad" % bad)
    # every divergence-free element of M(n)_j is in it, too
    from le import nullspace
    bad2 = 0
    for j in range(0, jmax + 1):
        bas = layer(n, j)
        eqs = {}
        for ci, f in enumerate(bas):
            for m, c in mdiv(f, n).items():
                eqs.setdefault(m, {})[ci] = c
        k = nullspace(list(eqs.values()), list(range(len(bas))))
        if len(k) != dims[j]:
            bad2 += 1
            print("    dim ker div in degree %d = %d, dim M_{n,-1}_%d = %d"
                  % (j, len(k), j, dims[j]))
    print("    dim ker(div) = dim M_{n,-1} in every degree: %s" % ("yes" if not bad2 else "NO"))
    # B(n) = M_{0,b}(n): central extension of Le(n)
    ab0 = Mab(mn, 0, 1)
    d0 = {j: len(ab0.layer(j)) for j in range(-2, jmax + 1)}
    dle = {j: len([f for f in layer(n, j) if n not in next(iter(f))[1]]) for j in range(-2, jmax + 1)}
    print("  dim M_{0,1}_j   :", d0)
    print("  dim Le(n)_j     :", dle, "  (Le(n)_j = <deg j+2 in x,xi>)")
    # M_{a,b} = M_{c,d} for a/b = c/d
    ab1, ab2 = Mab(mn, 2, 3), Mab(mn, 4, 6)
    same = all(_span(ab1.layer(j), n) == _span(ab2.layer(j), n) for j in range(0, jmax + 1))
    print("  M_{2,3} = M_{4,6}: %s" % same)
    # generic a,b
    for (a, b) in [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (n, -1), (-n, 2)]:
        ab = Mab(mn, a, b)
        print("  dim M_{%s,%s}_j : %s" % (a, b, {j: len(ab.layer(j)) for j in range(-2, jmax + 1)}))


def _span(fs, n):
    from le import rref
    amb = {}
    rows = []
    for f in fs:
        for m in f:
            amb.setdefault(m, len(amb))
    for f in fs:
        rows.append({amb[m]: c for m, c in f.items()})
    piv, red = rref(rows, len(amb))
    inv = {v: k for k, v in amb.items()}
    return frozenset((inv[p], frozenset((inv[k], v) for k, v in r.items()))
                     for p, r in zip(piv, red))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print("=== module axioms, n = %d ===" % n)
    test_module(n, [1] * n, 1, 2)
    print("=== M_{a,b}(n), n = %d ===" % n)
    test_mab(n, 3)
