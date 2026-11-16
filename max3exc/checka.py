#!/usr/bin/env python3
"""Independent check of the parameter a: instead of solving for the invariant
form, use the KILLING form  kappa(x,y) = str(ad x ad y)  computed inside u
(a supertrace, i.e. with the parity signs).  Both routes must agree, since u is
simple and its invariant form is unique up to a factor.
"""
from fractions import Fraction as F
import sys, pickle
import run3, ident3
from sweep3 import Ech, sdim_of
from ident3 import even_part, minimal_ideals, _canonical_H, _coords


def killing(g, ub):
    """matrix of str(ad x ad y) in the basis ub of the subalgebra u."""
    n = len(ub)
    # parity of each basis vector of u
    par = []
    for v in ub:
        ps = {g.PAR[i] for i, x in enumerate(v) if x}
        assert len(ps) == 1
        par.append(ps.pop())
    ad = []
    for x in ub:
        cols = []
        for y in ub:
            c = _coords(g, ub, g.br(x, y))
            assert c is not None, "u is not a subalgebra"
            cols.append(c)
        # ad[x][j][i] : coefficient of ub[i] in [x, ub[j]]
        ad.append(cols)
    K = [[F(0)] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            s = F(0)
            for j in range(n):
                # (ad a ad b)(ub[j]) : coefficient of ub[j]
                cb = ad[b][j]
                for i in range(n):
                    if cb[i]:
                        s += F(-1) ** par[j] * cb[i] * ad[a][i][j]
            K[a][b] = s
    return K, par


def a_ambient(g, Kg, ub):
    """a, computed from the AMBIENT Killing form restricted to u (independent of
       the invariant-form solver of ident3.py)."""
    E = Ech(g.N, ub)
    b = E.basis()
    mev = minimal_ideals(g, even_part(g, b))
    if len(mev) != 3 or any(I.dim() != 3 for I in mev):
        return None
    bs = []
    for I in mev:
        H = _canonical_H(g, I)
        bs.append(sum(H[i] * Kg[i][j] * H[j] for i in range(g.N)
                      for j in range(g.N) if H[i] and H[j]))
    if any(x == 0 for x in bs):
        return None
    cs = [F(1) / x for x in bs]
    s = sum(cs)
    if s != 0:
        cs = [c - s / 3 for c in cs]
    if cs[0] == 0:
        cs = [cs[1], cs[2], cs[0]]
    return cs[1] / cs[0], bs


def a_from_killing(g, ub):
    E = Ech(g.N, ub)
    b = E.basis()
    evb = even_part(g, b)
    mev = minimal_ideals(g, evb)
    if len(mev) != 3 or any(I.dim() != 3 for I in mev):
        return None
    K, par = killing(g, b)
    bs = []
    for I in mev:
        H = _canonical_H(g, I)
        co = _coords(g, b, H)
        val = sum(co[i] * K[i][j] * co[j] for i in range(len(co))
                  for j in range(len(co)) if co[i] and co[j])
        bs.append(val)
    if any(x == 0 for x in bs):
        return None
    cs = [F(1) / x for x in bs]
    s = sum(cs)
    if s != 0:
        cs = [c - s / 3 for c in cs]
    if cs[0] == 0:
        cs = [cs[1], cs[2], cs[0]]
    return cs[1] / cs[0], bs


if __name__ == "__main__":
    # NB: the Killing form of D(2,1;a) VANISHES identically (its dual Coxeter
    # number is 0), so the intrinsic Killing route is void; we use the ambient
    # Killing form instead, which is nonzero for ag(2) and ab(3).
    import d21a
    from check_d21a import D21a
    g = D21a(F(5))
    K, _ = killing(g, [g.e(i) for i in range(g.N)])
    print("D(2,1;5): nonzero entries of its own Killing form: %d"
          % sum(1 for r in K for x in r if x))
    # the copies found inside ag(2), ab(3)
    for p in sys.argv[1:]:
        with open(p, "rb") as f:
            d = pickle.load(f)
        which = "G3" if "G3" in p else "F4"
        g = run3.model(which)
        Kg, _ = killing(g, [g.e(i) for i in range(g.N)])
        for tags, basis, sd, pieces, verdict in d["answers"]:
            if tuple(sd) != (9, 8):
                continue
            ub = [[F(x) for x in v] for v in basis]
            r1 = a_ambient(g, Kg, ub)
            r2 = ident3.d21a_parameter(g, ub)
            print("%s  %s : ambient-Killing a = %s ;  invariant-form a = %s"
                  % (p, tags[0], r1[0] if r1 else None,
                     r2[0] if r2 else None))
