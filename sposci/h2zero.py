#!/usr/bin/env python3
"""h2zero.py -- over k^L(1|3), k^M(1|3), k^M(1|4): is [psi_V] zero in H^2
(all pairs of opposite weights, not only a Witt probe)?  Output h2zero.log."""
import sys
sys.path.insert(0, ".")
import mumford as mu
from mumford import *
def gb_from(fields, ell0, W):
    byE = {}
    for D in fields:
        if vzero(D):
            continue
        Dl = br(ell0, D)
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(Dl).get(piv, 0) / fD[piv]
        check(vzero(vadd(Dl, vsmul(-lam, D))), "not ell0-eigen")
        if abs(lam) <= W:
            byE.setdefault(lam, []).append(D)
    basis = [((lam, j), D) for lam, Ds in byE.items() for j, D in enumerate(Ds)]

    def coord(D):
        if vzero(D):
            return {}
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(br(ell0, D)).get(piv, 0) / fD[piv]
        vecs = byE.get(lam, [])
        if not vecs:
            return {}          # outside the window
        cols = sorted(set().union(*[flat(V).keys() for V in vecs]) | set(fD), key=str)
        x = solve([[flat(V).get(c, 0) for V in vecs] for c in cols],
                  [fD.get(c, 0) for c in cols])
        check(x is not None, "outside g")
        return {(lam, j): x[j] for j in range(len(vecs)) if x[j] != 0}
    return basis, coord


OUT = []
for n, sec in [(3, "L"), (3, "M"), (4, "M"), (2, "L")]:
    shift = (lambda S: Q(0)) if sec == "L" else ramond_shift(n)
    ell = contact_ell(n)
    l0 = ell(0)
    sh = shift
    fo = [K({(Q(k) + sh(S), S): Q(1)}, n) for S in subsets(n) for k in range(-12, 13)]
    Wm = 6
    gK = gb_from(fo, l0, Q(Wm))
    adj = Module("adj", gK[0], lambda D, X: br(D, X), None, flat)
    pars = {k: vpar(D) for k, D in gK[0]}
    adj.par = lambda k, pars=pars: pars[k]
    ber = density_module(n, ell, n % 2, contact_mult(n), Q(2 - n) / 2, Wm, shift)
    for lab, mod in [("Ber", ber), ("adj", (adj, gK[1]))]:
        g2 = gb_from(fo, l0, Q(2))
        r = h2_ratio(g2[0], g2[1], l0, {lab: mod}, Q(2), "k")
        s = "k^%s(1|%d), %s: class is a coboundary: %s (pairs %d)" % (
            sec, n, lab, r["ref_is_coboundary"], r["pairs"])
        print(s); OUT.append(s)
OUT.append("checks: %d of %d" % (mu.CHECKS[1], mu.CHECKS[0]))
print(OUT[-1])
open("h2zero.log", "w").write("\n".join(OUT) + "\n")
