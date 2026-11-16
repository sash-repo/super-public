"""
The Ramond table (74) and the basis (73) of the text.

Two things are checked:

 1) the basis (73): the degree column (the eigenvalue of ad_{K_t}), the
    parity column, and whether the 16 monomials of each t-power are all listed;
 2) the cocycle: the table printed in (74) is written in the names of the
    theta-basis of k^L, which do not occur in (73) at all, so what is
    checked instead is the complete table (cocycR1) of the text on spinor-oscillator representations, in the
    very basis (73) describes.
"""

from fractions import Fraction as Q
from itertools import combinations, product
import sys

from stringy import mono, parity
from kM4 import (bracket, monomials, weight, name_of, by_weight, jacobi_bad,
                 check_skew, check_cocycle, table_raw, Z, XI, ETA, TH, ODD)

# ---------------------------------------------------------- 1) the basis

# (73) as printed: name -> (a as a function of n, odd monomial, printed
# degree, printed parity)
PRINTED = [
    ("L_n",              lambda n: n + 1, (),               lambda n: 2 * n,     "ev"),
    ("T^{xi eta}_n",     lambda n: n,     (XI, ETA),        lambda n: 2 * n,     "ev"),
    ("T^{xi theta}_n",   lambda n: n,     (XI, TH),         lambda n: 2 * n,     "ev"),
    ("T^{eta theta}_n",  lambda n: n,     (ETA, TH),        lambda n: 2 * n,     "ev"),
    ("~T^{xi zeta}_n",   lambda n: n,     (Z, XI),          lambda n: 2 * n + 1, "ev"),
    ("~T^{eta zeta}_n",  lambda n: n,     (Z, ETA),         lambda n: 2 * n + 1, "ev"),
    ("S_n",              lambda n: n - 1, (Z, XI, ETA, TH), lambda n: 2 * n + 1, "ev"),
    ("G^xi_n",           lambda n: n + 1, (XI,),            lambda n: 2 * n + 1, "od"),
    ("G^eta_n",          lambda n: n + 1, (ETA,),           lambda n: 2 * n + 1, "od"),
    ("G^theta_n",        lambda n: n + 1, (TH,),            lambda n: 2 * n + 1, "od"),
    ("~F_n",             lambda n: n - 1, (XI, ETA, TH),    lambda n: 2 * n + 1, "od"),
    ("~G_n",             lambda n: n + 1, (Z,),             lambda n: 2 * n,     "od"),
    ("F^{xi eta}_n",     lambda n: n,     (Z, XI, ETA),     lambda n: 2 * n,     "od"),
    ("F^{xi theta}_n",   lambda n: n,     (Z, XI, TH),      lambda n: 2 * n,     "od"),
    ("F^{eta theta}_n",  lambda n: n,     (Z, ETA, TH),     lambda n: 2 * n,     "od"),
]


def check_basis():
    print("=== (73): the degree and parity columns ===")
    bad = 0
    for nm, af, S, degf, par in PRINTED:
        ok_deg = all(weight((af(n), tuple(sorted(S)))) == degf(n)
                     for n in range(-3, 4))
        real = weight((af(0), tuple(sorted(S))))
        p = "od" if len(S) % 2 else "ev"
        ok_par = (p == par)
        if not (ok_deg and ok_par):
            bad += 1
            # what the degree really is, as a linear function of n
            d0 = weight((af(0), tuple(sorted(S))))
            d1 = weight((af(1), tuple(sorted(S))))
            slope = d1 - d0
            print("   %-18s printed degree %s, parity %s   ->  true degree "
                  "%dn%+d, parity %s"
                  % (nm, degf(0) and ("2n%+d" % (degf(0))) or "2n",
                     par, slope, d0, p))
    if not bad:
        print("   all rows correct")
    print()
    print("=== (73): is every basis vector listed? ===")
    listed = {tuple(sorted(S)) for _, _, S, _, _ in PRINTED}
    allS = {S for r in range(5) for S in combinations(ODD, r)}
    missing = sorted(allS - listed, key=lambda S: (len(S), S))
    NM = {Z: "zeta", XI: "xi", ETA: "eta", TH: "theta"}
    print("   %d of the 16 monomials are listed" % len(listed))
    for S in missing:
        print("   MISSING: K_{t^a %s}, degree 2a-2+%d-%d = %s, parity %s"
              % (" ".join(NM[i] for i in S), len(S), 1 if Z in S else 0,
                 "2a%+d" % (-2 + len(S) - (1 if Z in S else 0)),
                 "od" if len(S) % 2 else "ev"))


# ------------------------------------------------- 2) the cocycle table

def check_table(B=3):
    print()
    print("=== the convention constants of the Ramond bracket ===")
    tiny = monomials(range(0, 2))
    good = [(s, k, kz) for s in (1, -1) for k in (Q(1), Q(-1))
            for kz in (Q(1), Q(-1))
            if jacobi_bad(tiny, sigma=s, k=k, kz=kz) == 0]
    print("   (sigma, k, kz) passing Jacobi on a small window:", good)
    # confirm on a bigger window
    good = [g for g in good
            if jacobi_bad(monomials(range(-1, 2)), sigma=g[0], k=g[1],
                          kz=g[2]) == 0]
    print("   ... and on |a| <= 1:", good)
    small = monomials(range(-B, B + 1))
    bad = check_skew(small)
    print()
    print("=== (cocycR1) of the text on spinor-oscillator representations, in the basis (73) ===")
    print("   super-skew-symmetry:", "OK" if not bad else "%d violations" % len(bad))
    for (s, k, kz) in good:
        f = check_cocycle(small, sigma=s, k=k, kz=kz)
        tot = sum(len(v) for v in f.values())
        print("   sigma=%s k=%s kz=%s : %d failing triples (window |a| <= %d, "
              "%d monomials)" % (s, k, kz, tot, B, len(small)))
        for sig in sorted(f, key=lambda x: -len(f[x]))[:6]:
            ex = f[sig][0]
            print("      %-22s %5d  e.g. %s %s %s -> %s"
                  % ("/".join(sig), len(f[sig]), name_of(ex[0])[:2],
                     name_of(ex[1])[:2], name_of(ex[2])[:2], ex[3]))


if __name__ == "__main__":
    check_basis()
    check_table(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
