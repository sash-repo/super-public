"""The structural half of Statement stSignRule, over a wide range.

sigma(u) = d(P(u)) s(u) with d depending only on P(u) mod 2, so the two Gram
matrices differ by the diagonal D = diag(d(P(u))).  D is a SCALAR matrix -- and
only then is Sh_mu changed by a scalar rather than distorted -- exactly when
all PBW monomials of one weight subspace have one and the same P(u) mod 2.

That is a statement about the PBW basis alone: no bracket, no determinant.  So
it is checked here over a far wider range than the symbolic determinants of
signrule.py can reach, which is what lets the Statement name a range at all.
"""

import sympy

from verma import NS1, NS2, Vir, Module


def scan(name, alg, levels, charges):
    mod = Module(alg)
    worst = 0
    checked = 0
    bad = []
    for lev in levels:
        for chg in charges:
            b = mod.basis(lev, chg)
            if not b:
                continue
            pars = {sum(alg.gen(k).parity for k in u) % 2 for u in b}
            checked += 1
            worst = max(worst, len(b))
            if len(pars) != 1:
                bad.append((lev, chg, sorted(pars), len(b)))
    tag = "%s: %d weight subspaces, largest of dim %d" % (name, checked, worst)
    if bad:
        print("  %s -- MIXED PARITY in %d of them:" % (tag, len(bad)))
        for lev, chg, pars, d in bad:
            print("      level %s charge %s: parities %s, dim %d"
                  % (lev, chg, pars, d))
    else:
        print("  %s -- parity constant on every one of them" % tag)


def main():
    half = [sympy.Rational(n, 2) for n in range(1, 17)]
    print("Is P(u) mod 2 constant on each weight subspace?")
    scan("vir  , levels 1..8", Vir(10), list(range(1, 9)), [None])
    scan("ns(1), levels 1/2..8", NS1(10), half, [None])
    scan("ns(2), levels 1/2..6, charges -4..4",
         NS2(10, sympy.Symbol("c")),
         [sympy.Rational(n, 2) for n in range(1, 13)],
         list(range(-4, 5)))


if __name__ == "__main__":
    main()
