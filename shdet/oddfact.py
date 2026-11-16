"""The linear factors of Sh_mu for svect^L_lambda(1|2), in closed form.

For an isotropic (odd) root gamma with root vectors e = e_gamma and
f = sigma(e), the weight subspace f v of the Verma module is 1-dimensional and

        (f v | f v) = chi([e, f]),          [e, f] in the torus t,

so the factor of Sh_mu that gamma contributes is the value of the highest
weight chi = (h, ell, kappa) on the toral element [e, f].  This is computed
here for EVERY odd generator at once, not just for the low levels of
``shsvect.py``, which is what turns the table of that engine into a formula.

The same is done for the even lowering generators, where [e, f] is again toral.
"""

import sympy

from svectlam import NAMES, PARITY, LAM, KAP, br
from shsvect import H, ELL, sigma, role, MIRROR

MU = (LAM + 1) / 2


def toral_value(terms, cen):
    """the value of chi = (h, ell, kappa) on a toral element."""
    out = cen
    for (name, m), co in terms.items():
        if m != 0:
            raise ValueError("not toral: %s" % ((name, m),))
        if name == "LL":
            out += co * H
        elif name == "T0":
            out += co * ELL
        else:
            raise ValueError("not toral: %s" % ((name, m),))
    return sympy.expand(out)


def factor_of(key):
    """chi([sigma(x), x]) for a lowering generator x."""
    e = sigma(key)
    t, c = br(e, key)
    return sympy.simplify(toral_value(t, c))


if __name__ == "__main__":
    log = open("run-oddfact.log", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")
        log.flush()

    say("=== chi([sigma(x), x]) for the lowering generators x, in closed form ===")
    say("    mu = (lambda+1)/2;  chi = (h, ell, kappa)\n")

    say("--- the odd (isotropic) ones ---")
    for name in ("P1", "P2", "S1", "S2"):
        for k in range(0, 4):
            key = (name, -k)
            if role(key) != "-":
                continue
            v = factor_of(key)
            say("   x = %-3s_%-2d  level %-8s  chi([sigma x, x]) = %s"
                % (name, -k, sympy.simplify(-(-k) + 0), sympy.factor(v)))

    say("\n--- the even ones ---")
    for name in ("LL", "T0", "Tp", "Tm"):
        for k in range(0, 4):
            key = (name, -k)
            if role(key) != "-":
                continue
            v = factor_of(key)
            say("   x = %-3s_%-2d  chi([sigma x, x]) = %s" % (name, -k, sympy.factor(v)))

    say("\n=== the closed forms, in terms of mu ===")
    k = sympy.Symbol("k", positive=True, integer=True)
    for name in ("P1", "P2", "S1", "S2"):
        vals = []
        for kk in range(0, 5):
            key = (name, -kk)
            if role(key) != "-":
                continue
            vals.append((kk, sympy.expand(factor_of(key))))
        if len(vals) < 3:
            continue
        #  fit  a*h + b*ell + q(k)*kappa  with q quadratic in k
        A = sympy.Symbol("A"), sympy.Symbol("B"), sympy.Symbol("C")
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        eqs = []
        for kk, v in vals:
            p = sympy.Poly(v, KAP)
            co = p.coeff_monomial(KAP)
            eqs.append(co - (c0 + c1 * kk + c2 * kk ** 2))
        sol = sympy.solve(eqs[:3], [c0, c1, c2], dict=True)
        if sol:
            s = sol[0]
            q = sympy.expand(s[c0] + s[c1] * k + s[c2] * k ** 2)
            ok = all(sympy.simplify(sympy.Poly(v, KAP).coeff_monomial(KAP)
                                    - q.subs(k, kk)) == 0 for kk, v in vals)
            hl = sympy.expand(vals[0][1].subs(KAP, 0))
            hl2 = sympy.expand(vals[1][1].subs(KAP, 0))
            say("   %-3s_-k : kappa-coefficient = %s   (fits all %d values: %s)"
                % (name, sympy.factor(q), len(vals), ok))
            say("           the (h, ell)-part is %s, independent of k: %s"
                % (sympy.factor(hl), sympy.simplify(hl - hl2) == 0))
    log.close()
