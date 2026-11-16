#!/usr/bin/env python3
"""
n2cons.py -- the N=2 super stress tensor is a conserved current.

n2.py determines T up to a scale from covariance alone.  Here the same three
coefficients are determined a SECOND time, and independently, from the equation
itself: T must be annihilated by the derivations of the OTHER half,

    d_- T = 0    modulo the system (IK2),

which is what makes it a current of the "+" half.  The two determinations agree,
which is the N=2 counterpart of the check that the two sectors of Statement
Theorem 4.2.1 return one and the same value.

Set-up.  In the "+" half put

    L := Dd^+ v,   Lb := Db^+ v+,   P := d_+ v,   Pb := d_+ v+,

in the "-" half put  M := Db^- v,  Mb := Dd^- v+,  and write E := exp(-v),
Eb := exp(-v+).  The constraints of (IK2) are Dd^- v = Db^+ v = 0 and their
conjugates Db^- v+ = Dd^+ v+ = 0, so those six objects exhaust the first
derivatives.  The dynamical half of (IK2) and its conjugate read

    Db^- Dd^+ v = mu1 Eb,          Dd^- Db^+ v+ = mu2 E,

with mu1 = -2im in \\cite{IvKr}; mu2 is left free, so that nothing below depends
on the normalization of the conjugate equation.  Everything then follows:

    Dd^- L = 0,   Db^- L = mu1 Eb,   Db^- Lb = 0,   Dd^- Lb = mu2 E,
    Dd^+ M = -mu1 Eb,               Db^+ Mb = -mu2 E,
    Dd^+ E = -L E,  Db^- E = -M E,  Db^+ Eb = -Lb Eb,  Dd^- Eb = -Mb Eb,

and, since  d_- = -1/4 (Dd^- Db^- + Db^- Dd^-),

    d_- L = mu1/4 Mb Eb,            d_- Lb = mu2/4 M E,
    d_- P = -1/4 Db^+ (d_- L),      d_- Pb = -1/4 Dd^+ (d_- Lb),

the last two because  P = d_+ v = -1/4 Db^+ Dd^+ v = -1/4 Db^+ L  and likewise
for Pb, and because d_- commutes with the derivations of the other half.

Usage:  ../venv/bin/python n2cons.py     (needs sympy for the solve)
"""

from fractions import Fraction as Q
from n2 import Alg, Elt, deriv


def build():
    A = Alg()
    L = A.gen("L", 1)
    Lb = A.gen("Lb", 1)
    M = A.gen("M", 1)
    Mb = A.gen("Mb", 1)
    E = A.gen("E", 0)
    Eb = A.gen("Eb", 0)
    return A, dict(L=L, Lb=Lb, M=M, Mb=Mb, E=E, Eb=Eb)


A, g = build()
e = lambda k: Elt.g(A, g[k])
zero = Elt(A)


def rules(mu1, mu2):
    """the four odd derivations, as dicts generator -> image."""
    Dp = {                                   # Dd^+
        g["L"]: zero,
        g["Lb"]: zero,                       # = -4 Pb, not needed below
        g["M"]: e("Eb").scal(-mu1),
        g["Mb"]: zero,
        g["E"]: (e("L") * e("E")).scal(-1),
        g["Eb"]: zero,
    }
    Dbp = {                                  # Db^+
        g["L"]: zero,                        # = -4 P, not needed below
        g["Lb"]: zero,
        g["M"]: zero,
        g["Mb"]: e("E").scal(-mu2),
        g["E"]: zero,
        g["Eb"]: (e("Lb") * e("Eb")).scal(-1),
    }
    return Dp, Dbp


def check(mu1, mu2, alpha, beta, gamma):
    """return d_- T for T = alpha L Lb + beta P + gamma Pb."""
    Dp, Dbp = rules(mu1, mu2)
    Dd = lambda x: deriv(A, Dp, 1, x)
    Db = lambda x: deriv(A, Dbp, 1, x)

    dL = (e("Mb") * e("Eb")).scal(Q(mu1) / 4)          # d_- L
    dLb = (e("M") * e("E")).scal(Q(mu2) / 4)           # d_- Lb
    dP = Db(dL).scal(Q(-1, 4))                         # d_- P
    dPb = Dd(dLb).scal(Q(-1, 4))                       # d_- Pb

    dT = (dL * e("Lb") + e("L") * dLb).scal(alpha) + dP.scal(beta) + dPb.scal(gamma)
    return dT, dict(dL=dL, dLb=dLb, dP=dP, dPb=dPb)


if __name__ == "__main__":
    import sympy as sp

    print("=" * 74)
    print("conservation: d_- T = 0 modulo (IK2)")
    print("=" * 74)

    # the four ingredients, printed once with symbolic mu's replaced by 1
    _, ing = check(1, 1, 1, 1, 1)
    print("   with mu1 = mu2 = 1:")
    for k in ("dL", "dLb", "dP", "dPb"):
        print(f"      d_- {k[2:]:<3} = {ing[k]}")

    # solve for alpha : beta : gamma over several (mu1, mu2)
    al, be, ga = sp.symbols("alpha beta gamma")
    print()
    for (mu1, mu2) in ((1, 1), (2, -3), (-2, 5), (7, 7)):
        parts = []
        for co, trip in ((al, (1, 0, 0)), (be, (0, 1, 0)), (ga, (0, 0, 1))):
            dT, _ = check(mu1, mu2, *trip)
            parts.append((co, dT))
        eqs = {}
        for co, dT in parts:
            for m, c in dT.d.items():
                eqs[m] = eqs.get(m, 0) + co * sp.Rational(c.numerator, c.denominator)
        sol = sp.solve([sp.expand(v) for v in eqs.values() if v != 0],
                       [al, be, ga], dict=True)
        print(f"   mu1 = {mu1:>3}, mu2 = {mu2:>3}:  d_- T = 0  iff  {sol}")

    print()
    print("   i.e. beta = 4 alpha and gamma = -4 alpha, for every mu1, mu2 --")
    print("   exactly the ratios covariance gives in n2.py, where the solution is")
    print("   alpha = -kappa2/4, beta = -kappa2, gamma = kappa2.")
    print()
    # and the explicit verification at those ratios
    nf = 0
    for (mu1, mu2) in ((1, 1), (2, -3), (-2, 5), (7, 7), (1, -1), (3, 4)):
        for a in (1, 2, -5):
            dT, _ = check(mu1, mu2, a, 4 * a, -4 * a)
            assert dT.is_zero(), (mu1, mu2, a, dT)
            nf += 1
    print(f"   d_- T = 0 verified identically               {nf} checks   OK")
