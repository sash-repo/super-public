"""E. Two supermanifolds with the same retract and the same class in
       H^1(CP^1; Theta (x) E^2(E)) -- the space that carries the degree-2
       obstructions in the form in which [LT] quote Palamodov -- but with
       different classes in H^1(M; T_2), the sheaf that actually classifies.

For m = 3 the sheaf T_2 is a non-split extension

        0 --> Lambda^3 E (x) E^*  -->  T_2  -->  Theta (x) Lambda^2 E --> 0,

and B--O's cocycle (41) has a non-zero component in the SUB.  So a class of the
sub is invisible in the quotient; if the corresponding supermanifolds are
non-isomorphic, the quotient alone cannot classify.  (For m <= 2 the sub is
zero and the two descriptions agree -- which is the case [LT] treat.)
"""

from fractions import Fraction
from ns import SM, g_from, g_mul, g_add, g_zero, lp

DEG = 10


def xi(j):
    return g_from((j,), lp((0, 1)))


def coef(e, c=1):
    return g_from((), lp((e, c)))


delta = g_mul(g_mul(xi(1), xi(2)), xi(3))
print(f"{'retract':>12} {'cocycle':>46} {'sdim v(M,O)':>13}  transitive")
print("-" * 90)

for ks in ([2, 2, 3], [2, 2, 5], [2, 2, 0]):
    k3 = ks[2]
    rows = [
        ("split (the retract itself)", g_zero(), {}),
        ("x^{-1} xi_1xi_2 d/dx   (the [LT] class alone)",
         g_mul(coef(-1), g_mul(xi(1), xi(2))), {}),
        (f"+ ({k3}/2) x^-2 delta d/dxi_3   (B--O's (41))",
         g_mul(coef(-1), g_mul(xi(1), xi(2))),
         {3: g_mul(coef(-2, Fraction(k3, 2)), delta)}),
        ("x^{-2} delta d/dxi_3 alone   (invisible in the quotient)",
         g_zero(), {3: g_mul(coef(-2), delta)}),
        ("x^{-1} delta d/dxi_3 alone   (ditto)",
         g_zero(), {3: g_mul(coef(-1), delta)}),
    ]
    for name, zx, zxi in rows:
        M = SM.split(ks) if (not zx and not zxi) else SM.from_cocycle(ks, zx, zxi)
        print(f"{str(tuple(ks)):>12} {name:>46} {str(M.sdim_vect(DEG)):>13}"
              f"  {M.transitive(DEG)}")
    print()
