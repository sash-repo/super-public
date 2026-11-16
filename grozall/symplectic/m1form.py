"""Check the corrected m = 1 formula, written in Grozman's own notation, against
the universal formula.

Grozman (p. 34):  omega = dx ^ dy,  A = a dx^2 + 2b dxdy + c dy^2, similarly A',
   g       = ac' - 2bb' + ca',
   dd A    = a_yy - 2 b_xy + c_xx.
The claim tested here is

  Gz(A,A') = g_xx dx^2 + 2 g_xy dxdy + g_yy dy^2
           + (dd A)(a'dx^2 + 2b'dxdy + c'dy^2) + (dd A')(a dx^2 + 2b dxdy + c dy^2)
           + ({a,b'}-{b,a'}) dx^2 + ({a,c'}-{c,a'}) dxdy + ({b,c'}-{c,b'}) dy^2 ,

which differs from the printed one in the middle Poisson term ONLY (printed:
({c,a'}-{a,c'})dxdy), the other three differences being missing primes and
a~missing plus sign.
"""

import random
from fractions import Fraction

from symp import Symp, random_sym_field, padd, pscale, pmul, pdiff, pzero
from verify import build, SOLUTION

S = Symp(1)
rng = random.Random(12345)

A = random_sym_field(S, 4, rng)
B = random_sym_field(S, 4, rng)
a, b, c = A[0][0], A[0][1], A[1][1]
a1, b1, c1 = B[0][0], B[0][1], B[1][1]

GZ = build(S, SOLUTION)
T = GZ(A, B)
# our normalisation is half of Grozman's
T = [[pscale(T[i][j], 2) for j in range(2)] for i in range(2)]

X, Y = 0, 1
pb = S.poisson


def dd(p, q, r):          # a_yy - 2 b_xy + c_xx
    return padd(padd(pdiff(pdiff(p, Y), Y), pscale(pdiff(pdiff(q, X), Y), -2)),
                pdiff(pdiff(r, X), X))


g = padd(padd(pmul(a, c1), pscale(pmul(b, b1), -2)), pmul(c, a1))
ddA, ddB = dd(a, b, c), dd(a1, b1, c1)

claim = [[None, None], [None, None]]
claim[0][0] = padd(padd(pdiff(pdiff(g, X), X), padd(pmul(ddA, a1), pmul(ddB, a))),
                   padd(pb(a, b1), pscale(pb(b, a1), -1)))
claim[1][1] = padd(padd(pdiff(pdiff(g, Y), Y), padd(pmul(ddA, c1), pmul(ddB, c))),
                   padd(pb(b, c1), pscale(pb(c, b1), -1)))
# the dxdy coefficient is 2 * (the (1,2) entry of the tensor)
claim[0][1] = padd(padd(pdiff(pdiff(g, X), Y), padd(pmul(ddA, b1), pmul(ddB, b))),
                   pscale(padd(pb(a, c1), pscale(pb(c, a1), -1)), Fraction(1, 2)))
claim[1][0] = claim[0][1]

ok = all(not padd(T[i][j], pscale(claim[i][j], -1)) for i in range(2) for j in range(2))
print("corrected m=1 formula matches the universal one:", ok)

# and the printed sign of the dxdy Poisson term does NOT match
bad = padd(padd(pdiff(pdiff(g, X), Y), padd(pmul(ddA, b1), pmul(ddB, b))),
           pscale(padd(pb(c, a1), pscale(pb(a, c1), -1)), Fraction(1, 2)))
print("printed sign of the dxdy Poisson term matches:",
      not padd(T[0][1], pscale(bad, -1)))
