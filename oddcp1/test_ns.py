"""Self-tests for ns.py: the Grassmann arithmetic, and the split models, where
the answer is known from the cohomology of line bundles over CP^1."""

from fractions import Fraction
from ns import (SM, g_from, g_mul, g_add, g_dxi, g_dx, lp, sdim_split_predicted,
                _apply, _pow_nilpotent, h0, h1, _pdet, _pgcd, _pmul)

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}: {got}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: got {got}, want {want}")


# ---------------------------------------------------------------- 1. Grassmann
x1 = g_from((1,), lp((0, 1)))
x2 = g_from((2,), lp((0, 1)))
check("xi1 xi2 = xi_{12}", g_mul(x1, x2), {(1, 2): {0: Fraction(1)}})
check("xi2 xi1 = - xi_{12}", g_mul(x2, x1), {(1, 2): {0: Fraction(-1)}})
check("xi1^2 = 0", g_mul(x1, x1), {})
check("d/dxi1 (xi1 xi2) = xi2", g_dxi(g_mul(x1, x2), 1), {(2,): {0: Fraction(1)}})
check("d/dxi2 (xi1 xi2) = -xi1", g_dxi(g_mul(x1, x2), 2), {(1,): {0: Fraction(-1)}})
check("d/dx x^3 = 3x^2", g_dx(g_from((), lp((3, 1)))), {(): {2: Fraction(3)}})

# (x + c xi1 xi2)^{-1} = x^{-1} - c x^{-2} xi1 xi2
nil = g_mul(g_from((), lp((-1, 1))), g_mul(x1, x2))       # x^{-1} xi1 xi2
elem = g_add(g_from((), lp((1, 1))), nil)
inv = _pow_nilpotent(elem, -1)
check("(x + x^{-1}xi1xi2)^{-1}", inv,
      {(): {-1: Fraction(1)}, (1, 2): {-3: Fraction(-1)}})
check("it really is the inverse", g_mul(inv, elem), {(): {0: Fraction(1)}})

# ------------------------------------------------- 2. B--O's own formula (47)
# cocycle z = x^{-1} xi1 xi2 d/dx + (1/2) k x^{-2} xi1xi2xi3 d/dxi3, k3 = k
for k in (0, 2, 3, 5):
    ks = [2, 2, k]
    zx = g_mul(g_from((), lp((-1, 1))), g_mul(x1, x2))
    d123 = g_mul(g_mul(x1, x2), g_from((3,), lp((0, 1))))
    zxi = {3: g_mul(g_from((), lp((-2, Fraction(k, 2)))), d123)}
    M = SM.from_cocycle(ks, zx, zxi)
    want_Y = {(): {-1: Fraction(1)}, (1, 2): {-3: Fraction(-1)}}
    want_H3 = {(3,): {-k: Fraction(1)}}
    if k:
        want_H3[(1, 2, 3)] = {-k - 2: Fraction(-k, 2)}
    check(f"(47) y for k={k}", M.Y, want_Y)
    check(f"(47) eta_1 for k={k}", M.H[0], {(1,): {-2: Fraction(1)}})
    check(f"(47) eta_3 for k={k}", M.H[2], want_H3)

# ------------------------------------------- 3. split models: dim v(M, Lambda E)
for ks in ([2, 2], [3, 1], [4, 0], [1, 1], [0, 0], [0, 0, 0], [2, 2, 0],
           [2, 2, 2], [3, 1, 0], [2, 2, 5], [2, 2, 1], [5, 3, 2], [1, 1, 1]):
    M = SM.split(ks)
    got = M.sdim_vect(degU=10)
    want = sdim_split_predicted(ks)
    check(f"sdim v(split{tuple(ks)})", got, want)

# ---------------------------------------------- 4. the split ones are transitive
for ks in ([2, 2], [3, 1], [4, 0], [2, 2, 0], [2, 2, 1]):
    check(f"split{tuple(ks)} homogeneous", SM.split(ks).transitive(degU=10), True)

# a bundle with a positive summand (k_j < 0) must NOT be homogeneous (B--O Prop 13)
check("split(2,-1) homogeneous", SM.split([2, -1]).transitive(degU=10), False)

# ------------------------------------------------------- 5. stability in degU
M = SM.from_cocycle([2, 2], g_mul(g_from((), lp((-1, 1))), g_mul(x1, x2)))
check("stability of sdim in degU", M.sdim_vect(degU=8), M.sdim_vect(degU=14))

# ------------------------------------------------ 6. the polynomial gcd machinery
one, X = [Fraction(1)], [Fraction(0), Fraction(1)]
check("det [[x,1],[1,x]] = x^2-1", _pdet([[X, one], [one, X]]),
      [Fraction(-1), Fraction(0), Fraction(1)])
check("det [[x,1,0],[0,x,1],[1,0,x]] = x^3+1",
      _pdet([[X, one, []], [[], X, one], [one, [], X]]),
      [Fraction(1), Fraction(0), Fraction(0), Fraction(1)])
check("gcd(x^2-1, x-1) = x-1", _pgcd([Fraction(-1), Fraction(0), Fraction(1)],
                                     [Fraction(-1), Fraction(1)]),
      [Fraction(-1), Fraction(1)])
check("gcd(x, 1) = 1", _pgcd(X, one), one)
check("x(x+1)", _pmul(X, [Fraction(1), Fraction(1)]),
      [Fraction(0), Fraction(1), Fraction(1)])

print(f"\n{PASS} passed, {FAIL} failed")
