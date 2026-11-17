"""The retract (3,3) over Spec Lambda(tau): the two families of Statement 9.1, 2).

H^1(T_1) = V (x) W with basis
    v0 = x^{-1} xi_1 d/dx,      v1 = x^{-1} xi_2 d/dx,
    v2 = x^{-1} xi_1xi_2 d/dxi_1,  v3 = x^{-1} xi_1xi_2 d/dxi_2,
the sl(2)-pairs being (v0, v3) and (v1, -v2), so that the class
c_0v_0 + c_1v_1 + c_2v_2 + c_3v_3 is the 2 x 2 matrix

        ( c_0   c_1 )
        ( c_3  -c_2 )        of determinant  -(c_0c_2 + c_1c_3) =: -Q,

and the orbits of SL(V) x GL(W) on P(V (x) W) are the two strata Q = 0 (rank 1)
and Q =/= 0 (rank 2).  The test: sdim_C v(M, O) over the base.
"""

from fractions import Fraction
import oddpar as O
import ns
from ns import SM, g_from, g_mul, g_add, g_zero, lp
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank

DEG = 10
TAU = 3                       # the odd parameter is the index m+1 = 3


def Qform(c):
    return c[0] * c[2] + c[1] * c[3]


def family(c):
    """The family over Lambda(tau) given by the class c in the basis above."""
    tau = g_from((TAU,), lp((0, 1)))
    xi1, xi2 = g_from((1,), lp((0, 1))), g_from((2,), lp((0, 1)))
    xm = lp((-1, 1))
    zx = g_zero()
    if c[0]:
        zx = g_add(zx, g_mul(tau, g_from((1,), lp((-1, c[0])))))
    if c[1]:
        zx = g_add(zx, g_mul(tau, g_from((2,), lp((-1, c[1])))))
    zxi = {}
    for j, cc in ((1, c[2]), (2, c[3])):
        if cc:
            zxi[j] = g_mul(tau, g_mul(g_from((), lp((-1, cc))),
                                      g_mul(xi1, xi2)))
    return SM.from_cocycle([3, 3], zx, zxi, npar=1)


def main():
    print(__doc__)
    L = O.Layer([3, 3], 1)

    print("A. Q is a semi-invariant: the sl(2) generators annihilate it, the")
    print("   gl(W) ones multiply it by a scalar (its weight).")
    for nm, _ in L.gens:
        M = L.mat(nm)
        # the derivative of Q along X, as a quadratic form in c
        # Q = c0c2 + c1c3; X.c has components sum_j M[i][j] c_j
        # dQ(X) = (X.c)_0 c_2 + c_0 (X.c)_2 + (X.c)_1 c_3 + c_1 (X.c)_3
        rows = {}
        pairs = [(0, 2), (2, 0), (1, 3), (3, 1)]
        for i, k in pairs:
            for j in range(4):
                if M[i][j]:
                    key = tuple(sorted((j, k)))
                    rows[key] = rows.get(key, Fraction(0)) + M[i][j]
        rows = {k: v for k, v in rows.items() if v}
        lam = rows.pop((0, 2), Fraction(0))
        ok = (not rows) and lam == rows.get((1, 3), lam)
        # recompute cleanly: dQ must be a multiple of Q itself
        rows2 = {}
        for i, k in pairs:
            for j in range(4):
                if M[i][j]:
                    key = tuple(sorted((j, k)))
                    rows2[key] = rows2.get(key, Fraction(0)) + M[i][j]
        rows2 = {k: v for k, v in rows2.items() if v}
        a, b = rows2.get((0, 2), Fraction(0)), rows2.get((1, 3), Fraction(0))
        rest = {k: v for k, v in rows2.items() if k not in ((0, 2), (1, 3))}
        ok = (a == b) and not rest
        print(f"   {nm:>18}:  X.Q = {a} * Q   {'OK' if ok else '<-- NOT'}"
              f"{'   (extra terms ' + str(rest) + ')' if rest else ''}")

    print()
    print("B. the orbit dimensions in P(H^1) = P^3")
    for c, tag in (([1, 0, 0, 0], "rank 1"), ([0, 1, 0, 0], "rank 1"),
                   ([1, 1, 0, 0], "rank 1"), ([1, 0, 1, 0], "rank 2"),
                   ([0, 1, 0, 1], "rank 2"), ([1, 1, 1, 1], "rank 2"),
                   ([2, 3, 5, 7], "rank 2")):
        v = [Fraction(x) for x in c]
        d = L.orbit_dim_proj(v)
        print(f"   c = {str(c):>14}  Q = {str(Qform(v)):>4}  {tag:>6}"
              f"   dim orbit in P^3 = {d}")

    print()
    print("C. the families themselves: sdim_C v(M, O) over Spec Lambda(tau)")
    sp = SM.split([3, 3], npar=1)
    d0 = sp.sdim_vect(DEG)
    print(f"   {'class':>16} {'Q':>5} {'rank':>5} {'sdim_C v':>12}  homogeneous?")
    print(f"   {'0 (split)':>16} {'-':>5} {'0':>5} {str(d0):>12}"
          f"  {sp.transitive(DEG)}")
    for c in ([1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0],
              [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1], [2, 3, 5, 7],
              [1, 0, 0, 1], [1, 2, 0, 0]):
        v = [Fraction(x) for x in c]
        q = Qform(v)
        M = family(c)
        d = M.sdim_vect(DEG)
        print(f"   {str(c):>16} {str(q):>5} {1 if q == 0 else 2:>5} {str(d):>12}"
              f"  {M.transitive(DEG)}")


if __name__ == "__main__":
    main()
