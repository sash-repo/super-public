"""Two odd parameters: the bracket of two odd classes is an even one.

Over Spec C with dim C_od = 1 the group of even automorphisms of the retract
that are the identity modulo the filtration is ABELIAN for m <= 2, because

    [T_1 tau, T_1 tau] = 0   (tau^2 = 0),
    [T_1 tau, T_2] subset T_3 tau = 0   (m <= 2),
    [T_2, T_2] subset T_4 = 0   (m <= 3),

and hence H^1 of it is the direct sum H^1(T_1) + H^1(T_2).  With two odd
parameters the first of these fails: tau_1 tau_2 =/= 0, so

    [tau_1 z_1, tau_2 z_2] = tau_1 tau_2 [z_1, z_2]  in  T_2 (x) C_ev,

and THAT is where the even obstructions of [LT] come from.  This module
computes the bracket of two odd degree-1 derivations and the class of the
result in H^1(T_2).
"""

from fractions import Fraction
import ns
import bashkin as B
import sl2inv as S
import quot


def bracket_odd(m, X, Y):
    """[X, Y] = XY + YX for two ODD vector fields."""
    A = ns.g_add(ns._apply(X[0], X[1], Y[0]), ns._apply(Y[0], Y[1], X[0]))
    Bs = [ns.g_add(ns._apply(X[0], X[1], Y[1][j]),
                   ns._apply(Y[0], Y[1], X[1][j])) for j in range(m)]
    return (A, Bs)


def bracket_classes(ks, coc1, coc2):
    """[z_1, z_2] as a cocycle of T_2, from two cocycles of T_1."""
    m = len(ks)
    X = S.coc_to_field(m, coc1)
    Y = S.coc_to_field(m, coc2)
    Z = bracket_odd(m, X, Y)
    return S.field_to_coc(m, 2, Z)


def main():
    print(__doc__)
    ks = [3, 3]
    Q1 = quot.Quot(ks, 1)
    Q2 = quot.Quot(ks, 2)
    print(f"retract {tuple(ks)}:  dim H^1(T_1) = {Q1.dim},"
          f"  dim H^1(T_2) = {Q2.dim}")
    print("   (the latter is the P^{a+b-4} = P^2 of [LT])")
    print(f"\n   basis of H^1(T_1): {Q1.basis}")
    print(f"   basis of H^1(T_2): {Q2.basis}\n")
    one = lambda i: {Q1.basis[i]: Fraction(1)}
    names = ["v0 = x^-1 xi_1 d/dx", "v1 = x^-1 xi_2 d/dx",
             "v2 = x^-1 xi_1xi_2 d/dxi_1", "v3 = x^-1 xi_1xi_2 d/dxi_2"]
    print(f"{'[z1, z2]':>28}  {'the bracket in T_2':>44}  class in H^1(T_2)")
    print("-" * 104)
    for i in range(4):
        for j in range(i, 4):
            br = bracket_classes(ks, one(i), one(j))
            txt = " + ".join(
                f"{c} x^({e}) " + (f"xi_{''.join(map(str,g[1]))} d/dx"
                                   if g[0] == "u" else
                                   f"xi_{''.join(map(str,g[1]))} d/dxi_{g[2]}")
                for (g, e), c in sorted(br.items(), key=str)) or "0"
            try:
                cl = Q2.coords(br) if br else [Fraction(0)] * Q2.dim
            except RuntimeError as ex:
                cl = f"<{ex}>"
            print(f"{'[' + names[i].split(' =')[0] + ', '
                      + names[j].split(' =')[0] + ']':>28}"
                  f"  {txt:>44}  {cl}")

    print("\nWhence: the even obstruction of the retract (3,3) -- the P^2 of [LT] --")
    print("is spanned by brackets of the odd ones; e.g. [v0, v2] is the class of")
    print("x^(-2) xi_1xi_2 d/dx, a basis vector of H^1(T_2).")


if __name__ == "__main__":
    main()
