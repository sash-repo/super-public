"""m = 3: the first case in which the group is NOT abelian, and the
Bunegina--Onishchik series (2,2,k) with one odd parameter.

For m = 3 the layers are T_1, T_2, T_3 (T_4 = 0, Lambda^4 E = 0), and over
Spec Lambda(tau) the Lie superalgebra of the group is

    T_1 tau + T_2 + T_3 tau,    [T_1 tau, T_2] subset T_3 tau =/= 0,

so, unlike m <= 2, it is not abelian: an odd class and an even one have
a~bracket, which is a~class of T_3.  The even class of [BO, (41)] is taken as
the even one.
"""

from fractions import Fraction
import oddpar as O
import famtools as F
import brack
import quot
import sl2inv as S
import ns
from ns import SM, g_from, g_mul, g_add, g_zero, lp

DEG = 8


def bo_cocycle(ks, i=1, j=2, r=3):
    """[BO, (41)]: z = x^{-1} xi_i xi_j d/dx + (k_r/2) x^{-2} xi_i xi_j xi_r d/dxi_r."""
    I = tuple(sorted((i, j)))
    out = {(("u", I), -1): Fraction(1)}
    kr = ks[r - 1]
    if kr:
        out[(("w", tuple(sorted(I + (r,))), r), -2)] = Fraction(kr, 2)
    return out


def bracket_mixed(m, coc_odd, coc_even):
    """[z_odd, z_even] = z_odd z_even - z_even z_odd, the even one being even."""
    X = S.coc_to_field(m, coc_odd)
    Y = S.coc_to_field(m, coc_even)
    return S.field_to_coc(m, 3, S.bracket_even(m, Y, X))


def main():
    print(__doc__, flush=True)
    print("A. the layers of the Bunegina--Onishchik series")
    print(f"{'retract':>11} {'dim H^1(T_1)':>13} {'dim H^1(T_2)':>13}"
          f" {'dim H^1(T_3)':>13}", flush=True)
    for k in [0, 1, 2, 3, 4, 5]:
        ks = [2, 2, k]
        d = []
        for q in (1, 2, 3):
            d.append(quot.Quot(ks, q).dim)
        print(f"{str(tuple(ks)):>11} {d[0]:>13} {d[1]:>13} {d[2]:>13}",
              flush=True)

    print()
    print("B. the bracket of an odd class with the even class of [BO, (41)],")
    print("   as a class of H^1(T_3) -- zero for m <= 2, where T_3 = 0")
    for k in [2, 3, 4]:
        ks = [2, 2, k]
        Q1, Q3 = quot.Quot(ks, 1), quot.Quot(ks, 3)
        ze = bo_cocycle(ks)
        print(f"\n   retract {tuple(ks)}:  dim H^1(T_3) = {Q3.dim};"
              f"  the even class z = {F.show(Q1, [])or''}"
              f"x^(-1)xi_1xi_2 d/dx + {Fraction(k,2)} x^(-2)xi_1xi_2xi_3 d/dxi_3",
              flush=True)
        for i in range(Q1.dim):
            one = {Q1.basis[i]: Fraction(1)}
            br = bracket_mixed(3, one, ze)
            try:
                cl = Q3.coords(br) if br else [Fraction(0)] * Q3.dim
                cl = [str(c) for c in cl]
            except RuntimeError as ex:
                cl = f"<{ex}>"
            nz = any(c != "0" for c in cl) if isinstance(cl, list) else True
            print(f"      [{F.show(Q1, [1 if j == i else 0 for j in range(Q1.dim)]):>34}"
                  f", z] = {str(cl):>28} {'  <-- non-zero' if nz else ''}",
                  flush=True)

    print()
    print("C. the purely odd families over (2,2,k): sdim_C v and homogeneity")
    for k in [0, 2, 3]:
        ks = [2, 2, k]
        L = O.Layer(ks, 1)
        sp = SM.split(ks, npar=1)
        print(f"\n   retract {tuple(ks)}: dim H^1(T_1) = {L.dim};"
              f" split family sdim_C v = {sp.sdim_vect(DEG)},"
              f" homog {sp.transitive(DEG)}", flush=True)
        seen = {}
        cls = [[1 if j == i else 0 for j in range(L.dim)] for i in range(L.dim)]
        cls.append([1] * L.dim)
        cls.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:L.dim])
        for v in cls:
            vv = [Fraction(c) for c in v]
            M = F.cocycle_to_family(ks, F.vec_to_coc(L.Q, vv))
            d, h = M.sdim_vect(DEG), M.transitive(DEG)
            od = L.orbit_dim_proj(vv)
            seen.setdefault((od, d, h), []).append(v)
        for (od, d, h), vs in sorted(seen.items()):
            print(f"      orbit dim {od:>2}  sdim_C v = {str(d):>9}"
                  f"  homogeneous {str(h):>5}   e.g. "
                  f"{F.show(L.Q, [Fraction(c) for c in vs[0]])}", flush=True)


if __name__ == "__main__":
    main()
