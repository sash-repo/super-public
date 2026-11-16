"""
Where the second component of the weight on S_c^{1|2} lives.

pair2.py established the two multipliers mu_G, nu_G of eq. (123) for a general
C-point of OSp(2|2).  The question is what is NEW for a modular
form, i.e., what happens on the ARITHMETIC group, and the answer turns on one
quantity alone: the ratio  r_G := mu_G nu_G^{-1}.  Here it is computed for the
C-points of the group -- all the odd parameters al_i, be_i set to 0, so that
the transformation is an honest element of SL(2;Z) x O(2;Z) -- which pair2.py
did not do.

What is checked (exact arithmetic in the Grassmann algebra over Q(t,A,B,C)):

 (1) such a map is contact and superconformal, U^t U = m 1_2, mu nu = m,
     and V = P^{-1} U P is diagonal for O in SO(2), antidiagonal for a
     reflection  -- i.e. pair2.py's headings 1), 2) survive the specialisation;

 (2) for O = 1_2 the ratio is EXACTLY

         r_G = 1 - 2i (C / (Ct+D)) th_1 th_2,        D = (1+BC)/A,

     so its BODY is 1 while the element itself is NOT 1 as soon as C != 0:
     with no odd parameter whatsoever, the two multipliers already differ --
     by a nilpotent built from the odd COORDINATES of the superstring;

 (3) the body of r_G is exp(2 i psi), psi the rotation angle; over O(2;Z) the
     rotations are +-1, +-J, so the body takes the two values +-1 only;

 (4) hence on the underlying manifold (th = 0) the transformation law of a
     modular form of weight (k_1,k_2) is the classical one of weight k_1+k_2
     up to a sign: the difference k_1-k_2 is invisible there, and the whole of
     it sits in the th_1 th_2 - component, where it is proportional to k_1-k_2
     and hence separates the pairs.

Run: ../venv/bin/python pair994.py
"""
import sympy as sp

from lfgrp import G, const, gen, Map, t

I = sp.I
A, B, C, ps = sp.symbols('A B C psi')

CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print("    %-58s %s" % (name, "ok" if cond else "FAILED"))


def body(x):
    """the C-valued part (th = 0)."""
    return sp.simplify(sp.cancel(sp.together(x.d.get((), 0))))


def comp(x, key):
    return sp.simplify(sp.cancel(sp.together(x.d.get(key, 0))))


def cpoint(O):
    """the C-point  t -> (At+B)/(Ct+D),  th -> O th/(Ct+D),  D = (1+BC)/A."""
    G.order = ['th1', 'th2']
    ths = ['th1', 'th2']
    Dd = (1 + B * C) / A                      # AD - BC = 1, no odd parameters
    den = const(C * t + Dd)
    dinv = den.inv()
    f = const(A * t + B) * dinv
    ph = [sum((O[i][j] * gen(ths[j]) for j in range(2)), const(0)) * dinv
          for i in range(2)]
    return Map(f, ph, ths)


def U_matrix(M):
    return [[M.D(i, M.ph[j]) for j in range(2)] for i in range(2)]


def conj_iso(U):
    """V = P^{-1} U P,  P = [[1,1],[i,-i]]:  the isotropic basis xi, eta."""
    P = [[sp.Integer(1), sp.Integer(1)], [I, -I]]
    Pi = [[sp.Rational(1, 2), -I / 2], [sp.Rational(1, 2), I / 2]]
    V = [[const(0)] * 2 for _ in range(2)]
    for a in range(2):
        for b in range(2):
            s = const(0)
            for c in range(2):
                for d in range(2):
                    s = s + U[c][d] * const(Pi[a][c] * P[d][b])
            V[a][b] = s
    return V


def multipliers(O):
    M = cpoint(O)
    m, ok_m = M.multiplier_D()
    U = U_matrix(M)
    V = conj_iso(U)
    return M, m, ok_m, U, V


def run(tag, O, diagonal=True):
    print("--- %s" % tag)
    M, m, ok_m, U, V = multipliers(O)
    check("superconformal", M.superconformal())
    check("U^t U = m 1_2", ok_m)
    if diagonal:
        check("V diagonal (mu, nu on the isotropic basis)",
              V[0][1].is_zero() and V[1][0].is_zero())
        mu, nu = V[0][0], V[1][1]
        check("mu nu = m", (mu * nu - m).is_zero())
        return mu, nu, m
    check("V ANTIdiagonal (the reflection swaps mu and nu)",
          V[0][0].is_zero() and V[1][1].is_zero())
    return None, None, m


def main():
    one = [[const(1), const(0)], [const(0), const(1)]]

    # (1) + (2) ------------------------------------------------------- O = 1_2
    mu, nu, m = run("O = 1_2: a C-point of the arithmetic group", one)
    r = mu * nu.inv()
    Dd = (1 + B * C) / A
    predicted = -2 * I * C / (C * t + Dd)
    print("    mu/nu   =", r)
    check("body(mu/nu) = 1", body(r) == 1)
    check("mu/nu = 1 - 2i C/(Ct+D) th_1 th_2, exactly",
          sp.simplify(comp(r, ('th1', 'th2')) - predicted) == 0
          and all(k in ((), ('th1', 'th2')) for k in r.d))
    check("mu/nu != 1 whenever C != 0",
          sp.simplify(predicted.subs({A: 1, B: 0, C: 1})) != 0)
    check("mu/nu = 1 for C = 0 (the affine subgroup)",
          sp.simplify(predicted.subs({C: 0})) == 0)

    # the weight character  mu^{-k_1} nu^{-k_2}  separates the pairs
    print("--- the character  mu^{-k_1} nu^{-k_2}  of the C-point")
    sep = True
    for k1, k2 in ((1, 1), (2, 0), (3, -1), (0, 2)):
        ch = const(1)
        for _ in range(abs(k1)):
            ch = ch * (mu.inv() if k1 > 0 else mu)
        for _ in range(abs(k2)):
            ch = ch * (nu.inv() if k2 > 0 else nu)
        half = k1 + k2                       # m^{-(k_1+k_2)/2} has no th-part
        co = comp(ch, ('th1', 'th2')) / comp(ch * const(0) + ch, ()) if ch.d.get((), 0) else 0
        co = sp.simplify(sp.cancel(sp.together(co)))
        print("    (k1,k2) = (%2d,%2d):  th_1th_2-part / body = %s" % (k1, k2, co))
        want = sp.simplify(-(k1 - k2) / sp.Integer(2) * predicted)
        if sp.simplify(co - want) != 0:
            sep = False
        assert half == k1 + k2
    check("th_1th_2-part of the character is -(k_1-k_2)/2 times that of mu/nu",
          sep)

    # (3) --------------------------------------------------- O = rotation psi
    rot = [[const(sp.cos(ps)), const(-sp.sin(ps))],
           [const(sp.sin(ps)), const(sp.cos(ps))]]
    mu2, nu2, m2 = run("O = rotation by psi", rot)
    r2 = mu2 * nu2.inv()
    check("body(mu/nu) = exp(2 i psi)",
          sp.simplify(body(r2) - sp.exp(2 * I * ps)) == 0)
    vals = sorted({sp.simplify(sp.exp(2 * I * ps).subs(ps, sp.pi * k / 2))
                   for k in range(4)}, key=str)
    print("    the four rotations of O(2;Z) give body(mu/nu) in", vals)
    check("over O(2;Z) the body of mu/nu is +-1", set(vals) == {1, -1})

    # (4) ------------------------------------------- the reflection swaps them
    refl = [[const(1), const(0)], [const(0), const(-1)]]
    run("O = diag(1,-1): the disconnected component", refl, diagonal=False)

    bad = [n for n, okk in CHECKS if not okk]
    print("\n%d checks, %d failed%s" % (len(CHECKS), len(bad),
                                        (": " + "; ".join(bad)) if bad else ""))


if __name__ == "__main__":
    main()
