"""
The weight PAIR on the contact superstring S_c^{1|2}.

Everything below is exact computation in the Grassmann algebra over Q(t, params)
of lfgrp.py (the C-points of OSp(2|2) acting on S_c^{1|2} by eq. (105) of the text).

What is checked:

 (1)  U := (D_i(th~_j))  satisfies  U^t U = m_G 1_2                    [known]
 (2)  in the ISOTROPIC basis  xi = (th_1 + i th_2)/sqrt2,
      eta = (th_1 - i th_2)/sqrt2,  the matrix  V = P^{-1} U P  of the same
      transformation is DIAGONAL, V = diag(mu_G, nu_G),  and  mu_G nu_G = m_G;
      so a transformation with O in SO(2) has TWO multipliers, not one.
 (3)  for the pure rotation (t fixed, th -> O th) one has m_G = 1 but
      mu_G = e^{i psi} != 1 : the pair is strictly finer than m_G alone.
 (4)  for the REFLECTION O = diag(1,-1) the matrix V is ANTIdiagonal:
      the disconnected part of O(2) swaps mu_G and nu_G, i.e. swaps the two
      components of the weight.  Hence the pair is a weight of the connected
      group only, and the full group permutes (k_1,k_2).
 (5)  the essential super point: with an EVEN NILPOTENT rotation angle
      eps = c*al_1*be_1 (which exists over any Grassmann ring A over Z, and is
      invisible over the C-points), one gets  m_G unchanged  but
      mu_G/nu_G = 1 + 2 i eps != 1.  So over A the two multipliers are
      independent even when the underlying group is the arithmetic one.
"""
import sympy as sp
from lfgrp import G, const, gen, Map, power, mat_det, mat_inv, t, osp_point

I = sp.I


def U_matrix(M):
    """U_ij = D_i(tilde theta_j)."""
    n = len(M.ths)
    return [[M.D(i, M.ph[j]) for j in range(n)] for i in range(n)]


def conj_iso(U):
    """V = P^{-1} U P for P = [[1,1],[i,-i]] (columns = isotropic directions).

    P^{-1} = 1/2 * [[1, -i],[1, i]].
    """
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


def osp_point_O(O):
    """as lfgrp.osp_point(2), but with an arbitrary orthogonal O (2x2 of G)."""
    A, B, C = sp.symbols('A B C')
    ths, als, bes = ['th1', 'th2'], ['al1', 'al2'], ['be1', 'be2']
    G.order = als + bes + ths
    al, be = [gen(a) for a in als], [gen(b) for b in bes]
    Delta = const(1) + al[0] * be[0] + al[1] * be[1]
    Dinv = Delta.inv()
    Dg = (Delta + const(B * C)) * const(1 / A)
    E = [row[:] for row in O]
    for _ in range(4):
        ga = [sum(((const(B) * al[i] - const(A) * be[i]) * E[i][j] for i in range(2)),
                  const(0)) * Dinv for j in range(2)]
        de = [sum(((Dg * al[i] - const(C) * be[i]) * E[i][j] for i in range(2)),
                  const(0)) * Dinv for j in range(2)]
        N = [[ga[i] * de[j] - de[i] * ga[j] for j in range(2)] for i in range(2)]
        S = [[const(1 if i == j else 0) for j in range(2)] for i in range(2)]
        Npow = [[const(1 if i == j else 0) for j in range(2)] for i in range(2)]
        for k in range(1, 8):
            Npow = [[sum((Npow[i][l] * N[l][j] for l in range(2)), const(0))
                     for j in range(2)] for i in range(2)]
            if all(not Npow[i][j].d for i in range(2) for j in range(2)):
                break
            cf = sp.binomial(sp.Rational(1, 2), k) * (-1) ** k
            for i in range(2):
                for j in range(2):
                    S[i][j] = S[i][j] + Npow[i][j] * cf
        E = [[sum((O[i][k] * S[k][j] for k in range(2)), const(0)) for j in range(2)]
             for i in range(2)]
    ok_E = all(((sum((E[k][i] * E[k][j] for k in range(2)), const(0))
                 - (const(1 if i == j else 0) - N[i][j])).is_zero())
               for i in range(2) for j in range(2))
    den = const(C) * const(t) + Dg + de[0] * gen(ths[0]) + de[1] * gen(ths[1])
    dinv = den.inv()
    f = (const(A) * const(t) + const(B) + ga[0] * gen(ths[0]) + ga[1] * gen(ths[1])) * dinv
    ph = [(al[i] * const(t) + be[i]
           + E[i][0] * gen(ths[0]) + E[i][1] * gen(ths[1])) * dinv for i in range(2)]
    return Map(f, ph, ths), den, ok_E


def report(tag, O, expect_diag=True):
    M, den, ok_E = osp_point_O(O)
    m, ok_D = M.multiplier_D()
    U = U_matrix(M)
    V = conj_iso(U)
    off = (V[0][1].is_zero() and V[1][0].is_zero())
    dia = (V[0][0].is_zero() and V[1][1].is_zero())
    mu, nu = V[0][0], V[1][1]
    print("--- %s" % tag)
    print("    E^t E = 1 - N                 :", "OK" if ok_E else "FAILED")
    print("    U^t U = m 1_2                 :", "OK" if ok_D else "FAILED")
    print("    V = P^-1 U P  diagonal        :", "YES" if off else "no (antidiagonal: %s)"
          % ("yes" if dia else "neither"))
    if off:
        print("    mu nu = m                     :",
              "OK" if (mu * nu - m).is_zero() else "FAILED")
        print("    mu = ", sp.simplify(mu.d.get((), 0)) if mu.d else 0)
        print("    nu = ", sp.simplify(nu.d.get((), 0)) if nu.d else 0)
        r = mu * nu.inv()
        print("    mu/nu = ", sp.simplify(r.d.get((), 0)) if r.d else 0,
              "   (other Grassmann components: %s)"
              % ({k: sp.simplify(v) for k, v in r.d.items() if k != ()} or "none"))
    else:
        print("    V[0][1], V[1][0] =",
              sp.simplify(V[0][1].d.get((), 0)) if V[0][1].d else 0,
              sp.simplify(V[1][0].d.get((), 0)) if V[1][0].d else 0)
    return m


if __name__ == "__main__":
    one = [[const(1), const(0)], [const(0), const(1)]]
    report("O = 1_2  (general C-point of the connected OSp(2|2))", one)

    ps = sp.Symbol('psi')
    rot = [[const(sp.cos(ps)), const(-sp.sin(ps))],
           [const(sp.sin(ps)), const(sp.cos(ps))]]
    report("O = rotation by psi", rot)

    refl = [[const(1), const(0)], [const(0), const(-1)]]
    report("O = diag(1,-1)  (the disconnected component)", refl)

    # a nilpotent rotation: exists over any Grassmann ring, invisible over C
    G.order = ['al1', 'al2', 'be1', 'be2', 'th1', 'th2']
    c = sp.Symbol('c')
    eps = gen('al1') * gen('be1') * const(c)          # even, eps^2 = 0
    nilrot = [[const(1), eps], [const(0) - eps, const(1)]]
    report("O = 1 + eps J,  eps = c*al_1*be_1 nilpotent (an A-point)", nilrot)
