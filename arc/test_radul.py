"""
Calibration of radul.py against what is known independently of Radul's text:
the contact field, the frame of the contact distribution, the multiplier and
its cocycle identity, and the classical (n = 0) Schwarzian.
"""
import sympy as sp
from itertools import combinations
import radul as R
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t, d_th,
                   VF, Diffeo, Composite, multiplier, U_matrix, einv, epow,
                   K_f, D_op, K_op, flow, body0)

OK = BAD = 0


def check(name, cond):
    global OK, BAD
    if cond:
        OK += 1
        print("  ok   %s" % name)
    else:
        BAD += 1
        print("  FAIL %s" % name)
    return cond


def gen(n, seed=0, deg=3):
    """An EVEN generating function with every theta-component non-zero.

    A component with an odd number of theta's carries an odd constant (the
    extra Grassmann generators 101, 102, ...): otherwise K_g is not an even
    field and exp(e K_g) is not a diffeomorphism -- the odd directions of a
    supergroup are reached with odd parameters only.
    """
    cs = [sp.Rational(a, b) for a, b in
          [(1, 1), (2, 3), (-1, 2), (3, 2), (-2, 5), (1, 3), (4, 3), (-3, 4),
           (2, 7), (5, 2), (-1, 5), (3, 7), (1, 6), (-2, 3), (7, 5), (1, 4)]]
    cs = cs[seed:] + cs[:seed]
    g, k = {}, 0
    for r in range(n + 1):
        for S in combinations(range(1, n + 1), r):
            poly = (cs[(2 * k) % len(cs)] * t ** deg
                    + cs[(2 * k + 1) % len(cs)] * t ** 2 + sp.Rational(1, 2))
            piece = {S: poly}
            if r % 2:
                piece = emul({(100 + k + 20 * seed,): sp.Integer(1)}, piece)
            g = eadd(g, piece)
            k += 1
    return g


def first_order(A, par=e):
    return {S: sp.expand(c.subs({e: 0, d: 0}) + par * sp.diff(c, par).subs({e: 0, d: 0}))
            for S, c in A.items()}


def gram(G, n, op=D_op):
    """(A_i(G^* theta_j)) . (A_i(G^* theta_l)) summed over i."""
    U = U_matrix(G, n, op)
    return [[eadd(*[emul(U[i][j], U[i][l]) for i in range(n)])
             for l in range(n)] for j in range(n)]


if __name__ == "__main__":
    print("=" * 70)
    print("radul.py calibration   (world %s)" % R.WORLD)
    print("=" * 70)

    # the contact field of theta_i is eq. (54)'s K_{theta_i}
    R.setup(1)
    X = K_f(th(1), 1)
    check("K_{theta_1} = theta_1 d_t - d_{theta_1}   (eq. (54))",
          eeq(X.a0, th(1)) and eeq(X.a[0], num(-1)))

    for n in (1, 2, 3):
        print("\nn = %d" % n)
        R.setup(max(n, 1))
        R.set_order(3)
        g = gen(n)
        G = Diffeo(g, e, n)
        m = multiplier(G, n)

        check("m = 1 + e . 2 d_t(g) + O(e^2)",
              eeq(first_order(m), eadd(num(1), escal(e, escal(2, d_t(g))))))

        if n:
            M = gram(G, n, D_op)
            scalar = all(eeq(M[j][l], M[0][0] if j == l else {})
                         for j in range(n) for l in range(n))
            check("U^tU is scalar in the D-frame", scalar)
            check("U^tU = m . 1_n   (eq. (112))", scalar and eeq(M[0][0], m))
            MK = gram(G, n, K_op)
            check("the K-frame fails it (n >= 2)",
                  n == 1 or not all(eeq(MK[j][l], MK[0][0] if j == l else {})
                                    for j in range(n) for l in range(n)))

        H = Diffeo(gen(n, seed=3), d, n)
        GH = Composite(G, H)
        check("m_{GH} = H^*(m_G) . m_H",
              eeq(multiplier(GH, n), emul(H.star(m), multiplier(H, n))))

    # the classical Schwarzian, n = 0, weight 2
    print("\nclassical Schwarzian (n = 0)")
    R.setup(1)
    R.set_order(3)

    def S0(G):
        m = multiplier(G, 0)
        mi = einv(m)
        return escal(sp.Rational(1, 2),
                     esub(emul(d_t(d_t(m)), mi),
                          escal(sp.Rational(3, 2),
                                emul(emul(d_t(m), d_t(m)), emul(mi, mi)))))

    G = Diffeo(gen(0, deg=4), e, 0)
    H = Diffeo(gen(0, seed=5, deg=4), d, 0)
    GH = Composite(G, H)
    mH = multiplier(H, 0)
    check("S_0 is not identically zero", bool(S0(G)))
    check("S_0(GH) = S_0(H) + m_H^2 . H^*(S_0(G))",
          eeq(S0(GH), eadd(S0(H), emul(epow(mH, 2), H.star(S0(G))))))
    Gq = Diffeo({(): t ** 2 + t + sp.Rational(1, 3)}, e, 0)
    check("S_0 = 0 on the Moebius flows (sl(2) = the annihilator)", not S0(Gq))

    print("\n%d ok, %d FAILED" % (OK, BAD))
