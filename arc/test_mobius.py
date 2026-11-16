"""
Calibration of mobius.py against what is known independently of the tables:
the field K^M_f of eq. (11), the Ramond bracket of eq. (14), the
frame of eq. (56) and its Gram matrix, and the multiplier of eq. (193).
"""
import sys
from itertools import combinations

import sympy as sp

import radul as R
import mobius as M
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t, d_th,
                   VF, einv, epow)
from mobius import (ZETA, cD_field, cE_field, D_M, K_M, K_f, Diffeo, Composite,
                    multiplier, bracket, anti, vf_eq, vf_scal, vf_add)

OK = BAD = 0


def check(name, cond):
    global OK, BAD
    if cond:
        OK += 1
        print("  ok   %s" % name)
    else:
        BAD += 1
        print("  FAIL %s" % name)
    sys.stdout.flush()
    return cond


def gen(n, seed=0, deg=3):
    """An EVEN generating function of (t | zeta, theta_i) with every component
    non-zero; a component with an odd number of odd coordinates carries an odd
    constant of its own, as in test_radul.gen -- without it exp(e K_g) is not
    an automorphism."""
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


def ramond(f, g, n):
    """{f, g}_{R.b.} = nabla(f) cD(g) - cD(f) nabla(g) + {f, g}_{MP.b.}."""
    cD = cD_field(n)

    def nabla(h):
        Eh = {}
        for i in range(1, n + 1):
            Eh = eadd(Eh, emul(th(i), d_th(i, h)))
        return esub(escal(2, h), Eh)

    fe, fo = R.parity_split(f)
    mp = {}
    for i in range(1, n + 1):
        w = sp.Rational(1, 1) / t if i == ZETA else sp.Integer(1)
        mp = eadd(mp, escal(w, emul(esub(d_th(i, fe), d_th(i, fo)), d_th(i, g))))
    return eadd(esub(emul(nabla(f), cD(g)), emul(cD(f), nabla(g))), mp)


def in_frame_span(X, n):
    """Is the odd field X an F-combination of the D^M_i?  The D^M_i are a free
    basis of the kernel of alpha~, so X is iff X - sum_i alpha~(X)-free part
    vanishes; concretely, subtract  sum_i c_i D^M_i  with  c_i  read off the
    d_theta_i components and see whether the d_t component survives."""
    Y = VF(X.a0, list(X.a))
    # the untwisted members first: their d_theta_i components are clean, while
    # every one of them contributes to the d_zeta component as well
    for i in range(2, n + 1):
        ci = Y.a[i - 1]
        Di = D_M(i, n)
        Y = VF(esub(Y.a0, emul(ci, Di.a0)),
               [esub(Y.a[j], emul(ci, Di.a[j])) for j in range(n)])
    if n >= 1:                          # D^M_zeta carries 1/t in front of d_zeta
        cz = escal(t, Y.a[ZETA - 1])
        Dz = D_M(ZETA, n)
        Y = VF(esub(Y.a0, emul(cz, Dz.a0)),
               [esub(Y.a[j], emul(cz, Dz.a[j])) for j in range(n)])
    return not Y.a0 and all(not Y.a[j] for j in range(n))


if __name__ == "__main__":
    print("=" * 70)
    print("mobius.py calibration   (world %s)" % M.WORLD)
    print("=" * 70)

    for n in (1, 2, 3):
        print("\nn = %d   (zeta, theta_1, ..., theta_%d)" % (n, n - 1))
        M.setup(n)
        R.set_order(3)
        cD, cE = cD_field(n), cE_field(n)

        # 1) K^M_1 = 2 cD, eq. (K^Ma)
        X1 = K_f(num(1), n)
        check("K^M_1 = 2 cD", vf_eq(X1, vf_scal(num(2), cD, n), n))

        # 2) the frame relations: the twisted member is NOT a square root of cD
        Dz = D_M(ZETA, n)
        zero = VF({}, [{} for _ in range(n)])
        check("(D^M_zeta)^2 = (1/t) cD - (zeta/2t) D^M_zeta",
              vf_eq(vf_scal(num(sp.Rational(1, 2)), anti(Dz, Dz, n), n),
                    vf_add(vf_scal(num(1 / t), cD, n),
                           vf_scal(escal(-sp.Rational(1, 2) / t, th(ZETA)), Dz, n), n), n))
        # ... whereas sqrt(t) D^M_zeta is: the frame of the two-sheeted cover
        Dzc = vf_scal(num(sp.sqrt(t)), Dz, n)
        check("(sqrt(t) D^M_zeta)^2 = cD",
              vf_eq(vf_scal(num(sp.Rational(1, 2)), anti(Dzc, Dzc, n), n), cD, n))
        if n >= 2:
            Dth = D_M(2, n)
            check("(D^M_theta)^2 = cD",
                  vf_eq(vf_scal(num(sp.Rational(1, 2)), anti(Dth, Dth, n), n),
                        cD, n))
            check("{D^M_zeta, D^M_theta} = -(theta/2t) D^M_zeta",
                  vf_eq(anti(Dz, Dth, n),
                        vf_scal(escal(-sp.Rational(1, 2) / t, th(2)), Dz, n), n))
            check("{sqrt(t) D^M_zeta, D^M_theta} = 0",
                  vf_eq(anti(Dzc, Dth, n), zero, n))

        # 3) the contact fields close on the Ramond bracket
        f = gen(n, seed=1, deg=2)
        g = gen(n, seed=4, deg=2)
        check("[K^M_f, K^M_g] = K^M_{f,g}_{R.b.}",
              vf_eq(bracket(K_f(f, n), K_f(g, n), n),
                    K_f(ramond(f, g, n), n), n))

        # 4) the fields preserve the frame (56): this is what fixes the world
        ok = all(in_frame_span(bracket(K_f(g, n), D_M(i, n), n), n)
                 for i in range(1, n + 1))
        check("[K^M_g, D^M_i] stays in the span of the D^M_j", ok)
        bad = all(in_frame_span(bracket(K_f(g, n, world='A'), D_M(i, n), n), n)
                  for i in range(1, n + 1))
        check("   and the mirror world A does not", not bad)

        # 5) the multiplier
        G = Diffeo(gen(n), e, n)
        H = Diffeo(gen(n, seed=3), d, n)
        m = multiplier(G, n)
        check("m_G = 1 + e . 2 cD(g) + O(e^2)",
              eeq(first_order(m), eadd(num(1), escal(e, escal(2, cD(G.g))))))
        check("m_{GH} = H^*(m_G) . m_H",
              eeq(multiplier(Composite(G, H), n), emul(H.star(m), multiplier(H, n))))

    print("\n%d ok, %d FAILED" % (OK, BAD))
    sys.exit(1 if BAD else 0)
