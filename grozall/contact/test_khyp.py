"""Checks for k(2n+1|m) in the SYMMETRIC (hyperbolic) shape of the contact form,

    alpha = dt + (1/2) sum_i (p_i dq_i - q_i dp_i)
               + sum_j (xi_j d eta_j + eta_j d xi_j) [+ theta d theta],

Everything the orthonormal shape was checked for is
checked again here, plus the two facts the shape exists for:

  *  D_xi^2 = D_eta^2 = 0, {D_xi,D_eta} = -2R  (the odd frame is isotropic-in-pairs),
  *  the maximal toral subalgebra of k_0 is diagonal on the frame -- see torus.py.

Exact rational arithmetic; the odd coordinates are numbered xi_1,eta_1,xi_2,eta_2,...
so that index 2j-1 is a xi and index 2j an eta, theta last when m is odd.
"""

import random
from fractions import Fraction as Fr

from kmix import KMix, Dens, add, scal, mul, parity, hyperbolic_gram

FAIL = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


def rand_f(alg, seed, maxdeg=3, nmon=7, par=None):
    rnd = random.Random(seed)
    out = {}
    for _ in range(nmon):
        e = [0] * alg.nev
        S = []
        for _ in range(rnd.randint(0, maxdeg)):
            if alg.m and rnd.random() < 0.4:
                j = rnd.randint(1, alg.m)
                if j not in S:
                    S.append(j)
            else:
                e[rnd.randrange(alg.nev)] += 1
        S = tuple(sorted(S))
        if par is not None and len(S) % 2 != par:
            if alg.m == 0:
                continue
            j = next((x for x in range(1, alg.m + 1) if x not in S), None)
            if j is None:
                continue
            S = tuple(sorted(S + (j,)))
        out[(tuple(e), S)] = out.get((tuple(e), S), Fr(0)) + Fr(rnd.randint(1, 5))
    return {k: v for k, v in out.items() if v}


def supercomm(alg, f, g, h):
    s = (-1) ** (parity(f) * parity(g)) if f and g else 1
    return add(alg.Kf(f, alg.Kf(g, h)), scal(-s, alg.Kf(g, alg.Kf(f, h))))


def divergence(alg, f):
    A, B, C, Th = alg.Kf_coeffs(f)
    out = alg.dt(A)
    for i in range(1, alg.n + 1):
        out = add(out, alg.de(alg.ip(i), B[i - 1]), alg.de(alg.iq(i), C[i - 1]))
    p = parity(f) if f else 0
    for j in range(1, alg.m + 1):
        out = add(out, scal(-(-1) ** p, alg.do(j, Th[j - 1])))
    return out


def main():
    for (n, m) in [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 2)]:
        alg = KMix(n, m, gram=hyperbolic_gram(m))
        print("k(%d|%d), symmetric shape:" % (2 * n + 1, m))
        fs = [f for f in (rand_f(alg, 100 + 7 * n + 3 * m + i, par=i % 2)
                          for i in range(4)) if f]
        hs = [h for h in (rand_f(alg, 500 + 7 * n + 3 * m + i) for i in range(3)) if h]

        # 1) f = alpha(K_f): the coefficient of d_t plus the contraction of the rest
        ok = True
        for f in fs:
            A, B, C, Th = alg.Kf_coeffs(f)
            acc = A
            for i in range(1, n + 1):
                acc = add(acc, scal(HALF := Fr(1, 2),
                                    mul(alg.evar(alg.ip(i)), C[i - 1])),
                          scal(-HALF, mul(alg.evar(alg.iq(i)), B[i - 1])))
            for a in range(1, m + 1):
                for b in range(1, m + 1):
                    g = alg.gram(a, b)
                    if g:
                        acc = add(acc, scal(-g * (-1) ** parity(f),
                                            mul(alg.ovar(a), Th[b - 1])))
            ok = ok and acc == f
        check("f = alpha(K_f)", ok)

        # 2) the isomorphism onto the Lagrange bracket -- the real test of the formula
        ok = all(supercomm(alg, f, g, h) == alg.Kf(alg.kbracket(f, g), h)
                 for f in fs for g in fs for h in hs)
        check("[K_f,K_g] = K_{{f,g}}", ok)

        # 3) the frame
        ok = True
        for a in range(1, 2 * n + 1):
            for b in range(1, 2 * n + 1):
                for h in hs:
                    lhs = add(alg.e(a, alg.e(b, h)), scal(-1, alg.e(b, alg.e(a, h))))
                    if lhs != scal(alg.varpi_low(a, b), alg.R(h)):
                        ok = False
        check("[e_a,e_b] = varpi_{ab} R", ok)
        ok = True
        for a in range(1, m + 1):
            for b in range(1, m + 1):
                for h in hs:
                    lhs = add(alg.D(a, alg.D(b, h)), alg.D(b, alg.D(a, h)))
                    if lhs != scal(-2 * alg.gram(a, b), alg.R(h)):
                        ok = False
        check("{D_a,D_b} = -2 G_{ab} R", ok)
        ok = all(alg.D(2 * j + 1, alg.D(2 * j + 1, h)) == {}
                 for j in range(m // 2) for h in hs)
        check("D_xi^2 = D_eta^2 = 0 (the odd frame is isotropic in pairs)", ok)
        ok = True
        for a in range(1, 2 * n + 1):
            for b in range(1, m + 1):
                for h in hs:
                    if add(alg.e(a, alg.D(b, h)),
                           scal(-1, alg.D(b, alg.e(a, h)))) != {}:
                        ok = False
        check("[e_a,D_b] = 0", ok)

        # 4) the frame commutes with k_{-2}+k_{-1}
        ok = True
        for g in alg.k_piece(-2) + alg.k_piece(-1):
            for h in hs:
                for a in range(1, 2 * n + 1):
                    if add(alg.Kf(g, alg.e(a, h)),
                           scal(-1, alg.e(a, alg.Kf(g, h)))) != {}:
                        ok = False
                for b in range(1, m + 1):
                    s = (-1) ** parity(g) if g else 1
                    if add(alg.Kf(g, alg.D(b, h)),
                           scal(-s, alg.D(b, alg.Kf(g, h)))) != {}:
                        ok = False
                if add(alg.Kf(g, alg.R(h)), scal(-1, alg.R(alg.Kf(g, h)))) != {}:
                    ok = False
        check("the frame commutes with k_{-2}+k_{-1}", ok)

        # 5) the grading
        z = alg.evar(0, 2)
        ok = True
        for i in range(-2, 4):
            for g in alg.k_piece(i):
                if alg.kbracket(z, g) != scal(i, g):
                    ok = False
        check("z = K_{2t} grades k_i by i", ok)

        # 6) the modules
        ok = True
        for lam in (Fr(0), Fr(-1), Fr(2, 3)):
            M = Dens(alg, lam)
            for f in fs:
                for g in fs:
                    s = (-1) ** (parity(f) * parity(g))
                    for h in hs:
                        lhs = add(M.act(f, M.act(g, h)),
                                  scal(-s, M.act(g, M.act(f, h))))
                        if lhs != M.act(alg.kbracket(f, g), h):
                            ok = False
        check("F_lam is a module", ok)
        lam = Fr(n + 1) - Fr(m, 2)
        check("div K_f = (%s) f_t, i.e. Vol = Pi^%d F_{%s}" % (lam, m, lam),
              all(divergence(alg, f) == scal(lam, alg.dt(f)) for f in fs))
        print()

    print("%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
