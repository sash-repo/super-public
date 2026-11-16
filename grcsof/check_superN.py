#!/usr/bin/env python3
"""Validation of superN.py:
  (1) K is a homomorphism:  [K_f, K_g] = K_{{f,g}}   (super-commutator);
  (2) [D_i, D_j] = 2 B_{ij} d_t   and  [K_f, D_i] lies in the D-distribution;
  (3) F_lam is a module:    [L_f, L_g] = L_{{f,g}};
  (4) regression: for n = 2, B = antidiag(1,1), superN reproduces k12.py.
"""
from fractions import Fraction as F
import random
from superN import (Contact, add, smul, mul, mono, parity, par_mon, d_t,
                    popcount, clean)

random.seed(11)


def randf(n, tmax=2, terms=4, par=None):
    """random homogeneous-parity superfunction."""
    while True:
        u = {}
        for _ in range(terms):
            a = random.randint(0, tmax)
            m = random.randint(0, (1 << n) - 1)
            if par is not None and popcount(m) % 2 != par:
                continue
            u[(a, m)] = u.get((a, m), 0) + F(random.randint(-3, 3))
        u = clean(u)
        if u and len({popcount(k[1]) % 2 for k in u}) == 1:
            return u


def commutator(op1, op2, s):
    return lambda u: add(op1(op2(u)), smul(-s, op2(op1(u))))


def check(alg, label):
    n = alg.n
    tests = alg.monos(3)
    ok = True

    # (1) homomorphism
    for _ in range(12):
        pf, pg = random.randint(0, 1), random.randint(0, 1)
        f, g = randf(n, par=pf), randf(n, par=pg)
        pf, pg = parity(f), parity(g)
        s = -1 if (pf and pg) else 1
        Kf, Kg, Kfg = alg.K(f), alg.K(g), alg.K(alg.kb(f, g))
        for m in tests:
            u = mono(*m)
            lhs = add(Kf(Kg(u)), smul(-s, Kg(Kf(u))))
            if add(lhs, smul(-1, Kfg(u))):
                print(f"  [{label}] HOMOMORPHISM FAILS at {m}")
                ok = False
                break
    print(f"  [{label}] (1) [K_f,K_g] = K_{{f,g}}: {'ok' if ok else 'FAIL'}")

    # (2) D-relations
    ok2 = True
    for i in range(n):
        for j in range(n):
            Di, Dj = alg.D(i), alg.D(j)
            for m in tests:
                u = mono(*m)
                lhs = add(Di(Dj(u)), Dj(Di(u)))
                rhs = smul(2 * alg.B[i][j], d_t(u))
                if add(lhs, smul(-1, rhs)):
                    print(f"  [{label}] D-relation fails for {i},{j} at {m}")
                    ok2 = False
    print(f"  [{label}] (2) [D_i,D_j] = 2B_ij d_t: {'ok' if ok2 else 'FAIL'}")

    # (2b) K_f preserves the distribution <D_1,...,D_n>:
    #      [K_f, D_i] = sum_j c_{ij} D_j  for some superfunctions c_{ij}.
    #      Test: [K_f,D_i](u) must vanish whenever we can detect a d_t-only part;
    #      we check it by expressing the commutator on the free module basis.
    ok3 = True
    for _ in range(6):
        f = randf(n, par=random.randint(0, 1))
        pf = parity(f)
        Kf = alg.K(f)
        for i in range(n):
            Di = alg.D(i)
            s = -1 if pf else 1          # D_i is odd
            com = commutator(Kf, Di, s)
            # the commutator is a first-order operator; write it as
            #     sum_j c_j D_j + c_0 d_t + c .
            # u = 1   gives  c        (D_j(1) = d_t(1) = 0);
            # u = y_k gives  c_k      (D_j(y_k) = delta_{jk}, d_t(y_k) = 0);
            # u = t   gives  sum_j c_j (sum_l B_{jl} y_l) + c_0, whence c_0.
            # Preservation of the distribution <D_1,...,D_n> means c_0 = c = 0.
            c = com(mono(0, 0))
            cj = [com(mono(0, 1 << k)) for k in range(n)]
            c0 = com(mono(1, 0))
            for j in range(n):
                Djt = {}
                for l in range(n):
                    if alg.B[j][l]:
                        Djt = add(Djt, mono(0, 1 << l, alg.B[j][l]))
                c0 = add(c0, smul(-1, mul(cj[j], Djt)))
            if c or c0:
                print(f"  [{label}] distribution not preserved: i={i}")
                ok3 = False
    print(f"  [{label}] (2b) [K_f,D_i] in <D_j>: {'ok' if ok3 else 'FAIL'}")

    # (3) module property for F_lam, symbolic lam
    from sympy import symbols
    lam = symbols('lam')
    ok4 = True
    for _ in range(8):
        f, g = randf(n, par=random.randint(0, 1)), randf(n, par=random.randint(0, 1))
        pf, pg = parity(f), parity(g)
        s = -1 if (pf and pg) else 1
        Lf, Lg = alg.L(f, lam), alg.L(g, lam)
        Lfg = alg.L(alg.kb(f, g), lam)
        for m in tests[:12]:
            u = mono(*m)
            lhs = add(Lf(Lg(u)), smul(-s, Lg(Lf(u))))
            res = add(lhs, smul(-1, Lfg(u)))
            from sympy import expand
            if any(expand(v) != 0 for v in res.values()):
                print(f"  [{label}] MODULE FAILS at {m}")
                ok4 = False
                break
    print(f"  [{label}] (3) F_lam is a module: {'ok' if ok4 else 'FAIL'}")
    return ok and ok2 and ok3 and ok4


def regression_n2():
    """superN with n=2, B=antidiag(1,1) vs the old k12.py."""
    import k12
    from super12 import mono as mono2, add as add2, smul as smul2
    alg = Contact(2, [[0, 1], [1, 0]], names=['xi', 'eta'])
    # dictionary of monomials: super12 key (a,e,f) <-> superN key (a, mask)
    def conv(u):                    # super12 -> superN
        return {(a, (e << 0) | (f << 1)): c for (a, e, f), c in u.items()}
    ok = True
    for a in range(3):
        for e in (0, 1):
            for f in (0, 1):
                fun2 = mono2(a, e, f)
                fun = conv(fun2)
                for b in range(3):
                    for e2 in (0, 1):
                        for f2 in (0, 1):
                            u2 = mono2(b, e2, f2)
                            u = conv(u2)
                            r_old = conv(k12.K(fun2)(u2))
                            r_new = alg.K(fun)(u)
                            if add(r_new, smul(-1, r_old)):
                                print(f"  regression FAILS: f={a,e,f} u={b,e2,f2}")
                                ok = False
                # D_xi, D_eta
    for b in range(3):
        for e2 in (0, 1):
            for f2 in (0, 1):
                u2 = mono2(b, e2, f2)
                u = conv(u2)
                if add(alg.D(0)(u), smul(-1, conv(k12.D_xi(u2)))):
                    print("  regression FAILS: D_xi")
                    ok = False
                if add(alg.D(1)(u), smul(-1, conv(k12.D_eta(u2)))):
                    print("  regression FAILS: D_eta")
                    ok = False
    print(f"  [n=2 vs k12.py] K_f, D_xi, D_eta agree: {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print("k(1|1), alpha = dt + theta d(theta):")
    check(Contact(1, [[1]], names=['th']), 'n=1')
    print("k(1|2), alpha = dt + xi d(eta) + eta d(xi):")
    check(Contact(2, [[0, 1], [1, 0]], names=['xi', 'eta']), 'n=2 anti')
    print("k(1|2), alpha = dt + th1 d(th1) + th2 d(th2):")
    check(Contact(2, [[1, 0], [0, 1]], names=['th1', 'th2']), 'n=2 diag')
    print("k(1|3), alpha = dt + th d(th) + xi d(eta) + eta d(xi):")
    check(Contact(3, [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                  names=['th', 'xi', 'eta']), 'n=3 mixed')
    print("k(1|4), alpha = dt + sum (xi_i d(eta_i) + eta_i d(xi_i)):")
    check(Contact(4, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                  names=['xi1', 'eta1', 'xi2', 'eta2']), 'n=4')
    print("regression against k12.py:")
    regression_n2()
