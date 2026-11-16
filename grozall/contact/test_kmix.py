"""Checks for kmix.py: the structure of k(2n+1|m), and agreement with the two engines
it generalizes -- contact.py (n=1, m=0) and super.py (n=0)."""

import random
from fractions import Fraction as Fr

from kmix import KMix, Dens, add, scal, mul, parity

FAIL = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


def rand_f(alg, seed, maxdeg=3, nmon=7, par=None):
    """A random function; par = 0/1 forces the parity."""
    rnd = random.Random(seed)
    out = {}
    for _ in range(nmon):
        e = [0] * alg.nev
        d = rnd.randint(0, maxdeg)
        S = []
        for _ in range(d):
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
    """[K_f,K_g] h  with the super commutator."""
    s = (-1) ** (parity(f) * parity(g)) if f and g else 1
    return add(alg.Kf(f, alg.Kf(g, h)), scal(-s, alg.Kf(g, alg.Kf(f, h))))


def main():
    for (n, m) in [(1, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]:
        alg = KMix(n, m)
        print("k(%d|%d):" % (2 * n + 1, m))
        fs = [rand_f(alg, 100 + 7 * n + 3 * m + i, par=i % 2) for i in range(4)]
        fs = [f for f in fs if f]
        hs = [rand_f(alg, 500 + 7 * n + 3 * m + i) for i in range(3)]

        # f = alpha(K_f).  On the odd part the contraction of a 1-form written with
        # its coefficient on the LEFT carries the sign (-1)^{p(f)+1}: the convention
        # is the one of super.py, and what it is FOR is the bracket identity below.
        ok = True
        for f in fs:
            A, B, C, Th = alg.Kf_coeffs(f)
            tot = A
            for i in range(1, n + 1):
                tot = add(tot, scal(HALF := Fr(1, 2), mul(alg.evar(alg.ip(i)), C[i - 1])),
                          scal(-Fr(1, 2), mul(alg.evar(alg.iq(i)), B[i - 1])))
            sgn = -(-1) ** (parity(f) if f else 0)
            for j in range(1, m + 1):
                tot = add(tot, scal(sgn, mul(alg.ovar(j), Th[j - 1])))
            ok = ok and tot == f
        check("f = alpha(K_f)", ok)

        # [K_f,K_g] = K_{{f,g}}
        ok = True
        for f in fs:
            for g in fs:
                br = alg.kbracket(f, g)
                for h in hs:
                    if supercomm(alg, f, g, h) != alg.Kf(br, h):
                        ok = False
        check("[K_f,K_g] = K_{{f,g}}", ok)

        # the frame relations
        ok = True
        for a in range(1, 2 * n + 1):
            for b in range(1, 2 * n + 1):
                for h in hs:
                    lhs = add(alg.e(a, alg.e(b, h)), scal(-1, alg.e(b, alg.e(a, h))))
                    if lhs != scal(alg.varpi_low(a, b), alg.R(h)):
                        ok = False
        check("[e_a,e_b] = varpi_{ab} R", ok)
        ok = True
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                for h in hs:
                    lhs = add(alg.D(i, alg.D(j, h)), alg.D(j, alg.D(i, h)))
                    if lhs != scal(-2 if i == j else 0, alg.R(h)):
                        ok = False
        check("{D_i,D_j} = -2 delta_{ij} R", ok)
        ok = True
        for a in range(1, 2 * n + 1):
            for j in range(1, m + 1):
                for h in hs:
                    if add(alg.e(a, alg.D(j, h)), scal(-1, alg.D(j, alg.e(a, h)))):
                        ok = False
        check("[e_a,D_j] = 0", ok)

        # the frame commutes with k_{-2}+k_{-1}
        kminus = ([alg.const()] + [alg.evar(i) for i in range(1, 2 * n + 1)]
                  + [alg.ovar(j) for j in range(1, m + 1)])
        ok = True
        for g in kminus:
            pg = parity(g)
            for h in hs:
                for a in range(1, 2 * n + 1):
                    if add(alg.Kf(g, alg.e(a, h)), scal(-1, alg.e(a, alg.Kf(g, h)))):
                        ok = False
                for j in range(1, m + 1):
                    s = (-1) ** pg
                    if add(alg.Kf(g, alg.D(j, h)), scal(-s, alg.D(j, alg.Kf(g, h)))):
                        ok = False
        check("the frame commutes with k_{-2}+k_{-1}", ok)

        # grading, and dim k_0 = dim osp(m|2n) + 1
        z = alg.evar(0, 2)
        ok = True
        for i in range(-2, 3):
            for g in alg.k_piece(i):
                if alg.kbracket(z, g) != scal(i, g):
                    ok = False
        check("z = K_{2t} grades k_i by i", ok)
        want = (n * (2 * n + 1) + m * (m - 1) // 2) + 2 * n * m + 1
        check("sdim k_0 = sdim (Kee z + osp(%d|%d)) = %d" % (m, 2 * n, want),
              len(alg.k_piece(0)) == want)

        # the densities are a module, and F_{-1} is the adjoint
        ok = True
        for lam in (Fr(0), Fr(2, 3), Fr(-1)):
            M = Dens(alg, lam)
            for f in fs:
                for g in fs:
                    br = alg.kbracket(f, g)
                    s = (-1) ** (parity(f) * parity(g))
                    for F in hs:
                        lhs = add(M.act(f, M.act(g, F)),
                                  scal(-s, M.act(g, M.act(f, F))))
                        if lhs != M.act(br, F):
                            ok = False
        check("F_lam is a module", ok)
        M = Dens(alg, -1)
        ok = all(M.act(f, g) == alg.kbracket(f, g) for f in fs for g in fs)
        check("F_{-1} = the adjoint module", ok)
        print()

    # ------------------------------------------------ against the two engines
    print("k(3) = kmix(1,0) against contact.py:")
    import contact as c3
    alg = KMix(1, 0)

    def to3(f):
        return {e: v for (e, S), v in f.items()}

    def fr3(p):
        return {(e, ()): v for e, v in p.items()}

    fs = [rand_f(alg, 900 + i) for i in range(4)]
    hs = [rand_f(alg, 950 + i) for i in range(4)]
    check("K_f agrees", all(to3(alg.Kf(f, h)) == c3.Kf(to3(f), to3(h))
                            for f in fs for h in hs))
    check("the frame agrees", all(to3(alg.e(1, h)) == c3.e1(to3(h))
                                  and to3(alg.e(2, h)) == c3.e2(to3(h)) for h in hs))

    print("\nk(1|n) = kmix(0,n) against super.py:")
    import super as sp
    for m in (1, 2, 3):
        alg = KMix(0, m)

        def tos(f):
            return {(e[0], S): v for (e, S), v in f.items()}

        fs = [rand_f(alg, 300 + m * 10 + i, par=i % 2) for i in range(4)]
        fs = [f for f in fs if f]
        hs = [rand_f(alg, 400 + m * 10 + i) for i in range(3)]
        check("m=%d: K_f agrees" % m,
              all(tos(alg.Kf(f, h)) == sp.Kf(m, tos(f), tos(h))
                  for f in fs for h in hs))
        check("m=%d: the bracket agrees" % m,
              all(tos(alg.kbracket(f, g)) == sp.kbracket(m, tos(f), tos(g))
                  for f in fs for g in fs))
        check("m=%d: D_j agrees" % m,
              all(tos(alg.D(j, h)) == sp.D(j, tos(h))
                  for j in range(1, m + 1) for h in hs))

    print("\n%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
