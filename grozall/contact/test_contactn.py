"""Checks for contactn.py: the structure of k(2n+1) for n = 1,2,3, and, for n = 1,
agreement with contact.py term by term."""

import random
from fractions import Fraction as Fr

from contactn import (Contact, Module, padd, pscal, pmul, pzero, pconst, pvar,
                      fiber_monomials)

FAIL = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


def rand_poly(alg, maxdeg, seed, nmon=8):
    rnd = random.Random(seed)
    out = {}
    for _ in range(nmon):
        e = [0] * alg.nv
        d = rnd.randint(0, maxdeg)
        for _ in range(d):
            e[rnd.randrange(alg.nv)] += 1
        out[tuple(e)] = out.get(tuple(e), Fr(0)) + Fr(rnd.randint(1, 5))
    return {k: v for k, v in out.items() if v}


def vfield(alg, f):
    """K_f as a list of coefficients (A, B^i, C^i) -- for the commutator test."""
    return alg.Kf_coeffs(f)


def commutator_on(alg, f, g, h):
    return padd(alg.Kf(f, alg.Kf(g, h)), pscal(-1, alg.Kf(g, alg.Kf(f, h))))


def main():
    for n in (1, 2, 3):
        alg = Contact(n)
        print("k(%d):" % (2 * n + 1))
        fs = [rand_poly(alg, 3, 100 + n * 10 + i) for i in range(3)]
        hs = [rand_poly(alg, 3, 500 + n * 10 + i) for i in range(3)]

        # 1) f = alpha(K_f):  A + (1/2) sum (p_i C^i - q_i B^i) = f
        ok = True
        for f in fs:
            A, B, C = alg.Kf_coeffs(f)
            tot = A
            for i in range(1, n + 1):
                tot = padd(tot,
                           pscal(Fr(1, 2), pmul(pvar(alg.ip(i), n), C[i - 1])),
                           pscal(-Fr(1, 2), pmul(pvar(alg.iq(i), n), B[i - 1])))
            ok = ok and tot == f
        check("f = alpha(K_f)", ok)

        # 2) [K_f,K_g] = K_{{f,g}}
        ok = True
        for f in fs:
            for g in fs:
                br = alg.kbracket(f, g)
                for h in hs:
                    if commutator_on(alg, f, g, h) != alg.Kf(br, h):
                        ok = False
        check("[K_f,K_g] = K_{{f,g}}", ok)

        # 3) Jacobi for the contact bracket
        ok = True
        for f in fs:
            for g in fs:
                for h in fs:
                    j = padd(alg.kbracket(f, alg.kbracket(g, h)),
                             pscal(-1, alg.kbracket(alg.kbracket(f, g), h)),
                             pscal(-1, alg.kbracket(g, alg.kbracket(f, h))))
                    if j:
                        ok = False
        check("Jacobi for {,}", ok)

        # 4) the frame: [e_a,e_b] = varpi_{ab} R, and [K_g,e_a] = 0 for g in k_{<=-1}
        ok = True
        for a in range(1, 2 * n + 1):
            for b in range(1, 2 * n + 1):
                for h in hs:
                    lhs = padd(alg.e(a, alg.e(b, h)), pscal(-1, alg.e(b, alg.e(a, h))))
                    rhs = pscal(alg.varpi_low(a, b), alg.R(h))
                    if lhs != rhs:
                        ok = False
        check("[e_a,e_b] = varpi_{ab} R", ok)

        kminus = [pconst(1, n)] + [pvar(i, n) for i in range(1, 2 * n + 1)]
        ok = True
        for g in kminus:
            for a in range(1, 2 * n + 1):
                for h in hs:
                    c = padd(alg.Kf(g, alg.e(a, h)), pscal(-1, alg.e(a, alg.Kf(g, h))))
                    if c:
                        ok = False
        check("[K_g,e_a] = 0 for g in k_{-2}+k_{-1}", ok)

        # 5) [K_f,e_a] = sum_b C_a^b e_b   (in particular no R-component)
        ok = True
        for f in fs:
            Cm = alg.Cmat(f)
            for a in range(1, 2 * n + 1):
                for h in hs:
                    lhs = padd(alg.Kf(f, alg.e(a, h)),
                               pscal(-1, alg.e(a, alg.Kf(f, h))))
                    rhs = pzero()
                    for b in range(1, 2 * n + 1):
                        rhs = padd(rhs, pmul(Cm[(a, b)], alg.e(b, h)))
                    if lhs != rhs:
                        ok = False
        check("[K_f,e_a] = C_a^b e_b", ok)

        # 6) trace of C = -f_t
        ok = True
        for f in fs:
            Cm = alg.Cmat(f)
            tr = pzero()
            for a in range(1, 2 * n + 1):
                tr = padd(tr, Cm[(a, a)])
            if tr != pscal(-n, alg.dt(f)):
                ok = False
        check("tr C = -n f_t", ok)

        # 7) grading: z = K_{2t} acts on k_i by i;  dim k_0 = binom(2n+1,2)+1
        z = pscal(2, pvar(0, n))
        ok = True
        for i in range(-2, 3):
            for g in alg.k_piece(i):
                if alg.kbracket(z, g) != pscal(i, g):
                    ok = False
        check("z = K_{2t} grades k_i by i", ok)
        check("dim k_0 = dim csp(2n) = %d" % (n * (2 * n + 1) + 1),
              len(alg.k_piece(0)) == n * (2 * n + 1) + 1)
        check("dim k_{-1} = 2n", len(alg.k_piece(-1)) == 2 * n)

        # 8) modules: L_{K_f} is a representation
        for k in (0, 1, 2):
            M = Module(alg, k, Fr(3, 5))
            Fs = []
            for s in range(2):
                rnd = random.Random(700 + n + s + 10 * k)
                F = {}
                for _ in range(6):
                    e = [0] * alg.nv
                    for _ in range(rnd.randint(0, 2)):
                        e[rnd.randrange(alg.nv)] += 1
                    mu = M.fmons[rnd.randrange(len(M.fmons))]
                    F[(tuple(e), mu)] = Fr(rnd.randint(1, 5))
                Fs.append(F)
            ok = True
            for f in fs:
                for g in fs:
                    br = alg.kbracket(f, g)
                    for F in Fs:
                        lhs = M.act(f, M.act(g, F))
                        for key, v in M.act(g, M.act(f, F)).items():
                            lhs[key] = lhs.get(key, Fr(0)) - v
                        lhs = {a: b for a, b in lhs.items() if b}
                        rhs = M.act(br, F)
                        if lhs != rhs:
                            ok = False
            check("T(%d,lam) is a module" % k, ok)

        # 9) Vol = F_2:  the density of weight 2 is invariant up to a divergence,
        #    i.e. the module F_{-1} is the adjoint: K_f <-> f.
        M = Module(alg, 0, -1)
        z0 = (0,) * alg.dim
        ok = True
        for f in fs:
            for g in fs:
                Fg = {(k, z0): v for k, v in g.items()}
                got = M.act(f, Fg)
                want = {(k, z0): v for k, v in alg.kbracket(f, g).items()}
                if got != want:
                    ok = False
        check("F_{-1} = the adjoint module", ok)
        print()

    # ------------------------------------------------- n = 1 against contact.py
    print("n = 1 against contact.py:")
    import contact as c3
    alg = Contact(1)

    def to3(p):
        """(t,p,q)-exponents from contactn's (t,p,q)."""
        return {(e[0], e[1], e[2]): v for e, v in p.items()}

    fs = [rand_poly(alg, 3, 900 + i) for i in range(4)]
    hs = [rand_poly(alg, 3, 950 + i) for i in range(4)]
    ok = all(to3(alg.Kf(f, h)) == c3.Kf(to3(f), to3(h)) for f in fs for h in hs)
    check("K_f agrees", ok)
    ok = all(to3(alg.e(1, h)) == c3.e1(to3(h)) and to3(alg.e(2, h)) == c3.e2(to3(h))
             for h in hs)
    check("the frame agrees", ok)
    ok = True
    for f in fs:
        Cm = alg.Cmat(f)
        (C11, C12), (C21, C22) = c3.Cmat(to3(f))
        if (to3(Cm[(1, 1)]) != C11 or to3(Cm[(1, 2)]) != C12
                or to3(Cm[(2, 1)]) != C21 or to3(Cm[(2, 2)]) != C22):
            ok = False
    check("C_a^b agrees", ok)

    # module action: contactn's u-monomials (i,j) <-> contact.py's index i
    ok = True
    for k in (0, 1, 2):
        M = Module(alg, k, Fr(2, 7))
        M3 = c3.Module(k, Fr(2, 7))
        rnd = random.Random(31 + k)
        F, F3 = {}, {}
        for _ in range(6):
            e = tuple(rnd.randint(0, 2) for _ in range(3))
            i = rnd.randrange(k + 1)
            v = Fr(rnd.randint(1, 5))
            F[(e, (i, k - i))] = v
            F3[e + (i,)] = v
        for f in fs:
            got = M.act(f, F)
            want = M3.act(to3(f), F3)
            got3 = {}
            for (e, mu), v in got.items():
                got3[e + (mu[0],)] = got3.get(e + (mu[0],), Fr(0)) + v
            got3 = {a: b for a, b in got3.items() if b}
            if got3 != want:
                ok = False
    check("the module action agrees", ok)

    print("\n%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
