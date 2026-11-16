"""Self-tests for mperi.py.  Nothing in the pericontact engine is assumed:
the bracket, the frame constants, the module structure and the subalgebra
b_{a,b}(n) are all verified here from the definitions.
"""

import random
import os, sys
from fractions import Fraction as Fr
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from mperi import (  # noqa: E402
    add, alpha0, apply_frame, b_piece, d_q, d_tau, d_xi, Dens, div_lambda,
    frame, frame_constants, m_piece, mbracket, Mf, mono, monomials, mul, one,
    parity, q, scal, sub, tau, vapply, vbracket, vparity_of, wdeg, xi,
)

OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print("%-58s %s" % (name, "ok" if cond else "FAIL"))
    return bool(cond)


def rand_fun(n, par, rnd, maxdeg=3, terms=4):
    out = {}
    for _ in range(terms):
        d = rnd.randint(0, maxdeg)
        cands = [m for m in monomials(n, d) if parity(m) == par]
        if not cands:
            continue
        out = add(out, scal(rnd.randint(-3, 3), rnd.choice(cands)))
    return out


def main():
    rnd = random.Random(11)
    allok = True

    for n in (1, 2, 3):
        print("\n--- n = %d -------------------------------------------------" % n)

        # 1. supercommutativity and Leibniz
        good = True
        for _ in range(40):
            p1, p2 = rnd.randint(0, 1), rnd.randint(0, 1)
            f, g = rand_fun(n, p1, rnd), rand_fun(n, p2, rnd)
            if not f or not g:
                continue
            if sub(mul(f, g), scal((-1) ** (p1 * p2), mul(g, f))):
                good = False
            for i in range(1, n + 1):
                lhs = d_xi(i, mul(f, g))
                rhs = add(mul(d_xi(i, f), g), scal((-1) ** p1, mul(f, d_xi(i, g))))
                if sub(lhs, rhs):
                    good = False
                lhs = d_q(i, mul(f, g))
                rhs = add(mul(d_q(i, f), g), mul(f, d_q(i, g)))
                if sub(lhs, rhs):
                    good = False
            lhs = d_tau(mul(f, g))
            rhs = add(mul(d_tau(f), g), scal((-1) ** p1, mul(f, d_tau(g))))
            if sub(lhs, rhs):
                good = False
        allok &= check("supercommutative, Leibniz for d_q, d_xi, d_tau", good)

        # 2. the bracket of vector fields is the commutator on functions
        good = True
        for _ in range(25):
            f = rand_fun(n, rnd.randint(0, 1), rnd)
            g = rand_fun(n, rnd.randint(0, 1), rnd)
            h = rand_fun(n, rnd.randint(0, 1), rnd)
            if not f or not g or not h:
                continue
            X, Y = Mf(n, f), Mf(n, g)
            if not X or not Y:
                continue
            pX, pY = vparity_of(X), vparity_of(Y)
            lhs = vapply(vbracket(X, Y), h)
            rhs = sub(vapply(X, vapply(Y, h)),
                      scal((-1) ** (pX * pY), vapply(Y, vapply(X, h))))
            if sub(lhs, rhs):
                good = False
        allok &= check("[X,Y] acts as XY - (-1)^{pp} YX", good)

        # 3. normalization alpha_0(M_f) = f, and p(M_f) = p(f)+1
        good, goodp = True, True
        for d in range(0, 5):
            for m in monomials(n, d):
                if sub(alpha0(n, Mf(n, m)), m):
                    good = False
                X = Mf(n, m)
                if X and vparity_of(X) != (parity(m) + 1) % 2:
                    goodp = False
        allok &= check("alpha_0(M_f) = f", good)
        allok &= check("p(M_f) = p(f) + 1", goodp)

        # 4. [M_f,M_g] = M_{{f,g}}, and Jacobi
        good = True
        for _ in range(20):
            f = rand_fun(n, rnd.randint(0, 1), rnd)
            g = rand_fun(n, rnd.randint(0, 1), rnd)
            if not f or not g:
                continue
            lhs = vbracket(Mf(n, f), Mf(n, g))
            rhs = Mf(n, mbracket(n, f, g))
            for k in set(lhs) | set(rhs):
                if sub(lhs.get(k, {}), rhs.get(k, {})):
                    good = False
        allok &= check("[M_f, M_g] = M_{{f,g}}  (m(n) is a subalgebra)", good)

        good = True
        for _ in range(12):
            fs = [rand_fun(n, rnd.randint(0, 1), rnd, maxdeg=2, terms=2) for _ in range(3)]
            if any(not f for f in fs):
                continue
            f, g, h = fs
            pf, pg, ph = parity(f) + 1, parity(g) + 1, parity(h) + 1
            t1 = mbracket(n, f, mbracket(n, g, h))
            t2 = scal((-1) ** (pf * pg), mbracket(n, g, mbracket(n, f, h)))
            t3 = mbracket(n, mbracket(n, f, g), h)
            if add(t1, scal(-1, t2), scal(-1, t3)):
                good = False
        allok &= check("super Jacobi for the pericontact bracket", good)

        # 5. the bracket in closed form:  {f,g} = M_f(g) - (-1)^{p(f)} f_tau g ?
        # the closed form is FITTED: the unique c(p(f)) that works on every sample,
        # and the number of non-degenerate samples is printed so that the test
        # cannot pass vacuously.
        cands = {0: {Fr(1), Fr(-1), Fr(0)}, 1: {Fr(1), Fr(-1), Fr(0)}}
        used = {0: 0, 1: 0}
        for _ in range(200):
            pf = rnd.randint(0, 1)
            f = rand_fun(n, pf, rnd, maxdeg=3)
            g = rand_fun(n, rnd.randint(0, 1), rnd, maxdeg=3)
            if not f or not g:
                continue
            corr = mul(d_tau(f), g)
            if not corr:
                continue
            used[pf] += 1
            rest = sub(mbracket(n, f, g), vapply(Mf(n, f), g))
            cands[pf] = {c for c in cands[pf] if not sub(rest, scal(c, corr))}
        allok &= check("{f,g} = M_f(g) + c(p(f)) f_tau g:  c unique, both parities",
                       len(cands[0]) == 1 and len(cands[1]) == 1
                       and used[0] > 3 and used[1] > 3)
        print("      c(even f) = %s, c(odd f) = %s   (%d + %d samples)"
              % (list(cands[0]), list(cands[1]), used[0], used[1]))

        # 6. frame
        g_, d_ = frame_constants(n)
        print("      frame constants: gamma = %s, delta = %s" % (g_, d_))
        fr = frame(n)
        good = all(not alpha0(n, e) for e in fr[:2 * n])
        allok &= check("the frame X_i, Th_i lies in D = ker alpha_0", good)
        good = True
        for e in fr:
            for h in [one(n)] + [q(n, i) for i in range(1, n + 1)] \
                    + [xi(n, i) for i in range(1, n + 1)]:
                if vbracket(e, Mf(n, h)):
                    good = False
        allok &= check("the frame commutes with m_{-2} + m_{-1}", good)

        R = fr[2 * n]
        rel = {}
        for i in range(2 * n + 1):
            for j in range(2 * n + 1):
                br = vbracket(fr[i], fr[j])
                if br:
                    rel[(i, j)] = br
        good = True
        for i in range(n):           # X_i X_j commute
            for j in range(n):
                if (i, j) in rel:
                    good = False
        for i in range(n, 2 * n):    # Th_i Th_j anticommute to 0
            for j in range(n, 2 * n):
                if (i, j) in rel:
                    good = False
        allok &= check("[X_i,X_j] = 0 and [Th_i,Th_j] = 0", good)
        good = True
        for i in range(n):
            for j in range(n, 2 * n):
                br = rel.get((i, j), {})
                if i + n == j:
                    if not br:
                        good = False
                else:
                    if br:
                        good = False
        allok &= check("[X_i,Th_j] ~ delta_ij R  (the periplectic pairing)", good)

        # R^2 = 0 and Th^2 = 0, the point of the odd Reeb field
        good = True
        for _ in range(10):
            f = rand_fun(n, rnd.randint(0, 1), rnd)
            if vapply(R, vapply(R, f)):
                good = False
            for i in range(n, 2 * n):
                if vapply(fr[i], vapply(fr[i], f)):
                    good = False
        allok &= check("R^2 = 0 and Th_i^2 = 0  (the Reeb field is odd)", good)

        # 7. dimensions of the graded pieces
        d0 = len(m_piece(n, 0))
        d1 = len(m_piece(n, -1))
        d2 = len(m_piece(n, -2))
        allok &= check("sdim m_{-2} = (0|1), dim m_{-1} = 2n, dim m_0 = 2n^2+1",
                       d2 == 1 and d1 == 2 * n and d0 == 2 * n * n + 1)

        # 8. the density module is a representation
        good = True
        D = Dens(n, Fr(2, 3), Fr(-1, 5))
        for _ in range(12):
            f = rand_fun(n, rnd.randint(0, 1), rnd, maxdeg=2, terms=2)
            g = rand_fun(n, rnd.randint(0, 1), rnd, maxdeg=2, terms=2)
            F = rand_fun(n, rnd.randint(0, 1), rnd, maxdeg=3)
            if not f or not g or not F:
                continue
            pf, pg = (parity(f) + 1) % 2, (parity(g) + 1) % 2
            lhs = sub(D.act(f, D.act(g, F)),
                      scal((-1) ** (pf * pg), D.act(g, D.act(f, F))))
            rhs = D.act(mbracket(n, f, g), F)
            if sub(lhs, rhs):
                good = False
        allok &= check("F_{a,b} is an m(n)-module", good)

        # 9. b_{a,b}(n) is a subalgebra
        for (a, b) in ((Fr(1), Fr(3)), (Fr(2), Fr(-1))):
            gens = []
            for i in range(-2, 3):
                gens += b_piece(n, i, a, b)
            good = True
            for f in gens[:14]:
                for g in gens[:14]:
                    br = mbracket(n, f, g)
                    if br and div_lambda(n, br, a, b):
                        good = False
            allok &= check("b_{%s,%s}(%d) closed under the bracket" % (a, b, n), good)

        # 10. the frame monomials are linearly independent as operators
        from exactlin import rank
        from mperi import frame_terms
        for W in (1, 2, 3):
            terms = frame_terms(n, W)
            Fs = [rand_fun(n, p, rnd, maxdeg=6, terms=8) for p in (0, 1) for _ in range(12)]
            Fs += [m for d in range(0, 6) for m in monomials(n, d)]
            rows = []
            cols = {}
            for t in terms:
                vec = {}
                for idx, F in enumerate(Fs):
                    for k, v in apply_frame(n, t, F).items():
                        cols[(idx, k)] = None
                        vec[(idx, k)] = v
                rows.append(vec)
            keys = sorted(cols, key=repr)
            mat = [[r.get(k, Fr(0)) for k in keys] for r in rows]
            if mat and keys:
                allok &= check("frame monomials of order %d independent (%d)"
                               % (W, len(terms)), rank(mat) == len(terms))

    print("\n%d/%d checks pass" % (sum(1 for _, v in OK if v), len(OK)))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
