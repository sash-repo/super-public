#!/usr/bin/env python3
"""Exact check of the realization of psl(2|2) by supermatrices from sl(2|2).

The realization of psl(2|2) in the text ends with

    [X,Y]_new = XY - (-1)^{p(X)p(Y)} XY - (2 str XY) . 1_4 .

Two things are tested here over Q, in the 4x4 matrix realization:

  1) the second product must be YX (as typed, the first two terms cancel on the
     even part);

  2) whatever the first two terms, no multiple of str(XY) can serve as the
     correcting coefficient: the correction must be the projection along C.1_4
     onto a chosen complement V of C.1_4 in sl(2|2), and str(X . 1_4) = str X = 0
     says that str cannot cut out such a complement.

     The complement used here is V = {X in sl(2|2) | tr A = 0}, where A is the
     upper-left 2x2 block; the coefficient is then lambda(X,Y) = (1/2) tr A([X,Y]).
     We verify: V is a complement, the corrected bracket closes on V, it is
     super anti-symmetric, it satisfies the super Jacobi identity, sdim V = (6|8),
     and V (with the new bracket) is isomorphic to sl(2|2)/C.1_4 = psl(2|2) as a
     superalgebra -- the projection sl(2|2) -> psl(2|2) restricted to V is an
     isomorphism intertwining the two brackets.

Everything is exact rational arithmetic; no floating point anywhere.
"""
from fractions import Fraction as F
from itertools import product

N = 4          # size
EVEN_IDX = (0, 1)   # rows/cols of the first block
ODD_IDX = (2, 3)


def parity(i, j):
    """Parity of the matrix unit E_ij in the 2|2 format."""
    return (0 if i in EVEN_IDX else 1) ^ (0 if j in EVEN_IDX else 1)


def zero():
    return [[F(0)] * N for _ in range(N)]


def eunit(i, j):
    M = zero()
    M[i][j] = F(1)
    return M


def mul(X, Y):
    return [[sum((X[i][k] * Y[k][j] for k in range(N)), F(0)) for j in range(N)]
            for i in range(N)]


def add(X, Y, c=F(1)):
    return [[X[i][j] + c * Y[i][j] for j in range(N)] for i in range(N)]


def scal(c, X):
    return [[c * X[i][j] for j in range(N)] for i in range(N)]


def str_(X):
    """Supertrace in the 2|2 format."""
    return X[0][0] + X[1][1] - X[2][2] - X[3][3]


def trA(X):
    """Trace of the upper-left block."""
    return X[0][0] + X[1][1]


def is_zero(X):
    return all(X[i][j] == 0 for i in range(N) for j in range(N))


ONE4 = [[F(1) if i == j else F(0) for j in range(N)] for i in range(N)]


# ---------------------------------------------------------------- bases
def basis_sl():
    """Homogeneous basis of sl(2|2) = {str X = 0}; returns (matrix, parity) pairs."""
    out = []
    for i, j in product(range(N), repeat=2):
        if i != j:
            out.append((eunit(i, j), parity(i, j)))
    # diagonal part of sl: str = 0, i.e. a11+a22 = a33+a44
    for (i, j, c) in ((0, 1, F(-1)), (2, 3, F(-1)), (0, 2, F(1))):
        X = zero()
        X[i][i] = F(1)
        X[j][j] = c
        out.append((X, 0))
    return out                       # 12 off-diagonal + 3 diagonal = 15


def basis_V():
    """Homogeneous basis of V = {X in sl(2|2) | tr A = 0} (so tr D = 0 too)."""
    out = []
    for i, j in product(range(N), repeat=2):
        if i != j:
            out.append((eunit(i, j), parity(i, j)))
    out.append((add(eunit(0, 0), eunit(1, 1), F(-1)), 0))   # E11 - E22
    out.append((add(eunit(2, 2), eunit(3, 3), F(-1)), 0))   # E33 - E44
    return out                       # 12 + 2 = 14


def brk_old(X, pX, Y, pY):
    """The Lie superalgebra bracket XY - (-1)^{p(X)p(Y)} YX."""
    return add(mul(X, Y), mul(Y, X), -F(-1) ** (pX * pY))


def brk_dl(X, pX, Y, pY):
    """the formula as typed: XY - (-1)^{pp} XY - (2 str XY) 1_4."""
    B = add(mul(X, Y), mul(X, Y), -F(-1) ** (pX * pY))
    return add(B, ONE4, -2 * str_(mul(X, Y)))


def brk_dl_yx(X, pX, Y, pY):
    """the formula with the product repaired: XY - (-1)^{pp} YX - (2 str XY) 1_4."""
    return add(brk_old(X, pX, Y, pY), ONE4, -2 * str_(mul(X, Y)))


def brk_new(X, pX, Y, pY):
    """The corrected bracket: project [X,Y] along C.1_4 onto V = {tr A = 0}."""
    B = brk_old(X, pX, Y, pY)
    return add(B, ONE4, -trA(B) / 2)


# ---------------------------------------------------------------- checks
def report(name, ok, detail=""):
    print("%-4s %s%s" % ("OK" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    return ok


def main():
    ok = True
    sl = basis_sl()
    V = basis_V()

    ok &= report("sl(2|2): sdim = (7|8), dim 15",
                 len(sl) == 15
                 and sum(1 for _, p in sl if p == 0) == 7
                 and sum(1 for _, p in sl if p == 1) == 8)
    ok &= report("1_4 lies in sl(2|2): str 1_4 = 0", str_(ONE4) == 0)
    ok &= report("V: sdim = (6|8), dim 14",
                 len(V) == 14
                 and sum(1 for _, p in V if p == 0) == 6
                 and sum(1 for _, p in V if p == 1) == 8)
    ok &= report("V is a complement of C.1_4 in sl(2|2): tr A(1_4) = 2 != 0",
                 trA(ONE4) == 2 and all(trA(X) == 0 for X, _ in V))

    # 1) the formula as typed kills the even part
    dead = []
    for X, pX in V:
        for Y, pY in V:
            if pX == 0 and pY == 0:
                B = brk_dl(X, pX, Y, pY)
                C = add(B, ONE4, 2 * str_(mul(X, Y)))     # the first two terms alone
                if not is_zero(C):
                    dead.append((X, Y))
    ok &= report("as typed, XY - (-1)^{pp} XY vanishes on all even pairs",
                 not dead, "(%d nonzero out of %d)" % (len(dead), 6 * 6))

    # ... and the resulting bracket is not even anti-symmetric / not a bracket
    bad = 0
    for X, pX in V:
        for Y, pY in V:
            L = brk_dl(X, pX, Y, pY)
            R = brk_dl(Y, pY, X, pX)
            if not is_zero(add(L, R, F(-1) ** (pX * pY))):
                bad += 1
    ok &= report("as typed, the bracket is not super anti-symmetric",
                 bad > 0, "(%d failing pairs of %d)" % (bad, 14 * 14))

    # 2) no multiple of str(XY) can work: str(X . 1_4) = 0 for all X in sl
    ok &= report("str(X.1_4) = 0 for every X in sl(2|2)  =>  str defines no complement",
                 all(str_(mul(X, ONE4)) == 0 for X, _ in sl))

    # with YX repaired, the bracket still leaves V (so the coefficient is wrong)
    out = 0
    for X, pX in V:
        for Y, pY in V:
            if trA(brk_dl_yx(X, pX, Y, pY)) != 0:
                out += 1
    ok &= report("with YX repaired, (2 str XY).1_4 still leaves V",
                 out > 0, "(%d pairs of %d land outside)" % (out, 14 * 14))

    # 3) the corrected bracket
    out = 0
    for X, pX in V:
        for Y, pY in V:
            if trA(brk_new(X, pX, Y, pY)) != 0 or str_(brk_new(X, pX, Y, pY)) != 0:
                out += 1
    ok &= report("corrected bracket closes on V", out == 0)

    bad = 0
    for X, pX in V:
        for Y, pY in V:
            L = brk_new(X, pX, Y, pY)
            R = brk_new(Y, pY, X, pX)
            if not is_zero(add(L, R, F(-1) ** (pX * pY))):
                bad += 1
    ok &= report("corrected bracket is super anti-symmetric", bad == 0)

    bad = 0
    for X, pX in V:
        for Y, pY in V:
            for Z, pZ in V:
                # (-1)^{pXpZ}[[X,Y],Z] + (-1)^{pYpX}[[Y,Z],X] + (-1)^{pZpY}[[Z,X],Y] = 0
                t1 = scal(F(-1) ** (pX * pZ),
                          brk_new(brk_new(X, pX, Y, pY), (pX + pY) % 2, Z, pZ))
                t2 = scal(F(-1) ** (pY * pX),
                          brk_new(brk_new(Y, pY, Z, pZ), (pY + pZ) % 2, X, pX))
                t3 = scal(F(-1) ** (pZ * pY),
                          brk_new(brk_new(Z, pZ, X, pX), (pZ + pX) % 2, Y, pY))
                if not is_zero(add(add(t1, t2), t3)):
                    bad += 1
    ok &= report("corrected bracket satisfies the super Jacobi identity",
                 bad == 0, "(%d triples checked)" % (14 ** 3))

    # 4) V with the new bracket IS psl(2|2): the projection is an isomorphism
    #    pi: sl -> sl/C1_4; on V it is bijective, and pi([X,Y]_new) = pi([X,Y]).
    bad = 0
    for X, pX in V:
        for Y, pY in V:
            d = add(brk_new(X, pX, Y, pY), brk_old(X, pX, Y, pY), F(-1))
            # the difference must be a multiple of 1_4
            c = d[0][0]
            if not is_zero(add(d, ONE4, -c)):
                bad += 1
    ok &= report("[.,.]_new and [.,.] differ by a multiple of 1_4  "
                 "=> V ~ sl(2|2)/C.1_4 = psl(2|2)", bad == 0)

    # 5) what the coefficient really is, in closed form, on odd x odd:
    #    lambda(X,Y) = (1/2) tr A([X,Y]) = (1/2) tr(B C' + B' C),  while
    #    str(XY)     = tr(B C') - tr(B' C).   They differ in a sign, so no
    #    normalization of str repairs the formula.
    diff = 0
    for X, pX in V:
        for Y, pY in V:
            if pX == 1 and pY == 1:
                lam = trA(brk_old(X, pX, Y, pY)) / 2
                if lam != 0 and str_(mul(X, Y)) != 0 and lam != str_(mul(X, Y)):
                    diff += 1
    ok &= report("on odd pairs, (1/2) tr A([X,Y]) differs from str(XY)",
                 diff > 0, "(%d pairs)" % diff)

    print()
    print("ALL CHECKS PASSED" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
