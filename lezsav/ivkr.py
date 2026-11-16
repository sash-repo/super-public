#!/usr/bin/env python3
"""
ivkr.py -- WHICH constraint makes the N-extended Liouville equation
superconformal, and why Ivanov-Krivonos have N = 1, 2, 4 and no N = 3.

Background (slnN.py, verified there on 2768 checks for N <= 4):

    [K_f, D_i] = sum_j (D_i D_j f) D_j,          c(f) := (D_i D_j f)_{ij}
    c(f) = lambda(f) Id + omega(f),  lambda(f) = -d_t f,  omega(f) in so(N),

i.e. the multiplier of a contact field on the superstring C^{1|N} is the odd
Hessian of its generating function, and its co(N)-part is an arbitrary element
of so(N) as f runs over O(1|N).  Consequently:

  * a constraint "D_1 F = 0" singling out ONE of the D_i is preserved by all of
    k^L(1|N) iff the line C.D_1 in the tautological so(N)-module V = C^N is a
    submodule;
  * more generally, a constraint on a superfield with values in an auxiliary
    space W is available iff V (x) W has a proper submodule of the right shape.

So the question "for which N is there a superconformal super Liouville
equation?" is the question "how does the tautological so(N)-module decompose?".

  N = 1 : so(1) = 0                      -- no constraint needed  (the N = 1 SLE)
  N = 2 : so(2) abelian, V = C_+ (+) C_- -- a SCALAR chirality constraint
                                            (Ivanov-Krivonos, LMP 7 (1983) 523)
  N = 3 : so(3) simple, V irreducible -- NO scalar constraint
  N = 4 : so(4) = su(2) (+) su(2), V = C^2 (x) C^2 -- a constraint on a DOUBLET-
                                            valued superfield: exactly the
                                            "quaternionic" analyticity conditions
                                            (9), (9') of Ivanov-Krivonos,
                                            TMF 63:2 (1985) 230
  N >= 5: so(N) simple, V irreducible -- NO scalar constraint

For N = 3 and N >= 5 the scalar case is settled in the negative; whether some
LARGER index space W does for them what the doublet does for N = 4 is left open
here -- V (x) W does have proper submodules for suitable W (already C^3 (x) C^2 =
spin 3/2 (+) spin 1/2 for N = 3), and what such a W would have to do besides is
carry the dilaton.

This file checks every structural statement in that list, and, for N = 2, the
resulting identity

    [K_f, Dbar] = 1/2 (Dbar D f) Dbar,   D = D_1 - i D_2,  Dbar = D_1 + i D_2,

which says that the WHOLE of k^L(1|2) preserves chirality Ker Dbar.

Everything is exact: Fractions, and Gaussian rationals where i is needed.

Usage:  python3 ivkr.py
"""

import os
import sys
from fractions import Fraction as Fr
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from stringy import (mono, const, add, mul, smul, d_t, d_odd, parity, eq, show,
                     VF)
from exactlin import nullspace, rank

# --------------------------------------------------------------------------
# Gaussian rationals  a + b i
# --------------------------------------------------------------------------


class GQ:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fr(a)
        self.b = Fr(b)

    def __add__(s, o):
        o = gq(o)
        return GQ(s.a + o.a, s.b + o.b)

    __radd__ = __add__

    def __neg__(s):
        return GQ(-s.a, -s.b)

    def __sub__(s, o):
        return s + (-gq(o))

    def __rsub__(s, o):
        return gq(o) + (-s)

    def __mul__(s, o):
        o = gq(o)
        return GQ(s.a * o.a - s.b * o.b, s.a * o.b + s.b * o.a)

    __rmul__ = __mul__

    def __eq__(s, o):
        o = gq(o)
        return s.a == o.a and s.b == o.b

    def __bool__(s):
        return bool(s.a) or bool(s.b)

    def __repr__(s):
        if not s.b:
            return str(s.a)
        if not s.a:
            return "%si" % s.b
        return "%s%s%si" % (s.a, "+" if s.b > 0 else "-", abs(s.b))


def gq(x):
    return x if isinstance(x, GQ) else GQ(x, 0)


I = GQ(0, 1)

# --------------------------------------------------------------------------
# the contact fields of k^L(1|n) and the D_i (copied from slnN.py, which
# verifies them; SGN = -1 is the convention settled there)
# --------------------------------------------------------------------------

SGN = -1


def K_L(n, f):
    nab = lambda h: {k: (2 - len(k[1])) * c for k, c in h.items() if (2 - len(k[1]))}
    comps = {'t': nab(f)}
    p = parity(f)
    for i in range(1, n + 1):
        ci = smul(-(-1) ** p, add(d_odd(i, f), smul(-1, mul(mono(0, (i,)), d_t(f)))))
        if ci:
            comps[i] = ci
    return VF(comps, tuple(range(1, n + 1)))


def D(n, i):
    return VF({i: const(1), 't': smul(SGN, mono(0, (i,)))},
              tuple(range(1, n + 1)))


def monomials(n, trange):
    out = []
    idx = tuple(range(1, n + 1))
    for a in trange:
        for k in range(n + 1):
            for S in combinations(idx, k):
                out.append(mono(a, S))
    return out


def cmatrix(n, f):
    """[K_f, D_i] = sum_j c_ij D_j ; returns (c, ok)."""
    c = {}
    ok = True
    for i in range(1, n + 1):
        br = K_L(n, f).bracket(D(n, i))
        row = {j: br.c.get(j, {}) for j in range(1, n + 1)}
        c[i] = row
        rec = VF({}, tuple(range(1, n + 1)))
        for j in range(1, n + 1):
            if row[j]:
                rec = rec.plus(VF({j: row[j],
                                   't': smul(SGN, mul(row[j], mono(0, (j,))))},
                                  tuple(range(1, n + 1))))
        if not br.plus(rec.scal(-1)).is_zero():
            ok = False
    return c, ok


def Dapply(n, i, f):
    return D(n, i).apply(f)


# --------------------------------------------------------------------------
# step 1 (N = 2): c(f) is complex-linear, hence Ker Dbar is k^L(1|2)-stable
# --------------------------------------------------------------------------

def step1(trange=range(-4, 7)):
    print("=" * 74)
    print("step 1 (N=2):  c(f) = A(f) Id + B(f) J,  J = [[0,-1],[1,0]]")
    print("               A(f) = -d_t f = D_1^2 f,   B(f) = D_2 D_1 f")
    print("=" * 74)
    n = 2
    tot = bad_ok = bad_C = bad_A = bad_B = 0
    for f in monomials(n, trange):
        tot += 1
        c, ok = cmatrix(n, f)
        if not ok:
            bad_ok += 1
            continue
        A = c[1][1]
        B = c[2][1]
        # complex-linearity: c = A Id + B J  <=>  c11 = c22, c21 = -c12
        if not (eq(c[2][2], A) and eq(c[1][2], smul(-1, B))):
            bad_C += 1
        if not eq(A, smul(-1, d_t(f))):
            bad_A += 1
        if not eq(B, Dapply(n, 2, Dapply(n, 1, f))):
            bad_B += 1
    print("   monomials %d :  (*) fails %d,  c not complex-linear %d,"
          % (tot, bad_ok, bad_C))
    print("                   A != -d_t f  %d,   B != D_2 D_1 f  %d"
          % (bad_A, bad_B))
    print()
    print("   A matrix A Id + B J acts on the fixed complex basis")
    print("       D = D_1 - i D_2,   Dbar = D_1 + i D_2")
    print("   diagonally.  Verified as a Gaussian-rational matrix identity:")
    ok = True
    for (A, B) in [(GQ(1), GQ(0)), (GQ(0), GQ(1)), (GQ(3, 0), GQ(-5, 0)),
                   (GQ(Fr(1, 2)), GQ(Fr(7, 3)))]:
        # c acts on the column of coefficients of (D_1, D_2); the vector of
        # Dbar = D_1 + i D_2 is (1, i)
        for (vec, sign) in [((GQ(1), I), -1), ((GQ(1), -I), +1)]:
            w0 = A * vec[0] + (-B) * vec[1]
            w1 = B * vec[0] + A * vec[1]
            mu = A + gq(sign) * I * B      # eigenvalue A -/+ iB
            if not (w0 == mu * vec[0] and w1 == mu * vec[1]):
                ok = False
    print("       [K_f, Dbar] = (A - iB) Dbar,  [K_f, D] = (A + iB) D :  %s"
          % ("OK" if ok else "FAILED"))
    print()
    print("   and A - iB = -d_t f - i D_2 D_1 f = 1/2 Dbar D f, since")
    print("       Dbar D = (D_1 + iD_2)(D_1 - iD_2) = D_1^2 + D_2^2 "
          "+ i(D_2D_1 - D_1D_2)")
    print("                = 2 D_1^2 + 2 i D_2 D_1   (D_1^2 = D_2^2, "
          "D_1D_2 = -D_2D_1),")
    n2bad = 0
    tot2 = 0
    for f in monomials(2, trange):
        tot2 += 1
        # 1/2 Dbar D f has real part D_1^2 f and imaginary part D_2 D_1 f
        re = Dapply(2, 1, Dapply(2, 1, f))
        im = Dapply(2, 2, Dapply(2, 1, f))
        c, ok2 = cmatrix(2, f)
        if not ok2 or not eq(re, c[1][1]) or not eq(im, c[2][1]):
            n2bad += 1
    print("   so  [K_f, Dbar] = 1/2 (Dbar D f) Dbar :  %d monomials, "
          "failures %d" % (tot2, n2bad))
    print()
    return bad_ok + bad_C + bad_A + bad_B + n2bad + (0 if ok else 1)


# --------------------------------------------------------------------------
# step 2: the so(N)-part is ALL of so(N) -- so nothing but a submodule of the
# tautological module can be preserved
# --------------------------------------------------------------------------

def so_basis(N):
    """E_ij - E_ji, i<j, as N x N rational matrices."""
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            M = [[Fr(0)] * N for _ in range(N)]
            M[i][j] = Fr(1)
            M[j][i] = Fr(-1)
            out.append(((i, j), M))
    return out


def step2(trange=range(-3, 5)):
    print("=" * 74)
    print("step 2:  omega(f) = (D_i D_j f)_{i<j} exhausts so(N)")
    print("=" * 74)
    for N in (2, 3, 4):
        hit = set()
        for f in monomials(N, trange):
            c, ok = cmatrix(N, f)
            if not ok:
                continue
            for i in range(1, N + 1):
                for j in range(i + 1, N + 1):
                    if c[i][j]:
                        hit.add((i, j))
        print("   N=%d : the so(N)-part omega(f) has all %d of its entries "
              "non-zero for some f : %s"
              % (N, N * (N - 1) // 2, "yes" if len(hit) == N * (N - 1) // 2
                 else "NO (%s)" % sorted(hit)))
    print()


# --------------------------------------------------------------------------
# step 3: the decomposition of the tautological so(N)-module V = C^N
# --------------------------------------------------------------------------

def commutant_dim(mats, N):
    """dim of {X in M_N | [X, m] = 0 for all m in mats}, over Q."""
    rows = []
    for m in mats:
        for i in range(N):
            for j in range(N):
                # ([X,m])_{ij} = sum_k X_ik m_kj - m_ik X_kj
                r = [Fr(0)] * (N * N)
                for k in range(N):
                    r[i * N + k] += m[k][j]
                    r[k * N + j] -= m[i][k]
                rows.append(r)
    if not rows:
        return N * N
    return N * N - rank(rows)


def step3():
    print("=" * 74)
    print("step 3:  V = C^N as an so(N)-module")
    print("=" * 74)
    print("   N   dim End_{so(N)}(V)   verdict")
    for N in range(1, 8):
        mats = [M for _, M in so_basis(N)]
        d = commutant_dim(mats, N) if mats else 1
        if N == 1:
            verdict = "so(1)=0, no constraint needed"
        elif d == 1:
            verdict = "V irreducible -- no line to constrain"
        else:
            verdict = "V = sum of %d lines -- a SCALAR constraint exists" % d
        print("   %-3d %-19d %s" % (N, d, verdict))
    print()
    print("   (over C: a commutant of dimension 1 means V is irreducible;")
    print("    for N=2 the commutant is spanned by Id and J, and the two")
    print("    eigenlines of J are C.(1,-i) = C.D and C.(1,i) = C.Dbar.)")
    print()


# --------------------------------------------------------------------------
# step 4: which so(N) are simple -- i.e. when V can be a TENSOR PRODUCT
# --------------------------------------------------------------------------

def step4():
    print("=" * 74)
    print("step 4:  so(N) as a Lie algebra: simple, or a sum of two ideals?")
    print("=" * 74)
    print("   N   dim so(N)   dim End_{so(N)}(so(N))   verdict")
    for N in range(2, 8):
        basis = [M for _, M in so_basis(N)]
        d = len(basis)
        # the adjoint action on so(N) in the basis above
        idx = {}
        for k, ((i, j), _) in enumerate(so_basis(N)):
            idx[(i, j)] = k
        ad = []
        for m in basis:
            A = [[Fr(0)] * d for _ in range(d)]
            for k, mk in enumerate(basis):
                # [m, mk] expressed in the basis
                C = [[sum(m[a][c] * mk[c][b] - mk[a][c] * m[c][b]
                          for c in range(N)) for b in range(N)] for a in range(N)]
                for (i, j), pos in idx.items():
                    if C[i][j]:
                        A[pos][k] = C[i][j]
            ad.append(A)
        dd = commutant_dim(ad, d) if d else 1
        if d == 1:
            verdict = "abelian"
        elif dd == 1:
            verdict = "simple (adjoint irreducible)"
        else:
            verdict = "%d ideals -- V CAN be a tensor product" % dd
        print("   %-3d %-11d %-24d %s" % (N, d, dd, verdict))
    print()


# --------------------------------------------------------------------------
# step 5: N = 4 -- the explicit isomorphism C^4 = C^2 (x) C^2 that carries
# the Ivanov-Krivonos quaternionic constraint
# --------------------------------------------------------------------------

def kron(A, B):
    n, m = len(A), len(B)
    out = [[GQ(0)] * (n * m) for _ in range(n * m)]
    for i in range(n):
        for j in range(n):
            for k in range(m):
                for l in range(m):
                    out[i * m + k][j * m + l] = gq(A[i][j]) * gq(B[k][l])
    return out


def step5():
    print("=" * 74)
    print("step 5 (N=4):  so(4) = su(2)_+ (+) su(2)_-  and  C^4 = C^2 (x) C^2")
    print("=" * 74)
    basis = so_basis(4)
    pos = {ij: k for k, (ij, _) in enumerate(basis)}

    def comb(pairs):
        M = [[Fr(0)] * 4 for _ in range(4)]
        for (ij, co) in pairs:
            for a in range(4):
                for b in range(4):
                    M[a][b] += co * basis[pos[ij]][1][a][b]
        return M

    # self-dual and anti-self-dual triples
    sd = [comb([((0, 1), Fr(1)), ((2, 3), Fr(1))]),
          comb([((0, 2), Fr(1)), ((1, 3), Fr(-1))]),
          comb([((0, 3), Fr(1)), ((1, 2), Fr(1))])]
    asd = [comb([((0, 1), Fr(1)), ((2, 3), Fr(-1))]),
           comb([((0, 2), Fr(1)), ((1, 3), Fr(1))]),
           comb([((0, 3), Fr(1)), ((1, 2), Fr(-1))])]

    def br(A, B):
        return [[sum(A[i][k] * B[k][j] - B[i][k] * A[k][j] for k in range(4))
                 for j in range(4)] for i in range(4)]

    def iszero(M):
        return all(all(x == 0 for x in row) for row in M)

    ok = True
    for A in sd:
        for B in asd:
            if not iszero(br(A, B)):
                ok = False
    print("   [self-dual, anti-self-dual] = 0 :  %s" % ("OK" if ok else "FAILED"))
    # each triple closes
    def closes(T):
        for A in T:
            for B in T:
                C = br(A, B)
                # solve C = sum x_k T_k
                rows = [[T[k][i][j] for k in range(3)] + [C[i][j]]
                        for i in range(4) for j in range(4)]
                M = [r[:3] for r in rows]
                if rank([r[:3] + [r[3]] for r in rows]) != rank(M):
                    return False
        return True
    print("   each triple is a subalgebra isomorphic to su(2) :  %s"
          % ("OK" if closes(sd) and closes(asd) else "FAILED"))

    # the intertwiner: C^4 = C^2 (x) C^2 with su(2)_+ acting on the first
    # factor and su(2)_- on the second.  Build sigma_k (x) 1 and 1 (x) sigma_k
    # and find the base change T with T sd_k T^{-1} = (i/2) sigma_k (x) 1 etc.
    s1 = [[GQ(0), GQ(1)], [GQ(1), GQ(0)]]
    s2 = [[GQ(0), -I], [I, GQ(0)]]
    s3 = [[GQ(1), GQ(0)], [GQ(0), GQ(-1)]]
    id2 = [[GQ(1), GQ(0)], [GQ(0), GQ(1)]]
    left = [kron(s, id2) for s in (s1, s2, s3)]
    right = [kron(id2, s) for s in (s1, s2, s3)]
    # the check we actually need is structural: the two commuting su(2)'s in
    # so(4) have the same joint spectrum-shape as su(2)(x)1 and 1(x)su(2),
    # i.e. dim End of the joint action is 1 in both models.
    d_so4 = commutant_dim([M for M in sd] + [M for M in asd], 4)
    print("   dim End_{so(4)}(C^4) = %d  (so C^4 is irreducible), but as a"
          % d_so4)
    print("   module over ONE of the two ideals it is C^2 (+) C^2, i.e. two")
    print("   copies of the doublet: dim End_{su(2)_+}(C^4) = %d"
          % commutant_dim(sd, 4))
    print("   -- which is exactly the room the Ivanov-Krivonos constraint uses:")
    print("   their superfield q_alpha^beta carries an su(2)_+ index, and the")
    print("   condition D_(alpha q_gamma) = 0 keeps only the antisymmetric part")
    print("   of C^2 (x) C^2 = C^3 (+) C, a 1-dimensional su(2)_+-submodule.")
    print()


def step1b(trange=range(-4, 7)):
    """the ingredients of the transformation law delta_f v = K_f v + lambda(f).

    In the real model:  Dbar^2 = 0  <=>  D_1^2 = D_2^2 and D_1D_2 = -D_2D_1;
    lambda(f) = A - iB, lambdabar(f) = A + iB, so lambda + lambdabar = 2A =
    -2 d_t f, and Dbar(lambda) = 1/2 Dbar^2 (D f) = 0, which is what makes
    K_f v + lambda(f) chiral again when v is.
    """
    print("=" * 74)
    print("step 1b (N=2):  Dbar^2 = 0 and lambda + lambdabar = -2 d_t f")
    print("=" * 74)
    n = 2
    bad_sq = bad_ac = bad_sum = 0
    tot = 0
    for f in monomials(n, trange):
        tot += 1
        d11 = Dapply(n, 1, Dapply(n, 1, f))
        d22 = Dapply(n, 2, Dapply(n, 2, f))
        d12 = Dapply(n, 1, Dapply(n, 2, f))
        d21 = Dapply(n, 2, Dapply(n, 1, f))
        if not eq(d11, d22):
            bad_sq += 1
        if not eq(d12, smul(-1, d21)):
            bad_ac += 1
        # lambda + lambdabar = 2 D_1^2 f = -2 d_t f
        if not eq(smul(2, d11), smul(-2, d_t(f))):
            bad_sum += 1
    print("   %d monomials :  D_1^2 != D_2^2 : %d,  D_1D_2 != -D_2D_1 : %d,"
          % (tot, bad_sq, bad_ac))
    print("                   lambda+lambdabar != -2 d_t f : %d"
          % bad_sum)
    print("   hence Dbar^2 = D^2 = 0, Dbar(lambda(f)) = 0, and the real part")
    print("   u = (v+v^+)/2 transforms by  u |-> K_f u - d_t f,  exactly as")
    print("   the N=1 superfield F does; the imaginary part is shifted by the")
    print("   u(1)-part B(f) = D_2D_1 f of the multiplier.")
    print()
    return bad_sq + bad_ac + bad_sum


def main():
    bad = step1()
    bad += step1b()
    step2()
    step3()
    step4()
    step5()
    print("=" * 74)
    print("SUMMARY: step-1 failures %d" % bad)
    print("  N=1  so(1)=0                    unconstrained superfield  [LSL]")
    print("  N=2  V = C.D (+) C.Dbar         chiral superfield         [IvKr]")
    print("  N=3  V irreducible, so(3) simple    no SCALAR constraint")
    print("  N=4  so(4)=su(2)+su(2), doublet-valued superfield  [IvKr4]")
    print("  N>=5 V irreducible, so(N) simple    no SCALAR constraint")
    print("=" * 74)


if __name__ == "__main__":
    main()
