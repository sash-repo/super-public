#!/usr/bin/env python3
"""oo.py -- the superization by "method 2" of [BLLS, Subsec. 3.3], and the
identification of the result for n = 1.

s(g, gr) is the Lie superalgebra structure on the restricted closure
(g, gr) = g + Span{x^2 | x in g_-}, the odd part being g_-, with x^2 := x^{[2]}.
Here gr is the Z/2-grading of the (2,4)-structure, i.e., the reduction mod 2 of
the Z-grading of o^{(1)}_I(2n+1): the X- and Y-blocks are odd.

For n = 1 the result is "oo(1|2)" of Sect. 4.  There are two ortho-orthogonal
Lie superalgebras of that format at p = 2 -- oo_{IPi}(1|2) and oo_{II}(1|2) --
and they differ exactly in the squaring on the odd part, which is what a
"method 2" superization is about.  Both are built here and compared with
s(o^{(1)}_I(3)) by an exhaustive search over the parity-preserving linear maps
over F_2 (|GL(3,2)| x |GL(2,2)| = 1008 of them).

A Lie superalgebra at p = 2 is a pair (bracket, squaring); an isomorphism must
respect both.  The squaring is 2-semilinear: (ax)^2 = a^2 x^2, and
(x+y)^2 = x^2 + y^2 + [x,y].
"""

import itertools

from wk import (GF, MatAlg, Space, mat_zero, mat_mul, mat_add, mat_comm, flat,
                oI, order_of, restricted_closure)


# --------------------------------------------------------------------------
# a Lie superalgebra at p = 2, by structure constants + squaring
# --------------------------------------------------------------------------

class SuperAlg:
    """Even basis vectors first, then odd ones."""

    def __init__(self, F, nev, nod, struct, sq):
        self.F, self.nev, self.nod = F, nev, nod
        self.n = nev + nod
        self.struct = struct          # struct[i][j] = coords of [e_i, e_j]
        self.sq = sq                  # sq[r] = coords of (odd basis r)^2

    @staticmethod
    def from_matalg(alg, odd):
        """alg a matrix Lie algebra, odd a list of booleans; reorder so that
        the even vectors come first."""
        F = alg.F
        order = [i for i in range(alg.n) if not odd[i]] + \
                [i for i in range(alg.n) if odd[i]]
        nev = sum(1 for o in odd if not o)
        nod = alg.n - nev
        basis = [alg.basis[i] for i in order]
        new = MatAlg(F, basis)
        new.build_struct()
        sq = []
        for r in range(nod):
            B = basis[nev + r]
            sq.append(new.coords(mat_mul(F, B, B)))
        return SuperAlg(F, nev, nod, new.struct, sq), new

    def square(self, v):
        """the square of an odd vector v (given in the full coordinates)."""
        F = self.F
        out = [0] * self.n
        for r in range(self.nod):
            a = v[self.nev + r]
            if a:
                a2 = F.mul(a, a)
                for k in range(self.n):
                    if self.sq[r][k]:
                        out[k] = F.add(out[k], F.mul(a2, self.sq[r][k]))
        for r in range(self.nod):
            for s in range(r + 1, self.nod):
                c = F.mul(v[self.nev + r], v[self.nev + s])
                if c:
                    row = self.struct[self.nev + r][self.nev + s]
                    for k in range(self.n):
                        if row[k]:
                            out[k] = F.add(out[k], F.mul(c, row[k]))
        return out

    def bracket(self, u, v):
        F = self.F
        out = [0] * self.n
        for i, a in enumerate(u):
            if not a:
                continue
            for j, b in enumerate(v):
                if not b:
                    continue
                c = F.mul(a, b)
                row = self.struct[i][j]
                for k in range(self.n):
                    if row[k]:
                        out[k] = F.add(out[k], F.mul(c, row[k]))
        return out

    def check_axioms(self):
        """[x^2, y] = [x,[x,y]] for odd x, and the additivity of the square."""
        F = self.F
        for r in range(self.nod):
            x = [1 if k == self.nev + r else 0 for k in range(self.n)]
            for j in range(self.n):
                y = [1 if k == j else 0 for k in range(self.n)]
                if self.bracket(self.square(x), y) != \
                        self.bracket(x, self.bracket(x, y)):
                    return False, ("sq-ad", r, j)
        for coefs in itertools.product(F.els, repeat=self.nod):
            v = [0] * self.nev + list(coefs)
            s = self.square(v)
            # recompute from the definition on a basis
            acc = [0] * self.n
            for r in range(self.nod):
                a = coefs[r]
                if a:
                    a2 = F.mul(a, a)
                    for k in range(self.n):
                        acc[k] = F.add(acc[k], F.mul(a2, self.sq[r][k]))
            for r in range(self.nod):
                for s2 in range(r + 1, self.nod):
                    c = F.mul(coefs[r], coefs[s2])
                    if c:
                        row = self.struct[self.nev + r][self.nev + s2]
                        for k in range(self.n):
                            acc[k] = F.add(acc[k], F.mul(c, row[k]))
            if acc != s:
                return False, ("add", coefs)
        return True, None

    def square_image_rank(self):
        S = Space(self.F, self.n)
        for coefs in itertools.product(self.F.els, repeat=self.nod):
            v = [0] * self.nev + list(coefs)
            S.add(self.square(v))
        return S.rank()

    def is_simple(self):
        """exhaustive over F_2-points: the ideal generated by each nonzero
        vector (closed under the bracket AND the square) is everything."""
        for coef in itertools.product([0, 1], repeat=self.n):
            if not any(coef):
                continue
            S = Space(self.F, self.n)
            S.add(list(coef))
            changed = True
            while changed:
                changed = False
                for w in list(S.basis()):
                    for i in range(self.n):
                        e = [1 if j == i else 0 for j in range(self.n)]
                        if S.add(self.bracket(e, w)):
                            changed = True
                    if any(w[self.nev:]):
                        if S.add(self.square([0] * self.nev + w[self.nev:])):
                            changed = True
            if S.rank() != self.n:
                return False, list(coef)
        return True, None


def iso(A, B):
    """Is there a parity-preserving isomorphism A -> B?  Exhaustive over the
    invertible maps with entries in F_2 (the algebras are defined over F_2)."""
    if (A.nev, A.nod) != (B.nev, B.nod):
        return False
    F = A.F
    ev_maps = _invertible(A.nev)
    od_maps = _invertible(A.nod)
    for P in ev_maps:
        for Q in od_maps:
            T = [[0] * A.n for _ in range(A.n)]
            for i in range(A.nev):
                for j in range(A.nev):
                    T[i][j] = P[i][j]
            for i in range(A.nod):
                for j in range(A.nod):
                    T[A.nev + i][A.nev + j] = Q[i][j]
            if _is_hom(A, B, T):
                return True
    return False


def _invertible(n):
    out = []
    for cols in itertools.product(itertools.product([0, 1], repeat=n),
                                  repeat=n):
        M = [[cols[j][i] for j in range(n)] for i in range(n)]
        S = Space(GF(1), n)
        if all(S.add(list(r)) for r in M):
            out.append(M)
    return out


def _apply(T, v, F):
    n = len(v)
    return [_dot(T[i], v, F) for i in range(n)]


def _dot(row, v, F):
    acc = 0
    for j, c in enumerate(row):
        if c and v[j]:
            acc = F.add(acc, F.mul(c, v[j]))
    return acc


def _is_hom(A, B, T):
    F = A.F
    for i in range(A.n):
        ei = [1 if k == i else 0 for k in range(A.n)]
        Ti = _apply(T, ei, F)
        for j in range(i + 1, A.n):
            ej = [1 if k == j else 0 for k in range(A.n)]
            if _apply(T, A.struct[i][j], F) != B.bracket(Ti, _apply(T, ej, F)):
                return False
    for r in range(A.nod):
        er = [1 if k == A.nev + r else 0 for k in range(A.n)]
        if _apply(T, A.sq[r], F) != B.square(_apply(T, er, F)):
            return False
    return True


# --------------------------------------------------------------------------
# the ortho-orthogonal Lie superalgebras at p = 2
# --------------------------------------------------------------------------

def oo(F, a, b, type1="Pi", derived=True):
    """oo_{I,type1}(a|2b) at p = 2 in the matrix realization: the even form is
    I on V_ev (dim a) and Pi (alternating) or I on V_od (dim 2b).
    Returns (SuperAlg, MatAlg, odd-flags)."""
    m = a + 2 * b
    G0 = [[1 if i == j else 0 for j in range(a)] for i in range(a)]
    n1 = 2 * b
    if type1 == "Pi":
        G1 = [[1 if i + j == n1 - 1 else 0 for j in range(n1)]
              for i in range(n1)]
    else:
        G1 = [[1 if i == j else 0 for j in range(n1)] for i in range(n1)]
    G = mat_zero(m)
    for i in range(a):
        for j in range(a):
            G[i][j] = G0[i][j]
    for i in range(n1):
        for j in range(n1):
            G[a + i][a + j] = G1[i][j]
    # the whole superalgebra: {X | GX + X^T G = 0}, X homogeneous
    basis, odd = [], []
    for par in (0, 1):
        for i in range(m):
            for j in range(m):
                pass
    # build by solving the linear condition on each parity block
    def solve_block(rows, cols, Gr, Gc):
        """X: cols -> rows block; condition Gr X + (X')^T Gc = 0 couples the
        two off-diagonal blocks, so handle even and odd separately below."""
        return None

    # even part: block-diagonal
    ev = []
    for (idx, Gb) in ((list(range(a)), G0), (list(range(a, m)), G1)):
        k = len(idx)
        if k == 0:
            continue
        # solve the linear condition Gb M + M^T Gb = 0 over F_2
        rows = []
        for r in range(k):
            for c in range(k):
                row = [0] * (k * k)
                for i in range(k):
                    for j in range(k):
                        val = F.add(F.mul(Gb[r][i], 1 if j == c else 0),
                                    F.mul(1 if j == r else 0, Gb[i][c]))
                        # (Gb M)_{rc} = sum_i Gb[r][i] M[i][c]
                        # (M^T Gb)_{rc} = sum_i M[i][r] Gb[i][c]
                        coef = 0
                        if j == c:
                            coef = F.add(coef, Gb[r][i])
                        if j == r:
                            coef = F.add(coef, Gb[i][c])
                        row[i * k + j] = coef
                rows.append(row)
        from wk import nullspace
        for v in nullspace(F, rows, k * k):
            A = mat_zero(m)
            for i in range(k):
                for j in range(k):
                    A[idx[i]][idx[j]] = v[i * k + j]
            ev.append(A)
    ev = _reduce_mats(F, ev, m)
    # odd part: X = [[0, Q],[R, 0]]; condition G0 Q + R^T G1 = 0
    od = []
    for i in range(a):
        for j in range(n1):
            Q = [[1 if (r, c) == (i, j) else 0 for c in range(n1)]
                 for r in range(a)]
            # R = G1^{-1} Q^T G0 ; both G1 and G0 are involutions here
            R = mat_mul_rect(F, mat_mul_rect(F, G1, _tr(Q)), G0)
            A = mat_zero(m)
            for r in range(a):
                for c in range(n1):
                    A[r][a + c] = Q[r][c]
            for r in range(n1):
                for c in range(a):
                    A[a + r][c] = R[r][c]
            od.append(A)
    od = _reduce_mats(F, od, m)
    alg = MatAlg(F, ev + od)
    odd = [False] * len(ev) + [True] * len(od)
    if derived:
        alg, odd = _derived(alg, odd)
    S, _ = SuperAlg.from_matalg(alg, odd)
    return S, alg, odd


def _derived(alg, odd):
    """the derived algebra [g,g] + Span{x^2 | x odd}, with its parities."""
    F = alg.F
    alg.build_struct()
    S = Space(F, alg.m * alg.m)
    ev, od = [], []
    cands = []
    for i in range(alg.n):
        for j in range(i + 1, alg.n):
            C = mat_comm(F, alg.basis[i], alg.basis[j])
            cands.append((C, odd[i] != odd[j]))
    for i in range(alg.n):
        if odd[i]:
            cands.append((mat_mul(F, alg.basis[i], alg.basis[i]), False))
    for C, o in cands:
        if any(any(r) for r in C) and S.add(flat(C)):
            (od if o else ev).append(C)
    new = MatAlg(F, ev + od)
    return new, [False] * len(ev) + [True] * len(od)


def _matrix_basis(F, k):
    for i in range(k):
        for j in range(k):
            yield [[1 if (r, c) == (i, j) else 0 for c in range(k)]
                   for r in range(k)]


def _tr(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def mat_mul_rect(F, A, B):
    n, k, m = len(A), len(B), len(B[0])
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for t in range(k):
            if A[i][t]:
                for j in range(m):
                    if B[t][j]:
                        C[i][j] = F.add(C[i][j], F.mul(A[i][t], B[t][j]))
    return C


def _reduce_mats(F, mats, m):
    S = Space(F, m * m)
    out = []
    for M in mats:
        if any(any(r) for r in M) and S.add(flat(M)):
            out.append(M)
    return out


# --------------------------------------------------------------------------
def method2(F, n):
    """s(o^{(1)}_I(2n+1)) -- the restricted closure superized."""
    alg, deg = oI(F, n)
    alg.build_struct()
    L, odd = restricted_closure(alg)
    L.build_struct()
    S, Lnew = SuperAlg.from_matalg(L, odd)
    return S, L, odd, alg
