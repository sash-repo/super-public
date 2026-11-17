#!/usr/bin/env python3
"""
Exact-arithmetic engine for the queer series q(n), sq(n), pq(n), psq(n) and
their Duflo--Serganova homology: Lemma 2.2.2, Remark 2.2.3 and table (8)
of the text, and the second family (7).

Realization (formula (6) of the text):
    T_{A,B} = [[A,B],[B,A]],   T_{A,0} even,  T_{0,B} odd,
    [T_{A,0},T_{A',0}] = T_{[A,A'],0},   [T_{A,0},T_{0,B'}] = T_{0,[A,B']},
    [T_{0,B},T_{0,B'}] = T_{BB'+B'B,0}.
    sq(n) = {T_{A,B} : tr B = 0},  pq(n) = q(n)/K.1,  psq(n) = sq(n)/K.1.

Homologic elements:
    in q(n), sq(n):   x = T_{0,B} with B^2 = 0;
    in pq(n), psq(n): x = T_{0,B} with B^2 in K.Id  -- the second family
                      (B^2 = c.Id, c != 0) has NO counterpart upstairs.

Usage: python3 qsq.py [nmax]      (default nmax = 7)
"""
import sys
from fractions import Fraction
from po import rank, null_space, independent_subset, rref


# ------------------------------------------------------------ matrix helpers
def mzero(n):
    return [[0] * n for _ in range(n)]

def mmul(X, Y):
    n = len(X)
    return [[sum(X[i][k] * Y[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]

def madd(X, Y, c=1):
    return [[X[i][j] + c * Y[i][j] for j in range(len(X))] for i in range(len(X))]

def mtr(X):
    return sum(X[i][i] for i in range(len(X)))


# -------------------------------------------------------- the algebra q(n)
class Q:
    """
    q(n) with the basis  (0,i,j) = T_{E_ij,0}  (even),  (1,i,j) = T_{0,E_ij} (odd).
    Index: even (i,j) -> i*n+j ; odd (i,j) -> n*n + i*n + j.
    """
    def __init__(self, n):
        self.n = n
        self.dim = 2 * n * n
        self.par = [0] * (n * n) + [1] * (n * n)

    def idx(self, p, i, j):
        return p * self.n * self.n + i * self.n + j

    def vec_from_pair(self, A, B):
        n = self.n
        v = [Fraction(0)] * self.dim
        for i in range(n):
            for j in range(n):
                if A[i][j]:
                    v[self.idx(0, i, j)] = Fraction(A[i][j])
                if B[i][j]:
                    v[self.idx(1, i, j)] = Fraction(B[i][j])
        return v

    def pair_from_vec(self, v):
        n = self.n
        A, B = mzero(n), mzero(n)
        for i in range(n):
            for j in range(n):
                A[i][j] = v[self.idx(0, i, j)]
                B[i][j] = v[self.idx(1, i, j)]
        return A, B

    def bracket(self, u, v):
        """u,v arbitrary vectors (sum of an even and an odd part)."""
        n = self.n
        A1, B1 = self.pair_from_vec(u)
        A2, B2 = self.pair_from_vec(v)
        # even-even
        Ae = madd(mmul(A1, A2), mmul(A2, A1), -1)
        # odd-odd  (anticommutator)
        Ao = madd(mmul(B1, B2), mmul(B2, B1))
        # even-odd and odd-even
        Bo = madd(madd(mmul(A1, B2), mmul(B2, A1), -1),
                  madd(mmul(B1, A2), mmul(A2, B1), -1))
        return self.vec_from_pair(madd(Ae, Ao), Bo)

    def ad_matrix_on(self, x, basis):
        """matrix of ad_x restricted to the span of `basis` (list of vectors),
        expressed in that basis; assumes the span is ad_x-stable."""
        cols = []
        for b in basis:
            cols.append(self.bracket(x, b))
        return cols   # as list of image vectors (ambient coords)


# --------------------------------------------------- subquotients of q(n)
class SubQuot:
    """
    The Lie superalgebra S/I, where S is a subalgebra of q(n) given by a basis
    of ambient vectors and I an ideal of S.  Carries the induced bracket.
    """
    def __init__(self, amb, S, I):
        self.amb = amb
        self.S = S
        self.I = I
        keep = independent_subset(I + S)
        self.reps = [(I + S)[i] for i in keep if i >= len(I)]
        self.basis_all = I + self.reps
        self.dim = len(self.reps)
        self.par = []
        for v in self.reps:
            ps = {amb.par[i] for i, c in enumerate(v) if c != 0}
            self.par.append(ps.pop() if len(ps) == 1 else None)
        self._proj = None

    def sdim(self):
        return (sum(1 for p in self.par if p == 0),
                sum(1 for p in self.par if p == 1))

    def project(self, v):
        """coordinates of v in S, modulo I."""
        if self._proj is None:
            B = self.basis_all
            K = len(B)
            rows = [[Fraction(B[j][i]) for j in range(K)] for i in range(self.amb.dim)]
            keepr = independent_subset(rows)
            S = [rows[i][:] for i in keepr]
            A = [S[i] + [Fraction(1) if i == j else Fraction(0) for j in range(K)]
                 for i in range(K)]
            for c in range(K):
                p = next(i for i in range(c, K) if A[i][c] != 0)
                A[c], A[p] = A[p], A[c]
                pv = A[c][c]
                A[c] = [t / pv for t in A[c]]
                for i in range(K):
                    if i != c and A[i][c] != 0:
                        f = A[i][c]
                        A[i] = [a - f * b for a, b in zip(A[i], A[c])]
            self._proj = (keepr, [r[K:] for r in A], len(self.I))
        keepr, inv, nI = self._proj
        sub = [Fraction(v[i]) for i in keepr]
        out = []
        for r in range(nI, len(sub)):
            s = Fraction(0)
            for j, sj in enumerate(sub):
                if sj:
                    s += inv[r][j] * sj
            out.append(s)
        return out

    def ambient(self, coords):
        v = [Fraction(0)] * self.amb.dim
        for c, b in zip(coords, self.reps):
            if c:
                for i, bi in enumerate(b):
                    v[i] += c * bi
        return v

    def ad_matrix(self, xcoords):
        """matrix of ad_x on S/I in the basis self.reps."""
        x = self.ambient(xcoords)
        cols = []
        for b in self.reps:
            cols.append(self.project(self.amb.bracket(x, b)))
        # columns -> matrix
        d = self.dim
        return [[cols[j][i] for j in range(d)] for i in range(d)]

    def ds(self, xcoords):
        """(rank ad_x, dim g_x, sdim g_x, representative vectors of g_x)."""
        M = self.ad_matrix(xcoords)
        r = rank(M)
        ker = null_space(M, self.dim)
        cols = [list(c) for c in zip(*M)]
        im = [cols[i] for i in independent_subset(cols)]
        keep = independent_subset(im + ker)
        reps = [(im + ker)[i] for i in keep if i >= len(im)]
        ev = od = 0
        for v in reps:
            ps = {self.par[i] for i, c in enumerate(v) if c != 0}
            if len(ps) == 1:
                if ps.pop() == 0:
                    ev += 1
                else:
                    od += 1
        return r, len(reps), (ev, od), reps, im


def build(n, kind):
    """kind in {'q','sq','pq','psq'} -> (ambient Q(n), SubQuot)."""
    amb = Q(n)
    Id = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    Z = mzero(n)
    S = []
    for i in range(n):
        for j in range(n):
            E = mzero(n); E[i][j] = 1
            S.append(amb.vec_from_pair(E, Z))              # even part: all of gl(n)
    # odd part
    for i in range(n):
        for j in range(n):
            if kind in ('q', 'pq') or i != j:
                E = mzero(n); E[i][j] = 1
                S.append(amb.vec_from_pair(Z, E))
    if kind in ('sq', 'psq'):
        for i in range(n - 1):
            E = mzero(n); E[i][i] = 1; E[i + 1][i + 1] = -1
            S.append(amb.vec_from_pair(Z, E))
    I = []
    if kind in ('pq', 'psq'):
        I.append(amb.vec_from_pair(Id, Z))
    return amb, SubQuot(amb, S, I)


def xk_matrix(n, k):
    """B = E_12 + E_34 + ... + E_{2k-1,2k}."""
    B = mzero(n)
    for j in range(k):
        B[2 * j][2 * j + 1] = 1
    return B


def run(nmax=7):
    print("Duflo--Serganova homology of the queer series, exact over Q")
    print("x_k = T_{0,B}, B = E_12 + E_34 + ... + E_{2k-1,2k}  (rank B = k)\n")
    for kind in ('q', 'sq', 'pq', 'psq'):
        print(f"--- {kind}(n) " + "-" * 60)
        print(f"{'n':>3} {'dim':>7} {'k':>3} {'rank ad_x':>10} {'dim g_x':>9} "
              f"{'sdim g_x':>10}   expected")
        for n in range(2, nmax + 1):
            amb, L = build(n, kind)
            for k in range(1, n // 2 + 1):
                B = xk_matrix(n, k)
                x = L.project(amb.vec_from_pair(mzero(n), B))
                r, dg, sd, reps, im = L.ds(x)
                # expected: same kind with n-2k, except at k = n//2
                if n - 2 * k >= 1:
                    _, L2 = build(n - 2 * k, kind)
                    exp = f"{kind}({n-2*k}): dim {L2.dim} = {L2.sdim()[0]}|{L2.sdim()[1]}"
                else:
                    exp = f"{kind}(0) = 0"
                mark = "  <== " if (n - 2 * k >= 1 and dg != L2.dim) or \
                                   (n - 2 * k < 1 and dg != 0) else "      "
                print(f"{n:>3} {L.dim:>7} {k:>3} {r:>10} {dg:>9} "
                      f"{str(sd[0])+'|'+str(sd[1]):>10} {mark}{exp}")
        print()

    # ---- the second family of homologic elements, present only in pq, psq
    print("--- homologic x with B^2 = c.Id, c != 0 (exists only in pq(n), psq(n))")
    print(f"{'kind':>5} {'n':>3} {'rank ad_x':>10} {'dim g_x':>9} {'sdim g_x':>10}"
          f"   [x,x] in q(n)")
    for kind in ('pq', 'psq'):
        for n in range(2, nmax + 1):
            B = mzero(n)
            for i in range(n):
                B[i][i] = 1 if i < (n + 1) // 2 else -1
            if kind == 'psq' and mtr(B) != 0:
                note = "  (tr B != 0: not in psq for odd n)"
                print(f"{kind:>5} {n:>3} {'--':>10} {'--':>9} {'--':>10}{note}")
                continue
            amb, L = build(n, kind)
            x = L.project(amb.vec_from_pair(mzero(n), B))
            r, dg, sd, reps, im = L.ds(x)
            print(f"{kind:>5} {n:>3} {r:>10} {dg:>9} "
                  f"{str(sd[0])+'|'+str(sd[1]):>10}   [x,x] = 2T_{{B^2,0}} = 2.Id != 0")


def blocks(nmax=7):
    """
    Remark 2.2.3: is rank ad_{x_k} on q(n) twice that of gl(n)?
    ad_x is block-antidiagonal: the even part goes to the odd one by
    A |-> [B,A] (the gl(n) adjoint), the odd part goes to the even one by
    A |-> BA + AB (the anticommutator).  Print the rank of each block.
    """
    print("\n--- ad_{x_k} on q(n) split into its two blocks")
    print(f"{'n':>3} {'k':>3} {'rk [B,-] on gl(n)':>18} {'rk (BA+AB)':>12} "
          f"{'total':>7} {'2k(n-k)':>9} {'4k(n-k)':>9}")
    for n in range(2, nmax + 1):
        for k in range(1, n // 2 + 1):
            B = xk_matrix(n, k)
            comm, anti = [], []
            for i in range(n):
                for j in range(n):
                    E = mzero(n); E[i][j] = 1
                    comm.append([c for row in madd(mmul(B, E), mmul(E, B), -1) for c in row])
                    anti.append([c for row in madd(mmul(B, E), mmul(E, B)) for c in row])
            rc, ra = rank([list(r) for r in zip(*comm)]), rank([list(r) for r in zip(*anti)])
            print(f"{n:>3} {k:>3} {rc:>18} {ra:>12} {rc+ra:>7} "
                  f"{2*k*(n-k):>9} {4*k*(n-k):>9}")


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    run(nmax)
    blocks(nmax)
