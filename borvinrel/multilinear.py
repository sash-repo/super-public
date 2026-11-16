r"""The multilinear component of g(A): do relations of degree n in all the x_j exist?

This backs Statement 10.1.15 of the paper (relations of degree n in all the
generators, and the wall that carries them).

What is computed here
---------------------
For the multidegree alpha = (1,...,1) -- "degree 1 in each generator", the
multilinear component -- we build

  * L^alpha, the multilinear component of the FREE Lie superalgebra on n
    generators of prescribed parities (its dimension should be (n-1)! whatever
    the parities);
  * the lowering operators D_i of eq. (lowering) of the paper and the map Phi,
    whose kernel is the radical r_+^alpha, i.e., the space of relations;
  * the Gram matrix G^alpha of the contravariant form, whose kernel is the
    same space.

and we report, for a random integer symmetric A,

  * dim L^alpha              (expect (n-1)!)
  * rank G^alpha, corank     (corank = number of relations at that A)
  * the homogeneity degree of det G^alpha in the entries of A, measured by
    scaling A |-> t A.

The point.  Every entry of D_i is linear in the A_{ij} and Phi is a composition
of n-1 of them, so every entry of G^alpha is homogeneous of degree n-1 in the
A_{ij} and det G^alpha is homogeneous of degree (n-1)*(n-1)!.  Hence, as soon as
det G^alpha is not identically zero -- which is what the generic rank measures --
its zero locus is a non-empty hypersurface (a cone through the origin), and
relations of degree n involving all the generators exist on it, for every n.
There is no bound on n.

For n = 3 this hypersurface is the one printed in the paper,
det G^(1,1,1) = 9abc(a+b+c), of degree 4 = (n-1)(n-1)! in (a,b,c).

Usage:  python multilinear.py [nmax]
"""

import random
import sys
from collections import defaultdict
from fractions import Fraction
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank as exact_rank  # noqa: E402


# ---------------------------------------------------------------- words ----

def pw(word, par):
    return sum(par[j] for j in word) % 2


def bracket(u, v, par):
    """[u,v] = uv - (-1)^{|u||v|} vu inside T(V); dicts word -> int."""
    out = defaultdict(int)
    for wu, cu in u.items():
        for wv, cv in v.items():
            sign = -1 if (pw(wu, par) * pw(wv, par)) % 2 else 1
            out[wu + wv] += cu * cv
            out[wv + wu] -= sign * cu * cv
    return {w: c for w, c in out.items() if c}


class Span:
    """Row-reduced span of dicts word -> Fraction, kept in echelon form."""

    def __init__(self):
        self.rows = {}

    def reduce(self, vec):
        vec = {w: Fraction(c) for w, c in vec.items()}
        for piv in sorted(self.rows, key=lambda w: (len(w), w)):
            c = vec.get(piv)
            if c:
                for w, rc in self.rows[piv].items():
                    nv = vec.get(w, Fraction(0)) - c * rc
                    if nv:
                        vec[w] = nv
                    else:
                        vec.pop(w, None)
        return vec

    def add(self, vec):
        vec = self.reduce(vec)
        if not vec:
            return False
        piv = min(vec, key=lambda w: (len(w), w))
        c = vec[piv]
        self.rows[piv] = {w: v / c for w, v in vec.items()}
        return True

    def basis(self):
        return [self.rows[p] for p in sorted(self.rows, key=lambda w: (len(w), w))]


def squarefree_free_lie(n, par):
    """L[S] = basis of the multidegree-chi_S component of the free Lie superalgebra.

    S runs over the non-empty subsets of {0,...,n-1}, encoded as frozensets.
    Only squarefree multidegrees are built -- that is all the multilinear
    component needs, and it keeps the computation small.
    """
    L = {}
    for i in range(n):
        L[frozenset([i])] = [{(i,): Fraction(1)}]
    for k in range(2, n + 1):
        for S in combinations(range(n), k):
            S = frozenset(S)
            span = Span()
            for j in range(1, k // 2 + 1):
                for T in combinations(sorted(S), j):
                    T = frozenset(T)
                    U = S - T
                    if j == k - j and min(T) > min(U):
                        continue
                    for u in L[T]:
                        for v in L[U]:
                            span.add(bracket(u, v, par))
            L[S] = span.basis()
    return L


# ------------------------------------------------------------ lowering D ---

def lower(vec, i, A, par):
    """D_i on a vector of T_k(V), k >= 2.  Returns (body, h_part)."""
    body = defaultdict(Fraction)
    hpart = defaultdict(Fraction)
    for w, c in vec.items():
        for t, j in enumerate(w):
            if j != i:
                continue
            sgn = -1 if (par[i] * sum(par[w[s]] for s in range(t))) % 2 else 1
            tail = sum(A[i][w[s]] for s in range(t + 1, len(w)))
            wdel = w[:t] + w[t + 1:]
            body[wdel] += -sgn * c * tail
            hpart[wdel] += -sgn * c
    return (_prune(body), _prune(hpart))


def _prune(d):
    """Drop the zero entries; expand first if the coefficients are symbolic."""
    out = {}
    for w, v in d.items():
        if not isinstance(v, (int, Fraction)):
            import sympy as sp
            v = sp.expand(v)
        if v:
            out[w] = v
    return out


def phi_all(basis, A, n, par):
    """Phi(u)_w for every u in basis and every word w = permutation of (1,...,1).

    Done breadth-first over the prefixes, so that the n-1 lowerings are shared
    by all the basis elements.
    """
    out = [dict() for _ in basis]
    # state: prefix (tuple of letters already lowered) -> list of vectors
    level = {(): list(basis)}
    for step in range(n - 1):
        nxt = {}
        for prefix, vecs in level.items():
            used = set(prefix)
            for i in range(n):
                if i in used:
                    continue
                new = []
                for v in vecs:
                    if not v:
                        new.append({})
                        continue
                    body, hpart = lower(v, i, A, par)
                    if hpart:
                        raise AssertionError("h-part did not cancel at %s" % (prefix,))
                    new.append(body)
                nxt[prefix + (i,)] = new
        level = nxt
    # after n-1 steps each vector sits in height 1
    for prefix, vecs in level.items():
        last = [i for i in range(n) if i not in prefix]
        assert len(last) == 1
        w = prefix + (last[0],)
        for a, v in enumerate(vecs):
            out[a][w] = v.get((last[0],), Fraction(0))
    return out


def gram(basis, A, n, par):
    phis = phi_all(basis, A, n, par)
    d = len(basis)
    G = [[Fraction(0)] * d for _ in range(d)]
    for a in range(d):
        pa = phis[a]
        for b in range(d):
            s = Fraction(0)
            for w, c in basis[b].items():
                pv = pa.get(w)
                if pv:
                    s += c * pv
            G[a][b] = s
    return G


def det(M):
    """Exact determinant by fraction-free elimination."""
    M = [[Fraction(x) for x in row] for row in M]
    d = len(M)
    res = Fraction(1)
    for c in range(d):
        piv = None
        for r in range(c, d):
            if M[r][c]:
                piv = r
                break
        if piv is None:
            return Fraction(0)
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            res = -res
        res *= M[c][c]
        inv = Fraction(1) / M[c][c]
        for r in range(c + 1, d):
            f = M[r][c] * inv
            if f:
                for k in range(c, d):
                    M[r][k] -= f * M[c][k]
    return res


# ------------------------------------------------------------------ run ----

def factorial(k):
    r = 1
    for i in range(2, k + 1):
        r *= i
    return r


def random_symmetric(n, diag, rng, lo=-9, hi=9):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = diag
    for i in range(n):
        for j in range(i + 1, n):
            v = 0
            while v == 0:
                v = rng.randint(lo, hi)
            A[i][j] = A[j][i] = v
    return A


def on_full_wall(n, diag, rng):
    """A random symmetric A on the wall  sum_{i<j} A_ij = 0, no entry zero."""
    while True:
        A = random_symmetric(n, diag, rng, lo=-4000, hi=4000)
        s = sum(A[i][j] for i in range(n) for j in range(i + 1, n))
        A[0][1] -= s
        A[1][0] = A[0][1]
        if all(A[p][q] != 0 for p in range(n) for q in range(n) if p != q):
            return A


def calibrate():
    """The one published value: det G^(1,1,1) = 9abc(a+b+c), see St3x3."""
    import sympy as sp
    a, b, c = sp.symbols("a b c")
    ok = 0
    for diag, parity, name in ((0, 1, "zero diagonal, odd generators"),
                               (0, 0, "zero diagonal, even generators"),
                               (2, 0, "main diagonal 2, even generators")):
        A = [[diag, a, b], [a, diag, c], [b, c, diag]]
        par = [parity] * 3
        basis = squarefree_free_lie(3, par)[frozenset(range(3))]
        G = sp.Matrix([[sp.expand(x) for x in row] for row in gram(basis, A, 3, par)])
        d = sp.factor(sp.expand(G.det()))
        assert d == sp.factor(9 * a * b * c * (a + b + c)), (name, d)
        print("  n=3, %-33s det G^(1,1,1) = %s" % (name + ":", d))
        ok += 1
    return ok


def size_four_determinant():
    """The 4 x 4 multilinear determinant, computed symbolically and factored."""
    import sympy as sp
    v = sp.symbols("a12 a13 a14 a23 a24 a34")
    A = [[0] * 4 for _ in range(4)]
    k = 0
    for i in range(4):
        for j in range(i + 1, 4):
            A[i][j] = A[j][i] = v[k]
            k += 1
    par = [1] * 4
    basis = squarefree_free_lie(4, par)[frozenset(range(4))]
    G = sp.Matrix([[sp.expand(x) for x in row] for row in gram(basis, A, 4, par)])
    d = sp.expand(G.det(method="berkowitz"))
    assert sp.Poly(d, *v).total_degree() == 3 * 6      # (n-1)(n-1)!
    print("  n=4, zero diagonal, odd generators:")
    print("       det G^(1,1,1,1) = %s" % sp.factor(d))
    return 1


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = random.Random(20260920)
    checks = 0
    print("The multilinear component of g(A): relations of degree n in all the x_j")
    print()
    print("Calibration against the determinant printed in the paper:")
    checks += calibrate()
    print()
    print("The next size, computed symbolically:")
    checks += size_four_determinant()
    print()
    for case, diag, parity in (("zero diagonal, odd generators", 0, 1),
                               ("main diagonal 2, even generators", 2, 0)):
        print("=== %s ===" % case)
        for n in range(3, nmax + 1):
            par = [parity] * n
            basis = squarefree_free_lie(n, par)[frozenset(range(n))]
            expect = factorial(n - 1)
            assert len(basis) == expect, (n, len(basis), expect)
            checks += 1
            generic = []
            for _ in range(4):
                A = random_symmetric(n, diag, rng, lo=-4000, hi=4000)
                generic.append(len(basis) - exact_rank(gram(basis, A, n, par)))
                checks += 1
            assert min(generic) == 0, (n, diag, generic)
            wall = []
            for _ in range(3):
                A = on_full_wall(n, diag, rng)
                wall.append(len(basis) - exact_rank(gram(basis, A, n, par)))
                checks += 1
            assert min(wall) == factorial(n - 2), (n, diag, wall)
            checks += 1
            homog = ""
            if diag == 0 and len(basis) <= 24:
                A = random_symmetric(n, diag, rng)
                d1 = det(gram(basis, A, n, par))
                d2 = det(gram(basis, [[2 * x for x in row] for row in A], n, par))
                assert d1
                q, e = d2 / d1, 0
                while q % 2 == 0:
                    q //= 2
                    e += 1
                assert q == 1 and e == (n - 1) * factorial(n - 1), (n, e, q)
                checks += 1
                homog = ",  deg det G = %d = (n-1)(n-1)!" % e
            print("  n=%d: dim L^(1..1) = %d = (n-1)!;  corank of G at a generic A: "
                  "%d;  on the wall sum_{i<j}A_ij = 0: %d = (n-2)!%s"
                  % (n, len(basis), min(generic), min(wall), homog))
        print()
    print("checks passed: %d" % checks)


if __name__ == "__main__":
    main()
