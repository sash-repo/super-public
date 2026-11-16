#!/usr/bin/env python3
"""Exact F_3 engine for the contact Lie algebra k(3; 1) and its derived k^(1)(3; 1).

Generating functions are divided-power monomials t^(a) p^(b) q^(c) with 0<=a,b,c<=2
(the shearing vector N = 1, so O(3;1) is spanned by 27 monomials).

Contact bracket (Leites' normalisation, see [GL3]):

    {f,g} = (2-E)(f) d_t g - d_t f (2-E)(g) + {f,g}_PB,
    E = p d_p + q d_q,     {f,g}_PB = d_p f d_q g - d_q f d_p g.

Gradings: deg(t)=2, deg(p)=deg(q)=1, and deg K_f = deg f - 2;
pq-weight of t^(a)p^(b)q^(c) is c-b.  These are the two weights of the maximal
torus Span(K_t, K_pq) used in the text.
"""
from itertools import combinations
from functools import lru_cache

P = 3
MONS = [(a, b, c) for a in range(P) for b in range(P) for c in range(P)]
IDX = {m: i for i, m in enumerate(MONS)}


# ---------------------------------------------------------------- polynomials
def binom(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def pmul(f, g):
    """product of two divided-power polynomials, truncated at exponent p-1."""
    out = {}
    for (a1, b1, c1), u in f.items():
        for (a2, b2, c2), v in g.items():
            a, b, c = a1 + a2, b1 + b2, c1 + c2
            if a >= P or b >= P or c >= P:
                continue           # binomial would be 0 mod p anyway
            co = u * v * binom(a, a1) * binom(b, b1) * binom(c, c1) % P
            if co:
                out[(a, b, c)] = (out.get((a, b, c), 0) + co) % P
    return {k: v for k, v in out.items() if v}


def dpart(f, var):
    """partial derivative d/d(t,p,q) in divided powers: d x^(a) = x^(a-1)."""
    out = {}
    for m, u in f.items():
        if m[var] == 0:
            continue
        n = list(m)
        n[var] -= 1
        n = tuple(n)
        out[n] = (out.get(n, 0) + u) % P
    return {k: v for k, v in out.items() if v}


def euler(f):
    """E = p d_p + q d_q acting as multiplication by (b+c)."""
    return {m: u * (m[1] + m[2]) % P for m, u in f.items() if u * (m[1] + m[2]) % P}


def lin(*terms):
    out = {}
    for co, f in terms:
        for m, u in f.items():
            out[m] = (out.get(m, 0) + co * u) % P
    return {k: v for k, v in out.items() if v}


def kbracket(f, g):
    twoE_f = lin((2, f), (-1, euler(f)))
    twoE_g = lin((2, g), (-1, euler(g)))
    pb = lin((1, pmul(dpart(f, 1), dpart(g, 2))), (-1, pmul(dpart(f, 2), dpart(g, 1))))
    return lin((1, pmul(twoE_f, dpart(g, 0))), (-1, pmul(dpart(f, 0), twoE_g)), (1, pb))


def mono(m):
    return {m: 1}


def degK(m):          # degree of the contact vector field K_f
    return 2 * m[0] + m[1] + m[2] - 2


def wpq(m):           # pq-weight
    return m[2] - m[1]


# ------------------------------------------------------- linear algebra mod p
def rref(rows, ncols):
    """row-reduce; returns (reduced rows, list of pivot columns)."""
    mat = [r[:] for r in rows]
    piv = []
    r = 0
    for c in range(ncols):
        pr = None
        for i in range(r, len(mat)):
            if mat[i][c]:
                pr = i
                break
        if pr is None:
            continue
        mat[r], mat[pr] = mat[pr], mat[r]
        inv = pow(mat[r][c], P - 2, P)
        mat[r] = [x * inv % P for x in mat[r]]
        for i in range(len(mat)):
            if i != r and mat[i][c]:
                f = mat[i][c]
                mat[i] = [(x - f * y) % P for x, y in zip(mat[i], mat[r])]
        piv.append(c)
        r += 1
        if r == len(mat):
            break
    return mat[:r], piv


def rank(rows, ncols):
    return len(rref(rows, ncols)[1])


def nullspace(rows, ncols):
    """basis of {x : M x = 0} for M given by rows."""
    red, piv = rref(rows, ncols)
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for fc in free:
        v = [0] * ncols
        v[fc] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-red[i][fc]) % P
        basis.append(v)
    return basis


def in_span(vec, rows, ncols):
    r0 = rank(rows, ncols)
    return rank(rows + [vec], ncols) == r0


# --------------------------------------------------------------- the algebras
class LieAlg:
    """Lie algebra with a fixed basis, structure constants over F_p, and
    an integer bigrading (deg, pq-weight)."""

    def __init__(self, basis, brk, wt, names):
        self.basis = basis                  # list of labels
        self.n = len(basis)
        self.wt = wt                        # list of (deg, w) per basis vector
        self.names = names
        self.c = [[brk(i, j) for j in range(self.n)] for i in range(self.n)]

    def bracket(self, x, y):
        out = [0] * self.n
        for i, xi in enumerate(x):
            if not xi:
                continue
            for j, yj in enumerate(y):
                if not yj:
                    continue
                cij = self.c[i][j]
                if cij:
                    f = xi * yj % P
                    for k, v in cij.items():
                        out[k] = (out[k] + f * v) % P
        return out

    def check_jacobi(self):
        for i, j, k in combinations(range(self.n), 3):
            e = lambda t: [1 if s == t else 0 for s in range(self.n)]
            s = [0] * self.n
            for a, b, cc in ((i, j, k), (j, k, i), (k, i, j)):
                v = self.bracket(e(a), self.bracket(e(b), e(cc)))
                s = [(x + y) % P for x, y in zip(s, v)]
            if any(s):
                return False, (i, j, k)
        return True, None


def build_k3():
    """k(3;1): 27-dimensional, basis = monomials."""
    def brk(i, j):
        r = kbracket(mono(MONS[i]), mono(MONS[j]))
        return {IDX[m]: v for m, v in r.items()}
    names = []
    for (a, b, c) in MONS:
        s = ""
        for e, v in ((a, "t"), (b, "p"), (c, "q")):
            if e == 1:
                s += v
            elif e == 2:
                s += v + "^{(2)}"
        names.append(s if s else "1")
    wt = [(degK(m), wpq(m)) for m in MONS]
    return LieAlg(list(MONS), brk, wt, names)


def subalgebra(alg, vectors, names=None):
    """Lie algebra structure on the span of the given (independent) vectors."""
    red, piv = rref([v[:] for v in vectors], alg.n)
    m = len(red)

    def coords(v):
        v = v[:]
        out = [0] * m
        for i, pc in enumerate(piv):
            co = v[pc]
            if co:
                out[i] = co
                v = [(x - co * y) % P for x, y in zip(v, red[i])]
        assert not any(v), "vector not in the span"
        return out

    def brk(i, j):
        r = coords(alg.bracket(red[i], red[j]))
        return {k: v for k, v in enumerate(r) if v}

    wt = []
    for r in red:
        ws = {alg.wt[k] for k, v in enumerate(r) if v}
        wt.append(ws.pop() if len(ws) == 1 else None)
    nm = names or [f"e{i}" for i in range(m)]
    a = LieAlg(list(range(m)), brk, wt, nm)
    a.vectors = red
    return a


def derived(alg):
    rows = []
    for i in range(alg.n):
        for j in range(i + 1, alg.n):
            v = [0] * alg.n
            for k, c in alg.c[i][j].items():
                v[k] = c
            if any(v):
                rows.append(v)
    red, _ = rref(rows, alg.n)
    return red


if __name__ == "__main__":
    k = build_k3()
    ok, w = k.check_jacobi()
    print("k(3;1): dim", k.n, " Jacobi:", ok, w or "")
    d1 = derived(k)
    print("dim k^(1)(3;1) =", len(d1))
    # which monomial is missing from the derived algebra?
    miss = [i for i in range(k.n) if not in_span([1 if j == i else 0 for j in range(k.n)], d1, k.n)]
    print("monomials not in k^(1):", [k.names[i] for i in miss])
    d2 = rref(derived(subalgebra(k, d1)), len(d1))[0]
    print("dim k^(2) =", len(d2))
