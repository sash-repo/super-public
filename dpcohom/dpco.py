r"""dpco.py -- the divided power cochain complex of a Lie superalgebra for p = 2
as the dual of an explicit free resolution over an enveloping algebra.

Everything is exact over F_2.  A Lie superalgebra g = g_0 + g_1 for p = 2 is,
following [Le], [BGL]:

    a Z/2-graded space with a bilinear bracket, [x,x] = 0 for all x,
    Jacobi, plus a squaring  s: g_1 -> g_0  with
        s(ax) = a^2 s(x),   s(x+y) = s(x) + s(y) + [x,y],   [s(x),y] = [x,[x,y]].

The enveloping algebra used here is

    U(g) = T(g) / ( xy + yx + [x,y]  for all x,y;   x*x + s(x)  for x in g_1 ),

i.e. the ordinary enveloping algebra of the underlying Lie algebra divided
further by the squaring relations.  Its PBW basis: ordered monomials, even
letters with arbitrary exponents, odd letters with exponent <= 1.

The resolution is  V_n = U(g) (x) C_n,  where

    C = Lambda(g_0) (x) Gamma(g_1)          (divided powers on the ODD part)

with the Chevalley-Eilenberg differential

    d(u (x) c) = sum_k  (u g_k) (x) partial_k(c)   +   u (x) delta(c),

    delta = mult o (bracket (x) id) o Delta_{2, n-2},

where the "bracket" on C_2 is  e/\e' |-> [e,e'],  e.gamma_1(f) |-> [e,f],
gamma_1(f)gamma_1(f') |-> [f,f'],  and  gamma_2(f) |-> s(f).  The last item is
the only new one; it is the Koszul dual of the relation x*x = s(x), and it is
what puts divided powers into the cochains.

Dualizing, Hom_U(V_n, M) = Hom_K(C_n, M) carries

    (d phi)(c) = sum_k g_k . phi(partial_k c) + phi(delta c),

which in degrees <= 2 is verbatim A. Lebedev's codifferential of [BGL, Sec. 4.1]
and in all degrees the codifferential of the package SuperLie.

Basis conventions: g has basis e_0..e_{m-1} (even), f_0..f_{n-1} (odd);
a vector of g is an int bitmask, bits 0..m-1 even, bits m..m+n-1 odd.
"""

import sys
from itertools import combinations

sys.setrecursionlimit(100000)


# ---------------------------------------------------------------- the algebra

class SLA2:
    """A Lie superalgebra over F_2 with a squaring."""

    def __init__(self, m, n, brackets, squares, name="g"):
        """brackets: dict (i,j) -> bitmask for i < j over ALL basis indices
        (evens 0..m-1, odds m..m+n-1); squares: list of n bitmasks, s(f_j)."""
        self.m, self.n, self.name = m, n, name
        self.dim = m + n
        self.br = [[0] * self.dim for _ in range(self.dim)]
        for (i, j), v in brackets.items():
            self.br[i][j] = v
            self.br[j][i] = v            # p = 2: no signs
        self.sq = list(squares)

    # -- parity and elementary operations ---------------------------------
    def odd(self, i):
        return i >= self.m

    def bracket(self, x, y):
        """bracket of two vectors given as bitmasks"""
        out = 0
        for i in bits(x):
            for j in bits(y):
                out ^= self.br[i][j]
        return out

    def squaring(self, x):
        """s(x) for x in g_1 given as a bitmask of odd bits"""
        out = 0
        idx = [i - self.m for i in bits(x)]
        for j in idx:
            out ^= self.sq[j]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out ^= self.br[idx[a] + self.m][idx[b] + self.m]
        return out

    # -- axioms ------------------------------------------------------------
    def check_axioms(self):
        problems = []
        d, m = self.dim, self.m
        for i in range(d):
            if self.br[i][i] != 0:
                problems.append("[e_%d,e_%d] != 0" % (i, i))
        # parity of the bracket
        for i in range(d):
            for j in range(d):
                par = (i >= m) + (j >= m)
                v = self.br[i][j]
                for k in bits(v):
                    if (k >= m) != (par % 2 == 1):
                        problems.append("parity of [%d,%d]" % (i, j))
        # Jacobi
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    v = (self.bracket(1 << i, self.br[j][k])
                         ^ self.bracket(1 << j, self.br[i][k])
                         ^ self.bracket(1 << k, self.br[i][j]))
                    if v:
                        problems.append("Jacobi(%d,%d,%d)" % (i, j, k))
        # s(x) even, [s(x),y] = [x,[x,y]], polarization
        for j in range(self.n):
            for k in bits(self.sq[j]):
                if k >= m:
                    problems.append("s(f_%d) not even" % j)
        for x in range(1, 1 << self.n):
            xm = x << m
            sx = self.squaring(xm)
            for y in range(d):
                lhs = self.bracket(sx, 1 << y)
                rhs = self.bracket(xm, self.bracket(xm, 1 << y))
                if lhs != rhs:
                    problems.append("[s(x),y] != [x,[x,y]] at x=%d,y=%d" % (x, y))
        for x in range(1, 1 << self.n):
            for y in range(1, 1 << self.n):
                lhs = self.squaring((x ^ y) << m)
                rhs = (self.squaring(x << m) ^ self.squaring(y << m)
                       ^ self.bracket(x << m, y << m))
                if lhs != rhs:
                    problems.append("polarization at (%d,%d)" % (x, y))
        return problems

    # -- the adjoint and trivial modules -----------------------------------
    def trivial_module(self):
        return Module(1, [[0] * self.dim for _ in range(1)], [0], "K")

    def adjoint_module(self):
        act = [[self.br[k][i] for k in range(self.dim)] for i in range(self.dim)]
        par = [0 if i < self.m else 1 for i in range(self.dim)]
        return Module(self.dim, act, par, "g")


class Module:
    """A g-module: act[i][k] = the bitmask of g_k . v_i."""

    def __init__(self, dim, act, par, name="M"):
        self.dim, self.act, self.par, self.name = dim, act, par, name

    def apply(self, k, v):
        out = 0
        for i in bits(v):
            out ^= self.act[i][k]
        return out


def bits(x):
    out = []
    i = 0
    while x:
        if x & 1:
            out.append(i)
        x >>= 1
        i += 1
    return out


# ------------------------------------------------- the enveloping algebra U

class Env:
    """U(g) with the PBW basis; words are tuples of generator indices."""

    def __init__(self, g, with_squaring=True):
        self.g = g
        self.with_squaring = with_squaring     # False = the naive U(g_Lie)
        self._memo = {}

    def is_normal(self, w):
        for a in range(len(w) - 1):
            if w[a] > w[a + 1]:
                return False
            if self.with_squaring and w[a] == w[a + 1] and self.g.odd(w[a]):
                return False
        return True

    def reduce(self, w):
        """reduce a word to a set of normal words (F_2 coefficients)"""
        w = tuple(w)
        if w in self._memo:
            return self._memo[w]
        pos = -1
        for a in range(len(w) - 1):
            if w[a] > w[a + 1]:
                pos = a
                break
            if self.with_squaring and w[a] == w[a + 1] and self.g.odd(w[a]):
                pos = a
                break
        if pos < 0:
            res = frozenset([w])
            self._memo[w] = res
            return res
        i, j = w[pos], w[pos + 1]
        head, tail = w[:pos], w[pos + 2:]
        out = set()
        if i == j:                                    # odd square -> s(f)
            v = self.g.sq[i - self.g.m]
            for k in bits(v):
                out ^= set(self.reduce(head + (k,) + tail))
        else:                                         # swap + bracket
            out ^= set(self.reduce(head + (j, i) + tail))
            for k in bits(self.g.br[i][j]):
                out ^= set(self.reduce(head + (k,) + tail))
        res = frozenset(out)
        self._memo[w] = res
        return res

    def mul_word_gen(self, w, k):
        return self.reduce(w + (k,))

    def check_associativity(self, maxdeg=3):
        """(uv)w = u(vw) on all words of length <= maxdeg"""
        d = self.g.dim
        words = [()]
        cur = [()]
        for _ in range(maxdeg):
            cur = [w + (k,) for w in cur for k in range(d)]
            words += cur
        bad = 0
        for w in words:
            for k in range(d):
                for l in range(d):
                    a = set()
                    for u in self.reduce(w + (k,)):
                        a ^= set(self.reduce(u + (l,)))
                    b = set()
                    for u in self.reduce((k, l)):
                        b ^= set(self.reduce(w + u))
                    if a != b:
                        bad += 1
        return bad


# ------------------------------------------------ the coalgebra C and V = U(x)C

def gamma_basis(n, deg):
    """all alpha in N^n with |alpha| = deg"""
    if n == 0:
        return [()] if deg == 0 else []
    out = []
    for a in range(deg + 1):
        for rest in gamma_basis(n - 1, deg - a):
            out.append((a,) + rest)
    return out


class CE:
    """the Chevalley-Eilenberg coalgebra C = Lambda(g_0) (x) Gamma(g_1)."""

    def __init__(self, g):
        self.g = g

    def basis(self, deg):
        m, n = self.g.m, self.g.n
        out = []
        for k in range(min(deg, m) + 1):
            for S in combinations(range(m), k):
                for al in gamma_basis(n, deg - k):
                    out.append((frozenset(S), al))
        return out

    def partial(self, k, c):
        """the (1, n-1) component of the comultiplication at the generator k"""
        S, al = c
        if k < self.g.m:
            return (S - {k}, al) if k in S else None
        j = k - self.g.m
        if al[j] == 0:
            return None
        return (S, al[:j] + (al[j] - 1,) + al[j + 1:])

    def mul_vec(self, v, c):
        """multiply the C-monomial c by the vector v of g.  The product is
        LINEAR in v: v = sum of basis vectors gives a SUM of products."""
        S, al = c
        out = set()
        for k in bits(v):
            if k < self.g.m:
                if k in S:
                    continue                            # e/\e = 0
                out ^= {(S | {k}, al)}
            else:
                j = k - self.g.m
                if (al[j] + 1) % 2 == 0:                # binomial C(a+1,1)
                    continue
                out ^= {(S, al[:j] + (al[j] + 1,) + al[j + 1:])}
        return out

    def delta(self, c):
        """the CE differential of the coalgebra: C_k -> C_{k-1}"""
        S, al = c
        m = self.g.m
        out = set()
        Sl = sorted(S)
        # two even letters
        for a in range(len(Sl)):
            for b in range(a + 1, len(Sl)):
                i, ip = Sl[a], Sl[b]
                v = self.g.br[i][ip]
                if v:
                    out ^= self.mul_vec(v, (S - {i, ip}, al))
        # one even, one odd
        for i in Sl:
            for j in range(self.g.n):
                if al[j] == 0:
                    continue
                v = self.g.br[i][j + m]
                if v:
                    rest = (S - {i}, al[:j] + (al[j] - 1,) + al[j + 1:])
                    out ^= self.mul_vec(v, rest)
        # two distinct odd letters
        for j in range(self.g.n):
            for l in range(j + 1, self.g.n):
                if al[j] == 0 or al[l] == 0:
                    continue
                v = self.g.br[j + m][l + m]
                if v:
                    a1 = list(al)
                    a1[j] -= 1
                    a1[l] -= 1
                    out ^= self.mul_vec(v, (S, tuple(a1)))
        # the divided square of an odd letter -- the squaring term
        for j in range(self.g.n):
            if al[j] < 2:
                continue
            v = self.g.sq[j]
            if v:
                a1 = list(al)
                a1[j] -= 2
                out ^= self.mul_vec(v, (S, tuple(a1)))
        return out


class Resolution:
    """V_n = U(g) (x) C_n with the differential described in the docstring."""

    def __init__(self, g, env=None, ce=None):
        self.g = g
        self.env = env or Env(g)
        self.ce = ce or CE(g)

    def d_gen(self, c):
        """d(1 (x) c) as a set of pairs (word, C-monomial)"""
        out = set()
        for k in range(self.g.dim):
            p = self.ce.partial(k, c)
            if p is not None:
                out ^= {((k,), p)}
        for c2 in self.ce.delta(c):
            out ^= {((), c2)}
        return out

    def d(self, elt):
        """d of a set of pairs (word, C-monomial), left U-linearly"""
        out = set()
        for (w, c) in elt:
            for (w2, c2) in self.d_gen(c):
                for u in self.env.reduce(w + w2):
                    out ^= {(u, c2)}
        return out

    # -- the dual (cochain) complex ---------------------------------------
    def codiff_matrix(self, deg, M):
        """the matrix of d: C^deg -> C^{deg+1} with values in M.
        Cochains are Hom(C_deg, M); basis: (c, i) |-> the map c |-> v_i."""
        src = self.ce.basis(deg)
        dst = self.ce.basis(deg + 1)
        src_idx = {c: a for a, c in enumerate(src)}
        rows = []                     # one row per basis cochain of C^deg
        for (a, c) in enumerate(src):
            for i in range(M.dim):
                col = [0] * (len(dst) * M.dim)
                for (b, cc) in enumerate(dst):
                    val = 0
                    for k in range(self.g.dim):
                        p = self.ce.partial(k, cc)
                        if p is not None and p == c:
                            val ^= M.apply(k, 1 << i)
                    for c2 in self.ce.delta(cc):
                        if c2 == c:
                            val ^= (1 << i)
                    for t in bits(val):
                        col[b * M.dim + t] ^= 1
                rows.append(col)
        return rows, len(src) * M.dim, len(dst) * M.dim


# ------------------------------------------------------------ linear algebra

def rank2(rows, ncols):
    """rank over F_2 of a list of rows given as lists of 0/1 (or ints)"""
    piv = {}
    r = 0
    for row in rows:
        v = 0
        for j, x in enumerate(row):
            if x:
                v |= 1 << j
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                r += 1
                break
    return r


def rank2_int(vecs):
    piv = {}
    r = 0
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                r += 1
                break
    return r


def span_int(vecs):
    """echelon basis (as ints) of the span"""
    piv = {}
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                break
    return piv


def in_span(piv, v):
    while v:
        h = v.bit_length() - 1
        if h in piv:
            v ^= piv[h]
        else:
            return False
    return True


def cohomology_dims(res, M, maxdeg):
    """dim H^n(g;M) from the dual complex, n = 0..maxdeg"""
    mats = {}
    dims = {}
    for n in range(maxdeg + 2):
        rows, ns, nd = res.codiff_matrix(n, M)
        mats[n] = (rows, ns, nd)
        dims[n] = ns
    out = []
    for n in range(maxdeg + 1):
        rows, ns, nd = mats[n]
        rk = rank2(rows, nd)
        if n == 0:
            prev = 0
        else:
            prowsrows, pns, pnd = mats[n - 1]
            prev = rank2(prowsrows, pnd)
        out.append(dims[n] - rk - prev)
    return out
