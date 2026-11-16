"""podd.py -- the same construction for p > 2, and the Hypothesis of [BGL, Sec. 4].

Engine for the case p > 2 (Theorem "two lattices" of the text).  Everything is
exact over F_p.

WHAT IS HERE.

1) U(g) for a Lie superalgebra over F_p, p > 2, with the signs restored:

      U(g) = T(g)/( x (x) y - (-1)^{p(x)p(y)} y (x) x - [x,y] ),

   in which every odd x satisfies x^2 = s(x) := (1/2)[x,x] -- so the odd part is
   again "restricted by force" and gr U(g) = S(g_ev) (x) Lambda(g_od).  PBW is
   checked by associativity of the rewriting.

2) The divided power complex: C = Lambda(g_ev) (x) Gamma(g_od), the resolution
   V = U (x) C, and the codifferential

      (d phi)(c) = sum_k eps_k(c) x_k . phi(partial_k c) + phi(delta c),
      delta = mult o (beta (x) id) o Delta_{2,.},
      beta(z_k z_l) = [x_k,x_l],   beta(gamma_2(z_k)) = s(x_k)  (x_k odd),

   the generator z_k = Pi x_k having parity p(x_k) + 1, so that the g_ev-slots
   are exterior and the g_od-slots carry divided powers.  The SIGNS eps, and
   those inside delta, are not guessed: they are searched for in the family
   (-1)^{a p(x_k) + b p(x_l) + c p(x_k) p(x_l) + ...} and pinned down by the
   requirement d^2 = 0, on all the samples at once.

3) The super anti-symmetric complex AC of [BGL, Sec. 4], verbatim: cochains are
   the n-linear functions on g with values in M that are anti-symmetric in the
   even arguments and symmetric in the odd ones, and

      d(f)(X_1,..,X_n) =
        sum_{i<j} (-1)^{p(X_j)(p(X_{i+1})+..+p(X_{j-1})) + j-1}
                  f(X_1,..,[X_i,X_j],..,^X_j,..,X_n)
      + sum_j (-1)^{p(X_j)(p(f)+p(X_1)+..+p(X_{j-1})) + j-1}
                  X_j f(X_1,..,^X_j,..,X_n).

4) THE ANSWER TO THE HYPOTHESIS of [BGL, Sec. 4] ("Supposedly, when p = 0 or
   n < p, there is a natural isomorphism between E^n(g^*) (x) M and the space of
   super anti-symmetric n-linear functions ... Open Problem: how are the two
   isomorphisms defined?").  The map is the RESTITUTION

      Phi(gamma_{a_1}(z_{j_1}) ... e_S (x) m) = the multilinear function whose
      value on the arguments is read off the monomial with the Koszul signs,

   i.e. the polarization of a divided power monomial; the composition of Phi with
   the obvious map back is multiplication by  prod_j a_j! .  Hence Phi is
   bijective if and only if every odd multiplicity a_j is < p -- and the sharp
   condition is on the ODD degree of the cochain, not on its total degree n:
   in degree n with b odd arguments the two theories agree as soon as b < p.
   Above that they differ, and it is the divided power one that computes Ext.
"""

import sys
from itertools import combinations, product

sys.setrecursionlimit(100000)


# --------------------------------------------------------------- F_p vectors

class Fp:
    def __init__(self, p):
        self.p = p

    def inv(self, a):
        return pow(a % self.p, self.p - 2, self.p)


def vadd(u, v, p):
    return tuple((a + b) % p for a, b in zip(u, v))


def vscal(a, v, p):
    return tuple((a * x) % p for x in v)


def vzero(d):
    return tuple([0] * d)


def vbasis(d, i):
    return tuple(1 if j == i else 0 for j in range(d))


def viszero(v):
    return all(x == 0 for x in v)


# ------------------------------------------------------------ the superalgebra

class SLAp:
    """A Lie superalgebra over F_p, p > 2.

    m even generators x_0..x_{m-1}, n odd ones x_m..x_{m+n-1};
    br[(i,j)] = the vector of [x_i,x_j] for i <= j (the rest by supersymmetry).
    """

    def __init__(self, p, m, n, brackets, name="g"):
        assert p > 2
        self.p, self.m, self.n, self.name = p, m, n, name
        self.dim = d = m + n
        self.br = [[vzero(d) for _ in range(d)] for _ in range(d)]
        for (i, j), v in brackets.items():
            v = tuple(x % p for x in v)
            self.br[i][j] = v
            sgn = -1 if (self.par(i) == 0 or self.par(j) == 0) else 1
            self.br[j][i] = vscal(sgn % p, v, p)
        # the squaring on the odd part: s(x) = (1/2)[x,x]
        self.sq = [vscal(pow(2, p - 2, p), self.br[m + j][m + j], p)
                   for j in range(n)]

    def par(self, i):
        return 0 if i < self.m else 1

    def bracket_vec(self, u, v):
        p, d = self.p, self.dim
        out = vzero(d)
        for i in range(d):
            if u[i] == 0:
                continue
            for j in range(d):
                if v[j] == 0:
                    continue
                out = vadd(out, vscal(u[i] * v[j] % p, self.br[i][j], p), p)
        return out

    def check_axioms(self):
        bad = []
        p, d, m = self.p, self.dim, self.m
        for i in range(d):
            for j in range(d):
                sgn = -1 if (self.par(i) == 0 or self.par(j) == 0) else 1
                if self.br[j][i] != vscal(sgn % p, self.br[i][j], p):
                    bad.append("supersymmetry (%d,%d)" % (i, j))
                # parity of the bracket
                par = (self.par(i) + self.par(j)) % 2
                for k in range(d):
                    if self.br[i][j][k] and self.par(k) != par:
                        bad.append("parity of [%d,%d]" % (i, j))
        # super Jacobi: [x,[y,z]] = [[x,y],z] + (-1)^{p(x)p(y)}[y,[x,z]]
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    lhs = self.bracket_vec(vbasis(d, i), self.br[j][k])
                    r1 = self.bracket_vec(self.br[i][j], vbasis(d, k))
                    s = (-1) ** (self.par(i) * self.par(j)) % p
                    r2 = vscal(s, self.bracket_vec(vbasis(d, j), self.br[i][k]), p)
                    if lhs != vadd(r1, r2, p):
                        bad.append("Jacobi(%d,%d,%d)" % (i, j, k))
        # [s(x),y] = [x,[x,y]] for x odd
        for j in range(self.n):
            for y in range(d):
                lhs = self.bracket_vec(self.sq[j], vbasis(d, y))
                rhs = self.bracket_vec(vbasis(d, m + j),
                                       self.br[m + j][y])
                if lhs != rhs:
                    bad.append("[s(f_%d),x_%d] != [f,[f,x]]" % (j, y))
        return bad

    # --- modules ---------------------------------------------------------
    def trivial(self):
        return ModP(self, 1, lambda k, t: vzero(1), [0], "K")

    def adjoint(self):
        d = self.dim
        return ModP(self, d, lambda k, t: self.br[k][t],
                    [self.par(i) for i in range(d)], "g")


class ModP:
    """a g-module over F_p: act(k, t) = x_k . v_t as a vector"""

    def __init__(self, g, dim, act, par, name="M"):
        self.g, self.dim, self.name = g, dim, name
        self.par = par
        self.mat = [[act(k, t) for t in range(dim)] for k in range(g.dim)]

    def apply(self, k, v):
        p = self.g.p
        out = vzero(self.dim)
        for t in range(self.dim):
            if v[t]:
                out = vadd(out, vscal(v[t], self.mat[k][t], p), p)
        return out

    def apply_vec(self, u, v):
        """the action of the element u of g (a vector) on v in M"""
        p = self.g.p
        out = vzero(self.dim)
        for k in range(self.g.dim):
            if u[k]:
                out = vadd(out, vscal(u[k], self.apply(k, v), p), p)
        return out

    def check_module(self):
        """rho([x,y]) = rho(x)rho(y) - (-1)^{p(x)p(y)} rho(y)rho(x)"""
        g, p = self.g, self.g.p
        bad = 0
        for i in range(g.dim):
            for j in range(g.dim):
                for t in range(self.dim):
                    v = vbasis(self.dim, t)
                    lhs = self.apply_vec(g.br[i][j], v)
                    s = (-1) ** (g.par(i) * g.par(j)) % p
                    rhs = vadd(self.apply(i, self.apply(j, v)),
                               vscal(-s % p, self.apply(j, self.apply(i, v)), p),
                               p)
                    if lhs != rhs:
                        bad += 1
        return bad


# ------------------------------------------------------ the enveloping algebra

class EnvP:
    """U(g) with the PBW normal form: nondecreasing words, odd letters at most
    once.  A word is reduced to a dict {word: coefficient mod p}."""

    def __init__(self, g):
        self.g = g
        self._memo = {}

    def is_normal(self, w):
        for a in range(len(w) - 1):
            if w[a] > w[a + 1]:
                return False
            if w[a] == w[a + 1] and self.g.par(w[a]) == 1:
                return False
        return True

    def reduce(self, w):
        w = tuple(w)
        if w in self._memo:
            return self._memo[w]
        g, p = self.g, self.g.p
        pos, kind = -1, None
        for a in range(len(w) - 1):
            if w[a] > w[a + 1]:
                pos, kind = a, "swap"
                break
            if w[a] == w[a + 1] and g.par(w[a]) == 1:
                pos, kind = a, "square"
                break
        if pos < 0:
            res = {w: 1}
            self._memo[w] = res
            return res
        out = {}
        head, tail = w[:pos], w[pos + 2:]
        i, j = w[pos], w[pos + 1]
        if kind == "swap":
            s = (-1) ** (g.par(i) * g.par(j)) % p
            _acc(out, self.reduce(head + (j, i) + tail), s, p)
            v = g.br[i][j]
            for k in range(g.dim):
                if v[k]:
                    _acc(out, self.reduce(head + (k,) + tail), v[k], p)
        else:
            v = g.sq[i - g.m]
            for k in range(g.dim):
                if v[k]:
                    _acc(out, self.reduce(head + (k,) + tail), v[k], p)
        res = {k: c for k, c in out.items() if c % p}
        self._memo[w] = res
        return res

    def check_associativity(self, maxlen=3):
        d, p = self.g.dim, self.g.p
        words = [()]
        cur = [()]
        for _ in range(maxlen):
            cur = [w + (k,) for w in cur for k in range(d)]
            words += cur
        bad = 0
        for w in words:
            for k in range(d):
                for l in range(d):
                    a = {}
                    for u, c in self.reduce(w + (k,)).items():
                        _acc(a, self.reduce(u + (l,)), c, p)
                    b = {}
                    for u, c in self.reduce((k, l)).items():
                        _acc(b, self.reduce(w + u), c, p)
                    a = {x: y % p for x, y in a.items() if y % p}
                    b = {x: y % p for x, y in b.items() if y % p}
                    if a != b:
                        bad += 1
        return bad


def _acc(dst, src, coeff, p):
    for k, c in src.items():
        dst[k] = (dst.get(k, 0) + coeff * c) % p




# =====================================================================
#  THE TWO LATTICES: the anti-symmetric and the divided power complexes
# =====================================================================
#
# For p > 2 both theories have the same dimension in each degree, and both are
# reductions mod p of ONE complex over Z: the multilinear one.  Write the
# anti-symmetric cochains in the basis e_X dual to the argument tuples X (X
# nondecreasing, an even index at most once), and put
#
#     c_X = prod_j a_j(X)!,    a_j(X) = the multiplicity of the odd index j.
#
# The divided power cochains are the lattice spanned by the f_X = c_X e_X,
# i.e. the coefficient functionals of the divided power monomials rather than
# the values of the multilinear functions.  Then
#
#     d(f_X) = sum_Y (c_X / c_Y) m_{XY} f_Y,
#
# and the two claims to check are: the numbers (c_X/c_Y) m_{XY} are INTEGERS
# (so the divided power complex exists over Z, hence over F_p for every p),
# and the comparison map  Phi = diag(c_X)  between the two lattices is
# invertible mod p exactly when every odd multiplicity a_j(X) is < p.  That is
# the isomorphism the Hypothesis of [BGL, Sec. 4] asks for, and the sharp
# condition is on the ODD degree of a cochain, not on its total degree.

from fractions import Fraction                                    # noqa: E402
from math import factorial                                        # noqa: E402


class SLAz:
    """the same Lie superalgebra with INTEGER structure constants"""

    def __init__(self, m, n, brackets, name="g"):
        self.m, self.n, self.name = m, n, name
        self.dim = d = m + n
        self.br = [[tuple([0] * d) for _ in range(d)] for _ in range(d)]
        for (i, j), v in brackets.items():
            self.br[i][j] = tuple(v)
            sgn = -1 if (self.par(i) == 0 or self.par(j) == 0) else 1
            self.br[j][i] = tuple(sgn * x for x in v)

    def par(self, i):
        return 0 if i < self.m else 1

    def mod(self, p):
        return SLAp(p, self.m, self.n,
                    {(i, j): self.br[i][j]
                     for i in range(self.dim) for j in range(i, self.dim)
                     if any(self.br[i][j])}, self.name)


class ModZ:
    """a module with integer matrices: mat[k][t] = x_k . v_t"""

    def __init__(self, g, dim, mat, name="M"):
        self.g, self.dim, self.mat, self.name = g, dim, mat, name

    @staticmethod
    def trivial(g):
        return ModZ(g, 1, [[tuple([0])] for _ in range(g.dim)], "K")

    @staticmethod
    def adjoint(g):
        return ModZ(g, g.dim,
                    [[g.br[k][t] for t in range(g.dim)] for k in range(g.dim)],
                    "g")


class ACZ:
    """the super anti-symmetric complex over Z, [BGL, Sec. 4] verbatim"""

    def __init__(self, g, M):
        self.g, self.M = g, M
        self._bas = {}

    def basis(self, deg):
        if deg in self._bas:
            return self._bas[deg]
        g = self.g
        out = []

        def rec(start, left, cur):
            if left == 0:
                out.append(tuple(cur))
                return
            for k in range(start, g.dim):
                if g.par(k) == 0:
                    rec(k + 1, left - 1, cur + [k])
                else:
                    rec(k, left - 1, cur + [k])
        rec(0, deg, [])
        self._bas[deg] = out
        return out

    def normalize(self, tup):
        g = self.g
        t = list(tup)
        sgn = 1
        for a in range(len(t)):
            for b in range(len(t) - 1 - a):
                if t[b] > t[b + 1]:
                    sgn *= -((-1) ** (g.par(t[b]) * g.par(t[b + 1])))
                    t[b], t[b + 1] = t[b + 1], t[b]
        for a in range(len(t) - 1):
            if t[a] == t[a + 1] and g.par(t[a]) == 0:
                return None
        return sgn, tuple(t)

    def dmat(self, deg):
        """the integer matrix of d in the basis e_X, rows = images"""
        g, M = self.g, self.M
        src = self.basis(deg)
        dst = self.basis(deg + 1)
        sidx = {c: a for a, c in enumerate(src)}
        rows = [[0] * (len(dst) * M.dim) for _ in range(len(src) * M.dim)]
        for (b, X) in enumerate(dst):
            n = len(X)
            for i in range(n):
                for j in range(i + 1, n):
                    e = sum(g.par(X[q]) for q in range(i + 1, j))
                    sgn = (-1) ** (g.par(X[j]) * e + j)
                    v = g.br[X[i]][X[j]]
                    for k in range(g.dim):
                        if not v[k]:
                            continue
                        arg = list(X)
                        arg[i] = k
                        del arg[j]
                        nz = self.normalize(tuple(arg))
                        if nz is None:
                            continue
                        s2, key = nz
                        a = sidx[key]
                        for t in range(M.dim):
                            rows[a * M.dim + t][b * M.dim + t] += \
                                sgn * s2 * v[k]
            for j in range(n):
                e = sum(g.par(X[q]) for q in range(j))
                sgn = (-1) ** (g.par(X[j]) * e + j)
                arg = list(X)
                del arg[j]
                nz = self.normalize(tuple(arg))
                if nz is None:
                    continue
                s2, key = nz
                a = sidx[key]
                for t in range(M.dim):
                    col = M.mat[X[j]][t]
                    for u in range(M.dim):
                        if col[u]:
                            rows[a * M.dim + t][b * M.dim + u] += \
                                sgn * s2 * col[u]
        return rows, len(src) * M.dim, len(dst) * M.dim

    # -- the scaling of the divided power lattice ------------------------
    def scale(self, deg):
        out = []
        for X in self.basis(deg):
            c = 1
            k = 0
            while k < len(X):
                if self.g.par(X[k]) == 1:
                    a = 1
                    while k + a < len(X) and X[k + a] == X[k]:
                        a += 1
                    c *= factorial(a)
                    k += a
                else:
                    k += 1
            out += [c] * self.M.dim
        return out

    def dmat_dp(self, deg):
        """the matrix of d in the divided power basis f_X = c_X e_X, and the
        number of non-integral entries (which must be 0)"""
        rows, ns, nd = self.dmat(deg)
        cs = self.scale(deg)
        ct = self.scale(deg + 1)
        bad = 0
        newrows = []
        for a, row in enumerate(rows):
            r = []
            for b, x in enumerate(row):
                q = Fraction(x * cs[a], ct[b])
                if q.denominator != 1:
                    bad += 1
                r.append(q)
            newrows.append(r)
        return newrows, ns, nd, bad

    # -- cohomology mod p ------------------------------------------------
    def cohom(self, maxdeg, p, divided=False):
        from exactlin import rank
        mats = {}
        for n in range(maxdeg + 1):
            if divided:
                rows, ns, nd, bad = self.dmat_dp(n)
                rows = [[int(x) % p for x in r] for r in rows]
            else:
                rows, ns, nd = self.dmat(n)
                rows = [[x % p for x in r] for r in rows]
            mats[n] = (rows, ns, nd)
        out = []
        for n in range(maxdeg):
            rows, ns, nd = mats[n]
            rk = rank(rows, p=p) if rows and nd else 0
            prev = 0
            if n > 0:
                prows, pns, pnd = mats[n - 1]
                prev = rank(prows, p=p) if prows and pnd else 0
            out.append(ns - rk - prev)
        return out

    def dsquared(self, maxdeg, p=None, divided=False):
        bad = 0
        for n in range(maxdeg):
            if divided:
                r1, s1, d1, b1 = self.dmat_dp(n)
                r2, s2, d2, b2 = self.dmat_dp(n + 1)
                r1 = [[int(x) for x in r] for r in r1]
                r2 = [[int(x) for x in r] for r in r2]
            else:
                r1, s1, d1 = self.dmat(n)
                r2, s2, d2 = self.dmat(n + 1)
            for row in r1:
                acc = [0] * d2
                for j, cf in enumerate(row):
                    if cf:
                        for i, x in enumerate(r2[j]):
                            if x:
                                acc[i] += cf * x
                if p is not None:
                    acc = [x % p for x in acc]
                if any(acc):
                    bad += 1
        return bad

    def phi_rank(self, deg, p):
        """the rank mod p of Phi = diag(c_X): the divided power lattice into the
        anti-symmetric one"""
        cs = self.scale(deg)
        return sum(1 for c in cs if c % p), len(cs)
