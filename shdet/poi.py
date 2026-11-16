"""The Poisson superalgebra po(0|2n): exact Shapovalov determinants.

Built to settle the relation between

  [GL] Grozman P., Leites D., The Shapovalov determinant for the Poisson
       superalgebras, J. Nonlinear Math. Phys. 8 (2001) 220-228, Theorem 2,
  [GS] Gorelik M., Serganova V., Shapovalov forms for Poisson Lie
       superalgebras, Contemp. Math. 391 (2005) 111-121, Thm 3.1 and Subsec 5.3.

Conventions are [GS] Subsec 2.6.1: generators xi_1..xi_n, eta_1..eta_n of
Lambda(2n) with

    [f,g] = (-1)^{p(f)+1} sum_i ( d_{xi_i}f d_{eta_i}g + d_{eta_i}f d_{xi_i}g ).

Everything is exact: coefficients are Fractions, the weights of a highest
vector are sympy symbols l_J, one per basis element h_J = prod_{i in J} xi_i eta_i
of the maximal toral subalgebra.
"""

from fractions import Fraction
from itertools import combinations
import sympy

# ---------------------------------------------------------------- Grassmann
# A monomial in Lambda(2n) is a bitmask over the 2n generators, in the fixed
# order xi_1..xi_n, eta_1..eta_n; generator i has bit 1<<i.


def gens(n):
    """(xi indices, eta indices) in the global generator order."""
    return list(range(n)), list(range(n, 2 * n))


def _sign_below(mask, i):
    """(-1)^{number of generators of mask with index < i}."""
    return -1 if bin(mask & ((1 << i) - 1)).count("1") % 2 else 1


def _sign_above(mask, i):
    """(-1)^{number of generators of mask with index > i}."""
    return -1 if bin(mask >> (i + 1)).count("1") % 2 else 1


def mono_mul(m1, m2):
    """Product of two Grassmann monomials: (sign, mask) or (0, 0)."""
    if m1 & m2:
        return 0, 0
    sign = 1
    m = m1
    # multiply m1 by the generators of m2 in increasing order
    for i in range(m2.bit_length()):
        if m2 >> i & 1:
            # move generator i to its place inside m
            sign *= _sign_above(m, i)
            m |= 1 << i
    # the generators of m2 were taken in increasing order, which is how they
    # already sit inside m2, so no extra sign from reordering m2 itself
    return sign, m


def deriv(mask, i):
    """Left derivative d/d(gen i) of a monomial: (sign, mask) or (0, 0)."""
    if not mask >> i & 1:
        return 0, 0
    sign = _sign_below(mask, i)
    return sign, mask ^ (1 << i)


def parity(mask):
    return bin(mask).count("1") % 2


class Po:
    """The Lie superalgebra po(0|2n) with the bracket of [GS] 2.6.1."""

    def __init__(self, n):
        self.n = n
        self.N = 2 * n
        self.xi, self.eta = gens(n)
        self.basis = list(range(1 << self.N))  # every monomial
        self.dim = 1 << self.N

    # --- the bracket, on monomials; returns dict mask -> Fraction
    def bracket_mono(self, f, g):
        res = {}
        sgn = -1 if parity(f) == 0 else 1  # (-1)^{p(f)+1}
        for i in range(self.n):
            for a, b in ((self.xi[i], self.eta[i]), (self.eta[i], self.xi[i])):
                s1, m1 = deriv(f, a)
                if not s1:
                    continue
                s2, m2 = deriv(g, b)
                if not s2:
                    continue
                s3, m3 = mono_mul(m1, m2)
                if not s3:
                    continue
                res[m3] = res.get(m3, 0) + sgn * s1 * s2 * s3
        return {k: v for k, v in res.items() if v}

    def bracket(self, f, g):
        """Bracket of two elements given as dicts mask -> coefficient."""
        res = {}
        for m1, c1 in f.items():
            for m2, c2 in g.items():
                for m, c in self.bracket_mono(m1, m2).items():
                    res[m] = res.get(m, 0) + c1 * c2 * c
        return {k: v for k, v in res.items() if v}

    # --- the invariant form: B(f,g) = integral of f*g, i.e. the coefficient
    #     of the top monomial
    def form(self, f, g):
        s, m = mono_mul(f, g)
        return s if m == (1 << self.N) - 1 else 0

    # --- the maximal toral subalgebra and the weights
    def h_elt(self, J):
        """h_J = prod_{i in J} xi_i eta_i, as (sign, mask).

        The generator order used internally is xi_1..xi_n, eta_1..eta_n, so the
        ordered product picks up a sign, which must be carried: lambda(h_J) is
        `sign` times the value on the monomial.
        """
        sgn, m = 1, 0
        for i in sorted(J):
            s, m = mono_mul(m, (1 << self.xi[i]) | (1 << self.eta[i]))
            sgn *= s
        return sgn, m

    def h_mask(self, J):
        return self.h_elt(J)[1]

    def weight(self, mask):
        """Weight of a monomial w.r.t. h_0 = span(h_{i}), as a tuple of 0,+-1."""
        w = []
        for i in range(self.n):
            has_x = mask >> self.xi[i] & 1
            has_e = mask >> self.eta[i] & 1
            w.append(1 if (has_x and not has_e) else (-1 if (has_e and not has_x) else 0))
        return tuple(w)

    def degree(self, mask):
        """The Z-grading of [GS] 2.6: the Lambda-degree shifted by -2."""
        return bin(mask).count("1") - 2

    # --- the triangular decomposition of [GS] 2.6.3
    def positive(self, w):
        for i, s in enumerate(w):
            if s:
                return s > 0
        return False

    def split(self):
        """(n_-, h, n_+) as lists of monomials."""
        neg, zero, pos = [], [], []
        for m in self.basis:
            w = self.weight(m)
            if all(s == 0 for s in w):
                zero.append(m)
            elif self.positive(w):
                pos.append(m)
            else:
                neg.append(m)
        return neg, zero, pos

    def roots(self):
        r = {}
        for m in self.basis:
            w = self.weight(m)
            if any(w):
                r.setdefault(w, []).append(m)
        return r


# ------------------------------------------------------------ self-tests
def check_algebra(g, verbose=True):
    """Jacobi, commutativity of h, invariance of B, h = the zero-weight space."""
    n, out = g.n, []
    # super Jacobi: [x,[y,z]] = [[x,y],z] + (-1)^{p(x)p(y)}[y,[x,z]]
    bad = 0
    for x in g.basis:
        for y in g.basis:
            for z in g.basis:
                lhs = g.bracket({x: 1}, g.bracket_mono(y, z))
                r1 = g.bracket(g.bracket_mono(x, y), {z: 1})
                r2 = g.bracket({y: 1}, g.bracket_mono(x, z))
                sg = -1 if parity(x) and parity(y) else 1
                rhs = dict(r1)
                for k, v in r2.items():
                    rhs[k] = rhs.get(k, 0) + sg * v
                rhs = {k: v for k, v in rhs.items() if v}
                if lhs != rhs:
                    bad += 1
    out.append(("Jacobi", bad == 0, f"{g.dim**3} triples, {bad} failures"))

    # h commutative
    hs = [g.h_mask(J) for k in range(n + 1) for J in combinations(range(n), k)]
    bad = sum(1 for a in hs for b in hs if g.bracket_mono(a, b))
    out.append(("h commutative", bad == 0, f"{len(hs)}^2 pairs, {bad} failures"))

    # h = the zero-weight space
    neg, zero, pos = g.split()
    out.append(("h = zero weight space", sorted(zero) == sorted(hs),
                f"dim h = {len(hs)}, zero-weight space = {len(zero)}"))
    out.append(("dim n_- = dim n_+", len(neg) == len(pos), f"{len(neg)}|{len(pos)}"))

    # invariance of B: B([x,y],z) = B(x,[y,z])
    bad = 0
    for x in g.basis:
        for y in g.basis:
            for z in g.basis:
                l = sum(c * g.form(m, z) for m, c in g.bracket_mono(x, y).items())
                r = sum(c * g.form(x, m) for m, c in g.bracket_mono(y, z).items())
                if l != r:
                    bad += 1
    out.append(("B invariant", bad == 0, f"{g.dim**3} triples, {bad} failures"))

    # B non-degenerate
    import itertools
    M = [[g.form(a, b) for b in g.basis] for a in g.basis]
    nz = all(any(row) for row in M)
    out.append(("B non-degenerate", nz, "every row non-zero"))

    # root multiplicities: dim g_alpha = 2^{n-k}, parity = k mod 2
    ok = True
    for w, ms in g.roots().items():
        k = sum(1 for s in w if s)
        if len(ms) != 2 ** (n - k):
            ok = False
        if any(parity(m) != k % 2 for m in ms):
            ok = False
    out.append(("root multiplicities 2^{n-k}", ok, f"{len(g.roots())} roots"))

    if verbose:
        for name, ok_, note in out:
            print(f"  [{'OK ' if ok_ else 'FAIL'}] {name}: {note}")
    return all(o[1] for o in out)


# ------------------------------------------------ U(n_-): PBW straightening
class Env:
    """The universal enveloping algebra of n_-, with a PBW basis.

    A basis vector of n_- is an index into `self.b`; a PBW word is a tuple of
    indices, non-decreasing, with no repetition among the odd ones.
    """

    def __init__(self, g, vecs):
        self.g = g
        self.b = list(vecs)                 # monomials of Lambda spanning n_-
        self.idx = {m: i for i, m in enumerate(self.b)}
        self.par = [parity(m) for m in self.b]
        self.wt = [g.weight(m) for m in self.b]

    def _expand(self, d):
        """dict mask->coef  ->  dict index->coef (must lie in the span)."""
        out = {}
        for m, c in d.items():
            assert m in self.idx, "bracket left the span"
            out[self.idx[m]] = out.get(self.idx[m], 0) + c
        return {k: v for k, v in out.items() if v}

    def straighten(self, word, coef=Fraction(1)):
        """Rewrite an arbitrary word into the PBW basis; dict word->coef."""
        res = {}
        stack = [(tuple(word), coef)]
        while stack:
            w, c = stack.pop()
            # find the first descent
            pos = -1
            for i in range(len(w) - 1):
                if w[i] > w[i + 1] or (w[i] == w[i + 1] and self.par[w[i]]):
                    pos = i
                    break
            if pos < 0:
                res[w] = res.get(w, 0) + c
                continue
            a, b = w[pos], w[pos + 1]
            pre, post = w[:pos], w[pos + 2:]
            br = self._expand(self.g.bracket_mono(self.b[a], self.b[b]))
            if a == b:  # odd square: x x = 1/2 [x,x]
                for k, v in br.items():
                    stack.append((pre + (k,) + post, c * Fraction(v) / 2))
            else:
                sg = -1 if (self.par[a] and self.par[b]) else 1
                stack.append((pre + (b, a) + post, c * sg))
                for k, v in br.items():
                    stack.append((pre + (k,) + post, c * Fraction(v)))
        return {k: v for k, v in res.items() if v}

    def height(self, w):
        """The functional of [GS] 2.6.3 that fixes the triangular decomposition."""
        n = self.g.n
        return sum(s * 2 ** (n - 1 - i) for i, s in enumerate(w))

    def basis_of_weight(self, nu):
        """PBW words of n_--weight -nu (nu a tuple of non-negative ints)."""
        target = tuple(-x for x in nu)
        hmin = self.height(target)          # every letter lowers the height
        out, cur = [], []

        def rec(start, w):
            if w == target:
                out.append(tuple(cur))
                return
            if self.height(w) <= hmin:
                return                      # cannot come back up
            for i in range(start, len(self.b)):
                nw = tuple(a + b for a, b in zip(w, self.wt[i]))
                if self.height(nw) < hmin:
                    continue
                cur.append(i)
                rec(i + 1 if self.par[i] else i, nw)
                cur.pop()

        rec(0, tuple(0 for _ in target))
        return out


class Verma:
    """The Verma module M(lambda) over po(0|2n) as U(n_-), with the action."""

    def __init__(self, g):
        self.g = g
        neg, zero, pos = g.split()
        self.env = Env(g, neg)
        self.neg, self.zero, self.pos = neg, zero, pos
        self.negset, self.posset = set(neg), set(pos)
        # lambda(monomial) = sign_J * l_J, where h_J = sign_J * monomial
        self.lam, self.hsym = {}, {}
        for k in range(g.n + 1):
            for J in combinations(range(g.n), k):
                sgn, m = g.h_elt(J)
                sym = sympy.Symbol("l" + ("".join(str(j + 1) for j in J) or "0"))
                self.hsym[J] = sym
                self.lam[m] = sympy.Integer(sgn) * sym

    def set_lambda(self, values):
        """Fix a numerical highest weight: values maps a subset J to a number.

        After this the Shapovalov matrix has numerical entries, which is what
        makes a large weight space affordable.
        """
        for k in range(self.g.n + 1):
            for J in combinations(range(self.g.n), k):
                sgn, m = self.g.h_elt(J)
                self.lam[m] = sympy.Integer(sgn) * sympy.Rational(values.get(tuple(J), 0))

    def lam_of(self, J):
        """lambda(h_J) as a symbol."""
        return self.hsym[tuple(sorted(J))]

    def act(self, x, word, memo=None):
        """Action of the monomial x of g on the basis vector `word` of M."""
        if memo is None:
            memo = {}
        key = (x, word)
        if key in memo:
            return memo[key]
        if not word:
            if x in self.lam:
                out = {(): self.lam[x]}
            elif x in self.posset:
                out = {}
            else:
                out = {(self.env.idx[x],): sympy.Integer(1)}
        else:
            y, rest = word[0], word[1:]
            ym = self.env.b[y]
            out = {}
            # [x, y] . rest
            for m, c in self.g.bracket_mono(x, ym).items():
                for w, cc in self.act(m, rest, memo).items():
                    out[w] = out.get(w, 0) + c * cc
            # (-1)^{p(x)p(y)} y . (x . rest)
            sg = -1 if (parity(x) and self.env.par[y]) else 1
            for w, cc in self.act(x, rest, memo).items():
                for w2, c2 in self.env.straighten((y,) + w).items():
                    out[w2] = out.get(w2, 0) + sg * cc * sympy.Rational(c2)
            out = {k: sympy.expand(v) for k, v in out.items()}
            out = {k: v for k, v in out.items() if v != 0}
        memo[key] = out
        return out

    def act_word(self, xs, vec, memo=None):
        """Action of a word in g (applied right to left) on an element of M."""
        if memo is None:
            memo = {}
        for x in reversed(xs):
            new = {}
            for w, c in vec.items():
                for w2, c2 in self.act(x, w, memo).items():
                    new[w2] = new.get(w2, 0) + c * c2
            vec = {k: sympy.expand(v) for k, v in new.items() if sympy.expand(v) != 0}
        return vec

    # --- the Shapovalov matrix
    def pos_words(self, nu):
        """PBW words in n_+ of weight nu, as tuples of monomials of Lambda."""
        wts = {m: self.g.weight(m) for m in self.pos}
        order = sorted(self.pos)
        hmax = self.env.height(tuple(nu))
        out, cur = [], []

        def rec(start, w):
            if w == tuple(nu):
                out.append(tuple(cur))
                return
            if self.env.height(w) >= hmax:
                return
            for i in range(start, len(order)):
                m = order[i]
                nw = tuple(a + b for a, b in zip(w, wts[m]))
                if self.env.height(nw) > hmax:
                    continue
                cur.append(m)
                rec(i + 1 if parity(m) else i, nw)
                cur.pop()

        rec(0, tuple(0 for _ in nu))
        return out

    def shap_matrix(self, nu):
        cols = self.env.basis_of_weight(nu)
        rows = self.pos_words(nu)
        assert len(rows) == len(cols), f"nu={nu}: {len(rows)} rows vs {len(cols)} columns"
        memo = {}
        M = []
        for x in rows:
            row = []
            for y in cols:
                v = self.act_word(x, {y: sympy.Integer(1)}, memo)
                row.append(sympy.expand(v.get((), 0)))
            M.append(row)
        return sympy.Matrix(M), rows, cols

    # --- the quadratic Casimir and its Harish-Chandra image
    def casimir_hc(self):
        """HC(C) for C the Casimir of the form B, as a polynomial in the l_J."""
        g = self.g
        # dual bases of g w.r.t. B
        basis = g.basis
        G = sympy.Matrix([[g.form(a, b) for b in basis] for a in basis])
        Ginv = G.inv()
        # C = sum_{a,b} (G^{-1})_{ab} e_a e_b  (as an element of U(g))
        # its HC image: push each product to U(h) modulo n_- U + U n_+
        hc = sympy.Integer(0)
        for i, a in enumerate(basis):
            for j, b in enumerate(basis):
                c = Ginv[i, j]
                if c == 0:
                    continue
                v = self.act_word([a, b], {(): sympy.Integer(1)})
                hc += c * v.get((), 0)
        return sympy.expand(hc)
