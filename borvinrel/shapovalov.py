"""
Shapovalov-form stratification driver for Borcherds (super)algebras g(A).

Setting
-------
A is an n x n symmetric matrix over Q(a,b,c,...).  The Chevalley generators
x_1..x_n are declared ODD (the case of interest for Subsec. 11.2 of the paper: zero
diagonal => all simple roots isotropic odd).  No relations are imposed by
hand: n_+ starts life as the FULL free Lie superalgebra L(V), V purely odd,
and

        g(A) = n_- (+) h (+) n_+ ,   n_+ = L(V) / r_+ ,

where r_+ is the maximal graded ideal with r_+ ^ h = 0, i.e. the radical of
the contravariant (Shapovalov) form.  So [x_i,x_i]=0 is *derived*, not
assumed, whenever A_ii = 0.

The lowering operator
---------------------
Inside U(g), for a word w = x_{j_1}...x_{j_k} (all letters odd),

    ad(f_i)(w) = - sum_{t : j_t = i} (-1)^{t-1} ( sum_{s>t} A_{i,j_s} ) w\t
                 - ( sum_{t : j_t = i} (-1)^{t-1} w\t ) h_i ,

where w\t is w with letter t deleted.  On a genuine Lie element of height
>= 2 the h_i-term cancels identically (the code asserts this), so

    D_i(w) = - sum_{t : j_t = i} (-1)^{t-1} ( sum_{s>t} A_{i,j_s} ) w\t

is a well-defined map L^alpha -> L^(alpha - alpha_i).  At height 1,
ad(f_i)(sum_j c_j x_j) = -c_i h_i.

Radical
-------
    r_+^alpha = { x in L^alpha : D_i(x) in r_+^(alpha-alpha_i) for all i },
    r_+ = 0 in height 1.

Equivalently r_+ = ker Phi, where for a word w = (i_1,...,i_k) that is a
permutation of alpha,

    Phi(x)_w = coefficient of x_{i_k} in D_{i_{k-1}} ... D_{i_1} (x).

The Gram matrix of the contravariant form in a basis (u_1..u_d) of L^alpha is
G_ab = sum_w coeff_w(u_b) * Phi(u_a)_w.  Its determinant is the Shapovalov
determinant in that degree; its irreducible factors cut out the strata.
"""

from fractions import Fraction
from itertools import product
from collections import defaultdict

import sympy as sp

# ---------------------------------------------------------------- words ----


def multideg(word, n):
    d = [0] * n
    for j in word:
        d[j] += 1
    return tuple(d)


def pw(word, par):
    """Parity of a word = sum of the parities of its letters."""
    return sum(par[j] for j in word) % 2


def bracket(u, v, par):
    """Super bracket of two elements of T(V).

    Elements are dicts word -> coefficient; `par` is the parity vector of the
    generators.  [u,v] = uv - (-1)^{|u||v|} vu.
    """
    out = defaultdict(Fraction)
    for wu, cu in u.items():
        for wv, cv in v.items():
            sign = -1 if (pw(wu, par) * pw(wv, par)) % 2 else 1
            out[wu + wv] += cu * cv
            out[wv + wu] -= sign * cu * cv
    return {w: c for w, c in out.items() if c != 0}


# --------------------------------------------------------- linear algebra --


class Span:
    """Row-reduced span of dicts word -> Fraction, kept in echelon form."""

    def __init__(self):
        self.rows = {}  # pivot word -> normalised row

    def reduce(self, vec):
        vec = dict(vec)
        for piv in sorted(self.rows, key=lambda w: (len(w), w)):
            c = vec.get(piv)
            if c:
                row = self.rows[piv]
                for w, rc in row.items():
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


def free_lie_superalgebra(n, maxheight, par=None):
    """L[height][multidegree] = list of basis elements (dicts word->Fraction).

    V is *free*: nothing is imposed, so for an odd generator [x_i,x_i]=2x_i^2
    is present in height 2 and is only killed later by the radical.
    """
    if par is None:
        par = [1] * n
    L = {1: {}}
    for i in range(n):
        alpha = multideg((i,), n)
        L[1][alpha] = [{(i,): Fraction(1)}]
    for k in range(2, maxheight + 1):
        spans = defaultdict(Span)
        for i in range(1, k):
            j = k - i
            for da, ua in L[i].items():
                for db, ub in L[j].items():
                    tgt = tuple(x + y for x, y in zip(da, db))
                    for u in ua:
                        for v in ub:
                            spans[tgt].add(bracket(u, v, par))
        L[k] = {d: s.basis() for d, s in spans.items() if s.rows}
    return L


# ------------------------------------------------------------ lowering D ---


def lower(vec, i, A, n, par):
    """D_i on an element of T_k(V), k >= 2.  Returns (body, h_coefficient)."""
    body = defaultdict(lambda: sp.Integer(0))
    hpart = defaultdict(lambda: sp.Integer(0))
    for w, c in vec.items():
        c = sp.nsimplify(sp.Rational(c.numerator, c.denominator)) if isinstance(c, Fraction) else c
        for t, j in enumerate(w):
            if j != i:
                continue
            sgn = -1 if (par[i] * sum(par[w[s]] for s in range(t))) % 2 else 1
            tail = sum(A[i][w[s]] for s in range(t + 1, len(w)))
            wdel = w[:t] + w[t + 1:]
            body[wdel] += -sgn * c * tail
            hpart[wdel] += -sgn * c
    body = {w: sp.expand(v) for w, v in body.items() if sp.expand(v) != 0}
    hpart = {w: sp.expand(v) for w, v in hpart.items() if sp.expand(v) != 0}
    return body, hpart


def phi(vec, A, n, alpha, par):
    """Phi(x)_w for every word w that is a permutation of alpha."""
    k = sum(alpha)
    out = {}

    def rec(cur, prefix, rem):
        if sum(rem) == 1:
            last = rem.index(1)
            out[prefix + (last,)] = cur.get((last,), sp.Integer(0))
            return
        for i in range(n):
            if rem[i] == 0:
                continue
            body, hpart = lower(cur, i, A, n, par)
            if hpart:
                raise AssertionError(
                    "h-part did not cancel: not a Lie element? prefix=%s" % (prefix,))
            nrem = list(rem)
            nrem[i] -= 1
            if body:
                rec(body, prefix + (i,), tuple(nrem))
            else:
                for w in perms_of(tuple(nrem), n):
                    out[prefix + (i,) + w] = sp.Integer(0)

    if k == 1:
        j = alpha.index(1)
        return {(j,): vec.get((j,), sp.Integer(0))}
    rec({w: sp.nsimplify(sp.Rational(c.numerator, c.denominator))
         if isinstance(c, Fraction) else c for w, c in vec.items()}, (), alpha)
    for w in perms_of(alpha, n):
        out.setdefault(w, sp.Integer(0))
    return out


def perms_of(alpha, n):
    """All words that are permutations of the multiset alpha."""
    k = sum(alpha)
    res = []

    def rec(prefix, rem):
        if len(prefix) == k:
            res.append(tuple(prefix))
            return
        for i in range(n):
            if rem[i]:
                nrem = list(rem)
                nrem[i] -= 1
                rec(prefix + [i], tuple(nrem))

    rec([], alpha)
    return res


# ----------------------------------------------------------------- Gram ----


def gram(basis, A, n, alpha, par):
    d = len(basis)
    phis = [phi(u, A, n, alpha, par) for u in basis]
    G = sp.zeros(d, d)
    for a in range(d):
        for b in range(d):
            s = sp.Integer(0)
            for w, c in basis[b].items():
                pv = phis[a].get(w)
                if pv is not None and pv != 0:
                    s += sp.Rational(c.numerator, c.denominator) * pv
            G[a, b] = sp.expand(s)
    return G
