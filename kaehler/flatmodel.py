r"""
Exact-arithmetic engine for the flat local model of Subsec. 1.5 of the text.

Operators on the space of (pseudo-)forms on the flat supermanifold C^{a|b} with
coordinates z_1..z_N of parity p_a are elements of the super-Weyl algebra on the
pairs

    c_a := \hat y_a = dz_a  (left multiplication),   b_a := \partial_{\hat y_a},
    x_a := z_a              (left multiplication),   t_a := \partial_{z_a},

with p(c_a) = p(b_a) = p_a + 1, p(x_a) = p(t_a) = p_a, and the only non-zero
super-commutators [b_a, c_a] = [t_a, x_a] = 1.  An extra odd central generator
h (= \hbar, with h^2 = 0) is available for the twisted case.

Monomials are normal-ordered (h, then c, x, then t, b; sorted by index) with
exact rational coefficients and Koszul signs.  Lie superalgebras are computed as
bracket closures over Q, with echelon forms kept separately in each parity.
"""

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction
from itertools import combinations

KIND_ORDER = {'h': 0, 'c': 1, 'x': 2, 't': 3, 'b': 4}


class Model:
    def __init__(self, parities):
        """parities[a] = parity of the base coordinate z_a (0 = even, 1 = odd)."""
        self.p = list(parities)
        self.N = len(parities)

    # ---- generators ---------------------------------------------------
    def par(self, g):
        kind, a = g
        if kind == 'h':
            return 1
        if kind in ('c', 'b'):
            return (self.p[a] + 1) % 2
        return self.p[a] % 2

    def key(self, g):
        return (KIND_ORDER[g[0]], g[1])

    def bracket_scalar(self, g1, g2):
        """[g1, g2] when it is a scalar; 0 otherwise."""
        if g1[0] == 'b' and g2[0] == 'c' and g1[1] == g2[1]:
            return Fraction(1)
        if g1[0] == 't' and g2[0] == 'x' and g1[1] == g2[1]:
            return Fraction(1)
        return Fraction(0)

    # ---- normal ordering ----------------------------------------------
    def normalize(self, gens, coeff=Fraction(1), out=None):
        if out is None:
            out = {}
        if coeff == 0:
            return out
        gens = list(gens)
        for i in range(len(gens) - 1):
            g1, g2 = gens[i], gens[i + 1]
            if self.key(g1) > self.key(g2):
                s = self.bracket_scalar(g1, g2)
                if s:
                    self.normalize(gens[:i] + gens[i + 2:], coeff * s, out)
                sign = -1 if (self.par(g1) and self.par(g2)) else 1
                swapped = gens[:i] + [g2, g1] + gens[i + 2:]
                self.normalize(swapped, coeff * sign, out)
                return out
        # canonical order reached: an odd generator repeated squares to zero
        for i in range(len(gens) - 1):
            if gens[i] == gens[i + 1] and self.par(gens[i]):
                return out
        m = tuple(gens)
        out[m] = out.get(m, Fraction(0)) + coeff
        if out[m] == 0:
            del out[m]
        return out

    # ---- algebra of elements (dict monomial -> Fraction) --------------
    def gen(self, kind, a):
        return {((kind, a),): Fraction(1)}

    def one(self):
        return {(): Fraction(1)}

    def add(self, *es):
        r = {}
        for e in es:
            for m, c in e.items():
                r[m] = r.get(m, Fraction(0)) + c
                if r[m] == 0:
                    del r[m]
        return r

    def smul(self, s, e):
        s = Fraction(s)
        return {m: s * c for m, c in e.items() if s * c != 0}

    def mul(self, e1, e2):
        r = {}
        for m1, c1 in e1.items():
            for m2, c2 in e2.items():
                self.normalize(list(m1) + list(m2), c1 * c2, r)
        return {m: c for m, c in r.items() if c != 0}

    def mmul(self, *es):
        r = self.one()
        for e in es:
            r = self.mul(r, e)
        return r

    def parity(self, e):
        pars = set()
        for m in e:
            pars.add(sum(self.par(g) for g in m) % 2)
        if len(pars) > 1:
            raise ValueError('inhomogeneous element')
        return pars.pop() if pars else 0

    def br(self, e1, e2):
        s = -1 if (self.parity(e1) and self.parity(e2)) else 1
        return self.add(self.mul(e1, e2), self.smul(-s, self.mul(e2, e1)))


# ---------------------------------------------------------------------------
# linear algebra over Q on the monomial basis
# ---------------------------------------------------------------------------
class Echelon:
    def __init__(self):
        self.rows = {}      # pivot monomial -> reduced element
        self.elements = []  # the accepted (unreduced) generators, in order

    def reduce(self, e):
        e = dict(e)
        while e:
            piv = max(e, key=lambda m: (len(m), m))
            if piv in self.rows:
                c = e[piv]
                row = self.rows[piv]
                for m, rc in row.items():
                    e[m] = e.get(m, Fraction(0)) - c * rc
                    if e[m] == 0:
                        del e[m]
            else:
                return e, piv
        return {}, None

    def add(self, e):
        red, piv = self.reduce(e)
        if not red:
            return False
        red = {m: c / red[piv] for m, c in red.items()}
        self.rows[piv] = red
        self.elements.append(e)
        return True

    def __len__(self):
        return len(self.rows)


def closure(M, gens, limit=400):
    """Lie superalgebra generated by `gens` (parity-homogeneous elements)."""
    ech = {0: Echelon(), 1: Echelon()}
    basis = []
    queue = []
    for g in gens:
        if not g:
            continue
        p = M.parity(g)
        if ech[p].add(g):
            basis.append(g)
            queue.append(g)
    i = 0
    while i < len(basis):
        for j in range(0, i + 1):
            b = M.br(basis[i], basis[j])
            if not b:
                continue
            p = M.parity(b)
            if ech[p].add(b):
                basis.append(b)
                if len(basis) > limit:
                    raise RuntimeError('closure exceeded limit %d' % limit)
        i += 1
    return basis, (len(ech[0]), len(ech[1]))


def sdim(M, gens, limit=400):
    return closure(M, gens, limit)[1]


# ---------------------------------------------------------------------------
# convenience constructors
# ---------------------------------------------------------------------------
def d_operator(M):
    r"""the exterior differential d = sum \hat x_a \partial_{x_a}"""
    return M.add(*[M.mul(M.gen('c', a), M.gen('t', a)) for a in range(M.N)])


def form_to_multiplication(M, terms):
    r"""terms: list of (coeff, (a, b)) meaning coeff * \hat y_a \hat y_b."""
    return M.add(*[M.smul(c, M.mul(M.gen('c', a), M.gen('c', b)))
                   for c, (a, b) in terms])


def form_to_convolution(M, terms):
    r"""the dual bivector: coeff * \partial_{\hat y_b} \partial_{\hat y_a}."""
    return M.add(*[M.smul(c, M.mul(M.gen('b', b), M.gen('b', a)))
                   for c, (a, b) in terms])
