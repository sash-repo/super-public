"""
orth.py -- riggings that do not split.

A contact structure on a split superstring (X, Lambda^. E) forces
E = E^* (x) K (Lemma 20.2).  Fixing
a theta characteristic L, L^2 = K, and setting V := E (x) L^{-1} turns that
K-valued symmetric form into an ORDINARY one:

      V (x) V --> O,  non-degenerate symmetric,

i.e. V is a holomorphic O(n)-bundle and E = V (x) L.  The DIAGONAL riggings
of srs.py are the V whose structure group reduces to O(1)^n = (Z/2)^n, i.e.
V = (+) nu_i; a general O(n)-bundle does not split into line bundles, and
that is what this module is for.

What is computed
----------------
The layers of the contact sheaf become

      Theta (x) Lambda^q E = Lambda^q V (x) L^{q-2},

and since V = V^*, also Lambda^q V = (Lambda^q V)^*, so Serre duality gives

      q = 2r   : h^1 = h^0(Lambda^{2r}V (x) K^{2-r})
      q = 2r+1 : h^1 = h^0(Lambda^{2r+1}V (x) L^{3-2r}).

Every Lambda^q V has degree 0 (det V is 2-torsion), whence for V semistable
all the terms with r >= 3 (even) and r >= 2 (odd) vanish on degree grounds,
and what is left is

  sdim B = ( 3g-3 + h^1(ad V) + h^0(Lambda^4 V) | n(2g-2) + h^0(Lambda^3 V (x) L) )

with ad V = Lambda^2 V.  This module verifies that on every SPLIT orthogonal
bundle -- a sum of anisotropic lines nu_i and of hyperbolic planes
M (+) M^{-1} -- against the direct layer-by-layer computation, and against
srs.py on the diagonal ones.

Bookkeeping
-----------
A line bundle is a monomial in named generators: 'L' (the theta
characteristic, deg = g-1), any number of free generators of prescribed
degree, and 2-torsion generators of degree 0.  The genus is a CONCRETE
integer here (srs.py keeps it symbolic); the only symbols left are the
dimensions s(nu) = h^0(L (x) nu) of the theta characteristics, which no
degree argument determines.

A free generator is by default GENERAL in its component of Pic, so that
h^0 = max(0, deg + 1 - g) (Brill-Noether); `special` generators keep
a symbol instead, and that is how the jumping loci are exhibited.

Nothing here is approximate and nothing is sampled.
"""

from itertools import combinations

import sympy as sp

from srs import s as theta_s          # s(nu) = h^0(L (x) nu), shared symbols


# ------------------------------------------------------------- a line bundle
class Line:
    """The line bundle prod gen^exp (x) prod nu, written multiplicatively."""

    __slots__ = ('free', 'tors')

    def __init__(self, free=None, tors=()):
        self.free = {k: v for k, v in (free or {}).items() if v}
        self.tors = frozenset(tors)

    def __mul__(self, o):
        f = dict(self.free)
        for k, v in o.free.items():
            f[k] = f.get(k, 0) + v
        return Line(f, self.tors ^ o.tors)

    def inv(self):
        return Line({k: -v for k, v in self.free.items()}, self.tors)

    def pow(self, m):
        return Line({k: v * m for k, v in self.free.items()},
                    self.tors if m % 2 else frozenset())

    def key(self):
        return (tuple(sorted(self.free.items())), tuple(sorted(self.tors)))

    def __eq__(self, o):
        return self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        bits = []
        for k, v in sorted(self.free.items()):
            bits.append(k if v == 1 else '%s^%d' % (k, v))
        bits += ['.' + t for t in sorted(self.tors)]
        return ''.join(bits) or 'O'


ONE = Line()
L = Line({'L': 1})
K = Line({'L': 2})
THETA = Line({'L': -2})


def torsion(*names):
    return Line(tors=names)


# ------------------------------------------------------------------ a curve
class Curve:
    """A compact Riemann surface of genus g, with the degrees of the free
    generators declared.  `special` names the generators that are NOT in
    general position, for which an undetermined h^0 stays a symbol."""

    def __init__(self, g, deg=None, special=(), L_at_g1_trivial=True):
        self.g = int(g)
        self.deg = dict(deg or {})
        self.special = set(special)
        # at g = 1, L itself is one of the four 2-torsion classes: trivial
        # is the odd theta characteristic (Ramond), the three others are NS.
        self.L1 = L_at_g1_trivial

    # ---------------------------------------------------------------- degree
    def degree(self, b):
        d = 0
        for k, v in b.free.items():
            d += v * (self.g - 1 if k == 'L' else self.deg[k])
        return d

    # ------------------------------------------------------------ triviality
    def _only_L(self, b):
        return all(k == 'L' for k in b.free)

    def _trivial(self, b):
        return not b.free and not b.tors

    def _trivial_g1(self, b):
        """At g = 1 every generator but L may still be non-trivial; a general
        one is, so only the pure-L-and-torsion bundles can be trivial."""
        if any(k != 'L' for k in b.free):
            return False
        cls = b.tors
        if b.free.get('L', 0) % 2 and not self.L1:
            cls = cls ^ frozenset({'@L'})   # L itself is a non-trivial class
        return not cls

    # ---------------------------------------------------------------- h^0
    def h0(self, b):
        g, D = self.g, self.degree(b)
        if g == 0:
            assert not b.tors, 'CP^1 carries no 2-torsion'
            return sp.Integer(max(0, D + 1))
        if g == 1:
            if D > 0:
                return sp.Integer(D)
            if D < 0:
                return sp.Integer(0)
            return sp.Integer(1 if self._trivial_g1(b) else 0)
        # ------------------------------------------------------------ g >= 2
        if D < 0:
            return sp.Integer(0)
        if D > 2 * g - 2:
            return sp.Integer(D + 1 - g)
        if self._only_L(b):
            k = b.free.get('L', 0)
            if k == 0:
                return sp.Integer(1 if not b.tors else 0)
            if k == 1:
                return theta_s(b.tors)          # THE undetermined one
            if k == 2:
                return sp.Integer(g if not b.tors else g - 1)
            raise AssertionError('unreachable: deg L^%d out of range' % k)
        if any(k in self.special for k in b.free):
            return sp.Symbol('h0_' + repr(b), integer=True, nonnegative=True)
        return sp.Integer(max(0, D + 1 - g))    # general in its component

    def h1(self, b):
        return self.h0(K * b.inv())

    def chi(self, b):
        return self.degree(b) + 1 - self.g


# ------------------------------------------------- split orthogonal bundles
#
# A split O(n)-bundle is an orthogonal direct sum of
#   * anisotropic lines  nu  (nu^2 = O), and
#   * hyperbolic planes  M (+) M^{-1}  (both summands isotropic).
# For the cohomology only the multiset of line bundles matters.

def diagonal(*nus):
    """V = (+) nu_i: the diagonal riggings of srs.py."""
    return [torsion(*n) if not isinstance(n, Line) else n for n in nus]


def hyperbolic(*gens):
    """V = (+)_j (M_j (+) M_j^{-1})."""
    out = []
    for m in gens:
        b = Line({m: 1}) if isinstance(m, str) else m
        out += [b, b.inv()]
    return out


def lam(V, q):
    """Lambda^q V as a multiset of line bundles."""
    out = []
    for S in combinations(range(len(V)), q):
        b = ONE
        for i in S:
            b = b * V[i]
        out.append(b)
    return out


def tw(bundles, A):
    return [b * A for b in bundles]


def h0V(C, bundles):
    return sp.expand(sum([C.h0(b) for b in bundles], sp.Integer(0)))


def h1V(C, bundles):
    return sp.expand(sum([C.h1(b) for b in bundles], sp.Integer(0)))


def det(V):
    b = ONE
    for w in V:
        b = b * w
    return b


# ---------------------------------------------------- the contact structure
def layers(V):
    """The layers Theta (x) Lambda^q E = Lambda^q V (x) L^{q-2} of the sheaf
    of contact fields, as (multiset of line bundles, parity)."""
    n = len(V)
    return [(tw(lam(V, q), L.pow(q - 2)), q % 2) for q in range(n + 1)]


def sdim_versal(C, V):
    """sdim of the base of the versal deformation = (h^1_ev | h^1_od)."""
    ev = od = sp.Integer(0)
    for pieces, p in layers(V):
        h = h1V(C, pieces)
        if p == 0:
            ev += h
        else:
            od += h
    return (sp.expand(ev), sp.expand(od))


def sdim_closed(C, V):
    """The closed form of formula (51) (Theorem 23.6), valid when the two tails vanish:

        ( 3g-3 + h^1(ad V) + h^0(Lambda^4 V) | n(2g-2) + h^0(Lambda^3 V (x) L) )

    with ad V = Lambda^2 V.  Here the first summand of the odd entry is
    written as h^0(V (x) L^3), which equals n(2g-2) exactly when V is
    semistable."""
    g, n = C.g, len(V)
    ev = C.h1(THETA) + h1V(C, lam(V, 2)) + h0V(C, lam(V, 4))
    od = h0V(C, tw(V, L.pow(3))) + h0V(C, tw(lam(V, 3), L))
    return (sp.expand(ev), sp.expand(od))


def tails(C, V):
    """The terms of the general formula beyond sdim_closed: the even ones
    h^0(Lambda^{2r}V (x) K^{2-r}) for r >= 3 and the odd ones
    h^0(Lambda^{2r+1}V (x) L^{3-2r}) for r >= 2."""
    n = len(V)
    ev = od = sp.Integer(0)
    r = 3
    while 2 * r <= n:
        ev += h0V(C, tw(lam(V, 2 * r), K.pow(2 - r)))
        r += 1
    r = 2
    while 2 * r + 1 <= n:
        od += h0V(C, tw(lam(V, 2 * r + 1), L.pow(3 - 2 * r)))
        r += 1
    return (sp.expand(ev), sp.expand(od))


def semistable_split(C, V):
    """Is the split orthogonal bundle V semistable?  A direct sum of line
    bundles is semistable iff they all have the same degree, and here that
    degree is 0 because det V is 2-torsion."""
    return all(C.degree(b) == 0 for b in V)


def sdim(pair):
    a, b = pair
    return '(%s | %s)' % (sp.simplify(a), sp.simplify(b))
