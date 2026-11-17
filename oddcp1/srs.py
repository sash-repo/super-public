"""
Versal deformations of superstrings of the twelve distinguished types.

A superstring here is a compact complex (1|n)-dimensional supermanifold whose
reduction is a curve X of genus g, carrying one of the structures whose sheaf
of infinitesimal automorphisms is, locally, one of the distinguished simple
stringy Lie superalgebras.  By Vaintrob's Theorem 3.4.3 (Itogi Nauki 32, 1988)
the versal deformation of such a structure exists, its base is smooth because
dim X = 1 kills H^2, and its superdimension is

      sdim B = ( dim H^1(X; K)_ev | dim H^1(X; K)_od ),

where K is that sheaf.  So everything reduces to Riemann-Roch on a curve, and
that is what this module does -- exactly, and symbolically in g.

Bookkeeping
-----------
Fix a theta characteristic L, i.e. L^2 = K_X, so deg L = g-1.  Every line
bundle that occurs below is L^k twisted by a 2-torsion class eta (an element of
H^1(X; Z/2), written multiplicatively, eta^2 = O).  A *rigging* of an
n-extended structure is a choice of such an eta for each odd coordinate: the
Neveu-Schwarz/Ramond dichotomy of the supercircle, spread over the 2g cycles of
X.

h^0 of L^k (x) eta, for g >= 2:

      k < 0 : 0                          (negative degree)
      k = 0 : 1 if eta trivial, else 0
      k = 1 : s(eta) := h^0(L (x) eta)   -- THE one unknown; deg = g-1 is the
                                            critical degree, and L (x) eta is
                                            again a theta characteristic
      k = 2 : g if eta trivial else g-1  (K (x) eta)
      k >= 3: (k-1)(g-1)                 (deg > 2g-2, so h^1 = 0)

and h^1(F) = h^0(K (x) F^{-1}) throughout; for F = L^k (x) eta this is
h^0(L^{2-k} (x) eta).  Riemann-Roch, h^0-h^1 = (k-1)(g-1), is verified by the
self-tests rather than assumed.

g = 1 and g = 0 are separate branches: at g = 1 every L^k (x) eta has degree 0
and h^0 is 1 or 0 according as the bundle is trivial; at g = 0 there is no
2-torsion, L = O(-1), and h^0(O(d)) = max(0, d+1).

Nothing here is approximate and nothing is sampled.
"""

import sympy as sp

# ------------------------------------------------------------------ the genus
G = sp.Symbol('g', integer=True)          # stands for a genus >= 2


# ------------------------------------------------- 2-torsion classes as a group
#
# A 2-torsion class is a frozenset of generator names; the group law is the
# symmetric difference, and the empty set is the trivial class O.

TRIV = frozenset()


def eta(*names):
    return frozenset(names)


def emul(*classes):
    out = TRIV
    for c in classes:
        out = out ^ c
    return out


# ------------------------------------------------------------- a line bundle
class LB:
    """The line bundle L^k (x) eta, L a fixed theta characteristic."""

    __slots__ = ('k', 'e')

    def __init__(self, k, e=TRIV):
        self.k = int(k)
        self.e = frozenset(e)

    def __mul__(self, other):
        return LB(self.k + other.k, emul(self.e, other.e))

    def inv(self):
        return LB(-self.k, self.e)

    def __repr__(self):
        tag = '' if not self.e else '.' + ''.join(sorted(self.e))
        return 'L^%d%s' % (self.k, tag)

    def __eq__(self, o):
        return self.k == o.k and self.e == o.e

    def __hash__(self):
        return hash((self.k, self.e))


THETA = LB(-2)          # Theta = K^{-1} = L^{-2}
CANON = LB(2)           # K = L^2
TRIVB = LB(0)           # O


def sdeg(b, g=G):
    """deg(L^k (x) eta) = k(g-1)."""
    return b.k * (g - 1)


# ------------------------------------- the spin structure of an elliptic curve
#
# At g = 1 the class of L itself is the datum: TRIV = Ramond, non-trivial = NS.
G1_L = [eta('L')]


def set_g1_sector(sector):
    """'R' : L = O (the odd theta characteristic);  'NS' : L non-trivial."""
    assert sector in ('R', 'NS')
    G1_L[0] = TRIV if sector == 'R' else eta('L')


# ----------------------------------------------------------- the unknowns s(eta)
_S = {}


def s(e):
    """s(eta) = h^0(L (x) eta), the dimension of the theta characteristic."""
    e = frozenset(e)
    if e not in _S:
        tag = 'o' if not e else ''.join(sorted(e))
        _S[e] = sp.Symbol('s_' + tag, integer=True, nonnegative=True)
    return _S[e]


# ------------------------------------------------------------------- h^0, h^1
def h0(b, g=G):
    """h^0(X; L^k (x) eta) on a curve of genus g."""
    if g == 0:
        assert not b.e, 'CP^1 carries no 2-torsion'
        d = -b.k                      # L = O(-1), so L^k = O(-k)
        return max(0, d + 1)
    if g == 1:
        # deg = 0 throughout and L^2 = K = O, so L is itself 2-torsion: one of
        # the four spin structures of an elliptic curve.  L = O is the ODD
        # theta characteristic (h^0 = 1) -- periodic in both cycles, i.e. the
        # Ramond one; the three others are even (h^0 = 0), i.e. Neveu-Schwarz.
        # G1_L carries the class of L itself.
        cls = emul(b.e, TRIV if b.k % 2 == 0 else G1_L[0])
        return 1 if cls == TRIV else 0
    # ------------------------------------------------------------ g >= 2
    k, e = b.k, b.e
    if k < 0:
        return sp.Integer(0)
    if k == 0:
        return sp.Integer(1) if e == TRIV else sp.Integer(0)
    if k == 1:
        return s(e)
    if k == 2:
        return g if e == TRIV else g - 1
    return (k - 1) * (g - 1)


def h1(b, g=G):
    """h^1(F) = h^0(K (x) F^{-1}); for F = L^k (x) eta that is h^0(L^{2-k} (x) eta)."""
    return h0(LB(2 - b.k, b.e), g)


# ------------------------------------------------------ a sheaf = graded pieces
class Sheaf:
    """A list of (line bundle, parity) -- the graded pieces of K, with parities.

    `split` records whether the sheaf really IS the direct sum of its pieces.
    It is True for the (peri)contact sheaves, where the isomorphism
    X |-> alpha(X) carries K onto (T/D) (x) Lambda E and Lambda E splits with E;
    it is FALSE for vect and svect, where T_q is a non-split extension
    0 -> Lambda^{q+1}E (x) E^* -> T_q -> Theta (x) Lambda^q E -> 0
    (for q = 0 the Atiyah algebroid of E).  Under an extension the Euler
    characteristic is still additive, so chi() is safe in either case, whereas
    H(1) is only an upper bound when split is False.
    """

    def __init__(self, pieces, name='', split=True):
        self.pieces = list(pieces)
        self.name = name
        self.split = split

    def rank(self):
        ev = sum(1 for _, p in self.pieces if p == 0)
        od = sum(1 for _, p in self.pieces if p == 1)
        return (ev, od)

    def degree(self, g=G):
        ev = sum(sdeg(b, g) for b, p in self.pieces if p == 0)
        od = sum(sdeg(b, g) for b, p in self.pieces if p == 1)
        return (sp.expand(ev), sp.expand(od))

    def H(self, i, g=G):
        f = h0 if i == 0 else h1
        ev = sum(f(b, g) for b, p in self.pieces if p == 0)
        od = sum(f(b, g) for b, p in self.pieces if p == 1)
        return (sp.expand(sp.sympify(ev)), sp.expand(sp.sympify(od)))

    def sdim_versal(self, g=G):
        """sdim of the base of the versal deformation = sdim H^1."""
        return self.H(1, g)

    def chi(self, g=G):
        """chi = deg + rank(1-g), piece by piece: ADDITIVE, hence safe under
        extensions, so this is the one number an unsplit sheaf still gives."""
        ev = sum(sdeg(b, g) + (1 - g) for b, p in self.pieces if p == 0)
        od = sum(sdeg(b, g) + (1 - g) for b, p in self.pieces if p == 1)
        return (sp.expand(sp.sympify(ev)), sp.expand(sp.sympify(od)))

    def chi_from_cohomology(self, g=G):
        ev0, od0 = self.H(0, g)
        ev1, od1 = self.H(1, g)
        return (sp.expand(ev0 - ev1), sp.expand(od0 - od1))


# ===================================================================== structures
#
# In every case the sheaf of structure-preserving vector fields is written as
# a sum of line bundles with parities, DERIVED (not asserted) in Sections 20-25 of the text; the
# derivations are recorded in the comments.


def contact(n, riggings=None, g=G):
    """The n-extended contact structure on a (1|n)-superstring.

    The contact distribution D < T has rank 0|n and the Frobenius form
    S^2 D -> T/D is non-degenerate; in the split model D = Pi(E^*) and
    T/D = Theta, so E = E^* (x) K, i.e. E carries a K-valued symmetric
    non-degenerate form.  The diagonal such E are

          E = (+)_i L (x) eta_i,      eta_i^2 = O,

    the eta_i being the rigging (Neveu-Schwarz vs Ramond around each cycle).

    The map X |-> alpha(X) is an isomorphism of sheaves K -> (T/D) (x) Lambda E,
    so the graded pieces are Theta (x) Lambda^q E, of parity q mod 2, and

          Theta (x) Lambda^q E = (+)_{|S|=q} L^{q-2} (x) eta_S.
    """
    if riggings is None:
        riggings = [TRIV] * n
    assert len(riggings) == n
    pieces = []
    for q in range(n + 1):
        for S in _subsets(range(n), q):
            e = emul(*[riggings[i] for i in S])
            pieces.append((LB(q - 2, e), q % 2))
    return Sheaf(pieces, 'k(1|%d)' % n)


def pericontact(rig=TRIV, g=G):
    """m(1): the pericontact structure on a (1|2)-superstring.

    alpha_0 = d tau + q d xi + xi d q is ODD; ker alpha_0 has rank 1|1 and
    T/ker = Pi B for a line bundle B.  Reading alpha_0 on the frame:
    alpha_0(d_tau) = 1 gives E_1 = B; alpha_0(d_q) = xi gives E_2 = B (x) K;
    and the degree-0 part of the sheaf must be Theta (+) O, which forces
    B^2 = Theta, i.e. B = L^{-1} (x) eta.  Hence

        E = L^{-1}eta (+) L eta,   pieces B (x) Lambda^q E of parity q+1:
        q=0: L^{-1}eta      (odd)
        q=1: Theta, O       (even)
        q=2: L^{-1}eta      (odd)
    """
    B = LB(-1, rig)
    E1, E2 = LB(-1, rig), LB(1, rig)
    pieces = [(B, 1),
              (B * E1, 0), (B * E2, 0),
              (B * E1 * E2, 1)]
    return Sheaf(pieces, 'm(1)')


def vect(bundles, g=G, split=False):
    """vect(1|n) on the split (X, Lambda E), E = (+) bundles: NO structure.

    T_q = Hom(E, Lambda^{q+1} E) (+) Theta (x) Lambda^q E, of parity q mod 2.
    This is the sheaf whose H^1 Vaintrob computes in 3.3.1 and 3.3.2.
    """
    n = len(bundles)
    pieces = []
    for q in range(-1, n + 1):
        # Hom(E, Lambda^{q+1} E)
        if 0 <= q + 1 <= n:
            for i in range(n):
                for S in _subsets(range(n), q + 1):
                    e = emul(*[bundles[j].e for j in S])
                    k = sum(bundles[j].k for j in S) - bundles[i].k
                    pieces.append((LB(k, emul(e, bundles[i].e)), q % 2))
        # Theta (x) Lambda^q E
        if 0 <= q <= n:
            for S in _subsets(range(n), q):
                e = emul(*[bundles[j].e for j in S])
                k = sum(bundles[j].k for j in S) - 2
                pieces.append((LB(k, e), q % 2))
    return Sheaf(pieces, 'vect(1|%d)' % n, split=split)


def _subsets(it, q):
    from itertools import combinations
    return list(combinations(list(it), q))


# --------------------------------------------------------------- the functions
def sdim(pair):
    a, b = pair
    return '(%s | %s)' % (sp.simplify(a), sp.simplify(b))
