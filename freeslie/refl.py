#!/usr/bin/env python3
"""
refl.py -- machine check of the numbers B_kj = sup{m >= 0 | alpha_j + m alpha_k is a root}
for Lie (super)algebras with Cartan matrix over a field of characteristic p.

Compared, for every ordered pair (k, j) of nodes of every algebra in DATA:

  measured  -- read off the root system printed in
               [BLLS] Bouarroudj, Leites, Lozhechnyk, Shang, "The roots of exceptional
               modular Lie superalgebras with Cartan matrix", Arnold Math. J. 6 (2020)
               63--118, arXiv:1904.09578, Section 1;
  recursion -- Lebedev's Statement 2.1 of arXiv:2311.02710 run literally:
               d_{-1} = 0,  d_m = (-1)^{i_k} (d_{m-1} - A_kj - m A_kk),
               B_kj = inf{m >= 0 | d_m = 0};
  Leb(3)    -- his closed formula (3) of the same note;
  BGL(7.4)  -- the formula published in [BGL] = Bouarroudj, Grozman, Leites, SIGMA 5
               (2009) 060, (7.4), repeated verbatim as (44) in [BGLL] = Bouarroudj,
               Grozman, Lebedev, Leites, Homology Homotopy Appl. 12 (2010) 237--278.

Everything is exact: integers mod p, no floating point.

Also checked, for every node k, using the corrected B_kj:
  (a) is the image of the base under
          alpha_k |--> -alpha_k,  alpha_j |--> alpha_j + B_kj alpha_k    (rule (2))
      again a system of simple roots (every root a non-negative or non-positive
      integer combination of it)?
  (b) does the linear extension r_k of that map to the root lattice satisfy
      r_k(Delta) = Delta and r_k^2 = id, i.e. is it a reflection in the usual sense?

Run: python3 refl.py
"""

from fractions import Fraction
from itertools import product

# ---------------------------------------------------------------- scalars
# A Cartan matrix entry is an int (read mod p) or PARAM: a parameter of the
# family, an element of the ground field lying outside the prime field.


class Param:
    """A generic parameter alpha of a family (alpha != 0, alpha not in F_p)."""

    __slots__ = ("name",)

    def __init__(self, name="a"):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Param) and other.name == self.name

    def __hash__(self):
        return hash(("Param", self.name))


PARAM = Param()


def is_zero(x, p):
    return (not isinstance(x, Param)) and (x % p == 0 if p else x == 0)


def same(x, y, p):
    """x == y in the ground field."""
    if isinstance(x, Param) or isinstance(y, Param):
        return x == y
    return (x - y) % p == 0 if p else x == y


def ratio(num, den, p):
    """num/den as an element of the prime field Z/p, or None if it is not there.

    den is assumed non-zero.  A parameter is generic, so a ratio involving it
    lies in the prime field only when it cancels.
    """
    if isinstance(num, Param) or isinstance(den, Param):
        return 0 if is_zero(num, p) else (1 if same(num, den, p) else None)
    if p == 0:
        r = Fraction(num, den)
        return r  # an honest rational; caller decides whether it is an integer
    return (num % p) * pow(den % p, -1, p) % p


def minrep(x, p):
    """The minimal non-negative integer congruent to x mod p."""
    if p == 0:
        return int(x) if (isinstance(x, Fraction) and x.denominator == 1) else None
    return x % p


# ------------------------------------------------------- the three formulas


def B_recursion(p, ik, Akk, Akj, bound=None):
    """Lebedev's Statement 2.1, run as stated.  None means 'no zero found'.

    d_m is kept as a pair (c0, c1) standing for c0 + c1*a, where a is the
    parameter of a family: it is transcendental over the prime field, so
    d_m = 0 means c0 = c1 = 0.  Entries that are plain scalars have c1 = 0.
    """
    if bound is None:
        bound = 4 * p + 4 if p else 200

    def lift(x):
        return (0, 1) if isinstance(x, Param) else (x, 0)

    def red(c):
        return (c[0] % p, c[1] % p) if p else c

    sgn = 1 if ik == 0 else -1
    akk, akj = lift(Akk), lift(Akj)
    d = (0, 0)  # d_{-1}
    for m in range(bound):
        d = red((sgn * (d[0] - akj[0] - m * akk[0]),
                 sgn * (d[1] - akj[1] - m * akk[1])))
        if d == (0, 0):
            return m
    return None


BRANCH = {}  # which case of formula (3) was used, how often


def B_lebedev(p, ik, Akk, Akj, count=False):
    """Lebedev's closed formula (3), arXiv:2311.02710."""

    def hit(tag, val):
        if count:
            BRANCH[tag] = BRANCH.get(tag, 0) + 1
        return val

    if is_zero(Akj, p):
        return hit("A_kj = 0  ->  0", 0)
    if ik == 0:
        if is_zero(Akk, p):
            return hit("i_k even, A_kk = 0  ->  p-1", p - 1)
        if p != 2:
            r = ratio(Akj, Akk, p)
            if r is None:
                return hit("i_k even, A_kk != 0, ratio not in F_p  ->  p-1", p - 1)
            return hit("i_k even, A_kk != 0, p != 2  ->  (-2A_kj/A_kk)_Z",
                       minrep(-2 * r, p))
        if same(Akj, Akk, p):
            return hit("i_k even, p = 2, A_kj = A_kk != 0  ->  2", 2)
        return hit("i_k even, p = 2, A_kj not in {0, A_kk}  ->  3", 3)
    else:
        if is_zero(Akk, p):
            return hit("i_k odd, A_kk = 0 (odd reflection)  ->  1", 1)
        r = ratio(Akj, Akk, p)
        if r is None:
            return hit("i_k odd, A_kk != 0, ratio not in F_p  ->  2p-1", 2 * p - 1)
        return hit("i_k odd, A_kk != 0  ->  2(-A_kj/A_kk)_Z", 2 * minrep(-r, p))


def B_bgl(p, ik, Akk, Akj, Ajk=None):
    """The formula published as (7.4) in [BGL] and as (44) in [BGLL]."""
    if ik == 0:
        if not is_zero(Akk, p):
            r = ratio(Akj, Akk, p)
            if r is None:
                return p - 1
            val = minrep(-2 * r, p)
            # [BGL], Remark 7.3.1: "There is known just one exception: if p = 2
            # and A_kk = A_jk, the expression -2A_jk/A_kk should be understood
            # as 2, not 0."
            if p == 2 and Ajk is not None and same(Akk, Ajk, p):
                val = 2
            return val
        return 0 if is_zero(Akj, p) else p - 1
    else:
        if not is_zero(Akk, p):
            r = ratio(Akj, Akk, p)
            if r is None:
                return p - 1
            return minrep(-r, p)
        return 0 if is_zero(Akj, p) else 1


# ------------------------------------------------------------ the root data
# Every entry: p, the Cartan matrix A, the parities i_k of the Chevalley
# generators (0 = even, 1 = odd), and the positive roots in the basis of
# simple roots, as printed in the source named in 'src'.

DATA = {}


def add(name, p, A, parity, pos, src, note=""):
    DATA[name] = dict(p=p, A=A, parity=parity, pos=[tuple(r) for r in pos],
                      src=src, note=note)


# --- p = 5 ---------------------------------------------------------------
add("brj(2;5)", 5,
    [[0, -1], [-2, 1]], [1, 1],
    [(1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (2, 5)],
    "[BLLS] 1.1.1, sdim 10|12")

add("el(5;5)", 5,
    [[2, 0, -1, 0, 0],
     [0, 2, 0, 0, -1],
     [-1, 0, 0, -4, -4],
     [0, 0, -4, 0, -2],
     [0, -1, -4, -2, 0]], [0, 0, 1, 1, 1],
    [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
     (1, 0, 1, 0, 0), (0, 1, 0, 0, 1), (0, 0, 1, 1, 0), (0, 0, 1, 0, 1), (0, 0, 0, 1, 1),
     (0, 1, 1, 0, 1), (1, 0, 1, 1, 0), (0, 1, 0, 1, 1), (1, 0, 1, 0, 1), (0, 0, 1, 1, 1),
     (1, 1, 1, 0, 1), (1, 0, 1, 1, 1), (0, 1, 1, 1, 1), (0, 1, 0, 1, 2),
     (1, 1, 1, 1, 1), (1, 0, 2, 1, 1), (0, 1, 1, 1, 2), (0, 1, 0, 2, 2),
     (1, 1, 2, 1, 1), (1, 1, 1, 1, 2), (0, 1, 1, 2, 2),
     (1, 1, 1, 2, 2), (1, 1, 2, 1, 2), (0, 1, 1, 2, 3),
     (1, 1, 2, 2, 2), (1, 1, 1, 2, 3), (0, 2, 1, 2, 3),
     (1, 2, 1, 2, 3), (1, 1, 2, 2, 3),
     (1, 2, 2, 2, 3), (1, 1, 2, 3, 3),
     (1, 2, 2, 2, 4), (1, 2, 2, 3, 3),
     (1, 2, 2, 3, 4), (1, 2, 3, 3, 4), (2, 2, 3, 3, 4)],
    "[BLLS] 1.1.3, sdim 55|32")

# --- p = 3 ---------------------------------------------------------------
add("brj(2;3)", 3,
    [[0, -1], [-2, 1]], [1, 1],
    [(1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4)],
    "[BLLS] 1.1.2, sdim 10|8")

add("el(5;3)", 3,
    [[0, 0, 0, -2, 0],
     [0, 2, 0, -1, 0],
     [0, 0, 0, -1, -2],
     [-2, -1, -1, 0, 0],
     [0, 0, -2, 0, 0]], [1, 0, 1, 1, 1],
    [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
     (1, 0, 0, 1, 0), (0, 1, 0, 1, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 1),
     (1, 1, 0, 1, 0), (1, 0, 1, 1, 0), (0, 1, 1, 1, 0), (0, 0, 1, 1, 1),
     (1, 1, 1, 1, 0), (1, 0, 1, 1, 1), (0, 1, 1, 2, 0), (0, 1, 1, 1, 1),
     (0, 1, 1, 2, 1), (1, 1, 1, 2, 0), (0, 1, 2, 2, 0), (1, 1, 1, 1, 1),
     (1, 1, 1, 2, 1), (1, 1, 2, 2, 0), (0, 1, 2, 2, 1),
     (1, 1, 2, 3, 0), (1, 1, 2, 2, 1),
     (1, 2, 2, 3, 0), (1, 1, 2, 3, 1),
     (1, 1, 3, 3, 1), (1, 2, 2, 3, 1),
     (1, 2, 3, 3, 1), (1, 2, 3, 4, 1), (2, 2, 3, 4, 1)],
    "[BLLS] 1.1.4, sdim 39|32")

add("g(1,6)", 3,
    [[2, -1, 0], [-1, 1, -1], [0, -1, 0]], [0, 1, 1],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (0, 2, 0), (0, 1, 1),
     (1, 2, 0), (1, 1, 1), (0, 2, 1),
     (2, 2, 0), (1, 2, 1),
     (2, 2, 1), (1, 3, 1),
     (2, 3, 1), (2, 4, 1), (2, 4, 2)],
    "[BLLS] 1.1.5, sdim 21|14")

add("g(2,3)", 3,
    [[0, 0, -1], [0, 0, -1], [-1, -1, 0]], [1, 1, 1],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 2),
     (2, 1, 2), (1, 2, 2), (2, 2, 2), (2, 2, 3)],
    "[BLLS] 1.1.6, sdim 12/10|14")

add("g(4,3)", 3,
    [[0, -1, 0, 0], [-1, 0, -2, 0], [0, -1, 0, -1], [0, 0, -1, 0]], [1, 1, 1, 1],
    [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
     (1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1), (1, 1, 1, 0), (0, 1, 1, 1),
     (1, 1, 1, 1), (0, 1, 2, 1), (0, 2, 2, 1), (1, 1, 2, 1), (0, 1, 2, 2),
     (1, 1, 2, 2), (1, 2, 2, 1), (0, 2, 2, 2), (1, 2, 2, 2), (0, 2, 3, 2),
     (1, 2, 3, 2), (1, 3, 3, 2), (1, 3, 4, 2), (1, 3, 4, 3)],
    "[BLLS] 1.2, sdim 24|26")

add("g(8,3)", 3,
    [[2, -1, -1, 0, 0],
     [-1, 0, -1, -2, 0],
     [-1, -1, 0, 0, 0],
     [0, -2, 0, 0, -1],
     [0, 0, 0, -1, 0]], [0, 1, 1, 1, 1],
    [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1),
     (1, 1, 0, 0, 0), (1, 0, 1, 0, 0), (0, 1, 1, 0, 0), (0, 1, 0, 1, 0), (0, 0, 0, 1, 1),
     (1, 1, 1, 0, 0), (1, 1, 0, 1, 0), (0, 1, 1, 1, 0), (0, 1, 0, 1, 1),
     (2, 1, 1, 0, 0), (1, 1, 0, 1, 1), (1, 1, 1, 1, 0), (0, 1, 1, 1, 1),
     (2, 1, 1, 1, 0), (1, 2, 1, 1, 0), (1, 1, 1, 1, 1), (2, 1, 1, 1, 1),
     (2, 2, 1, 1, 0), (1, 2, 1, 1, 1), (1, 2, 1, 2, 0), (2, 2, 1, 2, 0),
     (2, 2, 2, 1, 0), (2, 2, 1, 1, 1), (1, 2, 1, 2, 1), (2, 3, 1, 2, 0),
     (2, 2, 2, 1, 1), (2, 2, 1, 2, 1), (2, 2, 2, 2, 0), (2, 3, 1, 2, 1),
     (2, 3, 2, 2, 0), (2, 2, 2, 2, 1), (2, 3, 1, 3, 1), (3, 3, 2, 2, 0),
     (2, 3, 2, 2, 1), (3, 3, 2, 2, 1), (2, 3, 2, 3, 1), (3, 3, 2, 3, 1),
     (2, 4, 2, 3, 1), (3, 4, 2, 3, 1), (4, 4, 2, 3, 1), (3, 4, 3, 3, 1),
     (4, 4, 3, 3, 1), (4, 5, 3, 3, 1), (4, 5, 3, 4, 1), (4, 5, 3, 4, 2)],
    "[BLLS] 1.2.1, sdim 55|50")

add("br(3)", 3,
    [[2, -1, 0], [-1, 2, -1], [0, -1, 0]], [0, 0, 0],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (0, 1, 1), (1, 1, 1), (0, 1, 2), (1, 1, 2),
     (1, 2, 2), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
    "[BLLS] 2.4, dim 29",
    "a Lie algebra: node 3 is even with A_33 = 0 (Brown's node)")

# The Brown algebra br(2; a), p = 3, dim 10: a Lie algebra, so both nodes are
# even, with the Cartan matrix ((2,-1),(a,2)) of [BGL] (9.9), where a is the
# parameter of the family.  It is the ONLY finite-dimensional g(A) in the
# classification of [BGL] whose parameter stands in a row with A_kk != 0; the
# ratio -2A_21/A_22 = -a then lies outside Z/3, which is the branch of
# Lebedev's (3) giving p-1.
#
# The root system is that of o(5) ([BLLS] 2.3), i.e. of type B_2, with four
# positive roots.  Which of the two simple roots is the long one is fixed
# without using the branch under test: B_12 = (-2A_12/A_11)_Z = 1 involves no
# parameter and is the same in every version of the formula, so alpha_2 + 2
# alpha_1 is not a root, and the four positive roots must be as below.  Then
# B_21 = 2 is read off the root system, independently of any formula.
add("br(2;a)", 3,
    [[2, -1], [PARAM, 2]], [0, 0],
    [(1, 0), (0, 1), (1, 1), (1, 2)],
    "[BGL] (9.9) + [BLLS] 2.3 (root system of o(5)), dim 10",
    "the parameter sits in a row with A_kk != 0: the branch 'ratio not in Z/p'")

# --- p = 2 ---------------------------------------------------------------
add("bgl(3;a)", 2,
    [[0, PARAM, 0], [PARAM, 0, 1], [0, 1, 0]], [1, 1, 1],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (0, 1, 1), (1, 1, 1), (1, 2, 1)],
    "[BLLS] 1.3.2, sdim 10/8|8")

BGL4_ROOTS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
              (1, 1, 0, 0), (1, 0, 1, 0), (0, 0, 1, 1),
              (1, 1, 1, 0), (1, 0, 1, 1),
              (2, 1, 1, 0), (1, 1, 1, 1),
              (2, 1, 1, 1), (2, 1, 2, 1), (3, 1, 2, 1), (3, 2, 2, 1)]

# The Cartan matrix printed in [BLLS] 1.3.3 next to this basis is
#     ((0,a,0,0),(a,0,1,0),(0,1,0,1),(0,0,1,0)),
# which cannot go with it: it has A_13 = 0, yet x6 = [x1,x3] is a root vector,
# and it has A_23 = 1, yet alpha_2 + alpha_3 is not a root.  The basis goes with
# the Cartan matrix of [BGLd, Lemma 9.12] -- the one used in [KLS] --
# which is the printed one with the nodes 1 and 2 interchanged.  Both are run below.
add("bgl(4;a)", 2,
    [[0, PARAM, 1, 0], [PARAM, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], [1, 0, 0, 0],
    BGL4_ROOTS,
    "[BGL] 13.1.3 (matrix, parities and basis), sdim 18|16",
    "the parities are those of [BGL] 13.1.3: x1 odd, x2, x3, x4 even, i.e. the "
    "diagonal (0, 0bar, 0bar, 0bar); at p = 2 no B_kj depends on this choice")

add("bgl(4;a) [matrix as printed in BLLS 1.3.3]", 2,
    [[0, PARAM, 0, 0], [PARAM, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], [1, 1, 1, 1],
    BGL4_ROOTS,
    "[BLLS] 1.3.3 verbatim",
    "matrix and basis disagree -- see the comment in the source; kept to show it")

# The desuperizations: same Cartan matrices with every node even (the
# distribution of the diagonal entries which [BGL] 13.1.2, 13.1.3 exclude from
# the list for bgl, namely the all-0bar one), and, by [BLLS] 2.1, the same root
# systems.  For p = 2, a node with A_kk = 0 gives B_kj = 1 whether it is even
# (branch 'i_k = ev, A_kk = 0 -> p-1') or odd (branch 'odd reflection'), so
# these runs must, and do, reproduce the numbers of bgl(3;a) and bgl(4;a); what
# they add is a test of the even branch in the presence of a parameter.
add("wk(3;a)", 2,
    [[0, PARAM, 0], [PARAM, 0, 1], [0, 1, 0]], [0, 0, 0],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (0, 1, 1), (1, 1, 1), (1, 2, 1)],
    "[BGL] 9.1 + [BLLS] 2.1 (same roots as bgl(3;a)), dim 18",
    "a Lie algebra: every node is even with A_kk = 0bar")

add("wk(4;a)", 2,
    [[0, PARAM, 1, 0], [PARAM, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], [0, 0, 0, 0],
    BGL4_ROOTS,
    "[BGL] 9.1 + [BLLS] 2.1 (same roots as bgl(4;a)), dim 34",
    "a Lie algebra; the matrix is the corrected one, see bgl(4;a) above")

# F(oo(1|2n)), the desuperization of the ortho-orthogonal superalgebra, denoted
# Delta_n in [WK]: a simple Lie algebra whose root system is that of o(2n+1)
# ([BLLS] 2.2).  Its normalized Cartan matrix is that of o(2n+1) with the last
# row divided by 2 ([BGL], Section 13, item 1, and Subsection 4.6.1) and reduced
# modulo 2; so A_nn = 1 with the nth node EVEN -- the node '*', the one type our
# earlier sample had no instance of.  Simple roots as usual: alpha_i =
# eps_i - eps_{i+1} for i < n, alpha_n = eps_n.
add("Delta_2 = F(oo(1|4))", 2,
    [[0, 1], [1, 1]], [0, 0],
    [(1, 0), (0, 1), (1, 1), (1, 2)],
    "[BLLS] 2.2 + [BGL] Section 13 (root system of o(5))",
    "node 2 is the node '*': even, A_22 = 1bar != 0")

add("Delta_3 = F(oo(1|6))", 2,
    [[0, 1, 0], [1, 0, 1], [0, 1, 1]], [0, 0, 0],
    [(1, 1, 1), (0, 1, 1), (0, 0, 1),          # eps_1, eps_2, eps_3
     (1, 0, 0), (1, 2, 2),                     # eps_1 -+ eps_2
     (1, 1, 0), (1, 1, 2),                     # eps_1 -+ eps_3
     (0, 1, 0), (0, 1, 2)],                    # eps_2 -+ eps_3
    "[BLLS] 2.2 + [BGL] Section 13 (root system of o(7))",
    "node 3 is the node '*': even, A_33 = 1bar != 0")

# --- p = 5, the one parametric family that exists for every p > 2 --------
# osp(4|2; a) = D(2,1;a).  Simple roots alpha_1 = eps_1 - eps_2 - eps_3 (odd,
# isotropic), alpha_2 = 2 eps_2, alpha_3 = 2 eps_3 (even), the invariant form
# being (eps_1,eps_1) = -(1+a)/2, (eps_2,eps_2) = 1/2, (eps_3,eps_3) = a/2.
# Then A_21 = 2(alpha_2,alpha_1)/(alpha_2,alpha_2) = -1 and likewise A_31 = -1,
# while the first row, isotropic and hence defined up to a factor, is
# (0, 1, a): the parameter stands in a row with A_kk = 0, as it does in every
# parametric family except br(2;a).
add("osp(4|2;a), p=5", 5,
    [[0, 1, PARAM], [-1, 2, 0], [-1, 0, 2]], [1, 0, 0],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (1, 0, 1), (1, 1, 1), (2, 1, 1)],
    "[BGL] Section 10, item 3; roots +-2eps_i (even), +-eps_1+-eps_2+-eps_3 (odd)")

# --- characteristic 0, for calibration -----------------------------------
# osp(1|4) = oo(1|4): simple roots a1 = e1 - e2 (even), a2 = e2 (odd, non-isotropic).
add("osp(1|4), p=0", 0,
    [[2, -1], [-2, 2]], [0, 1],
    [(1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)],
    "classical; roots +-e_i, +-2e_i, +-e_1+-e_2")

# osp(1|6): a1 = e1-e2, a2 = e2-e3 (even), a3 = e3 (odd, non-isotropic).
def _osp16():
    e = [(1, 1, 1), (0, 1, 1), (0, 0, 1)]  # e1, e2, e3 in the basis a1,a2,a3

    def add_(u, v, s):
        return tuple(a + s * b for a, b in zip(u, v))
    pos = []
    for i in range(3):
        pos.append(e[i])                       # e_i        (odd)
        pos.append(tuple(2 * c for c in e[i]))  # 2e_i       (even)
        for j in range(i + 1, 3):
            pos.append(add_(e[i], e[j], -1))    # e_i - e_j
            pos.append(add_(e[i], e[j], +1))    # e_i + e_j
    return pos


add("osp(1|6), p=0", 0,
    [[2, -1, 0], [-1, 2, -1], [0, -2, 2]], [0, 0, 1],
    _osp16(),
    "classical; roots +-e_i, +-2e_i, +-e_i+-e_j")


# ------------------------------------------------------------------ checks

def measured_B(pos, k, j):
    """sup{m >= 0 | alpha_j + m alpha_k is a root}, read off the root system."""
    n = len(next(iter(pos)))
    roots = set(pos) | {tuple(-c for c in r) for r in pos}
    m = 0
    while True:
        cand = [0] * n
        cand[j] = 1
        cand[k] += m
        if tuple(cand) not in roots:
            return m - 1 if m else 0
        m += 1
        if m > 200:
            return None


def rho(base, roots, k):
    """Rule (2) applied to an arbitrary base: the numbers B are read off the root
    system, so this needs no Cartan matrix and can be iterated."""
    n = len(base)
    new = []
    for j in range(n):
        if j == k:
            new.append(tuple(-c for c in base[k]))
            continue
        m = 0
        while True:
            cand = tuple(base[j][i] + (m + 1) * base[k][i] for i in range(n))
            if cand not in roots:
                break
            m += 1
        new.append(tuple(base[j][i] + m * base[k][i] for i in range(n)))
    return new


def involutivity(alg):
    """Is rho_k o rho_k the identity on bases?  (A groupoid needs this.)"""
    pos = alg["pos"]
    n = len(alg["A"])
    roots = set(pos) | {tuple(-c for c in r) for r in pos}
    base0 = [tuple(1 if i == j else 0 for i in range(n)) for j in range(n)]
    out = []
    for k in range(n):
        b1 = rho(base0, roots, k)
        b2 = rho(b1, roots, k)
        out.append((k, b2 == base0))
    return out


def base_check(alg, Bmat):
    """rule (2) with the given B: is the image again a system of simple roots,
    and does its linear extension preserve the root system?"""
    p, pos = alg["p"], alg["pos"]
    n = len(alg["A"])
    roots = set(pos) | {tuple(-c for c in r) for r in pos}
    out = []
    for k in range(n):
        # new base: beta_k = -alpha_k, beta_j = alpha_j + B_kj alpha_k
        beta = []
        for j in range(n):
            v = [0] * n
            if j == k:
                v[k] = -1
            else:
                v[j] = 1
                v[k] = Bmat[k][j]
            beta.append(tuple(v))
        allroots = all(b in roots for b in beta)
        # coordinates of every root in the new base.  The change of basis is
        # unitriangular up to the sign in place k, so it is invertible over Z:
        #   x = sum_j c_j beta_j  =>  c_j = x_j (j != k),
        #   c_k = -( x_k - sum_{j != k} x_j B_kj ).
        isbase = True
        for x in roots:
            c = [x[j] for j in range(n)]
            c[k] = -(x[k] - sum(x[j] * Bmat[k][j] for j in range(n) if j != k))
            if not (all(t >= 0 for t in c) or all(t <= 0 for t in c)):
                isbase = False
                break
        # linear extension r_k of alpha_j |-> beta_j
        def r(x):
            return tuple(sum(x[j] * beta[j][i] for j in range(n)) for i in range(n))
        preserves = all(r(x) in roots for x in roots)
        invol = all(r(r(x)) == x for x in roots)
        out.append((k, allroots, isbase, preserves, invol))
    return out


def run():
    p_bad_leb = p_bad_bgl = p_bad_rec = total = 0
    print("=" * 78)
    print("B_kj: measured (from the printed root system) vs the three formulas")
    print("=" * 78)
    summary = []
    for name, alg in DATA.items():
        skip = "as printed in BLLS" in name  # kept only to exhibit the clash
        p, A, par, pos = alg["p"], alg["A"], alg["parity"], alg["pos"]
        n = len(A)
        assert len(set(pos)) == len(pos), f"{name}: a repeated root (multiplicity > 1)"
        print(f"\n--- {name}   p = {p}   {alg['src']}"
              + (f"\n    {alg['note']}" if alg["note"] else ""))
        print(f"    {len(pos)} positive roots, parities i = {par}")
        rows = []
        Bmat = [[0] * n for _ in range(n)]
        for k in range(n):
            for j in range(n):
                if j == k:
                    continue
                meas = measured_B(pos, k, j)
                rec = B_recursion(p, par[k], A[k][k], A[k][j])
                leb = B_lebedev(p, par[k], A[k][k], A[k][j], count=not skip) if p else None
                bgl = B_bgl(p, par[k], A[k][k], A[k][j], A[j][k]) if p else None
                Bmat[k][j] = meas
                if not skip:
                    total += 1
                    if rec != meas:
                        p_bad_rec += 1
                    if leb is not None and leb != meas:
                        p_bad_leb += 1
                    if bgl is not None and bgl != meas:
                        p_bad_bgl += 1
                flag = ""
                if bgl is not None and bgl != meas:
                    flag += "   <== BGL (7.4) WRONG"
                if leb is not None and leb != meas:
                    flag += "   <== Leb (3) WRONG"
                if rec != meas:
                    flag += "   <== recursion 2.1 WRONG"
                rows.append((k + 1, j + 1, A[k][k], A[k][j], par[k], meas, rec, leb, bgl, flag))
        print("      k  j  A_kk  A_kj  i_k | measured  recursion  Leb(3)  BGL(7.4)")
        for (k, j, Akk, Akj, ik, meas, rec, leb, bgl, flag) in rows:
            print(f"      {k}  {j}  {str(Akk):>4}  {str(Akj):>4}   {ik}  |"
                  f" {meas:>8}  {str(rec):>9}  {str(leb):>6}  {str(bgl):>8}{flag}")
        # (a), (b)
        inv = dict(involutivity(alg))
        print("      node | image of the base:  all roots  a base?  r_k(D)=D  r_k^2=id"
              "   rho_k^2=id")
        for (k, allroots, isbase, preserves, invol) in base_check(alg, Bmat):
            print(f"      {k+1:>4} |{'  yes' if allroots else '   NO':>15}"
                  f"{'  yes' if isbase else '   NO':>9}"
                  f"{'  yes' if preserves else '   NO':>10}"
                  f"{'  yes' if invol else '   NO':>10}"
                  f"{'  yes' if inv[k] else '   NO':>12}"
                  f"      (A_kk = {A[k][k]})")
            if not skip:
                summary.append((name, k + 1, isbase, preserves, is_zero(A[k][k], p),
                                inv[k]))
    print("\n" + "=" * 78)
    print(f"pairs (k,j) tested: {total}")
    print(f"  Lebedev's recursion (Statement 2.1) disagrees with the root system: {p_bad_rec}")
    print(f"  Lebedev's closed formula (3)        disagrees with the root system: {p_bad_leb}")
    print(f"  [BGL] (7.4) = [BGLL] (44)           disagrees with the root system: {p_bad_bgl}")
    nb = sum(1 for s in summary if not s[2])
    npres = sum(1 for s in summary if not s[3])
    print(f"nodes tested: {len(summary)}")
    print(f"  rule (2) fails to produce a system of simple roots: {nb}")
    print(f"  the linear extension r_k does NOT preserve Delta:   {npres}")
    noniso = [s for s in summary if not s[4]]
    iso = [s for s in summary if s[4]]
    print(f"  non-isotropic nodes (A_kk != 0): {len(noniso)}, of them r_k(Delta) = Delta: "
          f"{sum(1 for s in noniso if s[3])}")
    print(f"  isotropic     nodes (A_kk  = 0): {len(iso)}, of them r_k(Delta) = Delta: "
          f"{sum(1 for s in iso if s[3])}")
    print(f"  rho_k o rho_k = id on bases: {sum(1 for s in summary if s[5])}"
          f" of {len(summary)} nodes")
    print("  isotropic nodes where it happens to hold anyway: "
          + ", ".join(f"{s[0]} node {s[1]}" for s in iso if s[3]))
    print("\nwhich branches of formula (3) the sample exercises:")
    for tag in sorted(BRANCH):
        print(f"   {BRANCH[tag]:>4}x  {tag}")
    print("   (a branch with no line above is NOT tested by this sample)")
    print("=" * 78)


if __name__ == "__main__":
    run()
