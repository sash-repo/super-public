"""Exact combinatorics of Donin's spaces S(A,B), T_B(A), and the number nu(A,B).

Everything here is built from the definitions printed in the paper (Sections 2-4 of
the text), with no shortcut through the Littlewood-Richardson rule, so that the
rule comes out as a *check* rather than as an input.

Conventions (the paper's own):
  * a cell is a pair (i, j): i = row, counted downwards from 1; j = column, from 1.
  * a skew diagram is any finite set of cells that is the difference of two Young
    diagrams; `Diagram` accepts either the cell set or the pair (lam, mu).
  * a diagram mapping A -> B is a bijection of cells.
  * Aut(A) is the FULL symmetric group on the cells of A (the paper, Sect. 2:
    "it is the permutation group of the cells of A"); R_A and C_A are the row and
    the column stabilizers.
  * the semi-standard A-tableau puts the number of its row into each cell.
  * an A-tableau of type B is a filling of A whose content is (|B|_1,...,|B|_r).
"""

from fractions import Fraction
from itertools import permutations, product
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank  # noqa: E402


# --------------------------------------------------------------------------- #
#  diagrams
# --------------------------------------------------------------------------- #

class Diagram:
    def __init__(self, cells, name=None):
        self.cells = tuple(sorted(set(cells)))          # standard scan order
        self.name = name
        self.index = {c: k for k, c in enumerate(self.cells)}

    # --- constructors ------------------------------------------------------ #
    @staticmethod
    def skew(lam, mu=(), name=None):
        """The skew diagram lam/mu."""
        mu = tuple(mu) + (0,) * (len(lam) - len(mu))
        cells = [(i + 1, j + 1) for i, li in enumerate(lam) for j in range(mu[i], li)]
        return Diagram(cells, name)

    @staticmethod
    def row(q, name=None):
        return Diagram([(1, j + 1) for j in range(q)], name)

    @staticmethod
    def column(q, name=None):
        return Diagram([(i + 1, 1) for i in range(q)], name)

    # --- basic data -------------------------------------------------------- #
    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        return self.name or "Diagram%s" % (self.cells,)

    @property
    def rows(self):
        """rows as lists of cells, indexed by the row number (only nonempty ones)."""
        out = {}
        for c in self.cells:
            out.setdefault(c[0], []).append(c)
        return [out[i] for i in sorted(out)]

    @property
    def columns(self):
        out = {}
        for c in self.cells:
            out.setdefault(c[1], []).append(c)
        return [sorted(out[j]) for j in sorted(out)]

    @property
    def row_sizes(self):
        return tuple(len(r) for r in self.rows)

    @property
    def row_number(self):
        """cell -> its index (1-based) among the nonempty rows."""
        order = {i: k + 1 for k, i in enumerate(sorted({c[0] for c in self.cells}))}
        return {c: order[c[0]] for c in self.cells}

    def transpose(self):
        return Diagram([(j, i) for (i, j) in self.cells],
                       (self.name + "'") if self.name else None)

    def is_young(self):
        for (i, j) in self.cells:
            if j > 1 and (i, j - 1) not in self.index:
                return False
            if i > 1 and (i - 1, j) not in self.index:
                return False
        return True

    # --- the three groups -------------------------------------------------- #
    def _stabilizer(self, blocks):
        """all permutations of the cells preserving each block, with their signs."""
        out = []
        for choice in product(*[permutations(b) for b in blocks]):
            perm = {}
            for b, img in zip(blocks, choice):
                for c, d in zip(b, img):
                    perm[c] = d
            out.append((perm, _sign(perm, self.cells)))
        return out

    def row_stabilizer(self):
        return self._stabilizer(self.rows)

    def column_stabilizer(self):
        return self._stabilizer(self.columns)


def _sign(perm, cells):
    seen, s = set(), 1
    for c in cells:
        if c in seen:
            continue
        length, d = 0, c
        while d not in seen:
            seen.add(d)
            d = perm[d]
            length += 1
        if length % 2 == 0:
            s = -s
    return s


# --------------------------------------------------------------------------- #
#  tableaux and tabloids
# --------------------------------------------------------------------------- #

def semistandard(D):
    """the semi-standard D-tableau: each cell carries the number of its row."""
    return dict(D.row_number)


def tabloid(D, tab):
    """the row class of the D-tableau `tab`: the multiset of entries in each row."""
    return tuple(tuple(sorted(tab[c] for c in r)) for r in D.rows)


def tableaux_of_type(A, B):
    """all A-tableaux of type B, i.e. all fillings of A of content (|B|_1,...,|B|_r)."""
    content = B.row_sizes
    out, cells = [], A.cells

    def rec(k, left, acc):
        if k == len(cells):
            out.append(dict(acc))
            return
        for v, n in enumerate(left):
            if n:
                acc[cells[k]] = v + 1
                left2 = list(left)
                left2[v] -= 1
                rec(k + 1, left2, acc)
                del acc[cells[k]]

    rec(0, list(content), {})
    return out


def is_semi_ordered(A, tab):
    """weakly increasing rightwards along rows, strictly increasing downwards."""
    for (i, j) in A.cells:
        if (i, j + 1) in A.index and tab[(i, j)] > tab[(i, j + 1)]:
            return False
        if (i + 1, j) in A.index and tab[(i, j)] >= tab[(i + 1, j)]:
            return False
    return True


def is_quasi_ordered(A, tab):
    """weakly increasing both rightwards and downwards."""
    for (i, j) in A.cells:
        if (i, j + 1) in A.index and tab[(i, j)] > tab[(i, j + 1)]:
            return False
        if (i + 1, j) in A.index and tab[(i, j)] > tab[(i + 1, j)]:
            return False
    return True


# --------------------------------------------------------------------------- #
#  the linear algebra: T_B(A), S_B(A), S(A,B)
# --------------------------------------------------------------------------- #

def _mapping_tabloid(A, B, g):
    """the A-tabloid of type B carried by a diagram mapping g: B -> A."""
    rb = B.row_number
    tab = {g[c]: rb[c] for c in B.cells}
    return tabloid(A, tab)


def hom(A, B):
    """all diagram mappings B -> A (as dicts on the cells of B)."""
    assert len(A) == len(B)
    return [dict(zip(B.cells, img)) for img in permutations(A.cells)]


def _span_dim(vectors):
    """dimension of the span of a list of dicts basis-element -> coefficient."""
    keys = sorted({k for v in vectors for k in v})
    if not keys:
        return 0
    rows = [[Fraction(v.get(k, 0)) for k in keys] for v in vectors]
    return rank(rows)


def _add(acc, key, coeff):
    acc[key] = acc.get(key, 0) + coeff
    if acc[key] == 0:
        del acc[key]


def e_of_mapping(A, B, f, CA=None):
    """e_f = sum_{phi in C_A} sgn(phi) [phi f]   (formula (3.2))."""
    CA = CA if CA is not None else A.column_stabilizer()
    acc = {}
    for phi, s in CA:
        g = {c: phi[f[c]] for c in B.cells}
        _add(acc, _mapping_tabloid(A, B, g), s)
    return acc


def rho_of_mapping(A, B, f, CA=None, CB=None):
    """rho_f = sum_{phi in C_A, psi in C_B} sgn(phi)sgn(psi) [phi f psi^{-1}]  ((3.5))."""
    CA = CA if CA is not None else A.column_stabilizer()
    CB = CB if CB is not None else B.column_stabilizer()
    acc = {}
    for phi, sa in CA:
        for psi, sb in CB:
            inv = {v: k for k, v in psi.items()}
            g = {c: phi[f[inv[c]]] for c in B.cells}
            _add(acc, _mapping_tabloid(A, B, g), sa * sb)
    return acc


def dim_T(A, B):
    return len({_mapping_tabloid(A, B, f) for f in hom(A, B)})


def dim_S_B_of_A(A, B):
    """dim S_B(A) subset T_B(A)."""
    CA = A.column_stabilizer()
    return _span_dim([e_of_mapping(A, B, f, CA) for f in hom(A, B)])


def dim_S_pair(A, B):
    """dim S(A,B) subset T_B(A)."""
    CA, CB = A.column_stabilizer(), B.column_stabilizer()
    return _span_dim([rho_of_mapping(A, B, f, CA, CB) for f in hom(A, B)])


def _transpose_tabloid(A, B, t):
    """the image of a tabloid of T_B(A) under T_B(A) -> T_A(B), [f] -> [f^{-1}].

    A tabloid is the contingency matrix M[i][j] = #(cells of row j of B sent to
    row i of A); inversion transposes it.  We return it in the shape produced by
    `_mapping_tabloid(B, A, .)`."""
    nb = len(B.rows)
    mat = [[0] * nb for _ in t]
    for i, rowmultiset in enumerate(t):
        for v in rowmultiset:
            mat[i][v - 1] += 1
    return tuple(tuple(sorted(sum(([i + 1] * mat[i][j] for i in range(len(t))), [])))
                 for j in range(nb))


def dim_intersection_S(A, B):
    """dim (S_B(A) cap S_A^{-1}(B)) inside T_B(A)  -- the right side of Theorem 3.4."""
    CA, CB = A.column_stabilizer(), B.column_stabilizer()
    # S_B(A) inside T_B(A)
    U = [e_of_mapping(A, B, f, CA) for f in hom(A, B)]
    # S_A(B) inside T_A(B), pulled back to T_B(A) by transposing the tabloids
    W = []
    for g in hom(B, A):
        v = e_of_mapping(B, A, g, CB)
        W.append({_transpose_tabloid(B, A, k): c for k, c in v.items()})
    return _dim_meet(U, W)


def _dim_meet(U, W):
    keys = sorted({k for v in U + W for k in v})
    if not keys:
        return 0
    du, dw = _span_dim(U), _span_dim(W)
    dsum = _span_dim(U + W)
    return du + dw - dsum


# --------------------------------------------------------------------------- #
#  nu(A,B) by the combinatorial definition of Theorem 3.4
# --------------------------------------------------------------------------- #

def has_cycles_for(A, B, tab):
    """Does the A-tableau `tab` of type B contain cycles corresponding to B?

    I.e. is there a diagram mapping f: B -> A carrying the semi-standard
    B-tableau onto `tab` and mapping every column of B of >1 cell onto a cycle:
    cells in pairwise distinct rows of A whose order down the column is the order
    down the rows of A?
    """
    rb = B.row_number
    # cells of A available for each value
    pool = {}
    for c in A.cells:
        pool.setdefault(tab[c], []).append(c)
    cols = [sorted(col) for col in B.columns]
    # order the columns so that the long ones come first (pruning)
    cols.sort(key=len, reverse=True)

    used = set()

    def place(ci, k):
        if ci == len(cols):
            return True
        col = cols[ci]
        if k == len(col):
            return place(ci + 1, 0)
        cell = col[k]
        prev_row = None
        if k and len(col) > 1:
            prev_row = _placed[col[k - 1]][0]
        for a in pool[rb[cell]]:
            if a in used:
                continue
            if len(col) > 1 and prev_row is not None and a[0] <= prev_row:
                continue
            used.add(a)
            _placed[cell] = a
            if place(ci, k + 1):
                return True
            used.discard(a)
            del _placed[cell]
        return False

    _placed = {}
    return place(0, 0)


def nu(A, B):
    """the number of semi-ordered A-tableaux of type B with cycles corresponding to B."""
    n = 0
    for tab in tableaux_of_type(A, B):
        if is_semi_ordered(A, tab) and has_cycles_for(A, B, tab):
            n += 1
    return n


def semi_ordered_of_type(A, B):
    return [t for t in tableaux_of_type(A, B) if is_semi_ordered(A, t)]


def show(A, tab=None):
    """a printable picture of a diagram, or of a filling of it."""
    if not A.cells:
        return "(empty)"
    imin = min(c[0] for c in A.cells); imax = max(c[0] for c in A.cells)
    jmin = min(c[1] for c in A.cells); jmax = max(c[1] for c in A.cells)
    out = []
    for i in range(imin, imax + 1):
        line = ""
        for j in range(jmin, jmax + 1):
            if (i, j) in A.index:
                line += str(tab[(i, j)]) if tab else "."
            else:
                line += " "
        out.append(line)
    return "\n".join(out)
