"""The sl(2)-action on H^1(CP^1, T_q) and its invariants.

Bashkin's classification runs over the seven pairwise non-conjugate copies
s' = <e', h', f'> of sl(2) inside v(CP^1, O_gr)_0 that split the sequence
End E -> v_0 -> v(M), and, for each of them, over the invariants
H^1(CP^1, T_q)^{s'} -- the cocycles that can be lifted, i.e., the
even-homogeneous supermanifolds.  This engine computes those invariants from
scratch: it brackets an honest vector field with an honest Cech cocycle and
asks whether the result is a coboundary.

Only the even degrees q are handled (q = 2, 4 are the ones the classification
needs, by Green's theorem), where every element of T_q is an even vector field
and the bracket carries no sign.

The coordinates on T_q(U_0 cap U_1) are those of bashkin.py:
    ('u', I) at exponent r   <->   x^r xi_I d/dx,
    ('w', J, t) at exponent r <->  x^r xi_J d/dxi_t.
"""

from fractions import Fraction
from itertools import combinations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank, nullspace                              # noqa: E402

import ns                                                          # noqa: E402
import bashkin as B                                                # noqa: E402


# ------------------------------------------------- fields as (A, [B_1..B_m])

def fld(m):
    return (ns.g_zero(), [ns.g_zero() for _ in range(m)])


def fld_add(X, Y):
    return (ns.g_add(X[0], Y[0]),
            [ns.g_add(a, b) for a, b in zip(X[1], Y[1])])


def fld_scal(X, c):
    return (ns.g_scal(X[0], c), [ns.g_scal(a, c) for a in X[1]])


def d_x(m, coef=None):
    """coef * d/dx."""
    A = coef if coef is not None else ns.g_from((), ns.lp((0, 1)))
    return (A, [ns.g_zero() for _ in range(m)])


def xi_d_xi(m, i, j):
    """xi_i d/dxi_j."""
    Bs = [ns.g_zero() for _ in range(m)]
    Bs[j - 1] = ns.g_from((i,), ns.lp((0, 1)))
    return (ns.g_zero(), Bs)


def nabla(m, ks=None):
    """sum_i k_i xi_i d/dxi_i  (k_i = 1 unless given)."""
    ks = ks or [1] * m
    X = fld(m)
    for i in range(1, m + 1):
        X = fld_add(X, fld_scal(xi_d_xi(m, i, i), ks[i - 1]))
    return X


def bracket_even(m, X, Y):
    """[X, Y] for X, Y even vector fields."""
    A = ns.g_add(ns._apply(X[0], X[1], Y[0]),
                 ns.g_scal(ns._apply(Y[0], Y[1], X[0]), -1))
    Bs = []
    for j in range(m):
        Bs.append(ns.g_add(ns._apply(X[0], X[1], Y[1][j]),
                           ns.g_scal(ns._apply(Y[0], Y[1], X[1][j]), -1)))
    return (A, Bs)


# ------------------------------------------- coordinates <-> vector fields

def coc_to_field(m, el):
    """A cocycle dict {(gen, exp): c} as a vector field."""
    X = fld(m)
    for (g, e), c in el.items():
        if g[0] == "u":
            X = fld_add(X, (ns.g_from(g[1], ns.lp((e, c))),
                            [ns.g_zero() for _ in range(m)]))
        else:
            _, J, t = g
            Bs = [ns.g_zero() for _ in range(m)]
            Bs[t - 1] = ns.g_from(J, ns.lp((e, c)))
            X = fld_add(X, (ns.g_zero(), Bs))
    return X


def field_to_coc(m, q, X):
    """The inverse, for a field homogeneous of degree q; raises if it is not."""
    out = {}
    A, Bs = X
    for S, f in A.items():
        if len(S) != q:
            raise ValueError(f"A-part of degree {len(S)}, expected {q}")
        for e, c in f.items():
            out[(("u", tuple(S)), e)] = out.get((("u", tuple(S)), e), 0) + c
    for t in range(1, m + 1):
        for S, f in Bs[t - 1].items():
            if len(S) != q + 1:
                raise ValueError(f"B-part of degree {len(S)}, expected {q+1}")
            for e, c in f.items():
                k = (("w", tuple(S), t), e)
                out[k] = out.get(k, 0) + c
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------- the seven copies of sl2

def s_prime(case, m=5):
    """(e', f') of Bashkin-5's list of seven, in the paper's own numbering."""
    e = d_x(m)
    # f = -x^2 d/dx - x nabla
    f = fld_add(fld_scal(d_x(m, ns.g_from((), ns.lp((2, 1)))), -1),
                fld_scal(_x_times(m, nabla(m)), -1))
    add_e, add_f = fld(m), fld(m)

    def E(i, j, c=1):
        return fld_scal(xi_d_xi(m, i, j), c)

    if case == 1:
        pass
    elif case == 2:
        add_e = E(2, 1)
        add_f = E(1, 2)
    elif case == 3:
        add_e = fld_add(E(2, 1), E(4, 3))
        add_f = fld_add(E(1, 2), E(3, 4))
    elif case == 4:
        add_e = fld_add(E(2, 1), E(3, 2))
        add_f = fld_add(E(1, 2, 2), E(2, 3, 2))
    elif case == 5:
        add_e = fld_add(fld_add(E(2, 1), E(3, 2)), E(5, 4))
        add_f = fld_add(fld_add(E(1, 2, 2), E(2, 3, 2)), E(4, 5))
    elif case == 6:
        add_e = fld_add(fld_add(E(2, 1), E(3, 2)), E(4, 3))
        add_f = fld_add(fld_add(E(1, 2, 3), E(2, 3, 4)), E(3, 4, 3))
    elif case == 7:
        add_e = fld_add(fld_add(fld_add(E(2, 1, 2), E(3, 2, 2)), E(4, 3, 2)),
                        E(5, 4, 2))
        add_f = fld_add(fld_add(fld_add(E(1, 2, 2), E(2, 3, 3)), E(3, 4, 3)),
                        E(4, 5, 2))
    else:
        raise ValueError(case)
    return fld_add(e, add_e), fld_add(f, add_f)


def _x_times(m, X):
    """x * X, coefficient-wise."""
    xp = ns.lp((1, 1))
    return ({S: ns.lp_mul(f, xp) for S, f in X[0].items()},
            [{S: ns.lp_mul(f, xp) for S, f in b.items()} for b in X[1]])


# ------------------------------------------------------- H^1 and the action

class H1:
    """H^1(T_q) as an explicit quotient, with the induced sl(2)-action."""

    def __init__(self, ks, q, window=None):
        self.ks, self.q, self.m = ks, q, len(ks)
        m = self.m
        self.W = window or (2 * max(ks) * (m + 2) + 8)
        self.gens = B.gens(m, q)
        self.cols = {}
        for g in self.gens:
            for e in range(-self.W, self.W + 1):
                self.cols[(g, e)] = len(self.cols)
        bnd = list(B.U0_sections(m, q, ks, self.W)) + \
            list(B.U1_sections(m, q, ks, self.W + 2 * sum(ks) + 4))
        self.Brows = [self._row(el) for el in bnd
                      if all(abs(e) <= self.W for (_, e) in el)]
        self.rB = rank(self.Brows)
        # a basis of H^1: greedily pick generators independent modulo B^1
        self.basis = []
        cur = list(self.Brows)
        r = self.rB
        W2 = self.W - (2 * sum(ks) + 6)
        for g in self.gens:
            for e in range(-W2, 0):
                row = self._row({(g, e): Fraction(1)})
                r2 = rank(cur + [row])
                if r2 > r:
                    cur.append(row)
                    r = r2
                    self.basis.append((g, e))
        self.dim = len(self.basis)

    def _row(self, el):
        row = [Fraction(0)] * len(self.cols)
        for (g, e), c in el.items():
            if (g, e) not in self.cols:
                raise KeyError("window too small")
            row[self.cols[(g, e)]] = Fraction(c)
        return row

    def classes(self, el):
        """Coordinates of [el] in self.basis (exact), or None if not reducible."""
        target = self._row(el)
        rows = list(self.Brows) + [self._row({(g, e): Fraction(1)})
                                   for (g, e) in self.basis]
        # solve rows^T c = target
        from exactlin import solve
        sol = solve([list(col) for col in zip(*rows)], target)
        if sol is None:
            return None
        return sol[len(self.Brows):]

    def matrix_of(self, X):
        """The matrix of ad_X on H^1 in self.basis (columns = images)."""
        cols = []
        for (g, e) in self.basis:
            el = {(g, e): Fraction(1)}
            Y = coc_to_field(self.m, el)
            Z = bracket_even(self.m, X, Y)
            cz = field_to_coc(self.m, self.q, Z)
            cz = {k: v for k, v in cz.items() if abs(k[1]) <= self.W}
            c = self.classes(cz)
            if c is None:
                raise RuntimeError("bracket not reducible: window too small")
            cols.append(c)
        return [[cols[j][i] for j in range(self.dim)] for i in range(self.dim)]

    def invariants(self, X, Y):
        """A basis of ker ad_X cap ker ad_Y on H^1, in self.basis coordinates."""
        MX = self.matrix_of(X)
        MY = self.matrix_of(Y)
        return nullspace(MX + MY)
