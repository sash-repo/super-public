"""Odd parameters over CP^1: the group that classifies, and its orbits.

Over a base Spec C with C_od =/= 0, an EVEN automorphism of the retract may
carry components of ODD degree in the Lambda-grading, paired with odd elements
of C.  Hence the layer H^1(M; T_q) with q odd enters the classification; over
Spec C it is invisible, q being odd.

The group that acts is the one of the graded automorphisms of the split model,

    g = v_0 = H^0(M; T_0)  =  aut(E)  +  sl(2),

aut(E) spanned by x^p xi_j d/dxi_i with 0 <= p <= k_i - k_j (so that the field
is regular in both charts), and sl(2) by

    e = d/dx,   f = -x^2 d/dx - x sum_r k_r xi_r d/dxi_r,   h = [e, f].

This module computes the matrices of g on H^1(M; T_q) for EVERY q, odd ones
included -- the bracket [X, Y] with X even is X(Y) - Y(X) whatever the parity
of Y, which is what sl2inv.bracket_even computes -- and, from them, the weight
decomposition and the dimension of the orbit through a given class.
"""

from fractions import Fraction
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank                                         # noqa: E402

import ns                                                          # noqa: E402
import bashkin as B                                                # noqa: E402
import sl2inv as S                                                 # noqa: E402
import quot                                                        # noqa: E402


# ---------------------------------------------------------------- the algebra g

def sl2(ks):
    """(e, h, f) inside v_0 of the split model."""
    m = len(ks)
    e = S.d_x(m)
    f = S.fld_add(S.fld_scal(S.d_x(m, ns.g_from((), ns.lp((2, 1)))), -1),
                  S.fld_scal(S._x_times(m, S.nabla(m, ks)), -1))
    h = S.bracket_even(m, e, f)
    return e, h, f


def autE(ks):
    """Basis of aut(E): the labelled fields x^p xi_j d/dxi_i."""
    m = len(ks)
    out = []
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            for p in range(0, ks[i - 1] - ks[j - 1] + 1):
                Bs = [ns.g_zero() for _ in range(m)]
                Bs[i - 1] = ns.g_from((j,), ns.lp((p, 1)))
                out.append((f"x^{p} xi_{j} d/dxi_{i}", (ns.g_zero(), Bs)))
    return out


def gzero(ks):
    """Basis of g = v_0, labelled."""
    e, h, f = sl2(ks)
    return [("e", e), ("h", h), ("f", f)] + autE(ks)


def check_v0(ks):
    """dim g against dim H^0(M; T_0), computed by the Cech engine."""
    h0, _ = B.h0_h1(ks, 0)
    return len(gzero(ks)), h0


# ------------------------------------------------------------- the action on H^1

class Layer:
    """H^1(M; T_q) of the split (CP^1, Lambda E) with the action of g."""

    def __init__(self, ks, q, outer=None):
        self.ks, self.q, self.m = list(ks), q, len(ks)
        self.Q = quot.Quot(ks, q, outer=outer)
        self.dim = self.Q.dim
        self.gens = gzero(ks)
        self.mats = {}

    def mat(self, name):
        if name not in self.mats:
            X = dict(self.gens)[name]
            self.mats[name] = self.Q.matrix_of(X)
        return self.mats[name]

    def all_mats(self):
        return {nm: self.mat(nm) for nm, _ in self.gens}

    # --- sl(2) structure ----------------------------------------------------

    def h_weights(self):
        """Eigenvalues of h on H^1 (they are integers), with multiplicity."""
        from exactlin import rref
        Mh = self.mat("h")
        n = self.dim
        out = {}
        # h is semisimple with integer eigenvalues: find them by kernels
        for lam in range(-4 * (max(self.ks) + 3), 4 * (max(self.ks) + 3) + 1):
            A = [[Mh[i][j] - (Fraction(lam) if i == j else 0)
                  for j in range(n)] for i in range(n)]
            d = n - rank(A) if n else 0
            if d:
                out[lam] = d
        return out

    def sl2_decomposition(self):
        """Multiplicities of the irreducibles R_d (dim d+1) from the weights."""
        w = dict(self.h_weights())
        out = {}
        while w:
            top = max(w)
            mult = w[top]
            out[top] = out.get(top, 0) + mult
            for lam in range(-top, top + 1, 2):
                w[lam] = w.get(lam, 0) - mult
                if w[lam] <= 0:
                    w.pop(lam, None)
        return out            # d -> multiplicity of R_d

    def check_sl2(self):
        """[e,f] = h, [h,e] = 2e, [h,f] = -2f on H^1."""
        E, H, F = self.mat("e"), self.mat("h"), self.mat("f")

        def mm(A, Bm):
            n = self.dim
            return [[sum(A[i][k] * Bm[k][j] for k in range(n))
                     for j in range(n)] for i in range(n)]

        def comm(A, Bm):
            n = self.dim
            P, Qm = mm(A, Bm), mm(Bm, A)
            return [[P[i][j] - Qm[i][j] for j in range(n)] for i in range(n)]

        def eq(A, Bm, c=1):
            n = self.dim
            return all(A[i][j] == c * Bm[i][j] for i in range(n)
                       for j in range(n))
        return (eq(comm(E, F), H), eq(comm(H, E), E, 2), eq(comm(H, F), F, -2))

    # --- orbits -------------------------------------------------------------

    def tangent(self, v):
        """The rows X.v for X in the basis of g."""
        rows = []
        for nm, _ in self.gens:
            M = self.mat(nm)
            rows.append([sum(M[i][j] * v[j] for j in range(self.dim))
                         for i in range(self.dim)])
        return rows

    def orbit_dim(self, v):
        """dim of the g-orbit through v in H^1 (the linear span of X.v)."""
        return rank(self.tangent(v))

    def orbit_dim_proj(self, v):
        """dim of the orbit of [v] in P(H^1): the span of X.v and v, less 1."""
        rows = self.tangent(v) + [list(v)]
        return rank(rows) - 1

    def moduli_bound(self, v):
        """An upper bound for the number of moduli, read at the class v."""
        return (self.dim - 1) - self.orbit_dim_proj(v)


class Sum:
    """H^1(T_1) + H^1(T_2) with the diagonal action of g and the two scalings.

    Over Spec Lambda(tau) an isomorphism of families may also rescale tau, which
    acts as the identity on the even layer and by a scalar on the odd one; the
    scalars of aut(E) act on T_q by the q-th power, so the two layers can be
    scaled independently and the relevant group is g + Cee (the extra scaling).
    """

    def __init__(self, ks, qs=(1, 2)):
        self.L = [Layer(ks, q) for q in qs]
        self.dims = [l.dim for l in self.L]
        self.dim = sum(self.dims)
        self.ks = list(ks)

    def mat(self, name):
        out = [[Fraction(0)] * self.dim for _ in range(self.dim)]
        off = 0
        for l in self.L:
            M = l.mat(name)
            for i in range(l.dim):
                for j in range(l.dim):
                    out[off + i][off + j] = M[i][j]
            off += l.dim
        return out

    def extra_scalings(self):
        """The diagonal matrices scaling one layer and fixing the other."""
        out = []
        for t in range(len(self.L)):
            M = [[Fraction(0)] * self.dim for _ in range(self.dim)]
            off = sum(self.dims[:t])
            for i in range(self.dims[t]):
                M[off + i][off + i] = Fraction(1)
            out.append(M)
        return out

    def orbit_dim_proj(self, v, with_extra=True):
        mats = [self.mat(nm) for nm, _ in self.L[0].gens]
        if with_extra:
            mats += self.extra_scalings()
        rows = []
        for M in mats:
            rows.append([sum(M[i][j] * v[j] for j in range(self.dim))
                         for i in range(self.dim)])
        return rank(rows + [list(v)]) - 1
