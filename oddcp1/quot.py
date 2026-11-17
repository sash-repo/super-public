"""A sparse incremental echelon basis, and H^1(T_q) as an explicit quotient
with the induced sl(2)-action.  Same mathematics as sl2inv.H1, but the linear
algebra is done once, incrementally, instead of by repeated rank calls.
"""

from fractions import Fraction
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import solve, nullspace                             # noqa: E402

import ns                                                          # noqa: E402
import bashkin as B                                                # noqa: E402
import sl2inv as S                                                 # noqa: E402


class Echelon:
    """Row space of sparse vectors (dicts key -> Fraction), reduced form."""

    def __init__(self):
        self.piv = {}                      # pivot key -> reduced row

    def reduce(self, v):
        v = {k: Fraction(c) for k, c in v.items() if c}
        while True:
            hit = None
            for k in v:
                if k in self.piv:
                    hit = k
                    break
            if hit is None:
                return v
            c = v[hit]
            row = self.piv[hit]
            for k, a in row.items():
                v[k] = v.get(k, Fraction(0)) - c * a
                if not v[k]:
                    del v[k]

    def add(self, v):
        """Reduce and, if non-zero, install as a new pivot. True if new."""
        r = self.reduce(v)
        if not r:
            return False
        k = sorted(r)[0]
        c = r[k]
        self.piv[k] = {kk: a / c for kk, a in r.items()}
        return True


class Quot:
    """H^1(T_q) for the split (CP^1, Lambda E), E = + L_{-k_j}."""

    def __init__(self, ks, q, outer=None):
        self.ks, self.q, self.m = ks, q, len(ks)
        m = self.m
        self.N = outer or (4 * max(ks) * (m + 2) + 12)
        self.inner = self.N - (2 * sum(ks) + 8)
        # --- the coboundaries
        self.E = Echelon()
        for el in B.U0_sections(m, q, ks, self.N):
            if all(abs(e) <= self.N for (_, e) in el):
                self.E.add(el)
        for el in B.U1_sections(m, q, ks, self.N + 2 * sum(ks) + 8):
            if el and all(abs(e) <= self.N for (_, e) in el):
                self.E.add(el)
        # --- a basis of the quotient, from the monomials of the inner window
        self.basis = []
        self.red = []                     # their reduced forms
        E2 = Echelon()
        E2.piv = dict(self.E.piv)
        for g in B.gens(m, q):
            for e in range(-self.inner, 1):
                mon = {(g, e): Fraction(1)}
                r = self.E.reduce(mon)
                if r and E2.add(mon):
                    self.basis.append((g, e))
                    self.red.append(r)
        self.dim = len(self.basis)
        self._keys = sorted({k for r in self.red for k in r})
        self._mat = [[r.get(k, Fraction(0)) for r in self.red]
                     for k in self._keys]

    # ------------------------------------------------------------------ classes

    def coords(self, el):
        """Coordinates of [el] in self.basis; raises if el is out of range."""
        r = self.E.reduce(el)
        if not r:
            return [Fraction(0)] * self.dim
        if any(k not in self._keys for k in r):
            raise RuntimeError("class outside the span of the chosen basis "
                               "(window too small)")
        b = [r.get(k, Fraction(0)) for k in self._keys]
        sol = solve(self._mat, b)
        if sol is None:
            raise RuntimeError("not in the span of the chosen basis")
        return sol

    # ------------------------------------------------------------- the action

    def matrix_of(self, X):
        cols = []
        for (g, e) in self.basis:
            Y = S.coc_to_field(self.m, {(g, e): Fraction(1)})
            Z = S.bracket_even(self.m, X, Y)
            cols.append(self.coords(S.field_to_coc(self.m, self.q, Z)))
        return [[cols[j][i] for j in range(self.dim)] for i in range(self.dim)]

    def invariants(self, *Xs):
        rows = []
        for X in Xs:
            rows += self.matrix_of(X)
        return nullspace(rows) if rows else []

    def as_cocycle(self, vec):
        """A class given in self.basis coordinates, as a cocycle dict."""
        out = {}
        for c, (g, e) in zip(vec, self.basis):
            if c:
                out[(g, e)] = out.get((g, e), Fraction(0)) + c
        return out
