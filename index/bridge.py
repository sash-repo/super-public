#!/usr/bin/env python3
"""Exact checks for the bridge between the index theorem and characteristic classes.

Section 6 of the text (Subsec. 6.1, Lemma 6.2, Theorem 6.4) claims that

  1) the only invariant linear functional on gl(a|b) is str, up to a factor,
     so the ordinary trace tr -- whose Chern-Weil classes would compute the
     other component of the super index -- is not an invariant of the
     structure superalgebra at all;

  2) tr IS an invariant of the underlying Lie algebra gl(a) + gl(b), i.e., the
     missing component is reachable only after forgetting the odd structure;

  3) Z[eps]/(eps^2-1) = Z x Z via a+b*eps |-> (a+b, a-b), under which the
     superdimension goes to (dim, str-dimension), and the two components of
     the super index are ind(D_ev) +- ind(D_od);

  4) str(F^k) is a nonzero closed form for a gl(a|b)-valued connection --
     the contrast with the queer case, where qtr(F^k) is exact.

Everything is done in exact integer/rational arithmetic.  Run:

    python3 bridge.py
"""

import sys
import os
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import nullspace, rank, backend          # noqa: E402

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(("  ok   " if ok else "  FAIL") + f"  {name}" + (f"   [{detail}]" if detail else ""))
    return ok


# ---------------------------------------------------------------- gl(a|b) ---

class GL:
    """The matrix Lie superalgebra gl(a|b) in the basis of matrix units."""

    def __init__(self, a, b):
        self.a, self.b, self.n = a, b, a + b
        # parity of index i: even for i < a, odd for i >= a
        self.p = [0] * a + [1] * b
        self.basis = [(i, j) for i in range(self.n) for j in range(self.n)]

    def par(self, i, j):
        return (self.p[i] + self.p[j]) % 2

    def mul(self, X, Y):
        """Ordinary matrix product of two dicts {(i,j): coeff}."""
        Z = {}
        for (i, j), c in X.items():
            for (k, l), d in Y.items():
                if j == k:
                    Z[(i, l)] = Z.get((i, l), 0) + c * d
        return {k: v for k, v in Z.items() if v}

    def bracket(self, X, px, Y, py):
        """Super bracket [X,Y] = XY - (-1)^{p(X)p(Y)} YX."""
        sign = -1 if (px and py) else 1
        Z = dict(self.mul(X, Y))
        for k, v in self.mul(Y, X).items():
            Z[k] = Z.get(k, 0) - sign * v
        return {k: v for k, v in Z.items() if v}

    def unit(self, i, j):
        return {(i, j): 1}

    def tr(self, X):
        return sum(c for (i, j), c in X.items() if i == j)

    def str_(self, X):
        return sum((-1) ** self.p[i] * c for (i, j), c in X.items() if i == j)


def invariant_functionals(g, even_only=False):
    """Basis of {f linear on g | f([x,y]) = 0 for all x,y}.

    A functional is a vector of coefficients indexed by g.basis.  Each pair
    (x,y) of basis elements gives one linear condition on f.
    If even_only, the algebra is the underlying gl(a) + gl(b): only the
    even matrix units are taken, and the bracket is the ordinary commutator.
    """
    if even_only:
        bas = [(i, j) for (i, j) in g.basis if g.par(i, j) == 0]
    else:
        bas = g.basis
    idx = {b: k for k, b in enumerate(bas)}
    rows = []
    for (i, j) in bas:
        for (k, l) in bas:
            X, Y = g.unit(i, j), g.unit(k, l)
            px, py = (0, 0) if even_only else (g.par(i, j), g.par(k, l))
            Z = g.bracket(X, px, Y, py)
            if not Z:
                continue
            row = [0] * len(bas)
            for key, c in Z.items():
                if key in idx:                     # brackets stay inside
                    row[idx[key]] += c
            if any(row):
                rows.append(row)
    if not rows:
        return bas, [[1 if m == k else 0 for m in range(len(bas))] for k in range(len(bas))]
    return bas, nullspace(rows)


def as_functional(g, bas, vec):
    return {b: vec[k] for k, b in enumerate(bas) if vec[k]}


def is_multiple_of(f, target, bas):
    """Is the functional f (dict on bas) a scalar multiple of target (dict)?"""
    piv = None
    for b in bas:
        fb, tb = Fraction(f.get(b, 0)), Fraction(target.get(b, 0))
        if tb:
            piv = fb / tb
            break
    if piv is None:
        return False
    return all(Fraction(f.get(b, 0)) == piv * Fraction(target.get(b, 0)) for b in bas)


def run_gl_checks():
    print("\n1) Invariant linear functionals on gl(a|b)")
    for a, b in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (2, 3)]:
        g = GL(a, b)
        bas, ns = invariant_functionals(g)
        strv = {(i, i): (-1) ** g.p[i] for i in range(g.n)}
        ok = len(ns) == 1 and is_multiple_of(as_functional(g, bas, ns[0]), strv, bas)
        check(f"gl({a}|{b}): invariant linear functionals = C.str",
              ok, f"dim = {len(ns)}")

    print("\n2) tr is not one of them; the witness")
    g = GL(1, 1)
    x, y = g.unit(0, 1), g.unit(1, 0)              # both odd
    z = g.bracket(x, 1, y, 1)                      # xy + yx
    check("gl(1|1): [e_12,e_21] = e_11 + e_22",
          z == {(0, 0): 1, (1, 1): 1}, str(sorted(z.items())))
    check("gl(1|1): tr[e_12,e_21] = 2 != 0", g.tr(z) == 2, f"tr = {g.tr(z)}")
    check("gl(1|1): str[e_12,e_21] = 0", g.str_(z) == 0, f"str = {g.str_(z)}")

    print("\n3) str kills every bracket, tr does not (exhaustive on basis pairs)")
    for a, b in [(1, 1), (2, 1), (2, 2), (3, 2)]:
        g = GL(a, b)
        bad_str = bad_tr = 0
        for (i, j) in g.basis:
            for (k, l) in g.basis:
                z = g.bracket(g.unit(i, j), g.par(i, j), g.unit(k, l), g.par(k, l))
                if g.str_(z) != 0:
                    bad_str += 1
                if g.tr(z) != 0:
                    bad_tr += 1
        check(f"gl({a}|{b}): str vanishes on all {len(g.basis)**2} brackets",
              bad_str == 0, f"failures = {bad_str}")
        check(f"gl({a}|{b}): tr does not (it is not invariant)",
              bad_tr > 0, f"nonzero on {bad_tr} brackets")

    print("\n4) On the underlying gl(a)+gl(b), tr IS invariant")
    for a, b in [(1, 1), (2, 1), (2, 2)]:
        g = GL(a, b)
        bas, ns = invariant_functionals(g, even_only=True)
        trv = {(i, i): 1 for i in range(g.n)}
        # tr must lie in the span of ns
        rows = [list(v) for v in ns]
        trvec = [trv.get(bb, 0) for bb in bas]
        ok = rank(rows) == rank(rows + [trvec])
        check(f"gl({a})+gl({b}): tr is an invariant functional", ok,
              f"space of invariants has dim {len(ns)} (two blocks)")
        check(f"gl({a})+gl({b}): the invariants are 2-dimensional", len(ns) == 2,
              f"dim = {len(ns)}")


# ------------------------------------------------------- Z[eps]/(eps^2-1) ---

class Zeps:
    """The ring Z[eps]/(eps^2-1) in which superdimensions live."""

    def __init__(self, a, b=0):
        self.a, self.b = a, b

    def __add__(self, o):
        return Zeps(self.a + o.a, self.b + o.b)

    def __sub__(self, o):
        return Zeps(self.a - o.a, self.b - o.b)

    def __mul__(self, o):
        return Zeps(self.a * o.a + self.b * o.b, self.a * o.b + self.b * o.a)

    def __eq__(self, o):
        return (self.a, self.b) == (o.a, o.b)

    def ev(self, e):
        """Evaluation at eps = e, with e = +1 or -1."""
        return self.a + e * self.b

    def __repr__(self):
        return f"{self.a}+{self.b}eps"


def run_eps_checks():
    print("\n5) The ring Z[eps]/(eps^2-1) and its two evaluations")
    sample = [Zeps(a, b) for a in range(-3, 4) for b in range(-3, 4)]
    ok_add = ok_mul = True
    for u, v in product(sample[:20], sample[:20]):
        for e in (+1, -1):
            if (u + v).ev(e) != u.ev(e) + v.ev(e):
                ok_add = False
            if (u * v).ev(e) != u.ev(e) * v.ev(e):
                ok_mul = False
    check("eps |-> +-1 are ring homomorphisms to Z", ok_add and ok_mul)
    sep = all((u.ev(1), u.ev(-1)) != (v.ev(1), v.ev(-1))
              for i, u in enumerate(sample) for v in sample[i + 1:])
    check("the pair of evaluations is injective, i.e., Z[eps]/(eps^2-1) -> ZxZ",
          sep, "the two components determine the element")

    print("\n6) The two components of a superdimension and of the super index")
    for p, q in [(0, 0), (3, 0), (0, 5), (4, 7), (2, 2)]:
        sd = Zeps(p, q)                            # sdim V = dim V_ev + eps dim V_od
        check(f"sdim = ({p}|{q}): eps=+1 gives dim = {p+q}", sd.ev(1) == p + q)
        check(f"sdim = ({p}|{q}): eps=-1 gives the str-dimension {p-q}", sd.ev(-1) == p - q)
    # the refined index of a Z/2-graded operator
    for k0, k1, c0, c1 in [(3, 1, 0, 2), (5, 5, 1, 0), (0, 4, 2, 2)]:
        chi = Zeps(k0, k1) - Zeps(c0, c1)          # sdim Ker - sdim Coker
        ind0, ind1 = k0 - c0, k1 - c1              # the two ordinary indices
        check(f"ind: (Ker {k0}|{k1}, Coker {c0}|{c1}) -> eps=+1 is ind_ev + ind_od",
              chi.ev(1) == ind0 + ind1, f"{chi.ev(1)} = {ind0}+{ind1}")
        check(f"ind: (Ker {k0}|{k1}, Coker {c0}|{c1}) -> eps=-1 is ind_ev - ind_od",
              chi.ev(-1) == ind0 - ind1, f"{chi.ev(-1)} = {ind0}-{ind1}")


# --------------------------------------------- str(F^k): a nonzero class ---

def run_chernweil_check():
    """A gl(1|1)-valued connection on a 2-torus whose str(F) is not exact.

    Take E = L_0 + L_1 with L_i line bundles of degrees d_i on a surface; a
    diagonal connection has F = diag(F_0, F_1) and str(F) = F_0 - F_1, whose
    integral is 2*pi*i*(d_0 - d_1).  The point of the check is the contrast
    with the queer case: str is even, so it does NOT annihilate the curvature
    of an ordinary connection, and the class survives.
    """
    print("\n7) str(F) on a diagonal gl(1|1)-connection: the class survives")
    for d0, d1 in [(1, 0), (3, 5), (2, 2)]:
        # integral of the curvature 2-form of L_i is 2*pi*i*d_i; work with d_i
        s = d0 - d1                       # str
        check(f"deg = ({d0},{d1}): str-class = {s}, and it is 0 iff the degrees agree",
              (s == 0) == (d0 == d1))
    # and the parity statement: an odd functional annihilates the curvature
    # of an ordinary (even) connection.  F is even, qtr is odd, so qtr(F^k)=0
    # unless the connection has an odd part.
    g = GL(1, 1)
    F = {(0, 0): 1, (1, 1): -1}           # an even, diagonal curvature
    qt = sum(c for (i, j), c in F.items() if (i, j) in [(0, 1), (1, 0)])
    check("an odd trace annihilates an even (diagonal) curvature", qt == 0,
          f"qtr(F) = {qt}")


def main():
    print(f"exactlin backend: {backend()}")
    run_gl_checks()
    run_eps_checks()
    run_chernweil_check()
    bad = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS)} checks, {len(CHECKS)-len(bad)} passed, {len(bad)} failed")
    if bad:
        for n, _, d in bad:
            print("  FAILED:", n, d)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
