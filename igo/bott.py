#!/usr/bin/env python3
"""Bott's theorem on compact Hermitian symmetric spaces X = G/P.

Exact integer arithmetic throughout (Fractions only in the Weyl dimension
formula).  Everything is derived from the Cartan matrix: the root system, the
symmetrizer, rho, the Weyl group action.  No tables of roots are typed in.

Conventions (calibrated by the tests at the bottom, which must all pass):

  * a weight mu is given in the basis of fundamental weights:
    mu[i] = <mu, alpha_i^vee>;
  * A[i][j] = <alpha_j, alpha_i^vee>  (rows indexed by the coroot);
  * a P-module V with l-highest weight lambda gives the homogeneous bundle
    E_lambda on X = G/P; Bott: if lambda+rho is singular, all cohomology
    vanishes; else H^{l(w)}(X; E_lambda) = V_{w(lambda+rho)-rho} and the other
    H^i vanish.

  With this convention Theta_X = E_theta (theta = the highest root) and
  H^0(X; Theta_X) = g, which is test 1.
"""

from fractions import Fraction
from itertools import product


# ---------------------------------------------------------------- Cartan data

def cartan(typ, rank):
    """A[i][j] = <alpha_j, alpha_i^vee>, Bourbaki numbering."""
    n = rank
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
    if typ == "A":
        for i in range(n - 1):
            A[i][i + 1] = A[i + 1][i] = -1
    elif typ in ("B", "C"):
        for i in range(n - 1):
            A[i][i + 1] = A[i + 1][i] = -1
        if typ == "B":            # alpha_n short: <alpha_{n-1},alpha_n^vee>=-2
            A[n - 1][n - 2] = -2
            A[n - 2][n - 1] = -1
        else:                     # C: alpha_n long
            A[n - 1][n - 2] = -1
            A[n - 2][n - 1] = -2
    elif typ == "D":
        for i in range(n - 2):
            A[i][i + 1] = A[i + 1][i] = -1
        A[n - 3][n - 1] = A[n - 1][n - 3] = -1
    elif typ == "E":
        # Bourbaki E6/E7/E8: chain 1-3-4-5-6(-7(-8)) with 2 attached to 4.
        edges = [(1, 3), (3, 4), (4, 5), (2, 4)]
        if n >= 6:
            edges.append((5, 6))
        if n >= 7:
            edges.append((6, 7))
        if n >= 8:
            edges.append((7, 8))
        for a, b in edges:
            A[a - 1][b - 1] = A[b - 1][a - 1] = -1
    else:
        raise ValueError(typ)
    return A


def symmetrizer(A):
    """d_i > 0 with d_i A_ij = d_j A_ji; then (alpha_i,alpha_j) := d_i A_ij."""
    n = len(A)
    d = [None] * n
    d[0] = Fraction(1)
    todo = [0]
    while todo:
        i = todo.pop()
        for j in range(n):
            if A[i][j] and d[j] is None:
                # d_i A_ij = d_j A_ji
                d[j] = d[i] * Fraction(A[i][j], A[j][i])
                todo.append(j)
    assert all(x is not None and x > 0 for x in d), d
    for i in range(n):
        for j in range(n):
            assert d[i] * A[i][j] == d[j] * A[j][i]
    return d


def positive_roots(A):
    """All positive roots as coefficient vectors over the simple roots."""
    n = len(A)
    simple = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
    roots = set(simple)
    frontier = set(simple)
    while frontier:
        new = set()
        for b in frontier:
            for i in range(n):
                c = sum(b[j] * A[i][j] for j in range(n))   # <b, alpha_i^vee>
                r = list(b)
                r[i] -= c
                r = tuple(r)
                if r not in roots and r not in new:
                    new.add(r)
        roots |= new
        frontier = new
    pos = [r for r in roots if all(c >= 0 for c in r) and any(r)]
    return sorted(pos, key=lambda r: (sum(r), r))


class Root:
    """A root system, everything computed from the Cartan matrix."""

    def __init__(self, typ, rank):
        self.typ, self.rank = typ, rank
        self.A = cartan(typ, rank)
        self.d = symmetrizer(self.A)
        self.pos = positive_roots(self.A)
        self.rho = [1] * rank                      # in the omega-basis
        # the highest root, in omega-coordinates
        hi = max(self.pos, key=lambda r: sum(r))
        self.theta_coeff = hi
        self.theta = self.to_omega(hi)

    def to_omega(self, coeff):
        """root given by coefficients over simple roots -> omega-coordinates."""
        return [sum(coeff[j] * self.A[i][j] for j in range(self.rank))
                for i in range(self.rank)]

    def pair(self, mu, coeff):
        """(mu, alpha) for mu in omega-coords, alpha given by coefficients."""
        return sum(Fraction(coeff[j]) * mu[j] * self.d[j]
                   for j in range(self.rank))

    def reflect(self, mu, i):
        a = self.to_omega(tuple(1 if j == i else 0 for j in range(self.rank)))
        return [mu[j] - mu[i] * a[j] for j in range(self.rank)]

    def dominate(self, mu):
        """Greedy: (w(mu), l(w)) with w(mu) dominant, or None if singular."""
        mu = list(mu)
        # regular = off every wall, i.e., (mu, alpha) non-zero for every root,
        # not merely for the simple ones
        if any(self.pair(mu, a) == 0 for a in self.pos):
            return None
        steps = 0
        while True:
            for i in range(self.rank):
                if mu[i] < 0:
                    mu = self.reflect(mu, i)
                    steps += 1
                    break
            else:
                return mu, steps
            assert steps <= 10 ** 6

    def dim(self, lam):
        """Weyl dimension formula for the dominant weight lam."""
        num = den = Fraction(1)
        lr = [lam[i] + 1 for i in range(self.rank)]
        for a in self.pos:
            num *= self.pair(lr, a)
            den *= self.pair(self.rho, a)
        r = num / den
        assert r.denominator == 1, r
        return int(r)

    def cohomology(self, lam):
        """H^*(X; E_lam) -> (degree, highest weight, dim) or None if zero."""
        mu = [lam[i] + 1 for i in range(self.rank)]     # lam + rho
        got = self.dominate(mu)
        if got is None:
            return None
        w, ell = got
        hw = [w[i] - 1 for i in range(self.rank)]
        return ell, hw, self.dim(hw)


# ------------------------------------------------------- the CHSS themselves

def omega(rank, s, k=1):
    return [k if i == s - 1 else 0 for i in range(rank)]


def addw(*ws):
    return [sum(x) for x in zip(*ws)]


class CHSS:
    """X = G/P for a cominuscule node s; P = stabilizer, d = the character."""

    def __init__(self, name, typ, rank, node, comment=""):
        self.name, self.node, self.comment = name, node, comment
        self.R = Root(typ, rank)
        self.typ, self.rank = typ, rank
        # dim X = number of positive roots with coefficient 1 at the node
        self.dimX = sum(1 for r in self.R.pos if r[node - 1] == 1)
        assert all(r[node - 1] <= 1 for r in self.R.pos), "not cominuscule"
        self.dimg = len(self.R.pos) * 2 + rank

    def d(self, k):
        return omega(self.rank, self.node, k)

    def theta(self):
        return list(self.R.theta)

    def min_irrep(self):
        """min dim of a non-trivial irreducible g-module."""
        best = None
        for lam in fundamental_and_small(self.R):
            dd = self.R.dim(lam)
            if best is None or dd < best[0]:
                best = (dd, lam)
        return best

    def tangent_twist(self, j):
        """H^*(X; Theta_X (x) E_{d^j})."""
        return self.R.cohomology(addw(self.theta(), self.d(j)))

    def line(self, j):
        """H^*(X; E_{d^j})."""
        return self.R.cohomology(self.d(j))


def fundamental_and_small(R):
    """candidates for the smallest non-trivial irreducible module."""
    out = []
    for i in range(R.rank):
        lam = [0] * R.rank
        lam[i] = 1
        out.append(lam)
    return out


def chss_list():
    L = []
    for n in range(2, 9):                      # Gr_s^n, 1 <= s <= n-1
        for s in range(1, n):
            L.append(CHSS("Gr_%d^%d" % (s, n), "A", n - 1, s))
    for k in range(2, 8):                      # Q_{2k-1} = B_k / P_1
        L.append(CHSS("Q_%d" % (2 * k - 1), "B", k, 1))
    for k in range(4, 8):                      # Q_{2k-2} = D_k / P_1
        L.append(CHSS("Q_%d" % (2 * k - 2), "D", k, 1))
    for n in range(2, 8):                      # LGr_n = C_n / P_n
        L.append(CHSS("LGr_%d" % n, "C", n, n))
    for k in range(4, 8):                      # OGr_k = D_k / P_k
        L.append(CHSS("OGr_%d" % k, "D", k, k))
    L.append(CHSS("OP^2", "E", 6, 1))          # E_6 / P_1
    L.append(CHSS("EE", "E", 7, 7))            # E_7 / P_7
    return L


# ------------------------------------------------------------------- testing

def tests():
    ok = True

    def check(what, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("  %-52s %s   %s" % (what, "ok " if good else "FAIL", got))

    print("Tests.")
    # 1) H^0(X; Theta_X) = g, in degree 0, for every CHSS.
    for X in chss_list():
        c = X.tangent_twist(0)
        assert c is not None, X.name
        ell, hw, dd = c
        if (ell, dd) != (0, X.dimg):
            check("H^*(%s; Theta)" % X.name, (ell, dd), (0, X.dimg))
    print("  H^0(X;Theta_X) = g in degree 0 for all %d CHSS"
          % len(chss_list()))

    # 2) the classical Serre computation on CP^1 = Gr_1^2
    P1 = CHSS("CP1", "A", 1, 1)
    check("H^*(CP^1; O(1))", P1.line(1), (0, [1], 2))
    check("H^*(CP^1; O(-1)) = 0", P1.line(-1), None)
    check("H^*(CP^1; O(-2))", P1.line(-2), (1, [0], 1))
    check("H^*(CP^1; O(-3))", P1.line(-3), (1, [1], 2))
    P3 = CHSS("CP3", "A", 3, 1)
    check("H^*(CP^3; O(1))", P3.line(1), (0, [1, 0, 0], 4))
    check("H^*(CP^3; O(-4))", P3.line(-4), (3, [0, 0, 0], 1))
    for j in range(-3, 0):
        check("H^*(CP^3; O(%d)) = 0" % j, P3.line(j), None)

    # 3) dim X against the closed formulas
    for X in chss_list():
        want = None
        if X.name.startswith("Gr_"):
            s, n = X.name[3:].split("^")
            s, n = int(s), int(n)
            want = s * (n - s)
        elif X.name.startswith("Q_"):
            want = int(X.name[2:])
        elif X.name.startswith("LGr_"):
            n = int(X.name[4:])
            want = n * (n + 1) // 2
        elif X.name.startswith("OGr_"):
            n = int(X.name[4:])
            want = n * (n - 1) // 2
        elif X.name == "OP^2":
            want = 16
        elif X.name == "EE":
            want = 27
        if want is not None and X.dimX != want:
            check("dim %s" % X.name, X.dimX, want)
    print("  dim X against the closed formulas: ok")

    # 4) the minimal non-trivial irreducible g-module
    for nm, typ, rk, want in [("sl(4)", "A", 3, 4), ("sp(8)", "C", 4, 8),
                              ("o(9)", "B", 4, 9), ("o(12)", "D", 6, 12),
                              ("o(8)", "D", 4, 8), ("e6", "E", 6, 27),
                              ("e7", "E", 7, 56), ("e8", "E", 8, 248)]:
        R = Root(typ, rk)
        got = min(R.dim(lam) for lam in fundamental_and_small(R))
        check("min dim of a non-trivial irrep of %s" % nm, got, want)
    print("Tests %s.\n" % ("passed" if ok else "FAILED"))
    return ok


# ------------------------------------------------------------------ the runs

def bound_table():
    """D_m = min(m_G, dim X + 1), row by row."""
    print("The bound D_m = min(m_G, dim X + 1).")
    print("  %-10s %8s %8s %8s %8s   %s"
          % ("X", "dim X", "dim g", "m_G", "D_m", "sdim >="))
    for X in chss_list():
        mG, _ = X.min_irrep()
        Dm = min(mG, X.dimX + 1)
        print("  %-10s %8d %8d %8d %8d   (%d|%d)%s"
              % (X.name, X.dimX, X.dimg, mG, Dm, X.dimX, Dm,
                 "   <- D_m = dim X + 1" if Dm == X.dimX + 1 else ""))
    print()


def obstruction_census(jrange=range(-14, 7)):
    """H^1(X; Theta_X (x) E_{d^j}): the rank-2 character obstruction."""
    print("H^*(X; Theta_X (x) E_{d^j}) for %d <= j <= %d: the non-zero ones."
          % (jrange.start, jrange.stop - 1))
    for X in chss_list():
        hits = []
        for j in jrange:
            c = X.tangent_twist(j)
            if c is not None and j != 0:
                ell, hw, dd = c
                hits.append((j, ell, dd))
        s = "; ".join("j=%d: H^%d, dim %d" % h for h in hits) or "none"
        print("  %-10s %s" % (X.name, s))
    print()


def line_census(jrange=range(-14, 7)):
    print("H^*(X; E_{d^j}) for %d <= j <= %d: the non-zero ones with j < 0."
          % (jrange.start, jrange.stop - 1))
    for X in chss_list():
        hits = []
        for j in jrange:
            if j >= 0:
                continue
            c = X.line(j)
            if c is not None:
                ell, hw, dd = c
                hits.append((j, ell, dd))
        s = "; ".join("j=%d: H^%d, dim %d" % h for h in hits) or "none"
        print("  %-10s %s" % (X.name, s))
    print()


if __name__ == "__main__":
    if tests():
        bound_table()
        obstruction_census()
        line_census()
