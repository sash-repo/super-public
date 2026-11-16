#!/usr/bin/env python3
"""grass3.py -- the Grassmann envelope does not see the identity [x,[x,x]] = 0.

Companion to freelie.py (which builds the free objects) and cube3.py (which
scans the root systems).

In characteristic 3 there are two candidate categories of Lie superalgebras:
with the identity T(x) = [x,[x,x]] = 0 for odd x imposed, and without it.
[BGLL] works without it, which is why its formula (46) has to IMPOSE
ad_X(X) = 0 at a node "bullet".

This script settles which of the two Berezin's definition -- a superalgebra is
a Lie superalgebra when its GRASSMANN ENVELOPE is an ordinary Lie algebra --
picks out in characteristic 3.  The answer is: the bigger one.  The reason is
one line: for z_i = xi_i (x) x, the three cyclic terms of the ordinary Jacobi
identity in the envelope are all equal to +- xi_1 xi_2 xi_3 (x) T(x), and their
sum is 3 T(x) = 0.  So the envelope cannot feel T.

What is verified here, exactly, over F_3:

  1. A = <y odd, h = [y,y], z = [y,h]>, sdim (1|2), satisfies super
     anti-symmetry and super Jacobi -- all 27 triples of basis vectors -- and
     T(y) = z is NOT zero.  (It is the free Lie superalgebra on one odd
     generator of the category WITHOUT the identity; see freelie.py.)
  2. Over Q the same structure constants do NOT satisfy super Jacobi: there
     the identity 3 T(y) = 0 forces z = 0.  So this is a p = 3 phenomenon and
     not a sign convention.
  3. The Grassmann envelope G(A) = Lambda_ev (x) A_ev + Lambda_od (x) A_od,
     with Lambda = Lambda(xi_1, xi_2, xi_3), is an ordinary Lie algebra:
     dim 12, anti-symmetry and Jacobi on all 12^3 triples.
  4. In it, [xi_1 (x) y, [xi_2 (x) y, xi_3 (x) y]] = +- xi_1 xi_2 xi_3 (x) z
     is nonzero: the cube survives in the envelope, and the envelope is a Lie
     algebra all the same.

Exact arithmetic over F_3 and over Q (Fraction); no floating point.
"""
import sys
import os
from fractions import Fraction
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import backend                                   # noqa: E402


# ---------------------------------------------------------------------------
# a small superalgebra given by structure constants
# ---------------------------------------------------------------------------

class SuperAlg:
    """Basis with parities and a bracket given on basis vectors."""

    def __init__(self, names, par, table, p):
        self.names = list(names)
        self.par = list(par)
        self.n = len(names)
        self.p = p                          # None = over Q
        self.table = {}                     # (i,j) -> {k: coeff}
        for (i, j), v in table.items():
            self.table[(i, j)] = dict(v)

    def norm(self, c):
        return c % self.p if self.p else Fraction(c)

    def br(self, u, v):
        """u, v are dicts basis-index -> coefficient."""
        out = {}
        for i, a in u.items():
            for j, b in v.items():
                for k, c in self.table.get((i, j), {}).items():
                    out[k] = out.get(k, 0) + a * b * c
        return {k: self.norm(c) for k, c in out.items() if self.norm(c)}

    def e(self, i):
        return {i: self.norm(1)}

    def sub(self, u, v):
        out = dict(u)
        for k, c in v.items():
            out[k] = out.get(k, 0) - c
        return {k: self.norm(c) for k, c in out.items() if self.norm(c)}

    def add(self, u, v):
        out = dict(u)
        for k, c in v.items():
            out[k] = out.get(k, 0) + c
        return {k: self.norm(c) for k, c in out.items() if self.norm(c)}

    def smul(self, s, u):
        return {k: self.norm(s * c) for k, c in u.items() if self.norm(s * c)}

    # -- the axioms ---------------------------------------------------------

    def check_antisym(self):
        """[a,b] = -(-1)^{|a||b|}[b,a], and [a,a] = 0 for a even."""
        bad = []
        for i in range(self.n):
            for j in range(self.n):
                s = -1 if (self.par[i] * self.par[j]) % 2 == 0 else 1
                lhs = self.br(self.e(i), self.e(j))
                rhs = self.smul(s, self.br(self.e(j), self.e(i)))
                if lhs != rhs:
                    bad.append((self.names[i], self.names[j]))
            if self.par[i] == 0 and self.br(self.e(i), self.e(i)):
                bad.append((self.names[i], self.names[i]))
        return bad

    def check_jacobi(self):
        """[a,[b,c]] = [[a,b],c] + (-1)^{|a||b|}[b,[a,c]]."""
        bad = []
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    s = 1 if (self.par[i] * self.par[j]) % 2 == 0 else -1
                    lhs = self.br(self.e(i), self.br(self.e(j), self.e(k)))
                    rhs = self.add(self.br(self.br(self.e(i), self.e(j)), self.e(k)),
                                   self.smul(s, self.br(self.e(j),
                                                        self.br(self.e(i), self.e(k)))))
                    if lhs != rhs:
                        bad.append((self.names[i], self.names[j], self.names[k]))
        return bad

    def show(self, u):
        if not u:
            return "0"
        return " + ".join("%s*%s" % (c, self.names[k]) for k, c in sorted(u.items()))


def free_one_odd(p):
    """<y odd, h = [y,y] even, z = [y,h] odd central>: the free object on one
    odd generator of the category WITHOUT the identity [x,[x,x]] = 0."""
    names = ["y", "h", "z"]
    par = [1, 0, 1]
    y, h, z = 0, 1, 2
    table = {
        (y, y): {h: 1},
        (y, h): {z: 1},
        (h, y): {z: -1},        # [h,y] = -(-1)^{0*1}[y,h] = -z
    }
    return SuperAlg(names, par, table, p)


# ---------------------------------------------------------------------------
# the Grassmann envelope
# ---------------------------------------------------------------------------

class Grassmann:
    """Lambda(xi_1, ..., xi_n) with monomials as sorted tuples of indices."""

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.mon = []
        for d in range(n + 1):
            for c in combinations(range(n), d):
                self.mon.append(c)
        self.index = {m: i for i, m in enumerate(self.mon)}

    def parity(self, m):
        return len(m) % 2

    def mul(self, a, b):
        """Product of two monomials: (sign, monomial) or (0, None)."""
        if set(a) & set(b):
            return 0, None
        word = list(a) + list(b)
        # sign of the permutation sorting the word (adjacent transpositions)
        sign = 1
        w = word[:]
        for i in range(len(w)):
            for j in range(len(w) - 1 - i):
                if w[j] > w[j + 1]:
                    w[j], w[j + 1] = w[j + 1], w[j]
                    sign = -sign
        return sign, tuple(w)


def envelope(A, nxi):
    """G(A) = Lambda_ev (x) A_ev + Lambda_od (x) A_od, as a SuperAlg with all
    parities 0 -- i.e. as an ORDINARY algebra, whose axioms we then test."""
    p = A.p
    L = Grassmann(nxi, p)
    basis = []                              # (grassmann monomial, index in A)
    for m in L.mon:
        for i in range(A.n):
            if L.parity(m) == A.par[i]:
                basis.append((m, i))
    index = {b: k for k, b in enumerate(basis)}
    names = ["%s@%s" % (A.names[i], "".join("x%d" % (t + 1) for t in m) or "1")
             for m, i in basis]
    table = {}
    for u, (m, i) in enumerate(basis):
        for v, (mm, j) in enumerate(basis):
            sgn, prod = L.mul(m, mm)
            if sgn == 0:
                continue
            # [lam (x) a, mu (x) b] = (-1)^{|a||mu|} lam*mu (x) [a,b]
            s = sgn * (-1 if (A.par[i] * L.parity(mm)) % 2 else 1)
            br = A.br(A.e(i), A.e(j))
            row = {}
            for k, c in br.items():
                key = (prod, k)
                if key not in index:        # parity mismatch cannot happen
                    raise AssertionError("envelope not closed at %s" % (key,))
                row[index[key]] = row.get(index[key], 0) + s * c
            row = {k: (c % p if p else Fraction(c)) for k, c in row.items()}
            row = {k: c for k, c in row.items() if c}
            if row:
                table[(u, v)] = row
    G = SuperAlg(names, [0] * len(basis), table, p)
    G.basis = basis
    G.index = index
    return G


def check_ordinary(G):
    """G is an ordinary Lie algebra: [u,v] = -[v,u], [u,u] = 0, Jacobi."""
    bad_a, bad_j = [], []
    for i in range(G.n):
        for j in range(G.n):
            lhs = G.br(G.e(i), G.e(j))
            rhs = G.smul(-1, G.br(G.e(j), G.e(i)))
            if lhs != rhs:
                bad_a.append((G.names[i], G.names[j]))
        if G.br(G.e(i), G.e(i)):
            bad_a.append((G.names[i], G.names[i]))
    for i in range(G.n):
        for j in range(G.n):
            for k in range(G.n):
                # [i,[j,k]] + [j,[k,i]] + [k,[i,j]] = 0
                s = G.br(G.e(i), G.br(G.e(j), G.e(k)))
                s = G.add(s, G.br(G.e(j), G.br(G.e(k), G.e(i))))
                s = G.add(s, G.br(G.e(k), G.br(G.e(i), G.e(j))))
                if s:
                    bad_j.append((G.names[i], G.names[j], G.names[k]))
    return bad_a, bad_j


# ---------------------------------------------------------------------------

def main():
    print("backend: %s" % backend())
    print()
    print("=" * 78)
    print("1.  A = <y, h = [y,y], z = [y,h]>, sdim (1|2), over F_3")
    print("=" * 78)
    A3 = free_one_odd(3)
    ba, bj = A3.check_antisym(), A3.check_jacobi()
    print("  super anti-symmetry on all %d pairs:   %s"
          % (A3.n ** 2, "ok" if not ba else "FAILS at %s" % ba))
    print("  super Jacobi on all %d triples:       %s"
          % (A3.n ** 3, "ok" if not bj else "FAILS at %s" % bj))
    y = A3.e(0)
    Ty = A3.br(y, A3.br(y, y))
    print("  T(y) = [y,[y,y]] = %s   (nonzero: the identity does NOT hold in A)"
          % A3.show(Ty))
    print("  3*T(y) = %s" % A3.show(A3.smul(3, Ty)))
    ok1 = not ba and not bj and bool(Ty)

    print()
    print("=" * 78)
    print("2.  the same structure constants over Q")
    print("=" * 78)
    A0 = free_one_odd(None)
    ba0, bj0 = A0.check_antisym(), A0.check_jacobi()
    print("  super anti-symmetry:  %s" % ("ok" if not ba0 else "FAILS at %s" % ba0))
    print("  super Jacobi:         %s"
          % ("ok" if not bj0 else "FAILS at %s -- as it must: 3T(y) = 0 forces z = 0"
             % (bj0[:3],)))
    ok2 = bool(bj0)

    print()
    print("=" * 78)
    print("3.  the Grassmann envelope G(A) over F_3, Lambda = Lambda(xi_1,xi_2,xi_3)")
    print("=" * 78)
    G = envelope(A3, 3)
    print("  dim G(A) = %d  (Lambda_ev (x) A_ev: 4*1, Lambda_od (x) A_od: 4*2)" % G.n)
    ga, gj = check_ordinary(G)
    print("  anti-symmetry and [u,u] = 0 on all %d pairs:  %s"
          % (G.n ** 2, "ok" if not ga else "FAILS at %s" % ga[:5]))
    print("  ordinary Jacobi on all %d triples:          %s"
          % (G.n ** 3, "ok" if not gj else "FAILS at %s" % gj[:5]))
    ok3 = not ga and not gj

    print()
    print("=" * 78)
    print("4.  the cube inside the envelope")
    print("=" * 78)
    z1 = G.e(G.index[((0,), 0)])            # xi_1 (x) y
    z2 = G.e(G.index[((1,), 0)])            # xi_2 (x) y
    z3 = G.e(G.index[((2,), 0)])            # xi_3 (x) y
    w = G.br(z1, G.br(z2, z3))
    print("  [xi_1(x)y, [xi_2(x)y, xi_3(x)y]] = %s" % G.show(w))
    cyc = G.add(G.add(G.br(z1, G.br(z2, z3)), G.br(z2, G.br(z3, z1))),
                G.br(z3, G.br(z1, z2)))
    print("  the three cyclic terms sum to    = %s   (they are 3 T(y) = 0)" % G.show(cyc))
    ok4 = bool(w) and not cyc

    print()
    print("=" * 78)
    verdict = ok1 and ok2 and ok3 and ok4
    print("VERDICT: %s" % ("all four checks pass" if verdict else "SOMETHING FAILED"))
    print()
    print("  A superalgebra whose Grassmann envelope is an ordinary Lie algebra need")
    print("  NOT satisfy [x,[x,x]] = 0 when p = 3.  Hence Berezin's definition -- and")
    print("  with it the whole 'the envelope is a Lie algebra' tradition -- selects the")
    print("  category WITHOUT the identity, which is the one [BGLL] works in.")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
