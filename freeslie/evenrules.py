#!/usr/bin/env python3
r"""evenrules.py -- in what precise sense the Even Rules fail when p = 2.

Companion to grass3.py, which did the same job for p = 3.

The observation to be made precise: for p = 2 or 3 the Even Rules fail.
grass3.py proves one half of it: at p = 3
the Grassmann envelope cannot feel the identity T(x) = [x,[x,x]] = 0, because
the three cyclic Jacobi terms in the envelope add up to 3 T(x) = 0.  This
script proves the other half, and the failure at p = 2 is the worse of the two.

The reason, in one line.  The Even Rules recover a superstructure from the
ordinary structures on (Lambda (x) g)_ev, and everything they can see is
MULTILINEAR.  At p = 2 the squaring x |-> x^2 on g_od is not: it is the
diagonal of the odd-odd bracket, which polarization recovers only when 2 is
invertible.  So the envelope sees the bracket and nothing else, and the
squaring -- which is part of the definition of [BLLS], Subsection 2.2 -- is
invisible to it.  Two consequences, and both are verified here exactly.

  A. NON-UNIQUENESS.  Two Lie superalgebras with the same bracket and
     different squarings have THE SAME Grassmann envelope, and need not be
     isomorphic.  Smallest example, sdim (1|1):
         H^0 = <c even, x odd>, all brackets zero, x^2 = 0    (abelian)
         H^1 = <c even, x odd>, all brackets zero, x^2 = c
     Both satisfy every axiom of [BLLS] 2.2; their envelopes are equal (both
     abelian of dimension 8 over Lambda(xi_1,xi_2,xi_3)); and they are not
     isomorphic -- by the derived algebra, which at p = 2 is DEFINED with the
     squares in it, [BLLS] 2.2: (H^1)^(1) = <c> while (H^0)^(1) = 0.

     In general the squarings compatible with a fixed bracket form a torsor
     over the Frobenius-semilinear additive maps g_od -> z(g) \cap g_ev.
     Checked by counting: for H the count is q (over F_2 and over F_4), for
     the algebra D = <h even, x odd | [h,x] = x>, whose even center is 0, it
     is 1.

  B. NON-EXISTENCE.  There is a superalgebra whose Grassmann envelope is an
     honest Lie algebra but which admits NO squaring at all, so it is not a
     Lie superalgebra in the sense of [BLLS] 2.2.  The smallest one has
     sdim (1|2), i.e. dimension 3:
         B = <u even; x, y odd | [u,x] = y, [x,y] = u>.
     Then ad_x^2 (u) = [x,[x,u]] = [x,y] = u =/= 0, so a squaring would need
     an s(x) in g_ev = <u> with [s(x),u] = u; but <u> is abelian.  Hence at
     p = 2 Berezin's definition is STRICTLY WEAKER than [BLLS] 2.2 -- not
     merely blind to part of the data, but admitting objects that carry no
     such data.

     Minimality is exhaustive, not a guess (part D): over F_2 every object of
     Berezin's category of dimension <= 2 carries a squaring, and of the 24
     of sdim (1|2) exactly 7 do not.  The census also shows TWO mechanisms of
     failure -- ad_v^2 not implemented by any even element, as in B, and the
     axiom [v^2,v] = 0, which is what fails in
         <u | [u,x] = x, [u,y] = y, [x,y] = u>,
     where s(x) = s(y) = 0 is forced, hence s(x+y) = [x,y] = u, and then
     [s(x+y), x+y] = [u,x+y] = x+y =/= 0.

Exact arithmetic over F_2 and over F_4; no floating point.
"""
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from exactlin import backend                                    # noqa: E402
import grass3                                                   # noqa: E402


# ---------------------------------------------------------------------------
# GF(2^k) for k = 1, 2:  elements are ints, addition is XOR
# ---------------------------------------------------------------------------

class GF:
    """GF(2) and GF(4) = F_2[t]/(t^2+t+1); elements 0..q-1 as bit patterns."""

    def __init__(self, k):
        self.k = k
        self.q = 1 << k
        self.name = "F_%d" % self.q

    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        if self.k == 1:
            return a & b
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if a & 0b100:                      # reduce by t^2 = t + 1
                a ^= 0b111
        return r

    def sq(self, a):
        return self.mul(a, a)

    def elems(self):
        return range(self.q)

    def nonzero(self):
        return range(1, self.q)


# ---------------------------------------------------------------------------
# a superalgebra over GF(2^k), given by structure constants on a basis
# ---------------------------------------------------------------------------

class SAlg2:
    def __init__(self, names, par, table, F):
        self.names = list(names)
        self.par = list(par)
        self.n = len(names)
        self.F = F
        self.table = {}
        for (i, j), v in table.items():
            self.table[(i, j)] = dict(v)
        self.od = [i for i in range(self.n) if par[i] == 1]
        self.ev = [i for i in range(self.n) if par[i] == 0]

    # vectors are tuples of length n over F
    def zero(self):
        return (0,) * self.n

    def e(self, i):
        v = [0] * self.n
        v[i] = 1
        return tuple(v)

    def vadd(self, u, v):
        return tuple(self.F.add(a, b) for a, b in zip(u, v))

    def smul(self, c, u):
        return tuple(self.F.mul(c, a) for a in u)

    def br(self, u, v):
        F = self.F
        out = [0] * self.n
        for i, a in enumerate(u):
            if not a:
                continue
            for j, b in enumerate(v):
                if not b:
                    continue
                ab = F.mul(a, b)
                for k, c in self.table.get((i, j), {}).items():
                    out[k] = F.add(out[k], F.mul(ab, c))
        return tuple(out)

    def show(self, u):
        parts = [("%s" % self.names[k]) if c == 1 else ("%d*%s" % (c, self.names[k]))
                 for k, c in enumerate(u) if c]
        return " + ".join(parts) if parts else "0"

    # -- the bracket axioms (p = 2: symmetric, alternating, Jacobi) ---------

    def check_bracket(self, alt_odd=True):
        """alt_odd=True demands [x,x] = 0 for x odd as well -- which holds
        automatically in [BLLS] 2.2, the odd-odd bracket being there the
        polarization of the squaring, but which the Grassmann envelope does
        NOT impose (see check_berezin)."""
        bad = []
        for i in range(self.n):
            for j in range(self.n):
                if self.br(self.e(i), self.e(j)) != self.br(self.e(j), self.e(i)):
                    bad.append(("sym", self.names[i], self.names[j]))
            if (alt_odd or self.par[i] == 0) and any(self.br(self.e(i), self.e(i))):
                bad.append(("alt", self.names[i], self.names[i]))
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    a, b, c = self.e(i), self.e(j), self.e(k)
                    s = self.vadd(self.vadd(self.br(a, self.br(b, c)),
                                            self.br(b, self.br(c, a))),
                                  self.br(c, self.br(a, b)))
                    if any(s):
                        bad.append(("jac", self.names[i], self.names[j],
                                    self.names[k]))
        return bad

    def check_berezin(self):
        """The axioms the Grassmann envelope really imposes at p = 2: the
        bracket is symmetric, [a,a] = 0 for a EVEN, and Jacobi holds."""
        return self.check_bracket(alt_odd=False)

    def odd_elements(self):
        """All elements of g_od (only used on tiny examples)."""
        out = []
        for coeffs in product(self.F.elems(), repeat=len(self.od)):
            v = list(self.zero())
            for c, i in zip(coeffs, self.od):
                v[i] = c
            out.append(tuple(v))
        return out

    def even_elements(self):
        out = []
        for coeffs in product(self.F.elems(), repeat=len(self.ev)):
            v = list(self.zero())
            for c, i in zip(coeffs, self.ev):
                v[i] = c
            out.append(tuple(v))
        return out

    def even_center(self):
        """{z in g_ev | [z, g] = 0}."""
        return [z for z in self.even_elements()
                if all(not any(self.br(z, self.e(i))) for i in range(self.n))]


# ---------------------------------------------------------------------------
# squarings
# ---------------------------------------------------------------------------

def build_squaring(A, values):
    """The quadratic map with s(e_i) = values[i] on the odd basis and the
    given bracket as polarization:  s(sum a_i e_i) = sum a_i^2 s(e_i)
    + sum_{i<j} a_i a_j [e_i,e_j]."""
    F = A.F
    s = {}
    for v in A.odd_elements():
        acc = A.zero()
        for t, i in enumerate(A.od):
            acc = A.vadd(acc, A.smul(F.sq(v[i]), values[t]))
        for t1 in range(len(A.od)):
            for t2 in range(t1 + 1, len(A.od)):
                i, j = A.od[t1], A.od[t2]
                acc = A.vadd(acc, A.smul(F.mul(v[i], v[j]),
                                         A.br(A.e(i), A.e(j))))
        s[v] = acc
    return s


def squaring_defects(A, s, strong=True):
    """The axioms of [BLLS] 2.2 for a squaring s: g_od -> g_ev.

    The last axiom comes in two strengths, and over F_2 they differ -- this is
    [KLLS], Subsec. 3.2, equations (11) and (14), and their Example 3.2.1:

        weak   (11):  [v^2, y] = [v,[v,y]] for y EVEN,  and  [v^2, v] = 0;
        strong (14):  [v^2, y] = [v,[v,y]] for ALL y.

    Over any field of characteristic 2 other than F_2 the two are equivalent
    (Part E below proves it in one line and checks it); over F_2 they are not,
    and [BLLS]/[KLLS] use the strong one, which is therefore our default.
    """
    F = A.F
    bad = []
    for v in A.odd_elements():
        if any(p != 0 for k, p in enumerate(s[v]) if A.par[k] == 1):
            bad.append(("values not even", A.show(v)))
        for c in F.elems():                       # (cv)^2 = c^2 v^2
            if s[A.smul(c, v)] != A.smul(F.sq(c), s[v]):
                bad.append(("semilinear", c, A.show(v)))
        for w in A.odd_elements():                # polarization = the bracket
            lhs = s[A.vadd(v, w)]
            rhs = A.vadd(A.vadd(s[v], s[w]), A.br(v, w))
            if lhs != rhs:
                bad.append(("polarization", A.show(v), A.show(w)))
        rng = range(A.n) if strong else A.ev
        for i in rng:                             # [v^2, y] = [v,[v,y]]
            y = A.e(i)
            if A.br(s[v], y) != A.br(v, A.br(v, y)):
                bad.append(("jacobi-%s" % ("all" if strong else "even"),
                            A.show(v), A.names[i]))
        if any(A.br(s[v], v)):                    # [v^2, v] = 0
            bad.append(("jacobi-odd", A.show(v)))
    return bad


def all_squarings(A, strong=True):
    """Every squaring compatible with the bracket of A."""
    out = []
    for values in product(A.even_elements(), repeat=len(A.od)):
        s = build_squaring(A, values)
        if not squaring_defects(A, s, strong=strong):
            out.append((values, s))
    return out


# ---------------------------------------------------------------------------
# the examples
# ---------------------------------------------------------------------------

def enumerate_brackets(m, n, odd_diag=False):
    """Every symmetric bracket with [a,a] = 0 on a superspace of sdim (m|n)
    over F_2: basis 0..m-1 even, m..m+n-1 odd.  The diagonal is zero and the
    bracket is symmetric, so the free data are the values on the pairs i < j,
    each in the component of the right parity.  Jacobi is NOT imposed here --
    the caller filters."""
    ev, od = list(range(m)), list(range(m, m + n))
    slots = []                                  # (i, j, list of allowed values)
    ev_vals = [tuple(1 if k == t else 0 for k in ev) for t in range(m)]
    od_vals = [tuple(1 if k == t else 0 for k in od) for t in range(n)]

    def spans(basisvals, dim):
        out = []
        for bits in product((0, 1), repeat=dim):
            out.append(bits)
        return out

    EV, OD = spans(ev_vals, m), spans(od_vals, n)
    for a in range(m):
        for b in range(a + 1, m):
            slots.append((a, b, EV, ev))
    for a in ev:
        for b in od:
            slots.append((a, b, OD, od))
    for a in range(m, m + n):
        for b in range(a + (0 if odd_diag else 1), m + n):
            slots.append((a, b, EV, ev))

    for choice in product(*[s[2] for s in slots]):
        table = {}
        for (i, j, _, target), val in zip(slots, choice):
            row = {target[t]: 1 for t, c in enumerate(val) if c}
            if row:
                table[(i, j)] = dict(row)
                table[(j, i)] = dict(row)
        yield table


def alg_H(F):
    """H = <c even, x odd>, all brackets zero.  sdim (1|1)."""
    return SAlg2(["c", "x"], [0, 1], {}, F)


def alg_D(F):
    """D = <h even, x odd | [h,x] = x>.  Even center 0.  sdim (1|1)."""
    h, x = 0, 1
    return SAlg2(["h", "x"], [0, 1], {(h, x): {x: 1}, (x, h): {x: 1}}, F)


B_TABLE = {
    (0, 1): {2: 1}, (1, 0): {2: 1},             # [u,x] = y
    (1, 2): {0: 1}, (2, 1): {0: 1},             # [x,y] = u
}
B_NAMES, B_PAR = ["u", "x", "y"], [0, 1, 1]


def alg_B(F):
    """B = <u even; x, y odd | [u,x] = y, [x,y] = u>.  sdim (1|2)."""
    return SAlg2(B_NAMES, B_PAR, B_TABLE, F)


def alg_B_grass3():
    """The same B as a grass3.SuperAlg over F_2, for the envelope code."""
    return grass3.SuperAlg(B_NAMES, B_PAR, B_TABLE, 2)


def alg_C(F):
    """C = <u even; x, y odd | [u,x] = x, [u,y] = y, [x,y] = u>: the other
    mechanism -- here ad_v^2 is inner for every v, and it is the axiom
    [v^2,v] = 0 that has no solution."""
    table = {
        (0, 1): {1: 1}, (1, 0): {1: 1},         # [u,x] = x
        (0, 2): {2: 1}, (2, 0): {2: 1},         # [u,y] = y
        (1, 2): {0: 1}, (2, 1): {0: 1},         # [x,y] = u
    }
    return SAlg2(B_NAMES, B_PAR, table, F)


def alg_H_grass3():
    return grass3.SuperAlg(["c", "x"], [0, 1], {}, 2)


def alg_osp12(F):
    """osp(1|2) over a field of characteristic 2, sdim (3|2):
    g_ev = <e, h, f> = sl(2) with [e,f] = h, [h,e] = [h,f] = 0 (p = 2);
    g_od = <X, Y> with [h,X] = X, [h,Y] = Y, [e,Y] = X, [f,X] = Y,
    [X,Y] = h, and the squaring X^2 = e, Y^2 = f.
    [KLLS], Subsec. 3.3.4, second problem, takes its 'derived algebra
    ignoring the squaring', Span{[a,b]} = <h, X, Y>."""
    names = ["e", "h", "f", "X", "Y"]
    par = [0, 0, 0, 1, 1]
    e, h, f, X, Y = 0, 1, 2, 3, 4
    tab = {}

    def put(i, j, row):
        tab[(i, j)] = dict(row)
        tab[(j, i)] = dict(row)

    put(e, f, {h: 1})
    put(h, X, {X: 1})
    put(h, Y, {Y: 1})
    put(e, Y, {X: 1})
    put(f, X, {Y: 1})
    put(X, Y, {h: 1})
    return SAlg2(names, par, tab, F)


def osp12_squaring(A):
    """The squaring of osp(1|2): X^2 = e, Y^2 = f."""
    return build_squaring(A, [A.e(0), A.e(2)])


def alg_KLLS_example(F):
    """[KLLS], Example 3.2.1, of sdim (2|3): g_ev = <A, B> commutative,
    [A,Y] = Z, [B,X] = Z, the odd-odd bracket zero, and the squaring
    (aX + bY + cZ)^2 = a^2 A + b^2 B.  It satisfies the weak axiom (11)
    but not the strong one (14)."""
    names = ["A", "B", "X", "Y", "Z"]
    par = [0, 0, 1, 1, 1]
    A_, B_, X, Y, Z = 0, 1, 2, 3, 4
    tab = {}

    def put(i, j, row):
        tab[(i, j)] = dict(row)
        tab[(j, i)] = dict(row)

    put(A_, Y, {Z: 1})
    put(B_, X, {Z: 1})
    return SAlg2(names, par, tab, F)


# ---------------------------------------------------------------------------

def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main():
    print("evenrules.py -- the Even Rules at p = 2")
    print("backend: %s" % backend())
    ok = True

    F2, F4 = GF(1), GF(2)

    # ---------------------------------------------------------------- A ----
    hr("A.  Same bracket, two squarings: H^0 abelian and H^1 with x^2 = c")

    for F in (F2, F4):
        H = alg_H(F)
        bad = H.check_bracket()
        sq = all_squarings(H)
        ctr = [z for z in H.even_center()]
        print("  over %s: bracket axioms %s; squarings compatible with it: %d;"
              % (F.name, "ok" if not bad else "FAIL %s" % bad, len(sq)))
        print("            even center of H has %d elements, and the semilinear"
              % len(ctr))
        print("            maps <x> -> z(H)_ev number q = %d" % F.q)
        ok &= (not bad) and len(sq) == F.q
        if F is F2:
            names = sorted(H.show(s[H.e(1)]) for _, s in sq)
            print("            the squarings are  x^2 in {%s}" % ", ".join(names))

    H2 = alg_H(F2)
    s0 = [s for _, s in all_squarings(H2) if not any(s[H2.e(1)])][0]
    s1 = [s for _, s in all_squarings(H2) if any(s[H2.e(1)])][0]
    print()
    print("  H^0: x^2 = %s      H^1: x^2 = %s"
          % (H2.show(s0[H2.e(1)]), H2.show(s1[H2.e(1)])))

    # derived algebras, [BLLS] 2.2:  g^(1) = [g,g] + Span{g^2 | g in g_od}
    def derived(A, s):
        gen = []
        for i in range(A.n):
            for j in range(A.n):
                gen.append(A.br(A.e(i), A.e(j)))
        for v in A.odd_elements():
            gen.append(s[v])
        return sorted(set(g for g in gen if any(g)))

    d0, d1 = derived(H2, s0), derived(H2, s1)
    print("  (H^0)^(1) = span{%s}   (H^1)^(1) = span{%s}"
          % (", ".join(H2.show(g) for g in d0) or "0",
             ", ".join(H2.show(g) for g in d1) or "0"))
    print("  => H^0 and H^1 are NOT isomorphic (the derived algebra at p = 2")
    print("     is defined with the squares in it, [BLLS] 2.2).")
    ok &= (not d0) and bool(d1)

    # brute force over all parity-preserving invertible maps, over F_4 too
    for F in (F2, F4):
        H = alg_H(F)
        sq = all_squarings(H)
        z = [s for _, s in sq if not any(s[H.e(1)])][0]
        nz = [s for _, s in sq if any(s[H.e(1)])][0]
        found = None
        for a in F.nonzero():                     # phi(x) = a x
            for b in F.nonzero():                 # phi(c) = b c
                good = True
                for v in H.odd_elements():
                    pv = H.smul(a, v)             # phi(v), v odd
                    lhs = H.smul(b, z[v])         # phi(v^2) in H^0
                    if lhs != nz[pv]:             # (phi v)^2 in H^1
                        good = False
                        break
                if good:
                    found = (a, b)
        print("  over %s: parity-preserving isomorphisms H^0 -> H^1: %s"
              % (F.name, "none" if found is None else str(found)))
        ok &= found is None

    # the envelopes coincide, being built from the bracket alone
    Hg = alg_H_grass3()
    G = grass3.envelope(Hg, 3)
    ba, bj = grass3.check_ordinary(G)
    zero_br = all(not G.br(G.e(i), G.e(j)) for i in range(G.n)
                  for j in range(G.n))
    print("  envelope over Lambda(xi_1,xi_2,xi_3): dim %d, abelian: %s,"
          % (G.n, zero_br))
    print("     Lie algebra axioms %s -- and it is the SAME for H^0 and H^1,"
          % ("ok" if not ba and not bj else "FAIL"))
    print("     the squaring never entering its structure constants.")
    ok &= zero_br and not ba and not bj

    # the torsor, other end: no center => at most one squaring
    hr("A'. D = <h, x | [h,x] = x>: even center 0, so the squaring is unique")
    for F in (F2, F4):
        D = alg_D(F)
        bad = D.check_bracket()
        ctr = D.even_center()
        sq = all_squarings(D)
        print("  over %s: bracket %s; even center = {%s}; squarings: %d %s"
              % (F.name, "ok" if not bad else "FAIL",
                 ", ".join(D.show(z) for z in ctr if any(z)) or "0",
                 len(sq),
                 "(x^2 = %s)" % D.show(sq[0][1][D.e(1)]) if sq else ""))
        ok &= (not bad) and len(ctr) == 1 and len(sq) == 1

    # ---------------------------------------------------------------- B ----
    hr("B.  B = <u even; x, y odd | [u,x] = y, [x,y] = u>: no squaring")

    for F in (F2, F4):
        B = alg_B(F)
        bad = B.check_bracket()
        print("  over %s: bracket axioms (symmetry, [a,a] = 0, Jacobi on all"
              % F.name)
        print("            %d triples): %s"
              % (B.n ** 3, "ok" if not bad else "FAIL %s" % bad[:3]))
        ok &= not bad
        cand = list(product(B.even_elements(), repeat=len(B.od)))
        sq = all_squarings(B)
        print("            candidate squarings tried: %d; valid: %d"
              % (len(cand), len(sq)))
        ok &= len(sq) == 0
        # the obstruction, printed
        u, x = B.e(0), B.e(1)
        print("            ad_x^2(u) = [x,[x,u]] = %s, while [g_ev, u] = {%s}"
              % (B.show(B.br(x, B.br(x, u))),
                 ", ".join(sorted(set(B.show(B.br(z, u))
                                      for z in B.even_elements())))))

    # the other mechanism: ad_v^2 is inner for every v, and [v^2,v] = 0 fails
    for F in (F2, F4):
        C = alg_C(F)
        bad = C.check_bracket()
        sq = all_squarings(C)
        u, x, y = C.e(0), C.e(1), C.e(2)
        forced = C.br(x, y)                       # s(x+y) = s(x)+s(y)+[x,y]
        print("  over %s: C = <u | [u,x] = x, [u,y] = y, [x,y] = u>: bracket %s,"
              % (F.name, "ok" if not bad else "FAIL"))
        print("            squarings: %d.  Here s(x) = s(y) = 0 is forced, so"
              % len(sq))
        print("            s(x+y) = [x,y] = %s and [s(x+y), x+y] = %s =/= 0."
              % (C.show(forced), C.show(C.br(forced, C.vadd(x, y)))))
        ok &= (not bad) and len(sq) == 0 and any(C.br(forced, C.vadd(x, y)))

    Bg = alg_B_grass3()
    for nxi in (2, 3):
        G = grass3.envelope(Bg, nxi)
        ba, bj = grass3.check_ordinary(G)
        print("  envelope of B over Lambda(%s): dim %d; [U,U] = 0 and anti-"
              % (", ".join("xi_%d" % (i + 1) for i in range(nxi)), G.n))
        print("     symmetry on all %d pairs: %s; Jacobi on all %d triples: %s"
              % (G.n ** 2, "ok" if not ba else "FAILS %s" % ba[:3],
                 G.n ** 3, "ok" if not bj else "FAILS %s" % bj[:3]))
        ok &= not ba and not bj

    # and the envelope does see the bracket that causes the trouble
    G = grass3.envelope(Bg, 3)
    i1 = G.index[((0,), 1)]                        # xi_1 (x) x
    i2 = G.index[((1,), 1)]                        # xi_2 (x) x
    iu = G.index[((), 0)]                          # 1 (x) u
    v = G.br(G.e(i1), G.br(G.e(i2), G.e(iu)))
    print("  in it, [xi_1 (x) x, [xi_2 (x) x, 1 (x) u]] = %s =/= 0:" % G.show(v))
    print("     the envelope feels ad_x^2 perfectly well -- what it cannot")
    print("     feel is that no element of B_ev implements it.")
    ok &= bool(v)

    # ---------------------------------------------------------------- C ----
    hr("C.  Exactly what 'the envelope is a Lie algebra' imposes at p = 2")

    # Claim: G(g) is an ordinary Lie algebra iff the bracket is symmetric,
    # [a,a] = 0 for a EVEN, and Jacobi holds.  The odd diagonal [x,x] is NOT
    # constrained -- at p = 2 both [xi (x) x, xi (x) x] = xi^2 (x) [x,x] and
    # the cross terms 2 xi_1 xi_2 (x) [x,x] vanish for free.  So we enumerate
    # with the odd diagonal free as well, and compare.
    for (m, n) in [(1, 1), (1, 2)]:
        names = ["u%d" % (i + 1) for i in range(m)] + \
                ["x%d" % (i + 1) for i in range(n)]
        par = [0] * m + [1] * n
        agree = disagree = odddiag = 0
        for table in enumerate_brackets(m, n, odd_diag=True):
            A = SAlg2(names, par, table, F2)
            berezin = not A.check_berezin()
            G = grass3.envelope(grass3.SuperAlg(names, par, table, 2), 3)
            ba, bj = grass3.check_ordinary(G)
            lie = not ba and not bj
            if berezin == lie:
                agree += 1
            else:
                disagree += 1
            if lie and any(any(A.br(A.e(i), A.e(i))) for i in A.od):
                odddiag += 1
        print("  sdim (%d|%d), all %d brackets over F_2 with the odd diagonal"
              % (m, n, agree + disagree))
        print("     free: 'symmetric + [a,a] = 0 on g_ev + Jacobi' <=> 'envelope")
        print("     is a Lie algebra' in %d cases, differ in %d;" % (agree, disagree))
        print("     and %d of them have [x,x] =/= 0 for some odd x." % odddiag)
        ok &= disagree == 0 and odddiag > 0

    print()
    print("  So the envelope does not even force [x,x] = 0 for x odd: already")
    print("  in dimension 2, <c even, x odd | [x,x] = c> has a Lie envelope,")
    print("  while in [BLLS] 2.2 the odd-odd bracket is the polarization of the")
    print("  squaring and [x,x] = (2x)^2 - 2x^2 = 0 automatically.  Below we")
    print("  give Berezin's category the benefit of that doubt and impose")
    print("  [x,x] = 0 by hand -- the failure in D. is not this cheap one.")
    Z = SAlg2(["c", "x"], [0, 1], {(1, 1): {0: 1}}, F2)
    Gz = grass3.envelope(grass3.SuperAlg(["c", "x"], [0, 1], {(1, 1): {0: 1}}, 2), 3)
    baz, bjz = grass3.check_ordinary(Gz)
    print("     check: its envelope (dim %d) is a Lie algebra: %s; and it has"
          % (Gz.n, "yes" if not baz and not bjz else "NO"))
    print("     [x,x] = %s, so it is no polarization." % Z.show(Z.br(Z.e(1), Z.e(1))))
    ok &= not baz and not bjz

    # ---------------------------------------------------------------- D ----
    hr("D.  Minimality of B: the smallest superalgebra with no squaring")

    print("  Exhaustive over F_2: all brackets with the axioms above, i.e. all")
    print("  objects of Berezin's category, of every sdim (m|n), m + n <= 4,")
    print("  n >= 1; for each, does a squaring exist?")
    print()
    print("   sdim   brackets  satisfy axioms   no squaring")
    first_bad = None
    for m, n in [(0, 1), (1, 1), (0, 2), (2, 1), (1, 2), (0, 3), (3, 1),
                 (2, 2), (1, 3), (0, 4)]:
        names = (["u%d" % (i + 1) for i in range(m)]
                 + ["x%d" % (i + 1) for i in range(n)])
        par = [0] * m + [1] * n
        tot = good = nosq = 0
        for table in enumerate_brackets(m, n):
            tot += 1
            A = SAlg2(names, par, table, F2)
            if A.check_bracket():
                continue
            good += 1
            if not all_squarings(A):
                nosq += 1
                if first_bad is None:
                    first_bad = (m, n, table)
        print("   (%d|%d) %9d %14d %13d" % (m, n, tot, good, nosq))
        ok &= (nosq == 0) if (m + n <= 2) else True
    print()
    print("  So every object of Berezin's category of dimension <= 2 carries a")
    print("  squaring; the first that does not has sdim (1|2), dimension 3 --")
    print("  and B is one of the 7.  That (m|1) is always safe is a theorem,")
    print("  not an accident of the census: if g_od = <x> then [x,u] is a")
    print("  multiple of x for every even u, so ad_x^2(u) = 0, and likewise")
    print("  ad_x^2 = 0 on g_od; hence s = 0 is a squaring.  So dim g_od >= 2")
    print("  and g_ev =/= 0 are both necessary, and (1|2) is the minimum.")
    ok &= first_bad is not None and sum(first_bad[:2]) == 3

    # ---------------------------------------------------------------- E ----
    hr("E.  Which axiom?  (11) and (14) of [KLLS], and where they differ")

    print("  [KLLS], Subsec. 3.2: over Z/2 the axiom [v^2,v] = 0 (11) must be")
    print("  replaced by the stronger [v^2,y] = [v,[v,y]] for all y odd (14);")
    print("  'for any other ground field this condition is equivalent to (11)'.")
    print()

    # E1.  Their Example 3.2.1 verified.
    K = alg_KLLS_example(F2)
    bad = K.check_bracket()
    sK = build_squaring(K, [K.e(0), K.e(1), K.zero()])     # X^2=A, Y^2=B, Z^2=0
    weak = squaring_defects(K, sK, strong=False)
    strong = squaring_defects(K, sK, strong=True)
    X, Y, Z = K.e(2), K.e(3), K.e(4)
    print("  E1. [KLLS] Example 3.2.1, sdim (2|3): bracket %s;" %
          ("ok" if not bad else "FAIL %s" % bad[:2]))
    print("      the squaring (aX+bY+cZ)^2 = a^2 A + b^2 B satisfies (11): %s"
          % ("yes" if not weak else "no, %s" % weak[:2]))
    print("      and violates (14): %s" % ("yes" if strong else "no"))
    print("      indeed [X^2, Y] = %s  while  [X,[X,Y]] = %s."
          % (K.show(K.br(sK[X], Y)), K.show(K.br(X, K.br(X, Y)))))
    ok &= (not bad) and (not weak) and bool(strong) \
        and K.br(sK[X], Y) != K.br(X, K.br(X, Y))

    # E2.  Over F_4 the two axioms agree -- the census run under both.
    print()
    print("  E2. the census of sdim (1|2) under each axiom:")
    print()
    print("        field   objects   no squaring under (11)   under (14)")
    for F in (F2, F4):
        names, par = ["u", "x", "y"], [0, 1, 1]
        good = weak_no = strong_no = 0
        for table in enumerate_brackets(1, 2):
            A = SAlg2(names, par, table, F)
            if A.check_bracket():
                continue
            good += 1
            if not all_squarings(A, strong=False):
                weak_no += 1
            if not all_squarings(A, strong=True):
                strong_no += 1
        print("        %-6s %8d %22d %12d" % (F.name, good, weak_no, strong_no))
        ok &= weak_no == strong_no          # at this size they still agree
    print()
    print("      (the same 24 brackets, read over F_2 and over F_4; at this")
    print("      size the two axioms still agree -- the first sdim at which")
    print("      they part is (1|3), see E5.)")
    print("      The one-line reason they agree unless K = F_2: put")
    print("      D(v,w) := [v^2,w] + [v,[v,w]].  Expanding [(v+w)^2, v+w] = 0")
    print("      with (v+w)^2 = v^2 + w^2 + [v,w] and Jacobi gives")
    print("      D(v,w) + D(w,v) = 0, and D(av,w) = a^2 D(v,w),")
    print("      D(w,av) = a D(w,v); so a^2 D(v,w) + a D(w,v) = 0 for every a,")
    print("      whence D = 0 as soon as some a has a^2 =/= a.")

    # the identity D(v,w) + D(w,v) = 0 itself, checked on every object
    viol = 0
    for F in (F2, F4):
        names, par = ["u", "x", "y"], [0, 1, 1]
        for table in enumerate_brackets(1, 2):
            A = SAlg2(names, par, table, F)
            if A.check_bracket():
                continue
            for _, s in all_squarings(A, strong=False):
                for v in A.odd_elements():
                    for w in A.odd_elements():
                        d1 = A.vadd(A.br(s[v], w), A.br(v, A.br(v, w)))
                        d2 = A.vadd(A.br(s[w], v), A.br(w, A.br(w, v)))
                        if any(A.vadd(d1, d2)):
                            viol += 1
    print("      D(v,w) + D(w,v) = 0 on every weak squaring of every object of")
    print("      sdim (1|2) over F_2 and F_4: %d violations." % viol)
    ok &= viol == 0

    # E3.  What this does to C, i.e. to [KLLS]'s own example.
    print()
    print("  E3. the algebra C = <u | [u,x] = x, [u,y] = y, [x,y] = u>:")
    for F in (F2, F4):
        C = alg_C(F)
        x, y = C.e(1), C.e(2)
        n_weak = len(all_squarings(C, strong=False))
        n_strong = len(all_squarings(C, strong=True))
        print("      over %s: squarings under (11): %d; under (14): %d"
              % (F.name, n_weak, n_strong))
        ok &= n_weak == 0 and n_strong == 0
    C = alg_C(F2)
    x, y, u = C.e(1), C.e(2), C.e(0)
    print("      ad_x^2 is not inner already as a map: ad_x^2(y) = %s, whereas"
          % C.show(C.br(x, C.br(x, y))))
    print("      [a u, y] = a y for every a -- so under (14) the value s(x)")
    print("      does not exist, and C fails for the SAME reason as B.")
    print("      Under the weak axiom (11) instead s(x) = s(y) = 0 is forced,")
    print("      s(x+y) = [x,y] = u, and [s(x+y), x+y] = %s =/= 0."
          % C.show(C.br(u, C.vadd(x, y))))
    ok &= C.br(x, C.br(x, y)) == x

    # E4.  C is [KLLS]'s example: the derived algebra of osp(1|2), p = 2.
    print()
    print("  E4. C is the example of [KLLS], Subsec. 3.3.4: the 'derived")
    print("      algebra ignoring the squaring' of osp(1|2) at p = 2.")
    O = alg_osp12(F2)
    badO = O.check_bracket()
    sO = osp12_squaring(O)
    defO = squaring_defects(O, sO, strong=True)
    span = set()
    for i in range(O.n):
        for j in range(O.n):
            v = O.br(O.e(i), O.e(j))
            if any(v):
                span.add(v)
    print("      osp(1|2) over F_2, sdim (3|2): bracket %s, and X^2 = e,"
          % ("ok" if not badO else "FAIL %s" % badO[:2]))
    print("      Y^2 = f is a squaring in the sense of (14): %s"
          % ("ok" if not defO else "FAIL %s" % defO[:2]))
    print("      Span{[a,b]} = {%s} -- i.e. <h, X, Y>, of sdim (1|2), with"
          % ", ".join(sorted(O.show(v) for v in span)))
    print("      [h,X] = X, [h,Y] = Y, [X,Y] = h: this is C.")
    print("      squarings on it: %d (their argument: X^2 would be an even z"
          % len(all_squarings(alg_C(F2), strong=True)))
    print("      with [z,Y] = X, and no such z exists).")
    ok &= (not badO) and (not defO) and \
        span == {O.e(1), O.e(3), O.e(4)}

    # E5.  and what it does to the census of part D
    print()
    print("  E5. part D redone under (14): sdim, objects, no squaring")
    print()
    print("       sdim   satisfy axioms   no squaring (11)   (14)")
    for m, n in [(0, 1), (1, 1), (0, 2), (2, 1), (1, 2), (0, 3), (3, 1),
                 (2, 2), (1, 3), (0, 4)]:
        names = (["u%d" % (i + 1) for i in range(m)]
                 + ["x%d" % (i + 1) for i in range(n)])
        par = [0] * m + [1] * n
        good = w_no = s_no = 0
        for table in enumerate_brackets(m, n):
            A = SAlg2(names, par, table, F2)
            if A.check_bracket():
                continue
            good += 1
            if not all_squarings(A, strong=False):
                w_no += 1
            if not all_squarings(A, strong=True):
                s_no += 1
        print("       (%d|%d) %13d %16d %8d" % (m, n, good, w_no, s_no))
        if m + n <= 2:
            ok &= s_no == 0
        if (m, n) == (1, 2):
            census12 = (good, w_no, s_no)
    print()
    print("      so dimension <= 2 is safe under either axiom, and (1|2) is")
    print("      still the least format in which a squaring can fail to exist;")
    print("      the count there is %d of %d under (11) and %d of %d under (14)."
          % (census12[1], census12[0], census12[2], census12[0]))

    # E6.  the smallest place where the choice of axiom changes the answer
    print()
    print("  E6. the least format in which (11) and (14) differ:")
    names, par = ["u", "x", "y", "z"], [0, 1, 1, 1]
    witness = None
    n_diff = n_map = 0
    first_map = None
    for table in enumerate_brackets(1, 3):
        A = SAlg2(names, par, table, F2)
        if A.check_bracket():
            continue
        w = all_squarings(A, strong=False)
        s = all_squarings(A, strong=True)
        if w and not s:
            n_diff += 1
            if witness is None:
                witness = (table, w[0][1])
        for _, sq in w:
            if squaring_defects(A, sq, strong=True):
                n_map += 1
                if first_map is None:
                    first_map = (table, sq)
                break
    print("      of the 736 objects of sdim (1|3) over F_2, %d carry a"
          % n_diff)
    print("      squaring in the sense of (11) but none in the sense of (14),")
    print("      and %d carry a map satisfying (11) but not (14)." % n_map)
    A = SAlg2(names, par, witness[0], F2)
    rel = []
    for i in range(A.n):
        for j in range(i, A.n):
            v = A.br(A.e(i), A.e(j))
            if any(v):
                rel.append("[%s,%s] = %s" % (A.names[i], A.names[j], A.show(v)))
    sq = witness[1]
    sqs = ", ".join("%s^2 = %s" % (A.names[i], A.show(sq[A.e(i)]))
                    for i in A.od)
    print("      one witness: <u even; x, y, z odd | %s>" % ", ".join(rel))
    print("      with the (11)-squaring %s;" % sqs)
    v = A.e(1)
    for w in A.odd_elements():
        if A.br(sq[v], w) != A.br(v, A.br(v, w)):
            print("      it fails (14) at [x^2, %s] = %s =/= %s = [x,[x,%s]]."
                  % (A.show(w), A.show(A.br(sq[v], w)),
                     A.show(A.br(v, A.br(v, w))), A.show(w)))
            break
    ok &= n_diff > 0 and witness is not None

    hr("VERDICT")
    print("  At p = 2 the Grassmann envelope (the Even Rules) determines the")
    print("  bracket and nothing more.  Hence Berezin's definition")
    print("    (i)  does not determine a Lie superalgebra: A. shows two")
    print("         non-isomorphic ones with the same envelope; and")
    print("    (ii) is strictly weaker than [BLLS] 2.2: B. is admitted by it")
    print("         and carries no squaring at all.")
    print("  Together with grass3.py (p = 3, where the failure is only of type")
    print("  (ii) and only for one identity), this is the precise sense in which")
    print("  the Even Rules fail for p = 2 and p = 3.")
    print()
    print("  ALL CHECKS PASSED" if ok else "  *** SOMETHING FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
