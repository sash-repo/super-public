#!/usr/bin/env python3
"""
dereven.py -- Problem (5) of the text (Subsec. 1.4) IN THE PRESENCE OF EVEN VARIABLES.

    What is the Lie algebra corresponding to ALL terms of eq. (13)
    and functorially depending on Lambda?

`dermeta.py' answered this for n = 0 even variables.  Here n > 0 is allowed:

    B  := K[x_1, ..., x_n] (x) Lambda(theta_1, ..., theta_m) (x) Lambda,
    d_{n|m}(Lambda) := { D in Der(B) | D(Lambda) = 0 },

Der taken in the ASSOCIATIVE sense, D(bc) = D(b)c + bD(c) with NO Koszul signs
-- the only rule a parity-mixing map obeys (Bavula's notion).

The point of the even variables: a derivation may now send an EVEN generator
x_i to an ODD element, which no inner derivation can do (x_i is central).  So a
new, non-inner piece of statistics-mixing symmetry appears, and it turns out to
be exactly the infinitesimal form of the doubly underlined factor
(1 + F_i theta_1 ... theta_m p(m)) of eq. (13) of the paper -- including the
parity factor p(m), which comes out of the computation rather than being put in.

Everything is exact (exactlin.py: FLINT when available, pure Python else).

Universality: Lambda = Lambda(N) suffices, every finitely generated
supercommutative superalgebra being a quotient of a polynomial algebra tensored
with one of these; the polynomial part of Lambda only enlarges K[x].

Run:  python3 dereven.py     (faster with python-flint installed)
"""

import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "."))

from exactlin import backend, nullspace, rank            # noqa: E402


# --------------------------------------------------------------------- algebra

class Alg:
    """K[x_1..x_n] (x) Lambda(theta_1..theta_m) (x) Lambda(xi_1..xi_N).

    A basis monomial is a pair (alpha, mask): alpha a tuple of n exponents,
    mask a bitmask over the m+N odd generators (thetas first, then xis).
    An element is a dict {(alpha, mask): coefficient}.  Products of x-degree
    above `deg' are truncated away; every computation below is arranged so
    that no term of degree <= deg is ever lost.
    """

    def __init__(self, n, m, N, deg):
        self.n, self.m, self.N, self.deg = n, m, N, deg
        self.nodd = m + N
        self.top = (1 << self.nodd) - 1          # the top odd monomial

    # --- monomials --------------------------------------------------------

    def xmons(self, deg=None):
        deg = self.deg if deg is None else deg
        out = []
        for alpha in product(range(deg + 1), repeat=self.n):
            if sum(alpha) <= deg:
                out.append(alpha)
        return out

    def basis(self, deg=None, parity=None):
        out = []
        for alpha in self.xmons(deg):
            for mask in range(1 << self.nodd):
                if parity is not None and bin(mask).count("1") % 2 != parity:
                    continue
                out.append((alpha, mask))
        return out

    def par(self, key):
        return bin(key[1]).count("1") % 2

    # --- arithmetic -------------------------------------------------------

    def sign(self, A, B):
        """Koszul sign of (odd monomial A)(odd monomial B), masks disjoint."""
        s = 1
        for b in range(self.nodd):
            if (B >> b) & 1:
                s *= (-1) ** bin(A >> (b + 1)).count("1")
        return s

    def mul(self, u, v):
        out = {}
        for (a1, M1), c1 in u.items():
            for (a2, M2), c2 in v.items():
                if M1 & M2:
                    continue
                alpha = tuple(i + j for i, j in zip(a1, a2))
                if sum(alpha) > self.deg:
                    continue                      # harmless truncation
                c = self.sign(M1, M2) * c1 * c2
                if c:
                    k = (alpha, M1 | M2)
                    out[k] = out.get(k, 0) + c
        return {k: c for k, c in out.items() if c}

    def add(self, u, v, t=1):
        out = dict(u)
        for k, c in v.items():
            out[k] = out.get(k, 0) + t * c
        return {k: c for k, c in out.items() if c}

    def x(self, i):
        alpha = [0] * self.n
        alpha[i] = 1
        return {(tuple(alpha), 0): 1}

    def theta(self, j):
        return {((0,) * self.n, 1 << j): 1}

    def xi(self, a):
        return {((0,) * self.n, 1 << (self.m + a)): 1}

    def one(self):
        return {((0,) * self.n, 0): 1}


# --------------------------------------------------------- derivations, by hand

def derivations(A, hdeg, parity=None):
    """Basis of { D in Der(B) | D(Lambda) = 0 }, D given by (h_i; k_j).

    h_i = D(x_i), k_j = D(theta_j), all of x-degree <= hdeg.

    parity = 0 : D preserves parity (h_i even, k_j odd)  -- "even" derivations
    parity = 1 : D reverses parity  (h_i odd,  k_j even) -- statistics-mixing
    parity = None : no restriction.
    """
    n, m, N = A.n, A.m, A.N

    cols = []
    for i in range(n):
        for key in A.basis(deg=hdeg,
                           parity=None if parity is None else parity):
            cols.append(("h", i, key))
    for j in range(m):
        for key in A.basis(deg=hdeg,
                           parity=None if parity is None else 1 - parity):
            cols.append(("k", j, key))
    idx = {c: i for i, c in enumerate(cols)}
    allkeys = A.basis()

    rows = []

    def emit(terms):
        """terms: (kind, index, left, right, sign), meaning sign * left u right.

        Applying D to a defining relation  sum_r  eps_r u_r v_r = 0  of B gives
        sum_r eps_r (D(u_r) v_r + u_r D(v_r)) = 0, one linear equation per basis
        monomial of B; the signs eps_r are what distinguishes a commutator
        relation from an anticommutator one and must not be dropped.
        """
        block = {}
        for (kind, ind, uL, uR, eps) in terms:
            for c in cols:
                if c[0] != kind or c[1] != ind:
                    continue
                key = c[2]
                val = A.mul(A.mul(uL, {key: 1}), uR)
                for k, cc in val.items():
                    block.setdefault(k, [0] * len(cols))[idx[c]] += eps * cc
        for r in block.values():
            if any(r):
                rows.append(r)

    one = A.one()
    # x_i x_j - x_j x_i = 0
    for i in range(n):
        for j in range(i, n):
            emit([("h", i, one, A.x(j), +1), ("h", j, A.x(i), one, +1),
                  ("h", j, one, A.x(i), -1), ("h", i, A.x(j), one, -1)])
    # x_i theta_j - theta_j x_i = 0
    for i in range(n):
        for j in range(m):
            emit([("h", i, one, A.theta(j), +1), ("k", j, A.x(i), one, +1),
                  ("k", j, one, A.x(i), -1), ("h", i, A.theta(j), one, -1)])
    # x_i xi_a - xi_a x_i = 0     (D kills xi_a)
    for i in range(n):
        for a in range(N):
            emit([("h", i, one, A.xi(a), +1), ("h", i, A.xi(a), one, -1)])
    # theta_i theta_j + theta_j theta_i = 0
    for i in range(m):
        for j in range(i, m):
            emit([("k", i, one, A.theta(j), +1), ("k", j, A.theta(i), one, +1),
                  ("k", j, one, A.theta(i), +1), ("k", i, A.theta(j), one, +1)])
    # theta_j xi_a + xi_a theta_j = 0
    for j in range(m):
        for a in range(N):
            emit([("k", j, one, A.xi(a), +1), ("k", j, A.xi(a), one, +1)])

    if not rows:
        rows = [[0] * len(cols)]
    ns = nullspace(rows)

    out = []
    for v in ns:
        hs = [dict() for _ in range(n)]
        ks = [dict() for _ in range(m)]
        for c, val in zip(cols, v):
            if not val:
                continue
            (kind, ind, key) = c
            (hs if kind == "h" else ks)[ind][key] = val
        out.append((tuple(hs), tuple(ks)))
    return out


# ------------------------------------------------------------------- the models

def apply_der(A, D, u):
    """Apply the derivation D = (h_i; k_j) to an element u, no Koszul signs.

    A monomial is written as the ordered word x^alpha theta_T xi_S and the
    product rule is applied to each letter in turn; the xis are killed.
    """
    hs, ks = D
    out = {}
    for (alpha, mask), c in u.items():
        # the letters, in order: x_1^{a_1} ... x_n^{a_n}, then the odd ones
        letters = []
        for i, a in enumerate(alpha):
            letters.extend([("x", i)] * a)
        for b in range(A.nodd):
            if (mask >> b) & 1:
                letters.append(("t", b))
        for pos, (kind, ind) in enumerate(letters):
            if kind == "t" and ind >= A.m:
                continue                            # D(xi) = 0
            d = hs[ind] if kind == "x" else ks[ind]
            if not d:
                continue
            left, right = A.one(), A.one()
            for (kk, ii) in letters[:pos]:
                left = A.mul(left, A.x(ii) if kk == "x" else
                             ({((0,) * A.n, 1 << ii): 1}))
            for (kk, ii) in letters[pos + 1:]:
                right = A.mul(right, A.x(ii) if kk == "x" else
                              ({((0,) * A.n, 1 << ii): 1}))
            for k, cc in A.mul(A.mul(left, d), right).items():
                out[k] = out.get(k, 0) + c * cc
    return {k: v for k, v in out.items() if v}


def bracket(A, D1, D2):
    """D1 D2 - D2 D1, as a derivation given by its values on the generators."""
    hs = tuple(A.add(apply_der(A, D1, D2[0][i]), apply_der(A, D2, D1[0][i]), -1)
               for i in range(A.n))
    ks = tuple(A.add(apply_der(A, D1, D2[1][j]), apply_der(A, D2, D1[1][j]), -1)
               for j in range(A.m))
    return (hs, ks)


def compose(A, D1, D2):
    """The (generally non-derivation) operator D1 D2, on the generators."""
    hs = tuple(apply_der(A, D1, D2[0][i]) for i in range(A.n))
    ks = tuple(apply_der(A, D1, D2[1][j]) for j in range(A.m))
    return (hs, ks)


def flat(A, D):
    keys = A.basis()
    v = []
    for h in D[0] + D[1]:
        v.extend([h.get(k, 0) for k in keys])
    return v


def dim_span(A, fam):
    rows = [flat(A, D) for D in fam]
    return rank(rows) if rows else 0


def same_span(A, F1, F2):
    r1, r2 = dim_span(A, F1), dim_span(A, F2)
    rows = [flat(A, D) for D in F1] + [flat(A, D) for D in F2]
    return r1 == r2 == (rank(rows) if rows else 0)


# ---------------------------------------------------------------- the predicted

def socle(A, hdeg):
    """Ann_B(B_od) = { b in B | b B_od = 0 }, computed, not assumed."""
    odd = [{k: 1} for k in A.basis(parity=1)]
    out = []
    for key in A.basis(deg=hdeg):
        b = {key: 1}
        if all(not A.mul(b, o) for o in odd):
            out.append(b)
    return out


def vect_even(A, hdeg):
    """(vect(n|m) (x) Lambda)_ev : h_i even, k_j odd, unconstrained."""
    out = []
    for i in range(A.n):
        for key in A.basis(deg=hdeg, parity=0):
            hs = [dict() for _ in range(A.n)]
            ks = [dict() for _ in range(A.m)]
            hs[i] = {key: 1}
            out.append((tuple(hs), tuple(ks)))
    for j in range(A.m):
        for key in A.basis(deg=hdeg, parity=1):
            hs = [dict() for _ in range(A.n)]
            ks = [dict() for _ in range(A.m)]
            ks[j] = {key: 1}
            out.append((tuple(hs), tuple(ks)))
    return out


def socle_fields(A, hdeg):
    """b_{n}(Lambda) = { sum f_i del_{x_i} | f_i odd, in the socle }.

    This is the piece that only even variables can carry: it is NOT inner.
    """
    out = []
    for s in socle(A, hdeg):
        if A.par(next(iter(s))) != 1:
            continue
        for i in range(A.n):
            hs = [dict() for _ in range(A.n)]
            ks = [dict() for _ in range(A.m)]
            hs[i] = s
            out.append((tuple(hs), tuple(ks)))
    return out


def inner(A, hdeg):
    """{ ad_g | g odd, ad_g(Lambda) = 0 }: h_i = 0, k_j = g theta_j - theta_j g."""
    out = []
    for key in A.basis(deg=hdeg, parity=1):
        g = {key: 1}
        if any(A.add(A.mul(g, A.xi(a)), A.mul(A.xi(a), g), -1)
               for a in range(A.N)):
            continue                                 # ad_g must kill Lambda
        hs = tuple(dict() for _ in range(A.n))       # ad_g(x_i) = 0 always
        ks = tuple(A.add(A.mul(g, A.theta(j)), A.mul(A.theta(j), g), -1)
                   for j in range(A.m))
        out.append((hs, ks))
    return out


# ------------------------------------------------------------------- the checks

def report(n, m, N, hdeg=1):
    A = Alg(n, m, N, hdeg + 1)
    allD = derivations(A, hdeg)
    evD = derivations(A, hdeg, parity=0)
    odD = derivations(A, hdeg, parity=1)

    V = vect_even(A, hdeg)
    Bf = socle_fields(A, hdeg)
    I = inner(A, hdeg)

    ok = True
    ok &= len(allD) == len(evD) + len(odD)
    ok &= same_span(A, V, evD)
    ok &= same_span(A, Bf + I, odD)

    print("  n=%d m=%d N=%d deg<=%d :  dim d = %3d = %3d (preserving) + %3d (mixing)"
          % (n, m, N, hdeg, len(allD), len(evD), len(odD)))
    print("        (vect(n|m) (x) L)_ev = %3d %s ;  socle fields = %3d ;  inner = %3d ;"
          "  together %s"
          % (dim_span(A, V), "OK" if same_span(A, V, evD) else "MISMATCH",
             dim_span(A, Bf), dim_span(A, I),
             "OK" if same_span(A, Bf + I, odD) else "MISMATCH"))
    return ok


def is_mixing(A, D):
    """Is D a statistics-mixing element of d_{n|m}(Lambda)?

    Membership is tested against the DEFINING constraints, not against the span
    of a basis truncated in the x-degree: a bracket raises that degree, and a
    span test in a truncated basis would report a false failure.  The
    constraints are the ones the linear system imposes:
        h_i odd and h_i B_od = 0            (h_i lies in the socle)
        k_j even, k_j Lambda_od = 0, and  k_i theta_j + k_j theta_i = 0.
    """
    hs, ks = D
    for h in hs:
        if any(A.par(k) != 1 for k in h):
            return False
        for o in A.basis(parity=1):
            if A.mul(h, {o: 1}):
                return False
    for k in ks:
        if any(A.par(key) != 0 for key in k):
            return False
        for a in range(A.N):
            if A.mul(k, A.xi(a)):
                return False
    for i in range(A.m):
        for j in range(A.m):
            if A.add(A.mul(ks[i], A.theta(j)), A.mul(ks[j], A.theta(i))):
                return False
    return True


def structure(n, m, N, hdeg=1):
    """The mixing part is an abelian ideal, and it squares to zero."""
    # a composite or a bracket of two degree-<= hdeg objects has degree
    # <= 2*hdeg, and the membership test multiplies by one more generator:
    # truncating at 2*hdeg+1 loses nothing and can mask no nonzero term.
    A = Alg(n, m, N, 2 * hdeg + 1)
    odD = derivations(A, hdeg, parity=1)
    evD = derivations(A, hdeg, parity=0)
    bad_ab = bad_sq = bad_id = 0

    for X in odD:
        for Y in odD:
            if any(any(c) for c in bracket(A, X, Y)[0] + bracket(A, X, Y)[1]):
                bad_ab += 1
            C = compose(A, X, Y)
            if any(c for c in C[0] + C[1]):
                bad_sq += 1
    for D in evD:
        for X in odD:
            if not is_mixing(A, bracket(A, D, X)):
                bad_id += 1
    print("  n=%d m=%d N=%d : abelian %s ; squares to zero %s ; ideal %s"
          % (n, m, N,
             "OK" if not bad_ab else "%d FAIL" % bad_ab,
             "OK" if not bad_sq else "%d FAIL" % bad_sq,
             "OK" if not bad_id else "%d FAIL" % bad_id))
    return not (bad_ab or bad_sq or bad_id)


def triviality(n, m, N, hdeg=1):
    """S(a,b) = [[a,x],[b,x]] for x mixing, a, b preserving: never nonzero."""
    # S has x-degree <= 4*hdeg; truncating below that could mask a nonzero S
    # and turn a real witness into a false pass.
    A = Alg(n, m, N, 4 * hdeg)
    odD = derivations(A, hdeg, parity=1)
    evD = derivations(A, hdeg, parity=0)
    worst = 0
    for x in odD:
        for a in evD:
            ax = bracket(A, a, x)
            for b in evD:
                bx = bracket(A, b, x)
                S = bracket(A, ax, bx)
                if any(c for c in S[0] + S[1]):
                    worst += 1
    print("  n=%d m=%d N=%d : witnesses of nontriviality among %d x %d x %d : %d"
          % (n, m, N, len(odD), len(evD), len(evD), worst))
    return worst == 0


def socle_shape(n, m, N, hdeg=1):
    """Ann_B(B_od) = K[x] . theta_1...theta_m . xi_1...xi_N -- and its parity."""
    A = Alg(n, m, N, hdeg + 1)
    S = socle(A, hdeg)
    masks = sorted({next(iter(s))[1] for s in S})
    pred = [A.top] if A.nodd else [0]
    odd = (m + N) % 2 == 1
    got_odd = any(A.par(next(iter(s))) == 1 for s in S)
    ok = masks == pred and got_odd == odd
    print("  n=%d m=%d N=%d : socle carried by %s (predicted %s), odd: %s "
          "(predicted %s) %s"
          % (n, m, N, masks, pred, got_odd, odd, "OK" if ok else "MISMATCH"))
    return ok


def eq4_match(n, m, N):
    """The shift x_i |-> x_i(1 + F theta_1...theta_m) of eq. (13): when is it
    an automorphism of B over Lambda, and does it match the socle answer?"""
    A = Alg(n, m, N, 2)
    ok = True
    # the F_i of eq. (13) are ODD superfields; an even F would give an ordinary,
    # statistics-preserving automorphism, which is not what is at stake
    for key in A.basis(deg=0, parity=1):
        F = {key: 1}
        incr = A.mul(F, {((0,) * n, (1 << m) - 1): 1})   # F theta_1...theta_m
        if not incr:
            continue
        img = [A.add(A.x(i), A.mul(A.x(i), incr)) for i in range(n)]
        good = True
        for i in range(n):
            for j in range(n):
                if A.add(A.mul(img[i], img[j]), A.mul(img[j], img[i]), -1):
                    good = False
            for j in range(m):
                if A.add(A.mul(img[i], A.theta(j)),
                         A.mul(A.theta(j), img[i]), -1):
                    good = False
            for a in range(N):
                if A.add(A.mul(img[i], A.xi(a)), A.mul(A.xi(a), img[i]), -1):
                    good = False
        # the infinitesimal prediction: x_i F theta_1..theta_m is in the socle
        insoc = all(not A.mul(A.mul(A.x(i), incr), {k: 1})
                    for i in range(n) for k in A.basis(deg=0, parity=1))
        if good != insoc:
            ok = False
    print("  n=%d m=%d N=%d : x_i |-> x_i(1 + F theta_1..theta_m) is an "
          "automorphism exactly when the increment lies in the socle : %s"
          % (n, m, N, "OK" if ok else "MISMATCH"))
    return ok


def table():
    """The dimensions per x-monomial (hdeg = 0), i.e. as K[x]-modules.

    Everything in sight is a free K[x]-module: the defining relations of B are
    x-homogeneous, so the linear system splits by the degree in the x's and the
    numbers below simply get multiplied by the number of monomials.
    """
    print("   n  m  N | dim d | (vect(n|m)(x)L)_ev | socle fields | inner")
    ok = True
    for (n, m, N) in [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (2, 0, 1),
                      (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (2, 1, 1),
                      (1, 3, 0), (2, 2, 1)]:
        A = Alg(n, m, N, 1)
        allD = derivations(A, 0)
        V, Bf, I = vect_even(A, 0), socle_fields(A, 0), inner(A, 0)
        dv, db, di = dim_span(A, V), dim_span(A, Bf), dim_span(A, I)
        good = len(allD) == dv + db + di
        ok &= good
        print("  %2d %2d %2d | %5d | %18d | %12d | %5d  %s"
              % (n, m, N, len(allD), dv, db, di, "OK" if good else "MISMATCH"))
    return ok


def main():
    print("backend:", backend())
    print()
    print("0) the dimensions per x-monomial")
    if not table():
        print("*** SOMETHING FAILED ***")
        return 1
    print()
    print("1) the structure of d_{n|m}(Lambda) = { D in Der(B) : D(Lambda) = 0 }")
    ok = True
    for (n, m, N) in [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (2, 0, 1),
                      (2, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 1),
                      (1, 2, 1), (1, 2, 2), (2, 2, 1), (1, 3, 0)]:
        ok &= report(n, m, N)
    print()
    print("2) the statistics-mixing part is an abelian ideal squaring to zero")
    for (n, m, N) in [(1, 0, 1), (1, 0, 3), (2, 0, 1), (1, 1, 0), (1, 1, 2),
                      (1, 2, 1), (2, 1, 1)]:
        ok &= structure(n, m, N)
    print()
    print("3) no Volichenko algebra arises: S(a,b) = [[a,x],[b,x]] = 0")
    for (n, m, N) in [(1, 0, 1), (2, 0, 1), (1, 1, 0), (1, 1, 2), (1, 2, 1)]:
        ok &= triviality(n, m, N)
    print()
    print("4) Ann_B(B_od) is the top monomial, odd iff m+N is odd")
    for (n, m, N) in [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 0),
                      (1, 1, 1), (1, 2, 0), (1, 2, 1), (2, 3, 0), (2, 2, 2)]:
        ok &= socle_shape(n, m, N)
    print()
    print("5) the doubly underlined factor of eq. (13) and the socle agree")
    for (n, m, N) in [(1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (2, 3, 0),
                      (1, 3, 1), (2, 2, 1)]:
        ok &= eq4_match(n, m, N)
    print()
    print("ALL CHECKS PASSED" if ok else "*** SOMETHING FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
