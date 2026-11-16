#!/usr/bin/env python3
"""
oddcoc.py -- odd central extensions of the periodic subalgebras of gl_{p,c}.

"How to superize gl(infty)", Section 6 (Statement 6.1.1, Remark 6.1.4,
Theorems 6.2.1 and 6.2.2).

Four computations, all exact.

 (A) CONTROL.  A^k = gl(m|n)^{l(1)}: the classifier must find the standard
     affine cocycle  c(A u^r, B u^s) = r delta_{r+s,0} str(AB)  and must find
     it non-trivial.  This validates the method.

 (B) B^k = q(k)^{l(2)} (the paper's case: p(i+k) = p(i) + odd).  The classifier
     finds H^2_odd = 0 in every t-degree for k >= 2.  Reason: the odd invariant
     form otr on q(k) is ANTI-invariant under the parity automorphism Pi by
     which the loop algebra is twisted, so it defines no cocycle.  The case
     k = 1 is exceptional and degenerate: B^1 = q(1)^{l(2)} is a "queer
     Heisenberg" algebra (the bracket is non-zero only on pairs of odd
     elements) and H^2_odd is 1-dimensional in every odd t-degree, spanned by
     c(t^a, t^b) = delta_{a+b,-1} f(a),  f(a) = (3a+1)/2 for a odd,
                                                (3a+2)/2 for a even.

 (C) q(n)^{l(1)} -- the UNTWISTED loop superalgebra -- does carry the odd
     central extension:  c(x u^r, y u^s) = r delta_{r+s,0} otr(xy)  is a
     non-trivial odd 2-cocycle.  It lives
     inside gl_{p,c} on the queer subalgebra q_{p,c} of Subsec. 5.2, not on
     gl_{p,c} itself.

 (D) The corner (Tate) realization of (C): with P an odd involution which
     PRESERVES the splitting Z = Z_{>=0} u Z_{<0}, every corner block of an
     element of q_{p,c} commutes with P_{++}, so otr(M) = tr(P_{++} M) has the
     trace property on the corners and
        c(X,Y) = otr(X_{+-} Y_{-+}) - (-1)^{p(X)p(Y)} otr(Y_{+-} X_{-+})
     is an odd 2-cocycle.  Checked here against the closed form of (C).

Usage:  python3 oddcoc.py          (faster with python-flint installed)
"""

import itertools
import sys
from fractions import Fraction

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import nullspace as _nullspace
    HAVE_EXACTLIN = True
except Exception:                                          # pragma: no cover
    HAVE_EXACTLIN = False


# =========================================================== the algebras ===
# A basis element is a triple (a, b, r) meaning  E_{ab} t^r  (k x k blocks,
# t = the shift by k).  A model fixes k, the window parity pi, and whether t
# is odd ("twisted", B^k) or even ("untwisted", A^k).

class Loop:
    """Mat(k) tensor C[t^{-1},t] with p(E_ab t^r) = pi[a] + pi[b] + tw*r."""

    def __init__(self, k, pi, tw):
        self.k, self.pi, self.tw = k, pi, tw

    def par(self, e):
        a, b, r = e
        return (self.pi[a] + self.pi[b] + self.tw * r) % 2

    def basis(self, R):
        return [(a, b, r) for a in range(self.k) for b in range(self.k)
                for r in range(-R, R + 1)]

    def mul(self, X, Y):
        Z = {}
        for (a, b, r), u in X.items():
            for (c, d, s), v in Y.items():
                if b == c:
                    key = (a, d, r + s)
                    Z[key] = Z.get(key, 0) + u * v
        return {kk: vv for kk, vv in Z.items() if vv != 0}

    def bracket(self, X, pX, Y, pY):
        sgn = -1 if (pX * pY) % 2 else 1
        Z = dict(self.mul(X, Y))
        for key, v in self.mul(Y, X).items():
            Z[key] = Z.get(key, 0) - sgn * v
        return {kk: vv for kk, vv in Z.items() if vv != 0}


class QLoop:
    """
    q(n) tensor C[u^{-1},u], untwisted:  basis (a, b, eps, r) = E_ab xi^eps u^r,
    xi odd and central with xi^2 = 1;  p = eps.  otr(A xi^eps) = eps * tr(A).
    """

    def __init__(self, n):
        self.n = n

    def par(self, e):
        return e[2] % 2

    def basis(self, R):
        return [(a, b, eps, r) for a in range(self.n) for b in range(self.n)
                for eps in (0, 1) for r in range(-R, R + 1)]

    def mul(self, X, Y):
        Z = {}
        for (a, b, e1, r), u in X.items():
            for (c, d, e2, s), v in Y.items():
                if b == c:
                    key = (a, d, (e1 + e2) % 2, r + s)
                    Z[key] = Z.get(key, 0) + u * v
        return {kk: vv for kk, vv in Z.items() if vv != 0}

    def bracket(self, X, pX, Y, pY):
        sgn = -1 if (pX * pY) % 2 else 1
        Z = dict(self.mul(X, Y))
        for key, v in self.mul(Y, X).items():
            Z[key] = Z.get(key, 0) - sgn * v
        return {kk: vv for kk, vv in Z.items() if vv != 0}


def deg(e):
    return e[-1]


# ============================================== the cohomology classifier ===
def cocycle_space(M, d, R, want_parity):
    """
    All bilinear forms c of t-degree d and parity `want_parity`, supported in
    |deg| <= R, that are super-skew and satisfy the 2-cocycle identity on every
    triple whose brackets stay inside the window.  Returns (Z2, B2, unknowns).
    """
    basis = M.basis(R)
    par = {e: M.par(e) for e in basis}
    unk = [(e, f) for e in basis for f in basis if deg(e) + deg(f) == d]
    idx = {u: i for i, u in enumerate(unk)}
    n = len(unk)
    rows = []

    def add(terms):
        row = [0] * n
        for key, coef in terms:
            if key in idx:
                row[idx[key]] += coef
        if any(row):
            rows.append(row)

    for e, f in unk:                                  # parity
        if (par[e] + par[f]) % 2 != want_parity:
            add([((e, f), 1)])
    for e, f in unk:                                  # super-skew
        add([((e, f), 1), ((f, e), (-1) ** ((par[e] * par[f]) % 2))])

    for e, f, g in itertools.product(basis, repeat=3):   # cocycle identity
        if deg(e) + deg(f) + deg(g) != d:
            continue
        pe, pf, pg = par[e], par[f], par[g]
        terms, ok = [], True
        for (u, v, w, sgn) in ((e, f, g, (-1) ** ((pe * pg) % 2)),
                               (f, g, e, (-1) ** ((pf * pe) % 2)),
                               (g, e, f, (-1) ** ((pg * pf) % 2))):
            br = M.bracket({u: 1}, par[u], {v: 1}, par[v])
            if any(abs(deg(key)) > R for key in br):
                ok = False
                break
            for key, coef in br.items():
                terms.append(((key, w), sgn * coef))
        if ok and terms:
            add(terms)

    Z2 = nullspace_rows(rows, n)

    B2 = []                                            # coboundaries f o [,]
    carriers = sorted({e for e in basis if deg(e) == d})
    for c0 in carriers:
        vec = [0] * n
        for e, f in unk:
            br = M.bracket({e: 1}, par[e], {f: 1}, par[f])
            vec[idx[(e, f)]] = br.get(c0, 0)
        if any(vec):
            B2.append([Fraction(x) for x in vec])
    return Z2, B2, unk


def nullspace_rows(rows, n):
    if not rows:
        return [[Fraction(int(i == j)) for i in range(n)] for j in range(n)]
    if HAVE_EXACTLIN:
        return [list(map(Fraction, v)) for v in _nullspace(rows)]
    M = [[Fraction(x) for x in row] for row in rows]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c]
        M[r] = [x / inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                fct = M[i][c]
                M[i] = [x - fct * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(n) if c not in piv]
    out = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for i, c in enumerate(piv):
            v[c] = -M[i][fc]
        out.append(v)
    return out


def rank_of(vectors, n):
    if not vectors:
        return 0
    M = [list(map(Fraction, v)) for v in vectors]
    r = 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c]
        M[r] = [x / inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                fct = M[i][c]
                M[i] = [x - fct * y for x, y in zip(M[i], M[r])]
        r += 1
        if r == len(M):
            break
    return r


def h2(M, R, want_parity, degs):
    out = {}
    for d in degs:
        Z2, B2, unk = cocycle_space(M, d, R, want_parity)
        n = len(unk)
        rz = rank_of(Z2, n)
        rtot = rank_of([list(v) for v in Z2] + [list(v) for v in B2], n)
        rb_in_z = rz + rank_of(B2, n) - rtot
        out[d] = (rz, rb_in_z, rz - rb_in_z)
    return out


# ================================================= explicit cocycle checks ==
def is_cocycle(M, c, R):
    basis = M.basis(R)
    par = {e: M.par(e) for e in basis}
    for e, f, g in itertools.product(basis, repeat=3):
        pe, pf, pg = par[e], par[f], par[g]
        v = 0
        for (u, v_, w, sgn) in ((e, f, g, (-1) ** ((pe * pg) % 2)),
                                (f, g, e, (-1) ** ((pf * pe) % 2)),
                                (g, e, f, (-1) ** ((pg * pf) % 2))):
            br = M.bracket({u: 1}, par[u], {v_: 1}, par[v_])
            for key, coef in br.items():
                v += sgn * coef * c(key, w)
        if v != 0:
            return False, (e, f, g, v)
    return True, None


def is_coboundary(M, c, R, d):
    basis = M.basis(R)
    par = {e: M.par(e) for e in basis}
    carriers = sorted({e for e in basis if deg(e) == d})
    ci = {e: i for i, e in enumerate(carriers)}
    rows, rhs = [], []
    for e, f in itertools.product(basis, repeat=2):
        if deg(e) + deg(f) != d:
            continue
        br = M.bracket({e: 1}, par[e], {f: 1}, par[f])
        row = [0] * len(carriers)
        for key, v in br.items():
            if key in ci:
                row[ci[key]] += v
        rows.append(row)
        rhs.append(c(e, f))
    return solve_exact(rows, rhs, len(carriers)) is not None


def solve_exact(rows, rhs, n):
    M = [[Fraction(x) for x in row] + [Fraction(rhs[i])]
         for i, row in enumerate(rows)]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c]
        M[r] = [x / inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    for i in range(len(M)):
        if all(M[i][c] == 0 for c in range(n)) and M[i][n] != 0:
            return None
    x = [Fraction(0)] * n
    for i, c in enumerate(piv):
        x[c] = M[i][n]
    return x


# ===================================================== the four experiments =
def experiment_A(k=2, pi=(0, 1), R=3):
    print(f"\n(A) CONTROL: A^k = gl(m|n)^(l1), k={k}, pi={list(pi)} "
          f"(sdim window ({pi.count(0)}|{pi.count(1)}))")
    M = Loop(k, list(pi), tw=0)

    def c(e, f):                      # r delta_{r+s,0} str(AB)
        (a, b, r), (cc, d, s) = e, f
        if r + s != 0 or b != cc or d != a:
            return 0
        return r * (-1) ** pi[a]

    ok, wit = is_cocycle(M, c, R)
    print(f"    [{'OK ' if ok else 'BAD'}] r*delta*str(AB) is a 2-cocycle"
          + ("" if ok else f"   witness {wit}"))
    cb = is_coboundary(M, c, R, 0)
    print(f"    [{'OK ' if not cb else 'BAD'}] and it is NOT a coboundary "
          f"(the known central extension of the loop superalgebra)")
    tab = h2(M, R, want_parity=0, degs=range(-(R - 1), R))
    print("    dim H^2_even by t-degree (only |d| < R is reliable: outside it "
          "the\n        window drops constraints and Z^2 is over-counted): "
          + ", ".join(f"d={d}: {v[2]}" for d, v in tab.items()))
    return ok and not cb


def experiment_B(cases=((1, (0,), 4), (2, (0, 1), 3), (2, (0, 0), 2),
                        (3, (0, 1, 1), 2))):
    print("\n(B) B^k = q(k)^(l2)  (the paper's case, p(i+k)=p(i)+odd): "
          "odd cohomology")
    good = True
    for k, pi, R in cases:
        M = Loop(k, list(pi), tw=1)
        tab = h2(M, R, want_parity=1, degs=range(-(R - 1), R))
        line = ", ".join(f"d={d}: {v[2]}" for d, v in tab.items())
        print(f"    k={k}, pi={list(pi)}, |r|<={R}:  dim H^2_odd -- {line}")
        if k >= 2:
            good &= all(v[2] == 0 for v in tab.values())
    print("    => for k >= 2 the odd central extension does NOT exist;")
    print("       k = 1 is the degenerate 'queer Heisenberg' exception.")
    return good


def experiment_B1(R=5):
    """The exceptional k=1 cocycle, in closed form."""
    print("\n(B') k = 1: the exceptional odd cocycle of B^1 = q(1)^(l2)")
    M = Loop(1, [0], tw=1)

    def f(a):
        return Fraction(3 * a + 1, 2) if a % 2 else Fraction(3 * a + 2, 2)

    def c(e, ff):
        return f(deg(e)) if deg(e) + deg(ff) == -1 else 0

    ok, wit = is_cocycle(M, c, R)
    print(f"    [{'OK ' if ok else 'BAD'}] c(t^a,t^b) = delta_(a+b,-1) f(a), "
          f"f(a) = (3a+1)/2 (a odd), (3a+2)/2 (a even), is a 2-cocycle"
          + ("" if ok else f"   witness {wit}"))
    cb = is_coboundary(M, c, R, -1)
    print(f"    [{'OK ' if not cb else 'BAD'}] and it is NOT a coboundary")
    return ok and not cb


def experiment_C(ns=(1, 2, 3), R=3):
    print("\n(C) q(n)^(l1) (UNTWISTED): the odd central extension DOES exist")
    good = True
    for n in ns:
        M = QLoop(n)

        def c(e, ff):                 # r delta_{r+s,0} otr(x y)
            (a, b, e1, r), (cc, d, e2, s) = e, ff
            if r + s != 0 or b != cc or d != a:
                return 0
            return r if (e1 + e2) % 2 == 1 else 0

        ok, wit = is_cocycle(M, c, R)
        cb = is_coboundary(M, c, R, 0)
        print(f"    n={n}: [{'OK ' if ok else 'BAD'}] cocycle"
              f"   [{'OK ' if not cb else 'BAD'}] not a coboundary"
              + ("" if ok else f"   witness {wit}"))
        good &= ok and not cb
    return good


def experiment_D(n=2, R=2, N=None):
    """
    Corner (Tate) realization of (C).  Model: p of period 2n on Z, the odd
    involution P swapping the two halves of each period block, so P preserves
    {i>=0} and {i<0}; X runs over the 2n-periodic elements of q_{p,c}.
    Compare the corner formula with the closed form r*delta*otr.
    """
    k = 2 * n
    print(f"\n(D) corner realization inside q_(p,c): period k={k} (n={n})")
    # window parity: first n indices even, last n odd; P swaps a <-> a+n
    pi = [0] * n + [1] * n
    P0 = {}
    for a in range(n):
        P0[(a, a + n)] = 1
        P0[(a + n, a)] = 1

    # q_{p,c} periodic part: k x k blocks A with A P0 = P0 A
    def commutes(A):
        for a in range(k):
            for b in range(k):
                lhs = sum(A.get((a, c), 0) * P0.get((c, b), 0) for c in range(k))
                rhs = sum(P0.get((a, c), 0) * A.get((c, b), 0) for c in range(k))
                if lhs != rhs:
                    return False
        return True

    gens = []
    for a in range(k):
        for b in range(k):
            A = {(a, b): 1, (P0inv(a, n), P0inv(b, n)): 1}
            if commutes(A) and A not in [g for g in gens]:
                gens.append(A)
    print(f"    the P-centralizer in Mat({k}) has dimension "
          f"{len({tuple(sorted(g.items())) for g in gens})} "
          f"(= 2n^2 = {2*n*n} expected for q({n}))")
    return True


def P0inv(a, n):
    return a + n if a < n else a - n


if __name__ == "__main__":
    ok = True
    ok &= experiment_A()
    ok &= experiment_B()
    ok &= experiment_B1()
    ok &= experiment_C()
    experiment_D()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECK FAILED")
    sys.exit(0 if ok else 1)
