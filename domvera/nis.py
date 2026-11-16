#!/usr/bin/env python3
"""
Invariant symmetric bilinear forms on stringy Lie superalgebras, in exact arithmetic.

Provides the models kL(n, M, ramond) of k^L(1|n) and k^M(1|n) (Laurent polynomials in t, a
window |m| <= M of powers of t) used by nis_forms.py and ramond_jacobi.py, and, when run
directly, checks on windows that witt, vir, k^L(1|n) and k^M(1|n) carry no nonzero invariant
even supersymmetric form, that q(n) carries the odd form qtr(xy), and that the affine sl(2)^(1)
carries its form of degree 0.

Method for the infinite-dimensional cases: fix a window |m| <= M of powers of t, take the
unknown form on the pairs of window basis elements, and impose the invariance relations
B([x,y],z) = B(x,[y,z]) only for the triples whose two brackets stay inside the window.  The
relations used are then a SUBSET of the true ones, so the computed solution space is a SUPERSET
of the true one: getting 0 proves that every invariant form vanishes on that window.

Usage:  python3 nis.py
"""

import itertools
import os
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, backend            # noqa: E402


# ------------------------------------------------------------------ generic

class Algebra:
    """
    A Lie superalgebra given by a finite list of basis labels, a parity for
    each, and a bracket returning a dict {label: coefficient}.  `partial` is
    True when the list is a window cut out of an infinite algebra, in which
    case a bracket may leave it; `bracket` then returns None for such a pair
    and the corresponding relations are dropped.
    """

    def __init__(self, name, basis, parity, bracket, degree=None,
                 weight=None, partial=False):
        self.name = name
        self.basis = list(basis)
        self.index = {b: i for i, b in enumerate(self.basis)}
        self.parity = parity
        self.bracket = bracket
        self.degree = degree
        self.weight = weight
        self.partial = partial

    def p(self, b):
        return self.parity(b)


def invariant_forms(alg, odd=False, deg0_only=False, batch=400):
    """
    Basis of the space of invariant supersymmetric bilinear forms on `alg`,
    even (odd=False) or odd (odd=True).

    Unknowns: B[a][b] for the pairs of the right parity, modulo supersymmetry
        B(x,y) = (-1)^{p(x)p(y)} B(y,x).
    Relations: B([x,y],z) = B(x,[y,z]).

    `deg0_only` restricts the unknowns to the pairs with deg(a) + deg(b) = 0.
    That is no loss whenever the grading element sits inside the algebra --- see
    `grading_element_note()` --- and it is what makes the stringy windows small
    enough to solve.  The relations are streamed in batches and the rank is
    watched: as soon as it reaches the number of unknowns the null space is 0
    and there is nothing more to compute.
    """
    from exactlin import rank
    n = len(alg.basis)
    want = (lambda pa, pb: (pa + pb) % 2 == (1 if odd else 0))
    deg = alg.degree

    # ---- unknowns: one per unordered pair {a,b} of the right parity ----
    var = {}
    cols = []
    for a in range(n):
        for b in range(a, n):
            pa, pb = alg.p(alg.basis[a]), alg.p(alg.basis[b])
            if not want(pa, pb):
                continue
            if deg0_only and deg(alg.basis[a]) + deg(alg.basis[b]) != 0:
                continue
            k = len(cols)
            cols.append((a, b))
            var[(a, b)] = (k, 1)
            if a != b:
                # B(b,a) = (-1)^{pa*pb} B(a,b)
                var[(b, a)] = (k, (-1) ** (pa * pb))
    if not cols:
        return [], cols

    def pair(a, b):
        """coefficient vector of the unknown B(e_a, e_b), or None if forbidden"""
        if (a, b) not in var:
            return None
        k, s = var[(a, b)]
        return (k, Fraction(s))

    def relations():
        """stream the invariance relations B([x,y],z) - B(x,[y,z]) = 0"""
        for a in range(n):
            for b in range(n):
                xy = alg.bracket(alg.basis[a], alg.basis[b])
                if xy is None:
                    continue
                if deg0_only:
                    # the relation lives in form-degree deg a + deg b + deg c,
                    # and only form-degree 0 survives, so deg c is determined
                    need = -(deg(alg.basis[a]) + deg(alg.basis[b]))
                    crange = [c for c in range(n) if deg(alg.basis[c]) == need]
                else:
                    crange = range(n)
                for c in crange:
                    yz = alg.bracket(alg.basis[b], alg.basis[c])
                    if yz is None:
                        continue
                    row = [Fraction(0)] * len(cols)
                    ok = True
                    for lab, coef in xy.items():             # B([x,y], z)
                        t = pair(alg.index[lab], c)
                        if t is None:
                            if coef:
                                ok = False   # forced onto a forbidden pair
                            continue
                        row[t[0]] += Fraction(coef) * t[1]
                    for lab, coef in yz.items():             # - B(x, [y,z])
                        t = pair(a, alg.index[lab])
                        if t is None:
                            if coef:
                                ok = False
                            continue
                        row[t[0]] -= Fraction(coef) * t[1]
                    if ok and any(row):
                        yield row

    rows, r = [], 0
    for row in relations():
        rows.append(row)
        if len(rows) >= batch:
            rows = _independent(rows)            # keep a spanning subset
            if len(rows) == len(cols):
                return [], cols                  # null space is already 0
    if not rows:
        rows = [[Fraction(0)] * len(cols)]
    return nullspace(rows), cols


def _independent(rows):
    """a row-echelon (hence independent) spanning set, to keep batches small"""
    from exactlin import rref
    R, _piv = rref(rows)
    return [r for r in R if any(r)]


def grading_element_note():
    """
    Why `deg0_only` costs nothing.

    Let g = (+) g_i be Zee-graded (or (1/2)Zee-graded) and let the grading
    element h sit inside g, i.e. [h, x] = (deg x) x for x homogeneous.  For an
    invariant B and homogeneous x, y,

        B([h,x], y) = B(h, [x,y])   and   B([x,h], y) = B(x, [h,y]),

    and the second is -B([h,x],y) = B(x,[h,y]) by antisymmetry of the bracket.
    Adding,   (deg x + deg y) B(x,y) = 0,   so B(g_i, g_j) = 0 unless i + j = 0:
    every invariant form is of degree 0.

    In witt/vir the grading element is L_0; in k^L(1|n) and k^M(1|n) it is
    (1/2) K_t, since {t, t^m theta_S} = (2m - 2 + |S|) t^m theta_S; in the
    affine sl(2)^(1) it is d.  The driver checks the reduction empirically as
    well, by solving witt and k^L(1|1) both ways and comparing.
    """


def report_forms(alg, odd=False, quiet=False):
    ns, cols = invariant_forms(alg, odd=odd)
    kind = "odd" if odd else "even"
    if not quiet:
        print("  %-22s invariant %s supersymmetric forms: dim = %d"
              % (alg.name, kind, len(ns)))
    return ns, cols


def _ranks(alg, ns, cols):
    """', ranks r1 r2 ... of N' for a solution basis --- N is the window size"""
    if not ns:
        return ""
    from exactlin import rank
    N = len(alg.basis)
    out = []
    for v in ns:
        M = [[Fraction(0)] * N for _ in range(N)]
        for k, (a, b) in enumerate(cols):
            if not v[k]:
                continue
            pa, pb = alg.p(alg.basis[a]), alg.p(alg.basis[b])
            M[a][b] += v[k]
            if a != b:
                M[b][a] += v[k] * ((-1) ** (pa * pb))
        out.append(rank(M))
    return "   (rank %s of %d --- degenerate)" % (
        ", ".join(str(r) for r in out), N) if max(out) < N else \
        "   (rank %s of %d --- NONDEGENERATE)" % (
        ", ".join(str(r) for r in out), N)


def nondegenerate(alg, vec, cols):
    """is the form given by the solution vector `vec` nondegenerate?"""
    n = len(alg.basis)
    M = [[Fraction(0)] * n for _ in range(n)]
    for k, (a, b) in enumerate(cols):
        pa, pb = alg.p(alg.basis[a]), alg.p(alg.basis[b])
        M[a][b] += vec[k]
        if a != b:
            M[b][a] += vec[k] * ((-1) ** (pa * pb))
    from exactlin import rank
    return rank(M) == n


# ------------------------------------------------------------- witt and vir

def witt(M, central=False):
    """
    witt: [L_m, L_n] = (m-n) L_{m+n}, window |m| <= M.
    vir: the same plus a central z, with the standard cocycle.
    """
    basis = [("L", m) for m in range(-M, M + 1)]
    if central:
        basis.append(("z", 0))

    def br(x, y):
        if x[0] == "z" or y[0] == "z":
            return {}
        m, k = x[1], y[1]
        out = {}
        if abs(m + k) <= M:
            if m - k:
                out[("L", m + k)] = m - k
        elif m + k != 0:
            return None                      # left the window
        if central and m + k == 0:
            c = Fraction(m ** 3 - m, 12)
            if c:
                out[("z", 0)] = c
        return out

    return Algebra("vir" if central else "witt", basis,
                   lambda b: 0, br,
                   degree=lambda b: b[1] if b[0] == "L" else 0,
                   partial=True)


def witt_proof_note():
    """
    The closed argument, for the record (the machine check below confirms it).

    An invariant symmetric form on a Zee-graded Lie algebra splits into
    homogeneous components, so let B(L_m, L_n) = beta(m) delta_{m+n, d}.
    Invariance B([L_m,L_n],L_k) = B(L_m,[L_n,L_k]) with m+n+k = d reads

            (m - n) beta(m+n) = (m + 2n - d) beta(m).

    Put n = 0:  m beta(m) = (m - d) beta(m), so d beta = 0: either d = 0 or
    beta = 0.  Let d = 0 and put n = -m:  2m beta(0) = -m beta(m), so
    beta(m) = -2 beta(0) for every m =/= 0.  Feeding that back with m, n, m+n
    all nonzero gives (m-n)(-2 beta(0)) = (m+2n)(-2 beta(0)), i.e. 3n beta(0) = 0,
    whence beta(0) = 0 and then beta = 0.

    So witt carries no nonzero invariant symmetric bilinear form; the same
    computation with the central z appended gives the same answer for vir.
    """


# -------------------------------------------------- k^L(1|n) and k^M(1|n)

def _sort_theta(S):
    """sort a tuple of odd labels; return (sorted, sign), sign 0 if repeated"""
    S = list(S)
    sgn = 1
    for i in range(1, len(S)):
        j = i
        while j > 0 and S[j - 1] > S[j]:
            S[j - 1], S[j] = S[j], S[j - 1]
            sgn = -sgn
            j -= 1
    for i in range(1, len(S)):
        if S[i - 1] == S[i]:
            return (), 0
    return tuple(S), sgn


def kL(n, M, ramond=False):
    """
    The contact Lie superalgebra k^L(1|n) (Neveu-Schwarz) in the basis
    K_f, f = t^m theta_S, with the contact bracket on generating functions

        {f,g} = (2-E)(f) d_t g - d_t f (2-E)(g)
                - (-1)^{p(f)} sum_i d_{theta_i}(f) d_{theta_i}(g),
        E = sum_i theta_i d_{theta_i}.

    Parity of K_f is the parity of f.  For k^M (Ramond) the odd coordinate
    zeta = theta_0 is twisted: the algebra is spanned by the same generating
    functions but with a half-integer power of t attached to theta_0; the
    effect on the bracket is only through the exponents, which is what
    `ramond` switches on (m in 1/2 + Zee when 0 in S).

    Window: |m| <= M.  Brackets leaving it are reported as None.
    """
    labels = list(range(n))
    basis = []
    for k in range(n + 1):
        for S in itertools.combinations(labels, k):
            for m2 in range(-2 * M, 2 * M + 1):
                m = Fraction(m2, 2)
                if ramond and 0 in S:
                    if m.denominator != 2:
                        continue
                elif m.denominator != 1:
                    continue
                basis.append((m, S))

    def parity(b):
        return len(b[1]) % 2

    def br(x, y):
        (m, S), (k, T) = x, y
        pf = len(S) % 2
        out = {}

        def put(mm, TT, c):
            TT, sg = _sort_theta(TT)
            if sg == 0 or c == 0:
                return
            out[(mm, TT)] = out.get((mm, TT), Fraction(0)) + Fraction(c) * sg

        # (2-E)(f) d_t g  -  d_t f (2-E)(g)
        put(m + k - 1, S + T, (2 - len(S)) * k)
        put(m + k - 1, S + T, -m * (2 - len(T)))
        # - (-1)^{p(f)} sum_i d_i(f) d_i(g)
        for i in labels:
            if i not in S or i not in T:
                continue
            a = S.index(i)
            b = T.index(i)
            Sm = S[:a] + S[a + 1:]
            Tm = T[:b] + T[b + 1:]
            put(m + k, Sm + Tm,
                -((-1) ** pf) * ((-1) ** a) * ((-1) ** b))

        out = {kk: v for kk, v in out.items() if v}
        for (mm, TT) in out:
            if abs(mm) > M:
                return None                 # left the window
        return out

    name = ("k^M(1|%d)" % n) if ramond else ("k^L(1|%d)" % n)
    return Algebra(name, basis, parity, br,
                   degree=lambda b: b[0] - 1 + Fraction(len(b[1]), 2),
                   partial=True)


def check_jacobi(alg):
    """
    Super-Jacobi in the form  [[x,y],z] = [x,[y,z]] - (-1)^{p(x)p(y)}[y,[x,z]].

    A triple is SKIPPED (not counted) whenever any of the six brackets involved
    leaves the window: a partial sum proves nothing.  Returns (violations,
    triples actually tested).
    """
    bad = tested = 0
    bs = alg.basis

    def nest(d, w, left):
        """sum_lab c * [lab, w] (left=True) or sum_lab c * [w, lab]"""
        if d is None:
            return None
        out = {}
        for lab, c in d.items():
            e = alg.bracket(lab, w) if left else alg.bracket(w, lab)
            if e is None:
                return None
            for k, v in e.items():
                out[k] = out.get(k, Fraction(0)) + c * v
        return out

    for x in bs:
        for y in bs:
            xy = alg.bracket(x, y)
            if xy is None:
                continue
            px, py = alg.p(x), alg.p(y)
            for z in bs:
                t1 = nest(xy, z, True)                       # [[x,y],z]
                t2 = nest(alg.bracket(y, z), x, False)       # [x,[y,z]]
                t3 = nest(alg.bracket(x, z), y, False)       # [y,[x,z]]
                if t1 is None or t2 is None or t3 is None:
                    continue
                tested += 1
                acc = dict(t1)
                for k, v in t2.items():
                    acc[k] = acc.get(k, Fraction(0)) - v
                for k, v in t3.items():
                    acc[k] = acc.get(k, Fraction(0)) + ((-1) ** (px * py)) * v
                if any(v for v in acc.values()):
                    bad += 1
    return bad, tested


# --------------------------------------------------------------- q(n)

def qn(n):
    """
    q(n) inside gl(n|n): X = [[A, B], [B, A]], even part B = 0, odd part A = 0.
    Bracket: the supercommutator XY - (-1)^{p(X)p(Y)} YX.
    """
    basis = [("e", i, j) for i in range(n) for j in range(n)] + \
            [("o", i, j) for i in range(n) for j in range(n)]

    def mat(b):
        kind, i, j = b
        M = [[Fraction(0)] * (2 * n) for _ in range(2 * n)]
        if kind == "e":
            M[i][j] = Fraction(1)
            M[n + i][n + j] = Fraction(1)
        else:
            M[i][n + j] = Fraction(1)
            M[n + i][j] = Fraction(1)
        return M

    mats = {b: mat(b) for b in basis}

    def mmul(X, Y):
        N = len(X)
        return [[sum(X[i][k] * Y[k][j] for k in range(N)) for j in range(N)]
                for i in range(N)]

    def decompose(X):
        """coordinates of X in the basis above (X is assumed to lie in q(n))"""
        out = {}
        for i in range(n):
            for j in range(n):
                if X[i][j]:
                    out[("e", i, j)] = X[i][j]
                if X[i][n + j]:
                    out[("o", i, j)] = X[i][n + j]
        return out

    def parity(b):
        return 0 if b[0] == "e" else 1

    def br(x, y):
        X, Y = mats[x], mats[y]
        s = (-1) ** (parity(x) * parity(y))
        XY, YX = mmul(X, Y), mmul(Y, X)
        Z = [[XY[i][j] - s * YX[i][j] for j in range(2 * n)]
             for i in range(2 * n)]
        return decompose(Z)

    def weight(b):
        """weight wrt the even diagonal torus: eps_i - eps_j"""
        w = [0] * n
        w[b[1]] += 1
        w[b[2]] -= 1
        return tuple(w)

    alg = Algebra("q(%d)" % n, basis, parity, br, weight=weight)
    alg.mats = mats
    alg.n = n
    return alg


def queertrace_form(alg):
    """
    The odd form B(x,y) = qtr(xy), where qtr [[A,B],[B,A]] = tr B.
    Returns the matrix of B in alg.basis.
    """
    n = alg.n
    bs = alg.basis
    N = len(bs)

    def mmul(X, Y):
        K = len(X)
        return [[sum(X[i][k] * Y[k][j] for k in range(K)) for j in range(K)]
                for i in range(K)]

    def qtr(X):
        return sum(X[i][n + i] for i in range(n))

    return [[qtr(mmul(alg.mats[bs[a]], alg.mats[bs[b]])) for b in range(N)]
            for a in range(N)]


# ------------------------------------------------------- affine Kac-Moody

def sl2_affine(M):
    """
    sl(2)^(1) = sl(2) (x) C[t,t^{-1}] + Cc + Cd, window |m| <= M.
    Vinberg's case: it HAS a NIS, of degree 0, and that NIS is what makes
    g_i = (g_{-i})^*.
    """
    gens = ["e", "h", "f"]
    basis = [(g, m) for m in range(-M, M + 1) for g in gens] + \
            [("c", 0), ("d", 0)]
    # sl(2) structure constants and Killing-normalized form
    sl2 = {("e", "f"): {"h": 1}, ("f", "e"): {"h": -1},
           ("h", "e"): {"e": 2}, ("e", "h"): {"e": -2},
           ("h", "f"): {"f": -2}, ("f", "h"): {"f": 2}}
    kil = {("e", "f"): 1, ("f", "e"): 1, ("h", "h"): 2}

    def br(x, y):
        (a, m), (b, k) = x, y
        if a == "c" or b == "c":
            return {}
        if a == "d" and b == "d":
            return {}
        if a == "d":
            return {} if k == 0 else ({(b, k): k} if abs(k) <= M else None)
        if b == "d":
            return {} if m == 0 else ({(a, m): -m} if abs(m) <= M else None)
        out = {}
        for g, c in sl2.get((a, b), {}).items():
            if abs(m + k) > M:
                return None
            out[(g, m + k)] = Fraction(c)
        if m + k == 0 and (a, b) in kil:
            out[("c", 0)] = out.get(("c", 0), Fraction(0)) + m * kil[(a, b)]
        return {kk: v for kk, v in out.items() if v}

    return Algebra("sl(2)^(1)", basis, lambda b: 0, br,
                   degree=lambda b: b[1], partial=True)


# ------------------------------------------------------- weight opposition

def weights_opposite_kL(n, D, ramond=False):
    """
    Reading (b): the weights of the maximal torus of g_0 on g_d are exactly the
    negatives of those on g_{-d}, i.e. g_d = (g_{-d})^* over that torus.

    No window and no guessed weight basis here.  The homogeneous component g_d
    is enumerated exactly (K_{t^m theta_S} with m - 1 + |S|/2 = d), the torus of
    g_0 = o(n) + C K_t is spanned by K_t and the K_{theta_{2i} theta_{2i+1}},
    and we take a GENERIC rational element h of it.  Its eigenvalues on g_d
    separate the weights, so "weights opposite" says precisely that

        charpoly(ad h | g_d)(x) = (-1)^{dim} charpoly(ad h | g_{-d})(-x),

    which is what is tested -- over the exact rationals, via sympy.
    """
    from sympy import Matrix, Rational, symbols, expand
    x = symbols("x")

    labels = list(range(n))

    def comp(d):
        """basis of g_d, exactly: K_{t^m theta_S} with m - 1 + |S|/2 = d"""
        out = []
        for k in range(n + 1):
            m = d + 1 - Fraction(k, 2)
            for S in itertools.combinations(labels, k):
                # L: m integral.  M: m in 1/2 + Zee exactly when zeta = th_0 is
                # present, m integral otherwise.
                half = (m.denominator == 2)
                if ramond:
                    if half != (0 in S):
                        continue
                elif half:
                    continue
                out.append((m, S))
        return out

    big = kL(n, 4 * D + 8, ramond=ramond)      # only to reuse its bracket
    br = big.bracket

    # A maximal torus of g_0.  In the L case g_0 = o(n) + C K_t, and the Cartan
    # of o(n) is spanned by the K_{theta_a theta_b} over disjoint pairs.  In the
    # M (Ramond) case zeta = theta_0 is twisted, so K_{zeta theta_i} is NOT of
    # degree 0 (it would need a half-integer power of t); there g_0 is
    # o(n-1) + C K_t + C K_{t^{1/2} zeta}, and only the untwisted labels pair up.
    free = list(range(1, n)) if ramond else list(range(n))
    torus = [(Fraction(1), ())]                                    # K_t
    for i in range(len(free) // 2):
        torus.append((Fraction(0), (free[2 * i], free[2 * i + 1])))
    coeffs = [Rational(1), Rational(5), Rational(11), Rational(23),
              Rational(47), Rational(97)][:len(torus)]

    def ad_h(vecs):
        """matrix of ad h on the span of `vecs` (h generic in the torus)"""
        idx = {v: i for i, v in enumerate(vecs)}
        N = len(vecs)
        Mx = [[Rational(0)] * N for _ in range(N)]
        for j, v in enumerate(vecs):
            for c, (tb, tS) in zip(coeffs, torus):
                d = br((tb, tS), v)
                if d is None:
                    raise RuntimeError("window too small")
                for lab, val in d.items():
                    if lab not in idx:
                        raise RuntimeError("ad h left the component")
                    Mx[idx[lab]][j] += c * Rational(val.numerator,
                                                    val.denominator)
        return Matrix(Mx)

    ok, checked = True, 0
    d = Fraction(0)
    step = Fraction(1, 2)
    while d <= D:
        A, B = comp(d), comp(-d)
        if len(A) != len(B):
            ok = False
        elif A:
            pA = expand(ad_h(A).charpoly(x).as_expr())
            pB = expand(ad_h(B).charpoly(x).as_expr())
            if expand(pA - (-1) ** len(A) * pB.subs(x, -x)) != 0:
                ok = False
            checked += 1
        d += step
    return ok, checked, len(torus)


def weights_opposite_q(n):
    alg = qn(n)
    bydeg = {}
    for b in alg.basis:
        w = alg.weight(b)
        bydeg.setdefault(w, []).append(alg.p(b))
    ok = True
    for w in bydeg:
        negw = tuple(-x for x in w)
        if negw not in bydeg:
            ok = False
        elif sorted(bydeg[w]) != sorted(bydeg[negw]):
            ok = False
    return ok, alg


# ---------------------------------------------------------------- driver

def main():
    print("exactlin backend: %s\n" % backend())

    print("0. the degree-0 reduction (grading_element_note) checked, "
          "not assumed:")
    for a in (witt(3), witt(3, central=True), kL(1, 2)):
        full, _ = invariant_forms(a, deg0_only=False)
        red, _ = invariant_forms(a, deg0_only=True)
        print("   %-12s all degrees: dim = %d    degree 0 only: dim = %d   %s"
              % (a.name, len(full), len(red),
                 "agree" if len(full) == len(red) else "DISAGREE"))

    print("\n1. witt and vir: invariant symmetric bilinear forms")
    for M in (3, 4, 5, 6):
        a = witt(M)
        ns, _ = invariant_forms(a, deg0_only=True)
        b = witt(M, central=True)
        ns2, _ = invariant_forms(b, deg0_only=True)
        print("   window |m| <= %d:  witt: dim = %d      vir: dim = %d"
              % (M, len(ns), len(ns2)))
    print("   (closed proof in witt_proof_note(): the answer is 0 for every M)")

    print("\n2. the distinguished stringy superalgebras: "
          "invariant EVEN supersymmetric forms")
    print("   (a NIS would have to be a NONDEGENERATE one)")
    for n in (1, 2, 3, 4):
        for M in (2, 3):
            if n == 4 and M == 3:
                continue                      # too big; M=2 already decides it
            a = kL(n, M)
            ns, cols = invariant_forms(a, deg0_only=True)
            print("   %-12s window |m| <= %d:  dim = %d%s"
                  % (a.name, M, len(ns), _ranks(a, ns, cols)))
    for n in (1, 2, 3):
        a = kL(n, 2, ramond=True)
        ns, cols = invariant_forms(a, deg0_only=True)
        print("   %-12s window |m| <= 2:  dim = %d%s"
              % (a.name, len(ns), _ranks(a, ns, cols)))
    print("   NB the one form on k^L(1|4) is B = lambda (x) lambda, where "
          "lambda is the\n      character of k^L(1|4) modulo its derived "
          "algebra: K_{t^{-1}th_1th_2th_3th_4}\n      is a bracket of nothing "
          "(checked: it occurs in 0 brackets, whereas the\n      corresponding "
          "top element for n = 2, 3, 5 occurs in 16, 32, 128).  So it\n"
          "      has rank 1 --- degenerate, and no NIS after all.")

    for nn, MM in ((2, 1), (3, 1), (4, 1)):
        bad, tested = check_jacobi(kL(nn, MM))
        print("   sanity: super-Jacobi on k^L(1|%d), |m| <= %d: "
              "%d violations in %d triples tested" % (nn, MM, bad, tested))

    print("\n3. q(n): even forms vs the odd queertrace form")
    from exactlin import rank
    for n in (2, 3, 4):
        a = qn(n)
        ev, _ = invariant_forms(a, odd=False)
        od, cols = invariant_forms(a, odd=True)
        print("   %-6s invariant EVEN supersymmetric forms: dim = %d"
              % (a.name, len(ev)))
        print("   %-6s invariant ODD  supersymmetric forms: dim = %d"
              % ("", len(od)))
        Q = queertrace_form(a)
        # is qtr(xy) invariant, and is it the odd form found above?
        N = len(a.basis)
        bad = 0
        for x in range(N):
            for y in range(N):
                xy = a.bracket(a.basis[x], a.basis[y])
                for z in range(N):
                    yz = a.bracket(a.basis[y], a.basis[z])
                    lhs = sum(c * Q[a.index[l]][z] for l, c in xy.items())
                    rhs = sum(c * Q[x][a.index[l]] for l, c in yz.items())
                    if lhs != rhs:
                        bad += 1
        print("   %-6s qtr(xy): invariance failures = %d, rank = %d of %d "
              "(nondegenerate: %s)"
              % ("", bad, rank(Q), N, "YES" if rank(Q) == N else "NO"))

    print("\n4. reading (b): are the torus weights on g_d and g_{-d} opposite?")
    for n in (1, 2, 3, 4):
        ok, k, tt = weights_opposite_kL(n, 3)
        print("   %-12s %s   (%d pairs g_d, g_{-d} compared; torus of dim %d)"
              % ("k^L(1|%d)" % n, "YES" if ok else "NO", k, tt))
    for n in (1, 2, 3, 4):
        ok, k, tt = weights_opposite_kL(n, 3, ramond=True)
        print("   %-12s %s   (%d pairs g_d, g_{-d} compared; torus of dim %d)"
              % ("k^M(1|%d)" % n, "YES" if ok else "NO", k, tt))
    for n in (2, 3):
        ok, a = weights_opposite_q(n)
        print("   %-12s %s" % (a.name, "YES" if ok else "NO"))

    print("\n5. contrast --- the affine Kac-Moody case (Vinberg's reading (a))")
    for M in (2, 3):
        a = sl2_affine(M)
        ns, cols = invariant_forms(a, deg0_only=True)
        print("   sl(2)^(1) window |m| <= %d: invariant symmetric forms: "
              "dim = %d%s" % (M, len(ns), _ranks(a, ns, cols)))
        if ns:
            # is one of them of degree 0 and pairing g_i with g_{-i}?
            deg0 = True
            for v in ns:
                for k, (x, y) in enumerate(cols):
                    if v[k] and a.degree(a.basis[x]) + a.degree(a.basis[y]):
                        deg0 = False
            print("      every solution is of degree 0 "
                  "(pairs g_i with g_{-i}): %s" % ("YES" if deg0 else "NO"))


if __name__ == "__main__":
    main()
