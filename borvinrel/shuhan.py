#!/usr/bin/env python3
"""Exact checks on the h-Shuhan matrices of arXiv:2501.01955 (Wu--Yang).

An h-Shuhan matrix is a real n x n matrix H with

    (S1)  H_ii = h for all i, where h >= 0 is real;
    (S2)  H_ij in Z, H_ij <= 0 for i != j;
    (S3)  for i != j:  H_ij = H_ji, or  H_ij < H_ji = -1.

So H + (2-h)E is a generalized Cartan matrix (GCM), and conversely every GCM
satisfying (S3) arises this way.  Everything below is exact (sympy / Fraction).

What is verified:

 1. The Coxeter formula.  For a finite-type Cartan matrix A_X with Coxeter
    number m(X):  lambda_max(2E - A_X) = 2 cos(pi/m(X)), exactly (minimal
    polynomial divides the characteristic polynomial).  For affine X the value
    is exactly 2.
 2. For a symmetrizable H with positive symmetrizing coefficients,
    "all main minors >= 0" (Wu--Yang's VIRTUAL positive semi-definite, and
    Vinberg's type Pos or Z) is equivalent to lambda_min(H) >= 0.  Checked
    exhaustively over all 2^n main submatrices.
 3. Hence the threshold of Wu--Yang Prop. 3.3 / Thm. 3.9 is h = 2cos(pi/m(X)),
    which reproduces each of their eight case-by-case answers.
 4. The type C_n is missing from their Thm. 3.9(2) and Prop. 3.15: C_n is the
    transpose of B_n, has the same main minors, but is NOT a permutation
    conjugate of B_n for n >= 3.
 5. The arithmetic mean (H+H')/2 of Wu--Yang's "generalized" notion is not the
    Lie-theoretic symmetrization D*H (geometric means): at h = 2 it declares
    the Cartan matrix of o(2n+1) non-definite for n >= 9, whereas the invariant
    form of o(2n+1) is positive definite for every n.
 6. The constant eps ~ 2.04998 and the sequence mu_n of Wu--Yang Sect. 3.
 7. (S3) is a restriction, not a generalization, on the off-diagonal: the GCM
    [[2,-2],[-3,2]] satisfies no clause of it; every finite and affine GCM does.
 8. h = 0 gives Vinberg's type Neg for every size >= 2 (Proposition 5.2.1 of the paper).
 9. The Vinberg type is invariant under rescaling of rows by positive scalars,
    so the parameter h is not a parameter of g(H): g(H) = g((2/h)H).

Usage:  tools/venv/bin/python shuhan.py   (falls back to system python3)
"""

import itertools
import sys
from fractions import Fraction

import sympy as sp

LOG = None


def say(msg=""):
    print(msg)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


PASS = [0]
FAIL = [0]


def check(name, cond, detail=""):
    if cond:
        PASS[0] += 1
        say("  ok   %-62s %s" % (name, detail))
    else:
        FAIL[0] += 1
        say("  FAIL %-62s %s" % (name, detail))


# ----------------------------------------------------------------- matrices

def eye(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def from_edges(n, edges):
    """Cartan matrix from an edge list of triples (i, j, a_ij, a_ji)."""
    A = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
    for (i, j, aij, aji) in edges:
        A[i][j], A[j][i] = aij, aji
    return A


def chain(k, start=0):
    return [(start + i, start + i + 1, -1, -1) for i in range(k - 1)]


def cartan(kind, n):
    """Finite-type Cartan matrix.  The diagrams are built from edge lists and
    validated below by their determinants (DET_FIN) and by Vinberg's type."""
    if kind == "A":
        return from_edges(n, chain(n))
    if kind == "B":                                    # ... => last node short
        e = chain(n)
        e[-1] = (n - 2, n - 1, -1, -2)
        return from_edges(n, e)
    if kind == "C":                                    # the transpose of B_n
        e = chain(n)
        e[-1] = (n - 2, n - 1, -2, -1)
        return from_edges(n, e)
    if kind == "D":                                    # T(2,2,n-2): fork at one end
        e = chain(n - 1) + [(n - 3, n - 1, -1, -1)]
        return from_edges(n, e)
    if kind == "E":                                    # T(2,3,n-3): branch at node 2
        assert n in (6, 7, 8)
        e = chain(n - 1) + [(2, n - 1, -1, -1)]
        return from_edges(n, e)
    if kind == "F":
        assert n == 4
        return from_edges(4, [(0, 1, -1, -1), (1, 2, -1, -2), (2, 3, -1, -1)])
    if kind == "G":
        assert n == 2
        return from_edges(2, [(0, 1, -1, -3)])
    raise ValueError(kind)


DET_FIN = {"A": lambda n: n + 1, "B": lambda n: 2, "C": lambda n: 2,
           "D": lambda n: 4, "E": lambda n: {6: 3, 7: 2, 8: 1}[n],
           "F": lambda n: 1, "G": lambda n: 1}


def cartan_affine(name):
    """Affine Cartan matrices, by the extended Dynkin diagram.  Validated below
    by det = 0 together with Vinberg's type Z."""
    if name == "A1(1)":
        return [[2, -2], [-2, 2]]
    if name.startswith("A") and name.endswith("(1)"):
        k = int(name[1:-3])                            # A_k^(1), size k+1
        m = k + 1
        e = chain(m) + [(0, m - 1, -1, -1)]
        return from_edges(m, e)
    if name.startswith("D") and name.endswith("(1)"):
        k = int(name[1:-3])                            # D_k^(1), size k+1
        m = k + 1
        # nodes 0, 1 fork onto node 2; chain 2,...,k-2; nodes k-1, k fork onto k-2
        e = ([(0, 2, -1, -1), (1, 2, -1, -1)] + chain(k - 3, start=2) +
             [(k - 2, k - 1, -1, -1), (k - 2, k, -1, -1)])
        return from_edges(m, e)
    if name in ("E6(1)", "E7(1)", "E8(1)"):
        k = int(name[1])
        m = k + 1
        # E_k = chain(k-1) + branch at node 2; the affine node extends
        #   E6: the branch arm (node k-1);  E7: the short arm (node 0);
        #   E8: the long arm (node k-2)
        attach = {6: k - 1, 7: 0, 8: k - 2}[k]
        e = chain(k - 1) + [(2, k - 1, -1, -1), (attach, k, -1, -1)]
        return from_edges(m, e)
    if name in ("B2(1)", "C2(1)"):
        return from_edges(3, [(0, 1, -1, -2), (1, 2, -2, -1)])
    if name.startswith("B") and name.endswith("(1)"):
        k = int(name[1:-3])                            # fork at one end, => at the other
        m = k + 1
        # nodes 0 and 1 both attached to node 2; chain 2,...,k; last edge double
        e = [(0, 2, -1, -1), (1, 2, -1, -1)] + chain(k - 1, start=2)
        e[-1] = (k - 1, k, -1, -2)
        return from_edges(m, e)
    if name.startswith("C") and name.endswith("(1)"):
        k = int(name[1:-3])                            # => at both ends
        m = k + 1
        e = chain(m)
        # the marks of C_k^(1) are (1, 2, ..., 2, 1); the opposite choice of
        # the two arrows gives the dual diagram D_{k+1}^(2), whose marks are
        # all 1 -- the two have the same main minors, so the checks below do
        # not tell them apart, but the marks do.
        e[0] = (0, 1, -1, -2)
        e[-1] = (m - 2, m - 1, -2, -1)
        return from_edges(m, e)
    if name == "G2(1)":
        return from_edges(3, [(0, 1, -1, -1), (1, 2, -1, -3)])
    if name == "F4(1)":
        return from_edges(5, [(0, 1, -1, -1), (1, 2, -1, -1),
                              (2, 3, -1, -2), (3, 4, -1, -1)])
    if name == "A2(2)":
        return [[2, -1], [-4, 2]]
    raise ValueError(name)


AFFINE = ("A1(1)", "A3(1)", "A5(1)", "D4(1)", "D5(1)", "D6(1)", "E6(1)",
          "E7(1)", "E8(1)", "B2(1)", "B4(1)", "C3(1)", "G2(1)", "F4(1)",
          "A2(2)")


COXETER = {"A": lambda n: n + 1, "B": lambda n: 2 * n, "C": lambda n: 2 * n,
           "D": lambda n: 2 * n - 2, "E": lambda n: {6: 12, 7: 18, 8: 30}[n],
           "F": lambda n: 12, "G": lambda n: 6}


def finite_types(maxrank=10):
    out = []
    for n in range(1, maxrank + 1):
        out.append(("A", n))
        if n >= 2:
            out.append(("B", n))
            out.append(("C", n))
        if n >= 4:
            out.append(("D", n))
    for n in (6, 7, 8):
        out.append(("E", n))
    out.append(("F", 4))
    out.append(("G", 2))
    return out


# ------------------------------------------------------------ linear algebra

def det_frac(M):
    """Exact determinant over Fraction, fraction-free-ish Gauss."""
    n = len(M)
    if n == 0:
        return Fraction(1)
    A = [[Fraction(x) for x in row] for row in M]
    sign = 1
    d = Fraction(1)
    for c in range(n):
        p = None
        for r in range(c, n):
            if A[r][c] != 0:
                p = r
                break
        if p is None:
            return Fraction(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            sign = -sign
        d *= A[c][c]
        inv = Fraction(1) / A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] * inv
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
    return sign * d


def sub(M, idx):
    return [[M[i][j] for j in idx] for i in idx]


def main_minors(M, proper_only=False):
    """All main (principal) minors, in Vinberg's sense: strike out any set of
    rows and the columns of the same numbers."""
    n = len(M)
    rng = range(1, n) if proper_only else range(1, n + 1)
    for k in rng:
        for idx in itertools.combinations(range(n), k):
            yield idx, det_frac(sub(M, idx))


def vinberg_type(M):
    """Pos / Z / Neg, by Theorem V (conditions a)-d) assumed)."""
    n = len(M)
    for idx, d in main_minors(M, proper_only=True):
        if d <= 0:
            return "Neg"
    D = det_frac(M)
    if D > 0:
        return "Pos"
    if D == 0:
        return "Z"
    return "Neg"


def virtual_psd(M):
    return all(d >= 0 for _, d in main_minors(M))


def virtual_pd(M):
    return all(d > 0 for _, d in main_minors(M))


def shift(A, h):
    """A + (h-2)E, i.e. the h-Shuhan matrix over the diagram of A."""
    n = len(A)
    return [[A[i][j] + (h - 2 if i == j else 0) for j in range(n)] for i in range(n)]


def symmetrizing(A):
    """Positive rationals d_i with d_i A_ij = d_j A_ji, or None."""
    n = len(A)
    d = [None] * n
    d[0] = Fraction(1)
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(n):
            if i == j or A[i][j] == 0:
                continue
            val = d[i] * Fraction(A[i][j], A[j][i])
            if d[j] is None:
                d[j] = val
                stack.append(j)
            elif d[j] != val:
                return None
    if any(x is None or x <= 0 for x in d):
        return None
    return d


def sym_gm(A):
    """The Lie-theoretic symmetrization D*A (off-diagonal = -sqrt(A_ij A_ji)
    after rescaling); returned as the rational matrix diag(d) * A."""
    d = symmetrizing(A)
    if d is None:
        return None
    return [[d[i] * Fraction(A[i][j]) for j in range(len(A))] for i in range(len(A))]


def sym_am(A):
    n = len(A)
    return [[Fraction(A[i][j] + A[j][i], 2) for j in range(n)] for i in range(n)]


def psd_sym(S):
    """Symmetric S psd  <=>  every principal minor >= 0."""
    return all(d >= 0 for _, d in main_minors(S))


def pd_sym(S):
    return all(d > 0 for _, d in main_minors(S))


def charpoly(M, x):
    n = len(M)
    Msp = sp.Matrix(n, n, lambda i, j: sp.Rational(M[i][j]))
    return sp.Poly(Msp.charpoly(x).as_expr(), x)


# --------------------------------------------------------------------- 1--3

def sec0_validate():
    say("\n0. the diagrams are what they are called")
    for kind, n in finite_types(10):
        A = cartan(kind, n)
        d = det_frac(A)
        t = vinberg_type(A)
        s = symmetrizing(A)
        check("%s_%d: det = %s, Vinberg type %s, symmetrizable" % (kind, n, d, t),
              d == DET_FIN[kind](n) and t == "Pos" and s is not None,
              "det table: %s" % DET_FIN[kind](n))
    for name in AFFINE:
        A = cartan_affine(name)
        check("%s: det = 0 and Vinberg type Z (Kac's affine criterion)" % name,
              det_frac(A) == 0 and vinberg_type(A) == "Z", "size %d" % len(A))


def sec1_coxeter():
    say("\n1. lambda_max(2E - A_X) = 2 cos(pi/m(X)), m = the Coxeter number")
    x = sp.Symbol("x")
    for kind, n in finite_types(10):
        if kind in ("B", "C") and n < 2:
            continue
        A = cartan(kind, n)
        m = COXETER[kind](n)
        N = [[(2 if i == j else 0) - A[i][j] for j in range(n)] for i in range(n)]
        cp = charpoly(N, x)
        mp = sp.Poly(sp.minimal_polynomial(2 * sp.cos(sp.pi / m), x), x)
        div = sp.rem(cp, mp) == 0
        # and it is the LARGEST root: numerically
        lam = max(sp.Matrix(n, n, lambda i, j: sp.Rational(N[i][j])).eigenvals().keys(),
                  key=lambda t: sp.re(sp.N(t)))
        num_ok = abs(complex(sp.N(lam)) - float(sp.N(2 * sp.cos(sp.pi / m)))) < 1e-12
        check("%s_%d: minpoly(2cos(pi/%d)) | charpoly, and it is lambda_max"
              % (kind, n, m), div and num_ok, "m=%d" % m)


EXPONENTS = {
    "A": lambda n: list(range(1, n + 1)),
    "B": lambda n: list(range(1, 2 * n, 2)),
    "C": lambda n: list(range(1, 2 * n, 2)),
    "D": lambda n: sorted(list(range(1, 2 * n - 2, 2)) + [n - 1]),
    "E": lambda n: {6: [1, 4, 5, 7, 8, 11], 7: [1, 5, 7, 9, 11, 13, 17],
                    8: [1, 7, 11, 13, 17, 19, 23, 29]}[n],
    "F": lambda n: [1, 5, 7, 11],
    "G": lambda n: [1, 5],
}


def sec1b_spectrum():
    say("\n1b. the whole spectrum of 2E - A_X is 2cos(pi m_j/m), m_j the exponents")
    for kind, n in finite_types(8):
        A = cartan(kind, n)
        m = COXETER[kind](n)
        exps = EXPONENTS[kind](n)
        N = sp.Matrix(n, n, lambda i, j: sp.Rational((2 if i == j else 0) - A[i][j]))
        # 2E - A is similar to a symmetric matrix, so its spectrum is real
        x = sp.Symbol("x")
        cp = sp.Poly(sp.expand(N.charpoly(x).as_expr()), x)
        want = sorted(float(sp.N(2 * sp.cos(sp.pi * e / m), 30)) for e in exps)
        got = sorted(float(sp.re(r)) for r in cp.nroots(n=30, maxsteps=200))
        check("%s_%d: spectrum = {2cos(pi m_j/%d)}, m_j = %s"
              % (kind, n, m, exps),
              len(exps) == n and
              all(abs(a - b) < 1e-20 for a, b in zip(want, got)),
              "lambda_max = 2cos(pi/%d) = %.9f" % (m, got[-1]))


def sec2_equivalence():
    say("\n2. virtual psd  <=>  lambda_min >= 0   (symmetrizable, positive coeffs)")
    x = sp.Symbol("x")
    grid = [Fraction(k, 8) for k in range(0, 25)]          # h = 0 .. 3
    for kind, n in finite_types(6):
        A = cartan(kind, n)
        d = symmetrizing(A)
        check("%s_%d symmetrizable with positive coefficients" % (kind, n),
              d is not None, "d=%s" % (d,))
        bad = []
        for h in grid:
            H = shift(A, h)
            vp = virtual_psd(H)
            S = sym_gm(H)
            lam_ok = psd_sym(S)
            if vp != lam_ok:
                bad.append(h)
        check("%s_%d: all 2^%d main minors >= 0 <=> D*H psd, %d values of h"
              % (kind, n, n, len(grid)), not bad, "" if not bad else str(bad))


def sec3_thresholds():
    say("\n3. the threshold of [WY, Prop. 3.3] is 2cos(pi/m): their eight answers")
    claimed = {                       # exactly what [WY, Prop. 3.3] prints
        ("A", None): lambda n: sp.pi / (n + 1),
        ("B", None): lambda n: sp.pi / (2 * n),
        ("D", None): lambda n: sp.pi / (2 * (n - 1)),
        ("E", 6): lambda n: sp.pi / 12,
        ("E", 7): lambda n: sp.pi / 18,
        ("E", 8): lambda n: sp.pi / 30,
        ("F", 4): lambda n: sp.pi / 12,
        ("G", 2): lambda n: sp.pi / 6,
    }
    for kind, n in finite_types(8):
        key = (kind, n) if (kind, n) in claimed else (kind, None)
        if key not in claimed:
            continue
        theirs = 2 * sp.cos(claimed[key](n))
        ours = 2 * sp.cos(sp.pi / COXETER[kind](n))
        check("%s_%d: [WY] threshold = 2cos(pi/m(%s_%d))" % (kind, n, kind, n),
              sp.simplify(theirs - ours) == 0,
              "= %s" % sp.nsimplify(sp.N(ours, 8)))
    # and the threshold really is a threshold: exact test at h = 2cos(pi/m)
    for kind, n in finite_types(7):
        A = cartan(kind, n)
        m = COXETER[kind](n)
        # rational h just below / above the algebraic threshold
        t = float(sp.N(2 * sp.cos(sp.pi / m), 30))
        lo = Fraction(int((t - 1e-6) * 10 ** 9), 10 ** 9)
        hi = Fraction(int((t + 1e-6) * 10 ** 9) + 1, 10 ** 9)
        check("%s_%d: virtual psd fails below 2cos(pi/%d), holds above"
              % (kind, n, m),
              (not virtual_psd(shift(A, lo))) and virtual_psd(shift(A, hi)),
              "t=%.9f" % t)


def sec3b_affine():
    say("\n   affine: the threshold is exactly 2  ([WY, Prop. 3.10])")
    for name in AFFINE:
        A = cartan_affine(name)
        n = len(A)
        ok2 = virtual_psd(A)                       # h = 2
        below = virtual_psd(shift(A, Fraction(199, 100)))
        above = virtual_pd(shift(A, Fraction(201, 100)))
        check("%s: virtual psd at h=2, not below, pd above" % name,
              ok2 and not below and above, "size %d" % n)


# ----------------------------------------------------------------------- 4

def perm_conjugates(M):
    n = len(M)
    out = set()
    for p in itertools.permutations(range(n)):
        out.add(tuple(tuple(M[p[i]][p[j]] for j in range(n)) for i in range(n)))
    return out


def sec4_missing_C():
    say("\n4. C_n is absent from [WY, Thm. 3.9(2)] and [WY, Prop. 3.15]")
    for n in range(2, 8):
        B = cartan("B", n)
        C = cartan("C", n)
        Ct = tuple(tuple(r) for r in C)
        conj = Ct in perm_conjugates(B)
        transp = all(C[i][j] == B[j][i] for i in range(n) for j in range(n))
        check("C_%d = (B_%d)^T; permutation conjugate of B_%d: %s"
              % (n, n, n, "yes" if conj else "NO"),
              transp and (conj == (n == 2)),
              "n=2 is the only coincidence" if n == 2 else "")
    # same minors, hence same virtual-psd threshold
    for n in range(2, 8):
        B, C = cartan("B", n), cartan("C", n)
        same = True
        for h in [Fraction(k, 8) for k in range(0, 25)]:
            if virtual_psd(shift(B, h)) != virtual_psd(shift(C, h)):
                same = False
        check("C_%d^h and B_%d^h are virtual psd for exactly the same h" % (n, n),
              same)


# ----------------------------------------------------------------------- 5

def sec5_am_vs_gm():
    say("\n5. the arithmetic mean (H+H')/2 is not the Lie-theoretic symmetrization")
    say("   [WY, Prop. 3.15]: det of the AM-symmetrized Cartan matrix of o(2n+1)")
    for n in range(2, 14):
        B = cartan("B", n)
        am = sym_am(B)
        claimed = Fraction(2) - Fraction(n - 1, 4)
        check("B_%d: det((A+A')/2) = 2 - (n-1)/4" % n,
              det_frac(am) == claimed, "= %s" % claimed)
    for n in range(2, 14):
        B = cartan("B", n)
        am, gm = sym_am(B), sym_gm(B)
        check("B_%d: AM psd %-5s pd %-5s || GM (Lie) psd %-5s pd %-5s"
              % (n, psd_sym(am), pd_sym(am), psd_sym(gm), pd_sym(gm)),
              psd_sym(gm) and pd_sym(gm) and (pd_sym(am) == (n <= 8))
              and (psd_sym(am) == (n <= 9)),
              "<- the AM cuts at n=9")
    # AM-GM: the arithmetic mean is the larger in absolute value off-diagonal
    for kind, n in (("B", 5), ("C", 5), ("F", 4), ("G", 2)):
        A = cartan(kind, n)
        gm = sym_gm(A)
        d = symmetrizing(A)
        worse = True
        for i in range(n):
            for j in range(n):
                if i == j or A[i][j] == 0:
                    continue
                # compare after rescaling both to the same diagonal
                am_ij = float(abs(Fraction(A[i][j] + A[j][i], 2)))
                gm_ij = (abs(A[i][j]) * abs(A[j][i])) ** 0.5
                if am_ij < gm_ij - 1e-12:
                    worse = False
        check("%s_%d: |AM off-diagonal| >= |GM off-diagonal| (AM-GM)" % (kind, n),
              worse)


# ----------------------------------------------------------------------- 6

def lam_max_am(A, prec=40):
    """Largest eigenvalue of the arithmetic-mean symmetrization of the
    off-diagonal part, i.e. of (2E-A + (2E-A)')/2 --- the number mu_n, lambda_n,
    eta_n of [WY, Sect. 3]."""
    n = len(A)
    x = sp.Symbol("x")
    N = sp.Matrix(n, n, lambda i, j: (x if i == j else
                                      -sp.Rational(A[i][j] + A[j][i], 2)))
    rr = sp.Poly(sp.expand(N.det()), x).nroots(n=prec, maxsteps=200)
    return max(sp.re(r) for r in rr)


def sec6_epsilon():
    say("\n6. the constant eps ~ 2.04998 of [WY] and the sharp one")
    h = sp.Symbol("h", positive=True)
    g = h ** 3 - h ** 2 - sp.Rational(13, 4) * h + sp.Rational(9, 4)
    eps = max(sp.Poly(g, h).all_roots(), key=lambda r: sp.N(r))
    epsf = float(sp.N(eps, 30))
    check("eps is the largest root of h^3 - h^2 - 13h/4 + 9/4",
          abs(epsf - 2.04998) < 1e-4, "eps = %.9f" % epsf)

    # mu_n = lambda_max of the AM-symmetrized B_n with zero diagonal
    x = sp.Symbol("x")

    def bhat(n):
        B = cartan("B", n)
        S = sp.Matrix(n, n, lambda i, j: (x if i == j else
                                          sp.Rational(B[i][j] + B[j][i], 2)))
        return sp.Poly(sp.expand(S.det()), x)

    mus = []
    for n in range(2, 16):
        rr = [sp.re(r) for r in bhat(n).all_roots() if sp.im(sp.N(r)) == 0]
        mus.append(float(sp.N(max(rr, key=lambda r: sp.N(r)), 30)))
    check("mu_n strictly increases ([WY, Lemma 3.4])",
          all(mus[i] < mus[i + 1] for i in range(len(mus) - 1)),
          "mu_2..mu_6 = %s" % ", ".join("%.6f" % v for v in mus[:5]))
    check("mu_2 = 3/2, mu_3 = sqrt(13)/2, mu_9 = 2",
          abs(mus[0] - 1.5) < 1e-20 and
          abs(mus[1] - float(sp.N(sp.sqrt(13) / 2, 30))) < 1e-20 and
          abs(mus[7] - 2.0) < 1e-20, "mu_9 = %.15f" % mus[7])

    # the sharp constant: sup mu_n = 9/(2 sqrt 5) = 9 sqrt5 / 10
    sharp = sp.Rational(9, 2) / sp.sqrt(5)
    sharpf = float(sp.N(sharp, 30))
    check("9/(2 sqrt 5) = 9 sqrt(5)/10, and it is < eps",
          abs(sharpf - 2.012461180) < 1e-9 and sharpf < epsf,
          "sharp = %.9f, eps = %.9f, eps is %.2f%% too large"
          % (sharpf, epsf, 100 * (epsf / sharpf - 1)))
    check("mu_n < 9/(2 sqrt 5) for 2 <= n <= 15", all(v < sharpf for v in mus),
          "mu_15 = %.9f" % mus[-1])

    # the exact identity behind it:  bhat_n(9/(2 sqrt5)) = h * t^{1-n}, t = sqrt5/2
    t = sp.sqrt(5) / 2
    ok = True
    vals = []
    for n in range(2, 13):
        lhs = sp.simplify(bhat(n).as_expr().subs(x, sharp))
        rhs = sp.simplify(sharp * t ** (1 - n))
        vals.append(float(sp.N(lhs, 30)))
        if sp.simplify(lhs - rhs) != 0:
            ok = False
    check("bhat_n(9/(2 sqrt5)) = (9/(2 sqrt5)) * (sqrt5/2)^(1-n) > 0, n <= 12",
          ok, "values %s ..." % ", ".join("%.5f" % v for v in vals[:4]))
    # hence, by Sylvester (the leading minors are ahat_1..ahat_{n-1} > 0 at h>2),
    # the matrix is positive definite, i.e. mu_n < 9/(2 sqrt5)
    ahat_pos = all(sp.N(sp.Matrix(
        k, k, lambda i, j: (sharp if i == j else
                            (-1 if abs(i - j) == 1 else 0))).det(), 30) > 0
        for k in range(1, 13))
    check("the leading minors ahat_k(9/(2 sqrt5)) are > 0 (h > 2), k <= 12",
          ahat_pos, "so Sylvester gives positive definiteness")

    # mu_n -> 9/(2 sqrt5) numerically, at n = 400 (float, only the limit is at stake)
    try:
        import numpy as np
        n = 400
        N = np.zeros((n, n))
        for i in range(n - 1):
            w = 1.5 if i == n - 2 else 1.0
            N[i][i + 1] = N[i + 1][i] = w
        lam = max(np.linalg.eigvalsh(N))
        check("mu_400 = 9/(2 sqrt5) to 12 digits (the limit is attained)",
              abs(lam - sharpf) < 1e-12 and lam < sharpf,
              "mu_400 = %.12f, sharp = %.12f" % (lam, sharpf))
    except ImportError:
        say("  --   numpy absent, the n=400 limit check skipped")

    # the two affine constants of [WY] ARE sharp
    lams = [float(sp.N(lam_max_am(cartan_affine("B%d(1)" % n)), 30))
            for n in range(3, 10)]
    check("[WY, Prop. 3.13(i)]: sup_n lambda_n = sqrt(17)/2, attained at n=3",
          abs(lams[0] - float(sp.N(sp.sqrt(17) / 2, 30))) < 1e-20 and
          all(v <= lams[0] + 1e-20 for v in lams),
          "lambda_3..lambda_6 = %s" % ", ".join("%.6f" % v for v in lams[:4]))
    etas = [float(sp.N(lam_max_am(cartan_affine("C%d(1)" % n)), 30))
            for n in range(2, 10)]
    check("[WY, Prop. 3.13(ii)]: sup_n eta_n = 3 sqrt(2)/2, attained at n=2",
          abs(etas[0] - float(sp.N(3 * sp.sqrt(2) / 2, 30))) < 1e-20 and
          all(v <= etas[0] + 1e-20 for v in etas),
          "eta_2..eta_5 = %s" % ", ".join("%.6f" % v for v in etas[:4]))
    C5 = cartan_affine("C5(1)")
    m5 = len(C5)
    xx = sp.Symbol("xx")
    N5 = sp.Matrix(m5, m5, lambda i, j: (xx if i == j else
                                         -sp.Rational(C5[i][j] + C5[j][i], 2)))
    check("eps = eta_5 exactly: it is an eigenvalue of the C_5^(1) matrix",
          sp.simplify(sp.expand(N5.det()).subs(xx, eps)) == 0,
          "so [WY]'s eps is the sharp constant of another diagram")
    check("eta_n decreases to 9/(2 sqrt5) as well",
          all(etas[i] > etas[i + 1] for i in range(len(etas) - 1)) and
          etas[-1] > sharpf, "eta_9 = %.9f" % etas[-1])


# --------------------------------------------------------------------- 7--9

def satisfies_S3(A):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = A[i][j], A[j][i]
            if a == b:
                continue
            if min(a, b) < max(a, b) == -1:
                continue
            return False
    return True


def sec7_S3():
    say("\n7. (S3) is a restriction on the off-diagonal, not a generalization")
    bad = [[2, -2], [-3, 2]]
    check("[[2,-2],[-3,2]] is a GCM and satisfies no clause of (S3)",
          not satisfies_S3(bad), "it is of Vinberg's type %s" % vinberg_type(bad))
    allf = all(satisfies_S3(cartan(k, n)) for k, n in finite_types(9))
    check("every finite-type Cartan matrix satisfies (S3)", allf)
    names = AFFINE
    alla = all(satisfies_S3(cartan_affine(nm)) for nm in names)
    check("every affine Cartan matrix in the list satisfies (S3)", alla,
          "%d matrices" % len(names))


def find_u(M):
    """A positive u with Mu < 0.  For a matrix with a non-positive diagonal and
    non-positive off-diagonal the all-ones vector already does it; otherwise
    solve Mu = (-1,...,-1)."""
    n = len(M)
    one = [Fraction(1)] * n
    if all(sum(Fraction(M[i][j]) * one[j] for j in range(n)) < 0 for i in range(n)):
        return one
    Msp = sp.Matrix(n, n, lambda i, j: sp.Rational(M[i][j]))
    if Msp.det() == 0:
        return None
    u = Msp.solve(sp.Matrix([-1] * n))
    if all(v > 0 for v in u):
        return list(u)
    return None


def sec8_h_zero():
    say("\n8. h = 0 gives Vinberg's type Neg for every size >= 2 (Proposition 5.2.1)")
    for kind, n in finite_types(6):
        if n < 2:
            continue
        A = shift(cartan(kind, n), Fraction(0))
        t = vinberg_type(A)
        u = find_u(A)
        check("0-Shuhan over %s_%d is of type Neg, with an explicit u>0, Au<0"
              % (kind, n), t == "Neg" and u is not None,
              "u = %s" % (u if u is not None else "-"))


def sec9_rescaling():
    say("\n9. the Vinberg type is invariant under rescaling of rows by positives")
    say("   hence h is no parameter of g(H):  g(H) = g((2/h)H)")
    scales = [Fraction(1, 3), Fraction(1, 2), Fraction(2), Fraction(5, 2)]
    for kind, n in finite_types(5):
        A = cartan(kind, n)
        for h in (Fraction(3, 2), Fraction(2), Fraction(9, 4)):
            H = shift(A, h)
            t0 = vinberg_type(H)
            ok = True
            for s in scales:
                D = [[H[i][j] * s for j in range(n)] for i in range(n)]
                if vinberg_type(D) != t0:
                    ok = False
            check("%s_%d, h=%s: type %s unchanged by 4 positive rescalings"
                  % (kind, n, h, t0), ok)
    # the normalized form: (2/h) H has 2's on the diagonal
    A = cartan("E", 8)
    h = Fraction(7, 4)
    H = shift(A, h)
    Hn = [[Fraction(2, 1) / h * H[i][j] for j in range(8)] for i in range(8)]
    check("(2/h)H has 2 on the diagonal and the same Vinberg type",
          all(Hn[i][i] == 2 for i in range(8)) and
          vinberg_type(Hn) == vinberg_type(H),
          "E_8, h = 7/4: type %s" % vinberg_type(H))
    # the radius of the type-Pos region along the scaling ray
    say("\n   the radius of type Pos around a Dynkin point along the scaling ray:")
    for kind, n in (("A", 4), ("D", 5), ("E", 8), ("B", 3), ("G", 2)):
        m = COXETER[kind](n)
        t = float(sp.N(1 / sp.cos(sp.pi / m), 12))
        A = cartan(kind, n)
        # scale the off-diagonal by tt; type Pos iff tt < 1/cos(pi/m)
        def scaled(tt):
            return [[Fraction(2) if i == j else Fraction(A[i][j]) * tt
                     for j in range(n)] for i in range(n)]
        lo = Fraction(int((t - 1e-6) * 10 ** 9), 10 ** 9)
        hi = Fraction(int((t + 1e-6) * 10 ** 9) + 1, 10 ** 9)
        check("%s_%d: off-diagonal scaled by t is Pos iff t < 1/cos(pi/%d)"
              % (kind, n, m),
              vinberg_type(scaled(lo)) == "Pos" and
              vinberg_type(scaled(hi)) == "Neg",
              "1/cos(pi/%d) = %.9f" % (m, t))


def main():
    global LOG
    LOG = open(__file__.replace(".py", ".log"), "w")
    say("h-Shuhan matrices (arXiv:2501.01955) --- exact checks")
    say("sympy %s, python %s" % (sp.__version__, sys.version.split()[0]))
    sec0_validate()
    sec1_coxeter()
    sec1b_spectrum()
    sec2_equivalence()
    sec3_thresholds()
    sec3b_affine()
    sec4_missing_C()
    sec5_am_vs_gm()
    sec6_epsilon()
    sec7_S3()
    sec8_h_zero()
    sec9_rescaling()
    say("\n%d/%d checks pass" % (PASS[0], PASS[0] + FAIL[0]))
    LOG.close()
    return 0 if FAIL[0] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
