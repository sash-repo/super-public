#!/usr/bin/env python3
"""What a (generalized) Shuhan diagram is.  Exact checks.

Definition 1.1 of [WZ] (the term is [WLX]'s): a Shuhan diagram is an undirected
graph Gamma, with labels on the vertices only, such that

  (i)   there is a bijection phi from I = {1, ..., n} onto the vertices;
  (ii)  the vertex phi(i) carries a label x_i in Z_+;
  (iii) the number of edges between phi(i) and phi(j) is lambda_ij = lambda_ji,
        and lambda_ij is 0 or 1;
  (iv)  x_i = sum_{j != i} (1/2) lambda_ij x_j   for every i in I.

Definition 1.2 of [WZ] (generalized Shuhan diagram) keeps (i), (ii), (iv) and
relaxes (iii): the number of edges between phi(i) and phi(j) is k in N, and for
k > 1 the pair (lambda_ij, lambda_ji) is (1, k), (k, 1) or (k, k).

Put Lambda = (lambda_ij) and A := 2E - Lambda.  Then A is a matrix with 2 on
the main diagonal and non-positive integers off it -- a generalized Cartan
matrix -- and condition (iv) reads

                              A x = 0,   x in Z_+^n,

which is Kac's characterization of the matrices of affine type (Kac, Infinite
dimensional Lie algebras, Thm 4.3), and x is then the null root.  So a
generalized Shuhan diagram is a generalized Cartan matrix of affine type
together with a positive integer vector in its kernel; the kernel is a line, so
the label vector is determined by the diagram up to a positive factor, and
"minimal label 1" picks its primitive vector, the marks of Kac's Tables Aff 1,
Aff 2, Aff 3.

What is verified below, exactly (integer / Fraction arithmetic throughout):

 1. The bridge: on every diagram used here, condition (iv) holds for x if and
    only if A x = 0.
 2. Exhaustively over ALL graphs on n <= 7 vertices (Definition 1.1, so
    lambda in {0, 1}): a connected graph carries a positive integer label
    vector if and only if it is of Vinberg type Z, if and only if it is one of
    A~_n (n >= 2), D~_n (n >= 4), E~_6, E~_7 -- which is the list of [WLX].
 3. Exhaustively over ALL diagrams on n <= 4 vertices with multiplicities <= 4
    for n <= 3 and <= 3 for n = 4 (Definition 1.2, and also the pairs that
    Definition 1.2 forbids): the same equivalence, and no diagram having a pair
    with lambda_ij, lambda_ji >= 2, lambda_ij != lambda_ji, carries a positive
    label vector.  So the convention of Definition 1.2 excludes nothing.
 4. The affine families up to rank 10: the marks are the null vector.
 5. The label vector of a diagram is unique up to a positive rational factor
    (the kernel is one-dimensional), so the labels are the marks times a
    positive integer.

Usage:  tools/venv/bin/python shudiag.py   (falls back to system python3)
"""

import itertools
import sys
from fractions import Fraction

import shuhan as S

LOG = None
PASS = [0]
FAIL = [0]


def say(msg=""):
    print(msg)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


def check(name, cond, detail=""):
    if cond:
        PASS[0] += 1
        say("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
    else:
        FAIL[0] += 1
        say("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))


# --------------------------------------------------------------- definitions

def cond_iv(Lam, x):
    """Condition (iv) of Definitions 1.1 and 1.2, verbatim."""
    n = len(Lam)
    for i in range(n):
        s = sum(Fraction(Lam[i][j], 2) * x[j] for j in range(n) if j != i)
        if s != x[i]:
            return False
    return True


def gcm(Lam):
    """A := 2E - Lambda."""
    n = len(Lam)
    return [[-Lam[i][j] if i != j else 2 for j in range(n)] for i in range(n)]


def is_def11(Lam):
    n = len(Lam)
    return all(Lam[i][j] == Lam[j][i] and Lam[i][j] in (0, 1)
               for i in range(n) for j in range(i + 1, n))


def is_def12(Lam):
    """(iii) of Definition 1.2: k edges, and for k > 1 the pair is
    (1, k), (k, 1) or (k, k)."""
    n = len(Lam)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = Lam[i][j], Lam[j][i]
            if a < 0 or b < 0:
                return False
            if a == b:
                continue                      # k = a = b (covers k <= 1)
            if 1 in (a, b):
                continue                      # (1, k) or (k, 1)
            return False
    return True


def connected(Lam):
    n = len(Lam)
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for w in range(n):
            if w not in seen and (Lam[v][w] or Lam[w][v]):
                seen.add(w)
                stack.append(w)
    return len(seen) == n


# --------------------------------------------------------------- linear part

def det_int(M):
    """Exact determinant of an integer matrix (Bareiss)."""
    n = len(M)
    if n == 0:
        return 1
    A = [row[:] for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    A[k], A[r] = A[r], A[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def kernel(M):
    """Basis of the kernel of an integer matrix, as rows of Fractions."""
    n, m = len(M), len(M[0])
    A = [[Fraction(v) for v in row] for row in M]
    piv, where = [], [-1] * m
    r = 0
    for c in range(m):
        p = next((i for i in range(r, n) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = A[r][c]
        A[r] = [v / inv for v in A[r]]
        for i in range(n):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][j] - f * A[r][j] for j in range(m)]
        where[c] = r
        piv.append(c)
        r += 1
    free = [c for c in range(m) if where[c] == -1]
    basis = []
    for fc in free:
        v = [Fraction(0)] * m
        v[fc] = Fraction(1)
        for c in piv:
            v[c] = -A[where[c]][fc]
        basis.append(v)
    return basis


def primitive_positive_kernel(A):
    """The primitive positive integer vector in ker A, or None.

    Returns (x, dim ker).  x is normalized so that gcd(x) = 1."""
    K = kernel(A)
    if not K:
        return None, 0
    if len(K) > 1:
        # a positive vector may still exist; the affine case has dim 1, and
        # this branch is reported rather than silently dropped.
        return None, len(K)
    v = K[0]
    if all(c > 0 for c in v) or all(c < 0 for c in v):
        if v[0] < 0:
            v = [-c for c in v]
        den = 1
        for c in v:
            den = den * c.denominator // _gcd(den, c.denominator)
        w = [int(c * den) for c in v]
        g = 0
        for c in w:
            g = _gcd(g, c)
        return [c // g for c in w], 1
    return None, 1


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def vinberg(Lam):
    return S.vinberg_type(gcm(Lam))


# ------------------------------------------------------ the classical answers

def tilde_simply_laced(m):
    """The simply-laced affine diagrams on m vertices, as Lambda matrices:
    A~_{m-1} (a cycle, m >= 3), D~_{m-1} (m >= 5), E~_6 (m = 7), E~_7 (m = 8)."""
    out = {}
    if m >= 3:
        L = [[0] * m for _ in range(m)]
        for i in range(m):
            L[i][(i + 1) % m] = L[(i + 1) % m][i] = 1
        out["A~_%d" % (m - 1)] = L
    if m >= 5:                                  # D~_{m-1}: forks at both ends
        k = m - 1                               # D_k^{(1)} has k+1 = m nodes
        L = [[0] * m for _ in range(m)]
        ch = list(range(2, m - 2))              # spine 2, ..., m-3
        for a, b in zip(ch, ch[1:]):
            L[a][b] = L[b][a] = 1
        for a, b in ((0, 2), (1, 2), (m - 2, m - 3), (m - 1, m - 3)):
            L[a][b] = L[b][a] = 1
        out["D~_%d" % k] = L
    if m == 7:                                  # E~_6: three arms of length 2
        L = [[0] * 7 for _ in range(7)]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)):
            L[a][b] = L[b][a] = 1
        out["E~_6"] = L
    if m == 8:                                  # E~_7: arms 3, 3, 1
        L = [[0] * 8 for _ in range(8)]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7)):
            L[a][b] = L[b][a] = 1
        out["E~_7"] = L
    return out


def iso(L1, L2):
    """Are the two Lambda matrices equal up to a permutation of the nodes?"""
    n = len(L1)
    if n != len(L2):
        return False
    for p in itertools.permutations(range(n)):
        if all(L1[i][j] == L2[p[i]][p[j]] for i in range(n) for j in range(n)):
            return True
    return False


# ----------------------------------------------------------------- section 0

def sec0_bridge():
    say("\n0. The bridge: (iv) is A x = 0 for A = 2E - Lambda")
    tested = 0
    for m in range(3, 9):
        for name, L in tilde_simply_laced(m).items():
            A = gcm(L)
            x, d = primitive_positive_kernel(A)
            ok_iv = x is not None and cond_iv(L, x)
            ok_ker = x is not None and all(
                sum(A[i][j] * x[j] for j in range(len(L))) == 0
                for i in range(len(L)))
            check("(iv) <=> A x = 0 on %s" % name, ok_iv and ok_ker and d == 1,
                  "labels %s" % (x,))
            tested += 1
            # and a vector that is not in the kernel fails (iv)
            y = list(x)
            y[0] += 1
            check("(iv) fails off the kernel, %s" % name, not cond_iv(L, y))
    check("diagrams tested in section 0", tested >= 12, "%d" % tested)


# ----------------------------------------------------------------- section 1

def all_graphs(n):
    """All graphs on n labelled vertices, as Lambda matrices."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for bits in range(1 << len(pairs)):
        L = [[0] * n for _ in range(n)]
        for b, (i, j) in enumerate(pairs):
            if bits >> b & 1:
                L[i][j] = L[j][i] = 1
        yield L


def sec1_simply_laced():
    say("\n1. Definition 1.1, exhaustively for n <= 7")
    total = 0
    for n in range(2, 8):
        found, zero_det, conn = [], 0, 0
        for L in all_graphs(n):
            if not connected(L):
                continue
            conn += 1
            A = gcm(L)
            if det_int(A) != 0:
                continue                 # kernel 0: neither type Z nor labels
            zero_det += 1
            x, d = primitive_positive_kernel(A)
            t = vinberg(L)
            if (x is not None) != (t == "Z"):
                check("n=%d: labels <=> type Z" % n, False, "%s" % (L,))
            if x is not None:
                found.append((L, x))
        total += conn
        expected = tilde_simply_laced(n)
        names = []
        for L, x in found:
            nm = [k for k, M in expected.items() if iso(L, M)]
            names.append(nm[0] if nm else "UNKNOWN %s" % (L,))
        got = sorted(set(names))
        check("n=%d: the diagrams with positive labels" % n,
              got == sorted(expected.keys()),
              "%d connected, %d with det 0, %d with labels: %s"
              % (conn, zero_det, len(found), ", ".join(got) if got else "none"))
        for L, x in found:
            check("n=%d: min label 1, Definition 1.1" % n,
                  min(x) == 1 and is_def11(L) and cond_iv(L, x),
                  "labels %s" % (x,))
    say("  (%d connected graphs examined)" % total)


# ----------------------------------------------------------------- section 2

def all_diagrams(n, kmax):
    """All Lambda matrices on n nodes with 0 <= lambda_ij <= kmax, including
    the pairs that Definition 1.2 forbids."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    opts = [(0, 0)]
    for a in range(1, kmax + 1):
        for b in range(1, kmax + 1):
            opts.append((a, b))
    for combo in itertools.product(opts, repeat=len(pairs)):
        L = [[0] * n for _ in range(n)]
        for (i, j), (a, b) in zip(pairs, combo):
            L[i][j], L[j][i] = a, b
        yield L


def sec2_generalized():
    say("\n2. Definition 1.2, exhaustively for n <= 4")
    for n, kmax in ((2, 4), (3, 4), (4, 3)):
        conn = labels = bad_pair = off12 = 0
        mismatch = None
        for L in all_diagrams(n, kmax):
            if not connected(L):
                continue
            conn += 1
            A = gcm(L)
            if det_int(A) != 0:
                continue
            x, d = primitive_positive_kernel(A)
            t = vinberg(L)
            if (x is not None) != (t == "Z") and mismatch is None:
                mismatch = L
            if x is None:
                continue
            labels += 1
            if not is_def12(L):
                off12 += 1
            if any(L[i][j] >= 2 and L[j][i] >= 2 and L[i][j] != L[j][i]
                   for i in range(n) for j in range(i + 1, n)):
                bad_pair += 1
            if min(x) != 1 or d != 1:
                check("n=%d: min label 1 and corank 1" % n, False, "%s" % (L,))
        check("n=%d, multiplicities <= %d: labels <=> Vinberg type Z"
              % (n, kmax), mismatch is None,
              "%d connected diagrams, %d with labels" % (conn, labels))
        check("n=%d: every labelled diagram obeys Definition 1.2" % n,
              off12 == 0, "%d violations" % off12)
        check("n=%d: no labelled diagram has a pair 2 <= a != b >= 2" % n,
              bad_pair == 0,
              "so the convention of Definition 1.2 loses nothing here")


# ----------------------------------------------------------------- section 4

def sec4_families():
    say("\n4. The affine families up to rank 10: the labels are the marks")
    names = (["A%d(1)" % k for k in range(1, 10)] +
             ["B%d(1)" % k for k in range(2, 10)] +
             ["C%d(1)" % k for k in range(2, 10)] +
             ["D%d(1)" % k for k in range(4, 10)] +
             ["E6(1)", "E7(1)", "E8(1)", "G2(1)", "F4(1)", "A2(2)"])
    for nm in names:
        A = S.cartan_affine(nm)
        L = [[-A[i][j] if i != j else 0 for j in range(len(A))]
             for i in range(len(A))]
        x, d = primitive_positive_kernel(A)
        ok = (x is not None and d == 1 and min(x) == 1 and cond_iv(L, x)
              and is_def12(L))
        extra = ""
        if nm.endswith("(1)"):
            # for an untwisted affine diagram the marks sum to the Coxeter
            # number of the finite diagram it extends
            h = S.COXETER[nm[0]](int(nm[1:-3]))
            ok = ok and sum(x) == h
            extra = ", Coxeter number %d" % h
        check("%s" % nm, ok,
              "marks %s, sum %d%s" % (x, sum(x) if x else 0, extra))


# ----------------------------------------------------------------- section 5

def sec5_scaling():
    say("\n5. The label vector is the marks times a positive integer")
    for m in range(3, 9):
        for name, L in tilde_simply_laced(m).items():
            A = gcm(L)
            x, d = primitive_positive_kernel(A)
            ok = d == 1
            for c in (1, 2, 3, 7):
                ok = ok and cond_iv(L, [c * v for v in x])
            # nothing else: any solution is a multiple of x
            K = kernel(A)
            ok = ok and len(K) == 1
            check("%s: labels unique up to a factor" % name, ok,
                  "dim ker = %d" % len(K))


def main():
    global LOG
    LOG = open("shudiag.log", "w")
    say("shudiag.py -- what a (generalized) Shuhan diagram is")
    say("python %s" % sys.version.split()[0])
    sec0_bridge()
    sec1_simply_laced()
    sec2_generalized()
    sec4_families()
    sec5_scaling()
    say("\n%d/%d checks pass" % (PASS[0], PASS[0] + FAIL[0]))
    LOG.close()
    return 0 if FAIL[0] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
