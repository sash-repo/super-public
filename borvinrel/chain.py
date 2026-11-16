#!/usr/bin/env python3
"""The chain of Cartan matrices behind the tensor hierarchy algebras of [CP3].

For a finite-dimensional simple Lie algebra g of rank r one builds
    g^{(0)} = g,  g^{(1)} = g^+ (affine),  g^{(2)} = g^{++} (over-extended), ...
by attaching node -n+1 to node -n+2 with a single line; A^{(n)} is the Cartan
matrix of g^{(n)}, of size r+n.  The tensor hierarchy algebras S(g^{(n)}),
W(g^{(n)}), B(g^{(n)}) carry the Cartan matrix B^{(n+1)} of size r+n+1: it is
A^{(n+1)} with the outermost node -n turned grey (0 on the diagonal).

What is verified here, exactly over Q (Fraction arithmetic only):

  (C1) their eq. (4.3):  det B^{(n+1)} = - det A^{(n-1)};
  (C2) the grading element by the grey node lies in the span of ALL the h_i
       -- i.e. in the Cartan subalgebra of the contragredient g(B,I), and of
       W(g^{(n)}) -- if and only if det B^{(n+1)} != 0;
  (C3) it lies in the span of the h_i with i != grey -- the Cartan subalgebra
       of S(g^{(n)}), from which h_{-n} is deleted -- if and only if A^{(n)} is
       DEGENERATE, i.e. iff g^{(n)} is the affine algebra, i.e. iff n = 1;
  (C4) when B^{(n+1)} is singular (n = 2), its null vector is the affine null
       root delta of the subdiagram g^+ , extended by 0 on the other bosonic
       nodes and by -1 on the grey node.

Node order used everywhere: [grey, -n+1, -n+2, ..., -1, 0, 1, ..., r], i.e.
outermost first, so that the finite g occupies the last r positions and the
affine node is the one just before them.
"""

import sys
from fractions import Fraction

# ---------------------------------------------------------------------------
# The AFFINE Cartan matrices are the input: g^{(1)} is what the whole series is
# built on, and deleting its node 0 gives g = g^{(0)}.  Convention:
# A_ij = 2(alpha_i,alpha_j)/(alpha_i,alpha_i), so that the right null vector of
# an affine matrix is the vector of marks a_j of delta = sum a_j alpha_j.
# Node 0 (the affine node) comes first in every matrix below; the marks, taken
# from Kac's tables, are checked against the matrix rather than assumed.

AFFINE = {
    # A_1^{(1)} : 0 <=> 1
    "A_1": ([[2, -2],
             [-2, 2]], [1, 1]),
    # A_2^{(1)} : the triangle
    "A_2": ([[2, -1, -1],
             [-1, 2, -1],
             [-1, -1, 2]], [1, 1, 1]),
    # A_3^{(1)} : the square 0-1-2-3-0
    "A_3": ([[2, -1, 0, -1],
             [-1, 2, -1, 0],
             [0, -1, 2, -1],
             [-1, 0, -1, 2]], [1, 1, 1, 1]),
    # B_3^{(1)} : nodes 0 and 1 both attached to 2, then 2 => 3 (alpha_3 short)
    "B_3": ([[2, 0, -1, 0],
             [0, 2, -1, 0],
             [-1, -1, 2, -1],
             [0, 0, -2, 2]], [1, 1, 2, 2]),
    # C_3^{(1)} : 0 => 1 - 2 <= 3 (alpha_0, alpha_3 long)
    "C_3": ([[2, -1, 0, 0],
             [-2, 2, -1, 0],
             [0, -1, 2, -2],
             [0, 0, -1, 2]], [1, 2, 2, 1]),
    # D_4^{(1)} : the star, the central node last
    "D_4": ([[2, 0, 0, 0, -1],
             [0, 2, 0, 0, -1],
             [0, 0, 2, 0, -1],
             [0, 0, 0, 2, -1],
             [-1, -1, -1, -1, 2]], [1, 1, 1, 1, 2]),
    # G_2^{(1)} : 0 - 1 == 2 (triple bond, alpha_2 short)
    "G_2": ([[2, -1, 0],
             [-1, 2, -1],
             [0, -3, 2]], [1, 2, 3]),
    # E_8^{(1)} = E_9 : Bourbaki E_8 with node 0 attached to node 8
    "E_8": None,          # built below
}


def _affine_E8():
    # nodes: 0, then Bourbaki 1..8; edges 1-3, 3-4, 4-5, 5-6, 6-7, 7-8, 2-4,
    # and the affine node 0 attached to node 8
    n = 9
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 2
    edges = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (2, 4), (0, 8)]
    for a, b in edges:
        M[a][b] = M[b][a] = -1
    return M, [1, 2, 3, 4, 6, 5, 4, 3, 2]


AFFINE["E_8"] = _affine_E8()


# ---------------------------------------------------------------------------
# exact linear algebra over Q


def det(M):
    n = len(M)
    M = [[Fraction(x) for x in row] for row in M]
    d = Fraction(1)
    for c in range(n):
        p = None
        for rr in range(c, n):
            if M[rr][c] != 0:
                p = rr
                break
        if p is None:
            return Fraction(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        inv = Fraction(1) / M[c][c]
        for rr in range(c + 1, n):
            f = M[rr][c] * inv
            if f:
                for cc in range(c, n):
                    M[rr][cc] -= f * M[c][cc]
    return d


def solve(rows, rhs):
    """Solve x . rows = rhs, i.e. sum_i x_i rows[i][j] = rhs[j].

    rows is a list of m vectors of length n; returns a solution x (length m)
    or None.  Exact.
    """
    m = len(rows)
    n = len(rhs)
    # transpose: unknowns x_1..x_m, equations indexed by j
    M = [[Fraction(rows[i][j]) for i in range(m)] + [Fraction(rhs[j])]
         for j in range(n)]
    piv = []
    r = 0
    for c in range(m):
        p = None
        for rr in range(r, n):
            if M[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for rr in range(n):
            if rr != r and M[rr][c] != 0:
                f = M[rr][c]
                M[rr] = [a - f * b for a, b in zip(M[rr], M[r])]
        piv.append(c)
        r += 1
        if r == n:
            break
    for rr in range(r, n):
        if all(M[rr][c] == 0 for c in range(m)) and M[rr][m] != 0:
            return None
    x = [Fraction(0)] * m
    for k, c in enumerate(piv):
        x[c] = M[k][m]
    return x


def nullspace(M):
    """Basis of {v : M v = 0}, M a list of rows.  Exact, rows returned."""
    if not M:
        return []
    n = len(M[0])
    R = [[Fraction(x) for x in row] for row in M]
    piv = []
    r = 0
    for c in range(n):
        p = None
        for rr in range(r, len(R)):
            if R[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        inv = Fraction(1) / R[r][c]
        R[r] = [x * inv for x in R[r]]
        for rr in range(len(R)):
            if rr != r and R[rr][c] != 0:
                f = R[rr][c]
                R[rr] = [a - f * b for a, b in zip(R[rr], R[r])]
        piv.append(c)
        r += 1
        if r == len(R):
            break
    free = [c for c in range(n) if c not in piv]
    basis = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for k, c in enumerate(piv):
            v[c] = -R[k][fc]
        basis.append(v)
    return basis


# ---------------------------------------------------------------------------
# the chain


def A_chain(name, n):
    """Cartan matrix of g^{(n)}, nodes ordered [-n+1, ..., -1, 0, 1, ..., r].

    n = 0 gives g itself, n = 1 the affine algebra g^+; for n >= 2 a node is
    prepended at each step and joined by a single line to the previous one.
    """
    M, _marks = AFFINE[name]
    if n == 0:                        # delete the affine node
        return [[M[i][j] for j in range(1, len(M))] for i in range(1, len(M))]
    M = [row[:] for row in M]
    for _ in range(n - 1):
        size = len(M) + 1
        N = [[0] * size for _ in range(size)]
        for i in range(size - 1):
            for j in range(size - 1):
                N[1 + i][1 + j] = M[i][j]
        N[0][0] = 2
        N[0][1] = N[1][0] = -1
        M = N
    return M


def B_chain(name, n):
    """Cartan matrix B^{(n+1)} of S(g^{(n)}): A^{(n)} with a grey node in front.

    Nodes ordered [grey, -n+1, ..., 0, 1, ..., r]; grey is index 0.
    """
    A = A_chain(name, n)
    size = len(A) + 1
    B = [[0] * size for _ in range(size)]
    for i in range(len(A)):
        for j in range(len(A)):
            B[1 + i][1 + j] = A[i][j]
    B[0][0] = 0                        # isotropic (grey) node
    B[0][1] = B[1][0] = -1
    return B


def grading_element(B, grey, drop=None):
    """Solve sum_i c_i B_ij = delta_{j,grey} over the rows i != drop."""
    n = len(B)
    idx = [i for i in range(n) if i != drop]
    rows = [B[i] for i in idx]
    rhs = [1 if j == grey else 0 for j in range(n)]
    return solve(rows, rhs), idx


# ---------------------------------------------------------------------------


def main():
    log = open("chain.log", "w")

    def say(s=""):
        print(s)
        log.write(s + "\n")

    ok = 0
    bad = 0

    def check(cond, msg):
        nonlocal ok, bad
        if cond:
            ok += 1
            say("  OK   " + msg)
        else:
            bad += 1
            say("  FAIL " + msg)

    say("The chain of Cartan matrices of [CP3], exact over Q.")
    say("")

    names = ["A_1", "A_2", "A_3", "B_3", "C_3", "D_4", "G_2", "E_8"]

    say("Calibration of the input: every matrix listed as affine is singular,")
    say("and the marks of Kac's tables span its kernel and are positive.")
    for name in names:
        M, marks = AFFINE[name]
        check(det(M) == 0, "%s: the affine matrix is singular" % name)
        check(all(sum(M[i][j] * marks[j] for j in range(len(M))) == 0
                  for i in range(len(M))),
              "%s: the marks %s span its kernel" % (name, marks))
        check(all(a > 0 for a in marks),
              "%s: the marks are positive (Vinberg's type Z)" % name)
    say("")

    for name in names:
        say("=" * 70)
        say("g = %s" % name)
        dets_A = {}
        for m in range(0, 6):
            dets_A[m] = det(A_chain(name, m))
        say("  det A^{(m)}, m = 0..5 : " +
            ", ".join(str(dets_A[m]) for m in range(6)))
        check(dets_A[1] == 0, "%s: A^{(1)} (affine) is singular" % name)
        check(all(dets_A[m] != 0 for m in [0, 2, 3, 4, 5]),
              "%s: A^{(m)} is non-singular for m = 0, 2, 3, 4, 5" % name)
        check(all(dets_A[m] == (1 - m) * dets_A[0] for m in range(6)),
              "%s: det A^{(m)} = (1-m) det A for m = 0..5" % name)
        check(all(det(B_chain(name, n)) == (n - 2) * dets_A[0]
                  for n in range(1, 5)),
              "%s: det B^{(n+1)} = (n-2) det A for n = 1..4" % name)

        for n in range(1, 5):
            B = B_chain(name, n)
            dB = det(B)
            # (C1) their eq. (4.3)
            if n >= 1:
                expect = -dets_A[n - 1]
                check(dB == expect,
                      "%s, n=%d: det B^{(n+1)} = %s = -det A^{(n-1)}"
                      % (name, n, dB))
            # (C2) grading element in the span of ALL the h_i
            cAll, _ = grading_element(B, grey=0, drop=None)
            check((cAll is not None) == (dB != 0),
                  "%s, n=%d: grading element inner in g(B,I) iff det B != 0 "
                  "(%s, det B = %s)"
                  % (name, n, "yes" if cAll is not None else "no", dB))
            # (C3) grading element in the span of the h_i with i != grey
            cS, _ = grading_element(B, grey=0, drop=0)
            check((cS is not None) == (dets_A[n] == 0),
                  "%s, n=%d: grading element inner in S (no h_grey) iff "
                  "A^{(n)} singular (%s, det A^{(n)} = %s)"
                  % (name, n, "yes" if cS is not None else "no", dets_A[n]))
            # (C3') Cramer: the (p,p) entry of the inverse of A^{(n)}, p being
            # the outermost bosonic node -- the one the grey node is attached
            # to -- is det A^{(n-1)} / det A^{(n)} = (2-n)/(1-n).  It is what
            # decides everything, and it is a RATIO OF TWO NEIGHBORS.
            if dets_A[n] != 0:
                An = A_chain(name, n)
                m = len(An)
                minor = [[An[i][j] for j in range(1, m)] for i in range(1, m)]
                entry = det(minor) / dets_A[n]
                check(entry == Fraction(2 - n, 1 - n) if n != 1 else True,
                      "%s, n=%d: (A^{(n)}^{-1})_{pp} = %s = det A^{(n-1)}/"
                      "det A^{(n)} = (2-n)/(1-n)" % (name, n, entry))
            # (C4) the null vector of a singular B
            if dB == 0:
                ker = nullspace(B)
                check(len(ker) == 1,
                      "%s, n=%d: dim ker B^{(n+1)} = %d" % (name, n, len(ker)))
                v = ker[0]
                # normalize so that the affine-node entry is 1: the affine node
                # sits at index n (grey = 0, then nodes -n+1 .. 0)
                aff = n
                v = [x / v[aff] for x in v]
                # the affine null root of g^+ : kernel of A^{(1)}
                kerA = nullspace(A_chain(name, 1))
                d = kerA[0]
                d = [x / d[0] for x in d]         # a_0 = 1
                # compare: on the bosonic nodes v should be 0 on the nodes
                # -n+1..-1 and delta on the nodes 0..r
                tail = v[n:]
                check(tail == d,
                      "%s, n=%d: on the affine subdiagram the null vector is "
                      "delta = %s" % (name, n, [str(x) for x in d]))
                check(all(x == 0 for x in v[1:n]),
                      "%s, n=%d: it vanishes on the intermediate bosonic nodes"
                      % (name, n))
                check(v[0] == -1,
                      "%s, n=%d: and equals %s on the grey node"
                      % (name, n, v[0]))
        say("")

    say("=" * 70)
    say("checks: %d OK, %d FAIL" % (ok, bad))
    log.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
