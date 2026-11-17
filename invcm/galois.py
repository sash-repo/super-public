#!/usr/bin/env python3
"""Subsection 9.4: the Galois involution and the norms of the determinants, for the five
almost affine Lie superalgebras NS3_82 -- NS3_86 of Subsection 9.3.

Everything is exact, in the real quadratic field Q(alpha) of each algebra (sympy).
What is computed:

  1) formulas (55) and (56): alpha * (1/alpha) = 1 and alpha + 1/alpha is rational, so 1/alpha is
     the Galois conjugate of alpha (alpha is a unit of norm 1);
  2) the orbit of the Cartan matrix A_1 under repeated odd reflections (24), each matrix
     normalized only by making its non-zero diagonal elements equal to 2 (rows with a 0 on
     the diagonal are not rescaled, so this classification is finer than the equivalence
     of Subsection 2.4; for that one, see cases.py), with the determinant of each member;
  3) A_3 and A_4 lie in that orbit; sigma(A_3) = A_4 up to a permutation of the nodes;
     formula (57), det A_4 = sigma(det A_3); and Corollary 9.4.2, formula (58): the norms
     N(det A) = -3, -6, -9, -8, -16;
  4) the signs of the determinants met in the orbit of item 2).

Usage:  python galois.py
"""
import itertools
import sympy as sp

a = sp.Symbol("alpha")
M = sp.Matrix

# ---------------------------------------------------------------- the five algebras
# name -> (alpha, t), where t = 2 for NS3_82--84 and t = 3 for NS3_85, NS3_86
ALG = {
    "NS3_82": ((sp.sqrt(21) - 11) / 10, 2),
    "NS3_83": ((sp.sqrt(15) - 8) / 7, 2),
    "NS3_84": ((sp.sqrt(13) - 7) / 6, 2),
    "NS3_85": (2 * sp.sqrt(6) / 5 - sp.Rational(7, 5), 3),
    "NS3_86": ((sp.sqrt(5) - 3) / 2, 3),
}

# the printed entries of Subsection 9.3;
# name -> list of (label, A, printed inverse, printed det)
def norm(x, al):
    """N_{K/Q}(x): apply the Galois involution al -> 1/al and multiply."""
    return sp.radsimp(sp.simplify(x * gal(x, al)))


def gal(expr, al):
    """The Galois conjugate: alpha and 1/alpha are conjugate, so substitute."""
    return sp.radsimp(sp.simplify(expr.subs(sp.sqrt(rad(al)), -sp.sqrt(rad(al)))))


def rad(al):
    """The radicand of the quadratic irrationality al."""
    for x in sp.preorder_traversal(sp.nsimplify(al)):
        if x.is_Pow and x.exp == sp.Rational(1, 2):
            return x.base
    raise ValueError(al)


# ---------------------------------------------------------------- reflections
def refl(A, k):
    """The odd reflection r_k of formula (24); A[k][k] must be 0."""
    n = A.rows
    assert sp.simplify(A[k, k]) == 0
    b = [sp.Integer(0)] * n
    c = [sp.Integer(0)] * n
    for i in range(n):
        if i == k:
            b[i] = sp.Integer(-2)
            c[i] = sp.Integer(-2)
        else:
            b[i] = sp.simplify(A[i, k] / A[k, i]) if sp.simplify(A[i, k]) != 0 else sp.Integer(0)
            c[i] = sp.Integer(1) if sp.simplify(A[i, k]) != 0 else sp.Integer(0)
    R = sp.zeros(n)
    for i in range(n):
        for j in range(n):
            R[i, j] = sp.simplify(A[i, j] + b[i] * A[k, j] + c[j] * A[i, k])
    return R


def normalize(A):
    """Rescale a row only when its diagonal entry is non-zero, so as to make that
    entry equal to 2; a row with a 0 on the diagonal is left alone.  This is the
    convention the tables of Subsection 9.3 are printed in.  (Warning 2.4.1 allows
    rescaling the zero-diagonal rows as well; that coarser equivalence is the one
    used in cases.py.)"""
    n = A.rows
    R = A.as_mutable()
    for i in range(n):
        d = sp.simplify(R[i, i])
        lam = sp.simplify(2 / d) if d != 0 else sp.Integer(1)
        for j in range(n):
            R[i, j] = sp.radsimp(sp.simplify(lam * R[i, j]))
    return R


def perm(A, p):
    """Simultaneous permutation of rows and columns."""
    n = A.rows
    return M(n, n, lambda i, j: A[p[i], p[j]])


def key(A):
    """A canonical form modulo node permutation and rescaling."""
    best = None
    for p in itertools.permutations(range(A.rows)):
        B = normalize(perm(A, p))
        s = str([[sp.nsimplify(sp.radsimp(B[i, j])) for j in range(B.cols)] for i in range(B.rows)])
        if best is None or s < best:
            best = s
    return best


print("=" * 78)
print("1)  alpha is a unit of norm 1 of the real quadratic field Q(alpha)")
print("=" * 78)
for nm, (al, t) in ALG.items():
    n = sp.radsimp(sp.simplify(al * gal(al, al)))
    tr = sp.radsimp(sp.simplify(al + gal(al, al)))
    inv = sp.radsimp(sp.simplify(1 / al))
    ok = sp.simplify(inv - gal(al, al)) == 0
    print("   %-8s alpha = %-22s  N(alpha) = %s,  alpha + 1/alpha = %-6s  1/alpha = bar alpha: %s"
          % (nm, al, n, tr, ok))

EQ55 = {"NS3_82": (5, 11), "NS3_83": (7, 16), "NS3_84": (3, 7), "NS3_85": (5, 14), "NS3_86": (1, 3)}
for nm, (al, t) in ALG.items():
    c, b = EQ55[nm]
    print("   %-8s formula (55): %d alpha^2 + %d alpha + %d = %s"
          % (nm, c, b, c, sp.radsimp(sp.expand(c * al**2 + b * al + c))))

print()
print("=" * 78)
print("2)  the reflection orbit of the Cartan matrix 1)")
print("=" * 78)
ORBIT = {}
for nm, (al, t) in ALG.items():
    A1 = M([[0, 1, al], [-1, 0, t + al], [-1, t + 1 / al, 0]])
    orbit = {key(A1): ("1)", normalize(A1))}
    frontier = [A1]
    seen = set(orbit)
    while frontier:
        A = frontier.pop()
        for k in range(3):
            if sp.simplify(A[k, k]) != 0:
                continue
            B = normalize(refl(A, k))
            kb = key(B)
            if kb not in seen:
                seen.add(kb)
                orbit[kb] = ("r_%d" % (k + 1), B)
                frontier.append(B)
    ORBIT[nm] = orbit
    print("   %-8s the orbit has %d classes" % (nm, len(orbit)))
    for kb, (how, B) in orbit.items():
        d = sp.radsimp(sp.simplify(B.det()))
        print("        %-6s det = %-26s = %8.4f  %s"
              % (how, sp.nsimplify(d), float(d), "POSITIVE" if float(d) > 0 else "NEGATIVE"))
        for i in range(3):
            print("             %s" % [sp.nsimplify(sp.radsimp(B[i, j])) for j in range(3)])

print()
print("=" * 78)
print("3)  the matrices A_3 and A_4, and the Galois involution")
print("=" * 78)
# the printed A_3 and A_4 of each algebra
THREE = {
    "NS3_82": M([[2, -1, -1], [-1, 0, 2 + a], [-2, -1, 2]]),
    "NS3_83": M([[2, -1, -1], [-1, 0, 2 + a], [-3, -1, 2]]),
    "NS3_84": M([[2, -1, -1], [-1, 0, 2 + a], [-4, -1, 2]]),
    "NS3_85": M([[2, -1, -2], [-1, 0, 3 + a], [-1, -1, 2]]),
    "NS3_86": M([[2, -1, -2], [-1, 0, 3 + a], [-2, -1, 2]]),
}
FOUR = {
    "NS3_82": M([[2, -1, -1], [-2, 2, -1], [-1, 2 + 1 / a, 0]]),
    "NS3_83": M([[2, -1, -1], [-3, 2, -1], [-1, 2 + 1 / a, 0]]),
    "NS3_84": M([[2, -1, -1], [-4, 2, -1], [-1, 2 + 1 / a, 0]]),
    "NS3_85": M([[2, -2, -1], [-1, 2, -1], [-1, 3 + 1 / a, 0]]),
    "NS3_86": M([[2, -2, -1], [-2, 2, -1], [-1, 3 + 1 / a, 0]]),
}
NORMS = {}
for nm, (al, t) in ALG.items():
    A3 = THREE[nm].subs(a, al)
    A4 = FOUR[nm].subs(a, al)
    d3 = sp.radsimp(sp.simplify(A3.det()))
    d4 = sp.radsimp(sp.simplify(A4.det()))
    k3, k4 = key(A3), key(A4)
    inorb3 = k3 in ORBIT[nm]
    inorb4 = k4 in ORBIT[nm]
    # Galois: sigma(A3) should be a Cartan matrix of the same algebra, equal to A4
    sA3 = A3.applyfunc(lambda x: gal(x, al))
    same = key(sA3) == k4
    conj = sp.simplify(d4 - gal(d3, al)) == 0
    N = sp.radsimp(sp.simplify(d3 * gal(d3, al)))
    NORMS[nm] = N
    print("   %-8s det 3) = %-24s (%8.4f)   det 4) = %-24s (%8.4f)"
          % (nm, sp.nsimplify(d3), float(d3), sp.nsimplify(d4), float(d4)))
    print("            3) in the orbit of 1): %-5s   4) in the orbit of 1): %-5s" % (inorb3, inorb4))
    print("            sigma(3)) = 4) up to node permutation: %-5s   det 4) = sigma(det 3)): %-5s"
          % (same, conj))
    print("            N(det) = %s" % sp.nsimplify(N))

print()
print("=" * 78)
print("4)  the signs of the determinants in the orbit of item 2)")
print("=" * 78)
for nm in ALG:
    signs = []
    for kb, (how, B) in ORBIT[nm].items():
        d = float(sp.simplify(B.det()))
        signs.append("+" if d > 0 else "-")
    print("   %-8s %s   norm N(det 3)) = %s"
          % (nm, " ".join(sorted(signs)), sp.nsimplify(NORMS[nm])))
