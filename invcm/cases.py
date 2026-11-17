#!/usr/bin/env python3
"""Subsection 9.3 and Statement 9.4.3: the four Cartan matrices A_1, ..., A_4 of each of
the almost affine Lie superalgebras NS3_82 -- NS3_86.

For each of NS3_82 -- NS3_86, with alpha the irrational number of Subsection 9.3:

  * -1 < alpha < 0;
  * A_1 is the Cartan matrix all of whose main-diagonal elements are 0 (all three simple
    roots are odd isotropic);
  * A_{k+1} = r_k(A_1) for k = 1, 2, 3, formula (54), where r_k is the reflection (24) in
    the kth node -- up to the normalization of Subsection 2.4 (rescaling of rows, see
    Warning 2.4.1, and a simultaneous permutation of rows and columns);
  * the only isotropic node of A_{k+1} is the kth one;
  * det A_i is the printed value, and its sign at the actual (irrational) alpha;
  * the printed inverse is the inverse, and the mark !!! sits exactly on the inverses
    with no zero element;
  * Statement 9.4.3: A_1, ..., A_4 are pairwise non-equivalent for NS3_82 -- NS3_85,
    whereas for NS3_86 the matrices A_2, A_3, A_4 are pairwise equivalent;
  * the closure of {A_1} under the reflections (24), up to equivalence, consists of the
    classes of A_1, ..., A_4 only (four classes for NS3_82 -- NS3_85, two for NS3_86);
  * Lemma 9.4.1: sigma (alpha -> 1/alpha) maps A_1 and A_2 to matrices equivalent to
    themselves and interchanges the classes of A_3 and A_4; on A_1 it is the
    transposition of the nodes 2 and 3 followed by a rescaling of the first row.

Exact arithmetic in Q(alpha) (sympy).  Takes under a minute.

Usage:  python cases.py
"""
import itertools
import sympy as sp

a = sp.Symbol("alpha")
M = sp.Matrix

# the five values of alpha
ALPHA = {
    "NS3_82": (sp.sqrt(21) - 11) / 10,
    "NS3_83": (sp.sqrt(15) - 8) / 7,
    "NS3_84": (sp.sqrt(13) - 7) / 6,
    "NS3_85": (2 * sp.sqrt(6) - 7) / 5,
    "NS3_86": (sp.sqrt(5) - 3) / 2,
}

# the matrix 1) of each family: t = 2 for NS3_82--NS3_84, t = 3 for NS3_85, NS3_86
def one(t):
    return M([[0, 1, a], [-1, 0, t + a], [-1, t + sp.Rational(1, 1) / a, 0]])


PRINTED = {
    # name : (A, prefactor, numerator of A^{-1}, det, attn)
    "NS3_82-1": (one(2), -sp.Rational(1, 3) / (a + 1),
                 M([[-2 * a - 5 - 2 / a, 2 * a + 1, a + 2],
                    [-a - 2, a, -a],
                    [-2 - 1 / a, -1, 1]]), -3 * (a + 1), True),
    "NS3_82-2": (M([[0, 1, a], [-1, 2, -1], [-1, -1, 2]]),
                 -sp.Rational(1, 3) / (a + 1),
                 M([[-3, a + 2, 2 * a + 1], [-3, -a, a], [-3, 1, -1]]),
                 3 * (a + 1), True),
    "NS3_82-3": (M([[2, -1, -1], [-1, 0, 2 + a], [-2, -1, 2]]),
                 1 / (4 * a + 5),
                 M([[a + 2, 3, -a - 2], [-2 * a - 2, 2, -2 * a - 3], [1, 4, -1]]),
                 4 * a + 5, True),
    "NS3_82-4": (M([[2, -1, -1], [-2, 2, -1], [-1, 2 + 1 / a, 0]]),
                 1 / (5 * a + 4),
                 M([[2 * a + 1, -2 * a - 1, 3 * a], [a, -a, 4 * a],
                    [-2 * (a + 1), -3 * a - 2, 2 * a]]), 5 + 4 / a, True),
    "NS3_83-1": (one(2), -sp.Rational(1, 3) / (a + 1),
                 M([[-2 * a - 5 - 2 / a, 2 * a + 1, a + 2],
                    [-a - 2, a, -a],
                    [-2 - 1 / a, -1, 1]]), -3 * (a + 1), True),
    "NS3_83-2": (M([[0, 1, a], [-1, 2, -1], [-1, -1, 2]]),
                 -sp.Rational(1, 3) / (a + 1),
                 M([[-3, a + 2, 2 * a + 1], [-3, -a, a], [-3, 1, -1]]),
                 3 * (a + 1), True),
    "NS3_83-3": (M([[2, -1, -1], [-1, 0, 2 + a], [-3, -1, 2]]),
                 1 / (5 * a + 7),
                 M([[a + 2, 3, -a - 2], [-3 * a - 4, 1, -2 * a - 3], [1, 5, -1]]),
                 5 * a + 7, True),
    "NS3_83-4": (M([[2, -1, -1], [-3, 2, -1], [-1, 2 + 1 / a, 0]]),
                 1 / (7 * a + 5),
                 M([[2 * a + 1, -2 * a - 1, 3 * a], [a, -a, 5 * a],
                    [-4 * a - 3, -3 * a - 2, a]]), 7 + 5 / a, True),
    "NS3_84-1": (one(2), -sp.Rational(1, 3) / (a + 1),
                 M([[-2 * a - 5 - 2 / a, 2 * a + 1, a + 2],
                    [-a - 2, a, -a],
                    [-2 - 1 / a, -1, 1]]), -3 * (a + 1), True),
    "NS3_84-2": (M([[0, 1, a], [-1, 2, -1], [-1, -1, 2]]),
                 -sp.Rational(1, 3) / (a + 1),
                 M([[-3, a + 2, 2 * a + 1], [-3, -a, a], [-3, 1, -1]]),
                 3 * (a + 1), True),
    "NS3_84-3": (M([[2, -1, -1], [-1, 0, 2 + a], [-4, -1, 2]]),
                 sp.Rational(1, 3) / (2 * a + 3),
                 M([[a + 2, 3, -a - 2], [-2 * (2 * a + 3), 0, -2 * a - 3], [1, 6, -1]]),
                 3 * (2 * a + 3), False),
    "NS3_84-4": (M([[2, -1, -1], [-4, 2, -1], [-1, 2 + 1 / a, 0]]),
                 sp.Rational(1, 3) / (3 * a + 2),
                 M([[2 * a + 1, -2 * a - 1, 3 * a], [a, -a, 6 * a],
                    [-2 * (3 * a + 2), -3 * a - 2, 0]]), 3 * (3 + 2 / a), False),
    "NS3_85-1": (one(3), -sp.Rational(1, 4) / (a + 1),
                 M([[-3 * a - 10 - 3 / a, 3 * a + 1, a + 3],
                    [-a - 3, a, -a], [-3 - 1 / a, -1, 1]]), -4 * (a + 1), True),
    "NS3_85-2": (M([[0, 1, a], [-1, 2, -2], [-1, -2, 2]]),
                 -sp.Rational(1, 4) / (a + 1),
                 M([[0, 2 * (a + 1), 2 * (a + 1)], [-4, -a, a], [-4, 1, -1]]),
                 4 * (a + 1), False),
    "NS3_85-3": (M([[2, -1, -2], [-1, 0, 3 + a], [-1, -1, 2]]),
                 1 / (3 * a + 5),
                 M([[a + 3, 4, -a - 3], [-a - 1, 2, -2 * a - 4], [1, 3, -1]]),
                 3 * a + 5, True),
    "NS3_85-4": (M([[2, -2, -1], [-1, 2, -1], [-1, 3 + 1 / a, 0]]),
                 1 / (5 * a + 3),
                 M([[3 * a + 1, -3 * a - 1, 4 * a], [a, -a, 3 * a],
                    [-a - 1, -2 * (2 * a + 1), 2 * a]]), 5 + 3 / a, True),
    "NS3_86-1": (one(3), -sp.Rational(1, 4) / (a + 1),
                 M([[-3 * a - 10 - 3 / a, 3 * a + 1, a + 3],
                    [-a - 3, a, -a], [-3 - 1 / a, -1, 1]]), -4 * (a + 1), True),
    "NS3_86-2": (M([[0, 1, a], [-1, 2, -2], [-1, -2, 2]]),
                 -sp.Rational(1, 4) / (a + 1),
                 M([[0, 2 * (a + 1), 2 * (a + 1)], [-4, -a, a], [-4, 1, -1]]),
                 4 * (a + 1), False),
    "NS3_86-3": (M([[2, -1, -2], [-1, 0, 3 + a], [-2, -1, 2]]),
                 sp.Rational(1, 4) / (a + 2),
                 M([[a + 3, 4, -a - 3], [-2 * (a + 2), 0, -2 * (a + 2)], [1, 4, -1]]),
                 4 * (a + 2), False),
    "NS3_86-4": (M([[2, -2, -1], [-2, 2, -1], [-1, 3 + 1 / a, 0]]),
                 sp.Rational(1, 4) / (2 * a + 1),
                 M([[3 * a + 1, -3 * a - 1, 4 * a], [a, -a, 4 * a],
                    [-2 * (2 * a + 1), -2 * (2 * a + 1), 0]]), 8 + 4 / a, False),
}


def reflect(A, k):
    """The reflection (24) in the kth isotropic node."""
    n = A.rows
    B = sp.zeros(n, n)
    C = sp.zeros(n, n)
    for i in range(n):
        if i == k:
            B[i, k] = -2
        elif sp.simplify(A[i, k]) != 0:
            B[i, k] = sp.simplify(A[i, k] / A[k, i])
    for j in range(n):
        if j == k:
            C[k, j] = -2
        elif sp.simplify(A[j, k]) != 0:
            C[k, j] = 1
    return sp.simplify((sp.eye(n) + B) * A * (sp.eye(n) + C))


def equivalent(A, B, av):
    """Are A and B equivalent: B = diag(l) . P A P^T for a permutation P?"""
    A = sp.simplify(A.subs(a, av))
    B = sp.simplify(B.subs(a, av))
    for perm in itertools.permutations(range(A.rows)):
        P = M([[1 if perm[i] == j else 0 for j in range(A.rows)] for i in range(A.rows)])
        Ap = sp.simplify(P * A * P.T)
        ok = True
        for i in range(A.rows):
            piv = None
            for j in range(A.rows):
                if sp.simplify(Ap[i, j]) != 0:
                    piv = sp.radsimp(sp.simplify(B[i, j] / Ap[i, j]))
                    break
            if piv is None or piv == 0:
                ok = False
                break
            for j in range(A.rows):
                if sp.simplify(sp.radsimp(B[i, j] - piv * Ap[i, j])) != 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return perm
    return None


bad = 0
print("=" * 78)
print("alpha lies strictly between -1 and 0")
print("=" * 78)
for g, av in ALPHA.items():
    ok = (-1 < sp.N(av) < 0)
    bad += not ok
    print("   %-8s alpha = %-22s = %+.6f   %s"
          % (g, sp.nsimplify(av), float(sp.N(av)), "OK" if ok else "FAIL"))

print("\n" + "=" * 78)
print("A_{k+1} = r_k(A_1), and the only isotropic node of A_{k+1} is the kth")
print("=" * 78)
for g, av in ALPHA.items():
    A1 = PRINTED["%s-1" % g][0]
    d1 = [sp.simplify(A1[i, i].subs(a, av)) for i in range(3)]
    ok1 = all(x == 0 for x in d1)
    bad += not ok1
    print("   %-8s A_1 diagonal = %s  %s" % (g, d1, "all 0, OK" if ok1 else "FAIL"))
    for k in range(3):
        Ak = reflect(A1, k)
        Bk = PRINTED["%s-%d" % (g, k + 2)][0]
        perm = equivalent(Ak, Bk, av)
        diag = [sp.simplify(Bk[i, i].subs(a, av)) for i in range(3)]
        iso = [i + 1 for i in range(3) if diag[i] == 0]
        ok = perm is not None and iso == [k + 1]
        bad += not ok
        print("      r_%d(A_1) ~ A_%d : %-28s isotropic node(s) of A_%d: %s   %s"
              % (k + 1, k + 2, "yes, by the permutation %s" % (tuple(p + 1 for p in perm),)
                 if perm else "NO", k + 2, iso, "OK" if ok else "FAIL"))

print("\n" + "=" * 78)
print("det, its sign, the inverse, and the box !!!")
print("=" * 78)
for g, av in ALPHA.items():
    for k in range(1, 5):
        A, pref, num, det, attn = PRINTED["%s-%d" % (g, k)]
        dd = sp.simplify(sp.radsimp(A.det() - det))
        okd = dd == 0
        inv = sp.simplify(pref * num)
        R = (A * inv - sp.eye(3)).subs(a, av)
        oki = all(sp.simplify(sp.radsimp(R[i, j])) == 0 for i in range(3) for j in range(3))
        nz = all(sp.simplify(num[i, j]) != 0 for i in range(3) for j in range(3))
        okn = nz == attn
        s = sp.N(det.subs(a, av))
        bad += not (okd and oki and okn)
        print("   %-11s det = %-18s at alpha: %+.5f (%s)  A.A^{-1}=E: %s  !!!: %s %s"
              % ("%s-%d)" % (g, k), sp.factor(det), float(s), "positive" if s > 0 else "negative",
                 "OK" if oki else "FAIL", "yes" if nz else "no ",
                 "OK" if okn else "FAIL(printed %s)" % attn))

print("\n" + "=" * 78)
print("Statement 9.4.3: which of A_1, ..., A_4 are equivalent; the reflection closure of A_1")
print("=" * 78)
EXPECTED = {g: [] for g in ALPHA}
EXPECTED["NS3_86"] = [(2, 3), (2, 4), (3, 4)]
for g, av in ALPHA.items():
    Ms = [PRINTED["%s-%d" % (g, i)][0] for i in range(1, 5)]
    eq = [(i + 1, j + 1) for i in range(4) for j in range(i + 1, 4)
          if equivalent(Ms[i], Ms[j], av) is not None]
    orbit = [Ms[0].subs(a, av)]
    frontier = [orbit[0]]
    while frontier:
        A = frontier.pop()
        for k in range(3):
            if sp.simplify(A[k, k]) != 0:
                continue
            B = sp.simplify(reflect(A, k))
            if not any(equivalent(B, C, av) is not None for C in orbit):
                orbit.append(B)
                frontier.append(B)
    classes = [[i + 1 for i in range(4) if equivalent(Ms[i], C, av) is not None] for C in orbit]
    ok = eq == EXPECTED[g] and all(c for c in classes) and \
        sorted(i for c in classes for i in c) == [1, 2, 3, 4]
    bad += not ok
    print("   %-8s equivalent pairs among A_1..A_4: %-26s closure: %d classes %s  %s"
          % (g, eq if eq else "none", len(orbit), classes, "OK" if ok else "FAIL"))

print("\n" + "=" * 78)
print("Lemma 9.4.1: the Galois involution sigma (alpha -> 1/alpha) permutes A_1, ..., A_4")
print("=" * 78)
for g, av in ALPHA.items():
    Ms = [PRINTED["%s-%d" % (g, i)][0] for i in range(1, 5)]
    images = []
    for i in range(4):
        S = Ms[i].subs(a, 1 / a)
        images.append([j + 1 for j in range(4) if equivalent(S, Ms[j], av) is not None])
    # on A_1: transposition of the nodes 2 and 3, then rescaling of the first row
    S1 = Ms[0].subs(a, 1 / a)
    P = M([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    T = P * S1 * P.T
    T[0, :] = a * T[0, :]
    on_A1 = sp.simplify(T - Ms[0]) == sp.zeros(3, 3)
    ok = 1 in images[0] and 2 in images[1] and 4 in images[2] and 3 in images[3] and on_A1
    bad += not ok
    print("   %-8s sigma(A_i) ~ A_j for i = 1..4: j in %s;  sigma(A_1) = A_1 after (2 3) and "
          "rescaling row 1: %s  %s" % (g, images, on_A1, "OK" if ok else "FAIL"))

print("\n" + "=" * 78)
print("FAILURES: %d" % bad)
print("=" * 78)
