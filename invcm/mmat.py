#!/usr/bin/env python3
"""The adjugate as a sum over paths, the positivity of inverse M-matrices, and the
signs of the inverses of the 370 almost affine Cartan matrices.

Three parts, all in exact rational arithmetic.

PART 1.  Lemma 1.2.2: for every square matrix C,

    adj(C)_{ij} = sum over simple paths p: i --> j of
                     ( prod over the arcs (a,b) of p of (-C_{ab}) ) * det C[V \\ p],

the trivial path giving adj(C)_{ii} = det C[V\\{i}] (formula (2)).  Checked on random
matrices of sizes 1-5 with symmetric and non-symmetric zero patterns.  With a tree this is
the path formula of [LT] (Proposition 1.2.1).

PART 2.  Proposition 1.2.3: in Proposition 1.2.1 the hypothesis "the graph is a tree" may be
replaced with "the graph is connected".  Random search over Z-matrices with a connected
graph and all principal minors positive; plus the two samples showing that connectedness
and condition a) of (1) may not be dropped (Remark 1.2.5).

PART 3.  Statement 3.3.2: for each of the 370 Cartan matrices of Subsections 9.1 and 9.2,
the hypotheses used in the proof of Theorem 3.3.1 (a Z-matrix; connected graph; main
submatrices missing >= 2 nodes have positive determinant, those missing one node a
non-negative one), the path expansion of PART 1, the sign pattern of A^{-1} predicted by
Theorem 3.3.1, and: the 39 matrices marked !!! are exactly those for which every A_{\\hat i}
is of finite type.

Usage:  python mmat.py
"""

import itertools
import random
import sys
from fractions import Fraction

from tables import load


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)


# --------------------------------------------------------------- linear algebra
def det(M):
    n = len(M)
    M = [row[:] for row in M]
    d = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            d = -d
        d *= M[c][c]
        inv = Fraction(1) / M[c][c]
        for r in range(c + 1, n):
            if M[r][c]:
                f = M[r][c] * inv
                for k in range(c, n):
                    M[r][k] -= f * M[c][k]
    return d


def sub(M, rows, cols):
    return [[M[i][j] for j in cols] for i in rows]


def principal(M, keep):
    return sub(M, keep, keep)


def adjugate(M):
    """adj(M)_{ij} = (-1)^{i+j} det M(j|i)  (so that M * adj(M) = det(M) * 1)."""
    n = len(M)
    out = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rows = [r for r in range(n) if r != j]
            cols = [c for c in range(n) if c != i]
            out[i][j] = Fraction(-1) ** (i + j) * det(sub(M, rows, cols))
    return out


def inverse(M):
    d = det(M)
    if d == 0:
        return None
    A = adjugate(M)
    return [[A[i][j] / d for j in range(len(M))] for i in range(len(M))]


def mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


# --------------------------------------------------------------- graph helpers
def simple_paths(n, adj, i, j):
    """All simple paths i --> j as vertex lists (the trivial one when i == j)."""
    if i == j:
        return [[i]]
    out = []

    def walk(path, seen):
        v = path[-1]
        for w in range(n):
            if w in seen or not adj(v, w):
                continue
            if w == j:
                out.append(path + [w])
            else:
                walk(path + [w], seen | {w})

    walk([i], {i})
    return out


def connected(n, adj):
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for w in range(n):
            if w not in seen and (adj(v, w) or adj(w, v)):
                seen.add(w)
                stack.append(w)
    return len(seen) == n


# --------------------------------------------------------------- PART 1
def path_expansion(C):
    """The right-hand side of formula (2) of Lemma 1.2.2."""
    n = len(C)

    def adj(a, b):
        return a != b and C[a][b] != 0

    out = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            tot = Fraction(0)
            for p in simple_paths(n, adj, i, j):
                w = Fraction(1)
                for a, b in zip(p, p[1:]):
                    w *= -C[a][b]
                rest = [v for v in range(n) if v not in p]
                tot += w * det(principal(C, rest)) if rest else w
            out[i][j] = tot
    return out


def rand_matrix(rng, n, lo=-4, hi=4, zeros=0.35, zsym=True):
    M = [[Fraction(rng.randint(lo, hi)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < zeros:
                M[i][j] = Fraction(0)
                if zsym:
                    M[j][i] = Fraction(0)
    return M


def part1():
    say("=" * 78)
    say("PART 1.  Lemma 1.2.2: adj(C)_{ij} = sum over simple paths i-->j of prod(-C_ab) * det C[V\\p]")
    say("=" * 78)
    rng = random.Random(20180919)
    bad = 0
    trials = 0
    for n in (1, 2, 3, 4, 5):
        for _ in range(120):
            for zsym in (True, False):
                C = rand_matrix(rng, n, zsym=zsym)
                trials += 1
                if path_expansion(C) != adjugate(C):
                    bad += 1
                    if bad <= 3:
                        say("  COUNTEREXAMPLE n=%d zsym=%s: %s" % (n, zsym, C))
    say("  random matrices, sizes 1-5, symmetric and asymmetric zero patterns:")
    say("  %d trials, %d mismatches -> %s" % (trials, bad, "PASS" if not bad else "FAIL"))
    say("  (with a tree this is the path formula of [LT]: one path, one term)")
    return bad == 0


# --------------------------------------------------------------- PART 2
def is_Z(C):
    n = len(C)
    return all(C[i][j] <= 0 for i in range(n) for j in range(n) if i != j)


def all_principal_minors_pos(C):
    n = len(C)
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            if det(principal(C, list(S))) <= 0:
                return False
    return True


def pattern_symmetric(C):
    n = len(C)
    return all((C[i][j] == 0) == (C[j][i] == 0) for i in range(n) for j in range(n))


def graph_is_tree(C):
    n = len(C)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if C[i][j] != 0 or C[j][i] != 0]
    return connected(n, lambda a, b: C[a][b] != 0 or C[b][a] != 0) and len(edges) == n - 1


def part2():
    say("")
    say("=" * 78)
    say("PART 2.  Proposition 1.2.3: the tree hypothesis weakened to connectedness")
    say("=" * 78)
    rng = random.Random(1)
    tested = {"tree": 0, "nontree": 0}
    fails = []
    # random Z-matrices meeting conditions a), c), d), keeping the non-tree ones
    for _ in range(400000):
        n = rng.choice((3, 4, 4, 5))
        C = [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            C[i][i] = Fraction(rng.randint(1, 6))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.55:
                    v = Fraction(-rng.randint(1, 3))
                    C[i][j] = v
                    C[j][i] = Fraction(-rng.randint(1, 3))
        if not connected(n, lambda a, b: C[a][b] != 0):
            continue
        if not all_principal_minors_pos(C):
            continue
        key = "tree" if graph_is_tree(C) else "nontree"
        tested[key] += 1
        I = inverse(C)
        if any(I[i][j] <= 0 for i in range(n) for j in range(n)):
            fails.append(C)
        if tested["nontree"] >= 3000 and tested["tree"] >= 200:
            break
    say("  Z-matrices with a connected graph and all principal minors > 0:")
    say("    %d with a tree, %d with a NON-tree graph (cycles present)" % (tested["tree"], tested["nontree"]))
    say("    inverses with a non-positive entry: %d -> %s"
        % (len(fails), "PASS, the tree hypothesis is removable" if not fails else "FAIL"))
    for C in fails[:2]:
        say("    COUNTEREXAMPLE:", C)

    # connectedness may NOT be dropped
    C = [[Fraction(2), Fraction(-1), Fraction(0), Fraction(0)],
         [Fraction(-1), Fraction(2), Fraction(0), Fraction(0)],
         [Fraction(0), Fraction(0), Fraction(2), Fraction(-1)],
         [Fraction(0), Fraction(0), Fraction(-1), Fraction(2)]]
    I = inverse(C)
    nz = sum(1 for r in I for x in r if x == 0)
    say("  disconnected sample (A_2 + A_2): %d zero entries in the inverse -> %s"
        % (nz, "connectedness is needed" if nz else "FAIL"))

    # Remark 1.2.5: a non-symmetric zero pattern (condition a) fails) kills positivity
    C = [[Fraction(2), Fraction(-1), Fraction(0)],
         [Fraction(0), Fraction(2), Fraction(-1)],
         [Fraction(0), Fraction(0), Fraction(2)]]
    I = inverse(C)
    zeros = [(i, j) for i in range(3) for j in range(3) if I[i][j] == 0]
    ok = is_Z(C) and all_principal_minors_pos(C) and not pattern_symmetric(C) and zeros
    say("  sample of Remark 1.2.5 (c_{ij} != 0 = c_{ji}): conditions c), d) of (1) hold,")
    say("    a) fails, and the inverse has %d zeros at %s -> %s"
        % (len(zeros), zeros, "condition a) is needed" if ok else "FAIL"))
    return not fails


# --------------------------------------------------------------- PART 3
def part3():
    say("")
    say("=" * 78)
    say("PART 3.  Statement 3.3.2 and Theorem 3.3.1 on the 370 tables of Subsections 9.1, 9.2")
    say("=" * 78)
    entries = [(e["subsection"], e["name"], e["marked"], e["A"], e["inverse"]) for e in load()]
    say("  read %d matrices" % len(entries))

    bad = []
    nmark = 0
    stat = {"Z": 0, "conn": 0, "deep": 0, "one": 0, "pred": 0, "expansion": 0}
    for sect, name, marked, A, B in entries:
        n = len(A)
        nmark += marked
        dA = det(A)

        # the hypotheses of the argument
        if not is_Z(A):
            bad.append(("NOT-Z", name, "an off-diagonal entry is > 0"))
        else:
            stat["Z"] += 1
        if not connected(n, lambda a, b: A[a][b] != 0):
            bad.append(("DISCONNECTED", name, "the diagram is not connected"))
        else:
            stat["conn"] += 1
        if dA >= 0:
            bad.append(("DET", name, "det = %s is not negative" % dA))

        # every principal submatrix missing >= 2 nodes is of finite type (det > 0)
        deep_ok = True
        for k in range(0, n - 1):
            for S in itertools.combinations(range(n), k):
                if det(principal(A, list(S))) <= 0 and k:
                    deep_ok = False
        if not deep_ok:
            bad.append(("DEEP", name, "a submatrix missing >= 2 nodes has det <= 0"))
        else:
            stat["deep"] += 1

        # one node deleted: det >= 0, zero exactly for the affine ones
        ones = [det(principal(A, [v for v in range(n) if v != i])) for i in range(n)]
        if any(d < 0 for d in ones):
            bad.append(("ONE", name, "a submatrix missing one node has det < 0"))
        else:
            stat["one"] += 1

        # the path expansion reproduces the adjugate
        if path_expansion(A) == adjugate(A):
            stat["expansion"] += 1
        else:
            bad.append(("EXPANSION", name, "the path formula misses the adjugate"))

        # the predictions
        ok = True
        for i in range(n):
            for j in range(n):
                if i == j:
                    want_zero = ones[i] == 0
                    if (B[i][j] == 0) != want_zero or B[i][j] > 0:
                        ok = False
                    if B[i][j] != ones[i] / dA:
                        ok = False
                elif B[i][j] >= 0:
                    ok = False
        if ok:
            stat["pred"] += 1
        else:
            bad.append(("PRED", name, "the predicted sign pattern fails"))

        # the !!! mark
        want_mark = all(d > 0 for d in ones)
        if want_mark != marked:
            bad.append(("MARK", name, "!!! is %s but every deleted node gives det %s"
                        % (marked, [str(d) for d in ones])))

    say("  %d marked !!!" % nmark)
    say("")
    say("  hypotheses of the argument, out of %d matrices:" % len(entries))
    say("    off-diagonal entries <= 0 (a Z-matrix)            : %d" % stat["Z"])
    say("    diagram connected (indecomposable)                : %d" % stat["conn"])
    say("    submatrix missing >= 2 nodes has det > 0          : %d" % stat["deep"])
    say("    submatrix missing one node has det >= 0           : %d" % stat["one"])
    say("    the path expansion of PART 1 gives the adjugate   : %d" % stat["expansion"])
    say("")
    say("  conclusions:")
    say("    predicted signs of C^{-1} confirmed               : %d" % stat["pred"])
    say("    the !!! mark = every proper principal")
    say("    submatrix is of finite type (det > 0)             : %s"
        % ("all %d agree" % len(entries) if not [b for b in bad if b[0] == "MARK"]
           else "%d disagree" % len([b for b in bad if b[0] == "MARK"])))
    say("")
    say("  %d problems" % len(bad))
    for kind, name, msg in bad[:40]:
        say("    %-12s %-16s %s" % (kind, name, msg))
    return not bad


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    r1 = part1()
    r2 = part2()
    r3 = part3()
    say("")
    say("=" * 78)
    say("PART 1 %s   PART 2 %s   PART 3 %s"
        % ("PASS" if r1 else "FAIL", "PASS" if r2 else "FAIL", "PASS" if r3 else "FAIL"))
