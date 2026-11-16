"""What the Carter matrices become when the sign changes are divided out as well.

The census of `permcensus.py` is up to a renumbering of the indices only.
But e_i -> -e_i changes neither the lattice nor the reflection s_{e_i}, hence not
the Weyl group element w = w_1w_2 that Carter attaches to the matrix.  So the object
Carter classifies is the class of A modulo S_n x {+-1}^n, and that class is nothing
but the underlying GRAPH: by Corollary 8.5.6 of the paper every induced cycle
carries an odd number of +1's, the induced cycles span the cycle space over F_2, and
the number of +1's on a cycle is an F_2-linear functional of the cycle -- so the whole
sign pattern is forced, and the sign changes act transitively on it.

This module checks that on the census (partial Cartan matrices, i.e., connected and
with no odd cycle, 3 <= n <= 7) and computes, for each size and each lattice:
  * the number of classes up to renumbering,
  * the number of classes up to renumbering AND sign changes = the number of graphs,
  * the characteristic polynomial and the order of w = w_1w_2 for each graph.

The bipartition of a connected bipartite graph is unique, so w_1, w_2 (the products
of the reflections in the roots of each part -- involutions, the roots inside a part
being mutually orthogonal) are well defined up to interchanging them, i.e. up to
inverting w, which does not change the conjugacy class in a Weyl group.

Everything is exact integer arithmetic.  Run:  python3 carter.py
"""

import collections
from fractions import Fraction
from itertools import permutations, product

from vinb import is_connected, is_bipartite, det_int, canon
from permcensus import perm_canon
from lattice import root_system_type



# -------------------------------------------------------------- the underlying graph

def graph_refine(A):
    """Weisfeiler-Leman colours of the unsigned graph."""
    n = len(A)
    adj = [[1 if i != j and A[i][j] != 0 else 0 for j in range(n)] for i in range(n)]
    color = [sum(adj[i]) for i in range(n)]
    while True:
        sig = [(color[i], tuple(sorted(color[j] for j in range(n) if adj[i][j])))
               for i in range(n)]
        order = {s: k for k, s in enumerate(sorted(set(sig)))}
        new = [order[s] for s in sig]
        if new == color:
            return color
        color = new


def graph_canon(A):
    """Canonical form of the UNSIGNED graph: least adjacency triangle over permutations."""
    n = len(A)
    color = graph_refine(A)
    classes = {}
    for i, c in enumerate(color):
        classes.setdefault(c, []).append(i)
    best = None
    for choice in product(*(permutations(classes[k]) for k in sorted(classes))):
        perm = [v for blk in choice for v in blk]
        s = tuple(1 if A[perm[i]][perm[j]] else 0
                  for i in range(1, n) for j in range(i))
        if best is None or s < best:
            best = s
    return best


def degrees(A):
    n = len(A)
    return tuple(sorted(sum(1 for j in range(n) if j != i and A[i][j]) for i in range(n)))


# ------------------------------------------------------------- the Weyl group element

def bipartition(A):
    n = len(A)
    color = [None] * n
    color[0] = 0
    stack = [0]
    while stack:
        v = stack.pop()
        for w in range(n):
            if w != v and A[v][w] and color[w] is None:
                color[w] = 1 - color[v]
                stack.append(w)
    assert None not in color, 'graph not connected'
    return color


def matmul(P, Q):
    n = len(P)
    return [[sum(P[i][k] * Q[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def reflection(A, j):
    """s_j in the basis e_1,...,e_n: s_j(e_i) = e_i - A_{ij} e_j; columns are images."""
    n = len(A)
    M = [[1 if i == k else 0 for k in range(n)] for i in range(n)]
    for i in range(n):
        M[j][i] -= A[i][j]
    return M


def weyl_element(A):
    """w = w_1 w_2, the two involutions being the parts of the bipartition."""
    n = len(A)
    col = bipartition(A)
    w = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for part in (0, 1):
        u = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        for j in range(n):
            if col[j] == part:
                u = matmul(u, reflection(A, j))
        w = matmul(w, u)
    return w


def charpoly(M):
    """Coefficients of det(xI - M), highest degree first; Faddeev-LeVerrier, exact.

    Cross-checked below against det(xI - M) evaluated at integer points.
    """
    n = len(M)
    A = [[Fraction(x) for x in row] for row in M]
    coef = [Fraction(1)]
    N = [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        # N <- A (N + c_{k-1} I)
        S = [[N[i][j] + (coef[-1] if i == j else 0) for j in range(n)] for i in range(n)]
        N = [[sum(A[i][t] * S[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        c = -sum(N[i][i] for i in range(n)) / k
        coef.append(c)
    assert all(c.denominator == 1 for c in coef), coef
    out = tuple(int(c) for c in coef)
    for x in (2, -3):
        assert _peval(out, x) == det_int([[(x if i == j else 0) - M[i][j]
                                           for j in range(n)] for i in range(n)])
    return out


def _peval(cp, x):
    v = 0
    for c in cp:
        v = v * x + c
    return v


def order(M, cap=200):
    n = len(M)
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    P = M
    for k in range(1, cap + 1):
        if P == I:
            return k
        P = matmul(P, M)
    return None


# --------------------------------------------------------------------------- main

def main():
    print('backend: exact integer arithmetic\n')
    total = collections.OrderedDict()
    from permcensus import census
    reps, _ = census(7, verbose=False)
    for n in range(3, 8):
        mats = [A for A in reps[n] if is_connected(A) and is_bipartite(A)]
        assert all(is_connected(A) and is_bipartite(A) for A in mats)
        by_lat = collections.defaultdict(list)
        for A in mats:
            by_lat[root_system_type(A)[0]].append(A)

        print('size %d x %d : %d matrices up to a renumbering of the indices' % (n, n, len(mats)))
        for lat in sorted(by_lat):
            L = by_lat[lat]
            sign_classes = collections.defaultdict(list)   # up to renumbering AND sign
            for A in L:
                sign_classes[canon(A)].append(A)
            graph_classes = collections.defaultdict(list)  # up to renumbering, unsigned
            for A in L:
                graph_classes[graph_canon(A)].append(A)

            # the claim: the graph determines the matrix up to the sign changes
            same = len(sign_classes) == len(graph_classes)
            fine = all(len({canon(A) for A in v}) == 1 for v in graph_classes.values())

            wdata = {}
            for c, v in graph_classes.items():
                A = v[0]
                w = weyl_element(A)
                cp = charpoly(w)
                assert sum(cp) != 0, 'w has the eigenvalue 1'   # elliptic
                wdata[c] = (cp, order(w), degrees(A))
            polys = collections.Counter(d[0] for d in wdata.values())

            print('   %-4s %4d matrices  ->  %2d classes up to sign changes, '
                  '%2d graphs%s  ->  %2d distinct char. polynomials of w'
                  % (lat, len(L), len(sign_classes), len(graph_classes),
                     '' if same and fine else '   MISMATCH', len(polys)))
            for c in sorted(wdata, key=lambda c: (wdata[c][1], wdata[c][0])):
                cp, o, deg = wdata[c]
                print('        order %2d   char.poly %-28s degrees %s   (%d matrices)'
                      % (o, _fmt(cp), deg, len(graph_classes[c])))
            total[(n, lat)] = (len(L), len(graph_classes), len(polys))
        print()

    print('summary  (matrices up to renumbering | graphs = Carter diagrams | classes of w):')
    for (n, lat), (a, b, c) in total.items():
        print('   n=%d  %-4s  %6d | %3d | %3d' % (n, lat, a, b, c))


def _fmt(cp):
    """x^n + ... , as a compact string."""
    n = len(cp) - 1
    out = []
    for k, c in enumerate(cp):
        d = n - k
        if c == 0:
            continue
        term = ('' if abs(c) == 1 and d else str(abs(c))) + ('' if d == 0 else
                                                             ('x' if d == 1 else 'x^%d' % d))
        out.append(('-' if c < 0 else '+') + term)
    s = ''.join(out)
    return s[1:] if s.startswith('+') else s


if __name__ == '__main__':
    main()
