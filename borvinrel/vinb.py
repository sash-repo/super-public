"""Census of symmetric matrices with 2 on the main diagonal and 0, +-1 off it.

The census is built level by level (`census`): every matrix of type Pos of size n
is obtained by bordering one of size n-1 by a row (Proposition 7.3.1 of the paper).
`parse` reads lists in a plain text format: each block of n-1 lines is the strict
lower triangle of an n x n symmetric matrix, row i holding A_{i+1,1} ... A_{i+1,i};
the letter `-` stands for -1.

Everything here is exact integer arithmetic.  Positive definiteness is decided by
the leading principal minors computed by integer-preserving (Bareiss) elimination;
the extension test uses 2*det(A) - v^T adj(A) v > 0, again over the integers.

Canonical form is taken under the group S_n x {+-1}^n acting by
A -> P D A D P^T (simultaneous permutation of rows and columns, and simultaneous
sign change of the i-th row and column).  Signs are gauge-fixed on a spanning
forest, which uses up the whole sign freedom; the permutation is minimized over,
after a Weisfeiler-Leman refinement of the unsigned graph cuts the candidates down.
"""

import sys
from itertools import product, permutations

# --------------------------------------------------------------------------- parsing

SYM = {'1': 1, '0': 0, '-': -1}


def parse(path):
    """Read a census file; return a list of matrices as tuples of tuples."""
    raw = open(path, 'rb').read().decode('ascii')
    # a block separator is an empty line, so line terminators are normalized first
    raw = raw.replace('\r\r\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
    lines = [ln.strip() for ln in raw.split('\n')]
    mats, block = [], []
    for ln in lines:
        if ln.startswith(';'):
            continue
        if ln == '':
            if block:
                mats.append(from_triangle(block))
                block = []
        else:
            block.append(ln)
    if block:
        mats.append(from_triangle(block))
    return mats


def from_triangle(rows):
    """rows[i] is row i+2 of the strict lower triangle (i = 0, 1, ...)."""
    n = len(rows) + 1
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
    for i, r in enumerate(rows):
        assert len(r) == i + 1, (i, r)
        for j, ch in enumerate(r):
            A[i + 1][j] = A[j][i + 1] = SYM[ch]
    return tuple(tuple(r) for r in A)


def to_triangle(A):
    inv = {1: '1', 0: '0', -1: '-'}
    return '\n'.join(''.join(inv[A[i][j]] for j in range(i)) for i in range(1, len(A)))


# ------------------------------------------------------------------ integer linear algebra

def bareiss_minors(A):
    """Leading principal minors of an integer symmetric matrix, exactly."""
    n = len(A)
    M = [list(map(int, row)) for row in A]
    minors = []
    prev = 1
    for k in range(n):
        if M[k][k] == 0:
            # a zero pivot: fall back to exact fractions for the remaining minors
            return None
        minors.append(M[k][k] * 1 if k == 0 else M[k][k])
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    # accumulate: after Bareiss, M[k][k] IS the k-th leading principal minor
    return minors


def is_posdef(A):
    """Sylvester's criterion, exact.  A is integer symmetric."""
    n = len(A)
    M = [list(map(int, row)) for row in A]
    prev = 1
    for k in range(n):
        if M[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return True


def det_int(A):
    """Determinant of an integer matrix by fraction-free elimination."""
    n = len(A)
    M = [list(map(int, row)) for row in A]
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            piv = next((i for i in range(k + 1, n) if M[i][k] != 0), None)
            if piv is None:
                return 0
            M[k], M[piv] = M[piv], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def adjugate(A):
    """adj(A) = det(A) * A^{-1}, integer, via cofactors (n is small)."""
    n = len(A)
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [[A[r][c] for c in range(n) if c != i] for r in range(n) if r != j]
            adj[i][j] = ((-1) ** (i + j)) * (det_int(minor) if n > 1 else 1)
    return adj


# ------------------------------------------------------------------------ graph side

def graph(A):
    n = len(A)
    return [[j for j in range(n) if j != i and A[i][j] != 0] for i in range(n)]


def components(A):
    n, g = len(A), graph(A)
    seen, comps = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in g[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def is_connected(A):
    return len(components(A)) == 1


def is_bipartite(A):
    """No cycle of odd length -- Carter's condition on an admissible diagram."""
    n, g = len(A), graph(A)
    color = [None] * n
    for s in range(n):
        if color[s] is not None:
            continue
        color[s] = 0
        stack = [s]
        while stack:
            v = stack.pop()
            for w in g[v]:
                if color[w] is None:
                    color[w] = 1 - color[v]
                    stack.append(w)
                elif color[w] == color[v]:
                    return False
    return True


# ------------------------------------------------------------------ canonical form

def refine(A):
    """Weisfeiler-Leman colours of the UNSIGNED graph (signs are gauge-dependent)."""
    n = len(A)
    g = graph(A)
    color = [len(g[i]) for i in range(n)]
    while True:
        sig = [(color[i], tuple(sorted(color[j] for j in g[i]))) for i in range(n)]
        order = {s: k for k, s in enumerate(sorted(set(sig)))}
        new = [order[s] for s in sig]
        if new == color:
            return color
        color = new


def gauge_fix(A, perm):
    """Permute by `perm` then make every spanning-forest edge +1; return the string.

    For a connected graph the sign gauge {+-1}^n acts with the global flip as its
    only kernel, and fixing the n-1 edges of a spanning tree to +1 uses it all up,
    so this is a genuine canonical representative of the sign orbit.
    """
    n = len(A)
    B = [[A[perm[i]][perm[j]] for j in range(n)] for i in range(n)]
    eps = [0] * n           # 0 = not yet set
    for s in range(n):
        if eps[s]:
            continue
        eps[s] = 1
        stack = [s]
        while stack:
            v = stack.pop()
            for w in range(n):
                if w != v and B[v][w] != 0 and eps[w] == 0:
                    # want eps[v]*eps[w]*B[v][w] = +1
                    eps[w] = eps[v] * B[v][w]
                    stack.append(w)
    inv = {1: '1', 0: '0', -1: '-'}
    return ''.join(inv[eps[i] * eps[j] * B[i][j]] for i in range(1, n) for j in range(i))


def canon(A):
    """Lexicographically least gauge-fixed triangle over all admissible permutations."""
    n = len(A)
    color = refine(A)
    classes = {}
    for i, c in enumerate(color):
        classes.setdefault(c, []).append(i)
    keys = sorted(classes)
    best = None
    for choice in product(*(permutations(classes[k]) for k in keys)):
        perm = [v for blk in choice for v in blk]
        s = gauge_fix(A, perm)
        if best is None or s < best:
            best = s
    return best


# ---------------------------------------------------------------- incremental census

def extend(A):
    """All pos-def one-row extensions of a pos-def integer matrix A."""
    n = len(A)
    d = det_int(A)
    adj = adjugate(A)
    out = []
    for v in product((-1, 0, 1), repeat=n):
        # Schur complement: 2 - v^T A^{-1} v > 0  <=>  2d - v^T adj(A) v > 0  (d > 0)
        q = 0
        for i in range(n):
            vi = v[i]
            if vi:
                q += vi * sum(adj[i][j] * v[j] for j in range(n))
        if 2 * d - q > 0:
            B = [list(row) + [v[i]] for i, row in enumerate(A)]
            B.append(list(v) + [2])
            out.append(tuple(tuple(r) for r in B))
    return out


def census(nmax, connected_only=False, verbose=True):
    """Representatives, level by level, up to S_n x {+-1}^n."""
    levels = {1: {'': ((2,),)}}
    reps = {1: [((2,),)]}
    for n in range(2, nmax + 1):
        seen = {}
        for A in reps[n - 1]:
            for B in extend(A):
                c = canon(B)
                if c not in seen:
                    seen[c] = B
        reps[n] = list(seen.values())
        if verbose:
            allc = len(reps[n])
            conn = sum(1 for B in reps[n] if is_connected(B))
            bip = sum(1 for B in reps[n] if is_bipartite(B))
            cbip = sum(1 for B in reps[n] if is_connected(B) and is_bipartite(B))
            print(f'n={n:2d}  all={allc:7d}  connected={conn:7d} '
                  f' no-odd-cycle={bip:6d}  connected & no-odd-cycle={cbip:6d}', flush=True)
    return reps


if __name__ == '__main__':
    # the census up to S_n x {+-1}^n, i.e., counted as graphs (Statement 8.6.2, n <= 7)
    census(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
