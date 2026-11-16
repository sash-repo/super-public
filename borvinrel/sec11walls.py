"""Statement 12.1.11: the particular values of a for the Cartan matrix

        A = [[0, 1, a], [-1, 0, 2+a], [-1, 2+1/a, 0]]

(case a) of the matrices (2types) of Sect. 12), at which the dimensions of the
homogeneous components of g(A) are smaller than for generic a.

A is not symmetrizable (except at a=1), so there is
no contravariant form and the Shapovalov-determinant stratification does not
apply.  But dim n_+^alpha = rank Phi^alpha, where Phi is the iterated-lowering
map, and that needs no symmetry.  So:

  1. Build Phi^alpha over Q(a) for every multidegree alpha of height <= H.
  2. Fraction-free row reduction over Q(a): the generic rank r is the number of
     pivots, and every value of a at which the rank can drop is a root of one
     of the pivot numerators.  That gives a finite SUPERSET of candidates.
  3. Verify each candidate by substituting it and recomputing the rank
     honestly.  A candidate survives only if the rank really drops.

Both readings of the main diagonal are run: all three generators odd (what a
zero main diagonal means when the nodes are isotropic) and all three even.

    python sec11walls.py [maxheight]     (maxheight defaults to 5)
"""

import sys
import time
from collections import defaultdict

import sympy as sp

from shapovalov import free_lie_superalgebra, phi, perms_of

a = sp.Symbol('a')
N = 3
MAXH = int(sys.argv[1]) if len(sys.argv) > 1 else 5

A_SYM = [[sp.Integer(0), sp.Integer(1), a],
         [sp.Integer(-1), sp.Integer(0), 2 + a],
         [sp.Integer(-1), 2 + sp.Integer(1) / a, sp.Integer(0)]]


def phi_matrix(basis, A, n, alpha, par):
    words = perms_of(alpha, n)
    rows = []
    for u in basis:
        p = phi(u, A, n, alpha, par)
        rows.append([sp.cancel(sp.together(p.get(w, sp.Integer(0))))
                     for w in words])
    return sp.Matrix(rows), words


def pivots_over_Qa(M):
    """Fraction-free elimination over Q(a).

    Returns (generic_rank, [pivot numerators]).  The rank at a numeric a_0 can
    differ from the generic rank only if a_0 kills one of those numerators (or
    one of the denominators cleared on the way, which is why the caller must
    verify every candidate).
    """
    M = M.applyfunc(lambda e: sp.cancel(sp.together(e)))
    rows, cols = M.rows, M.cols
    mat = [[M[i, j] for j in range(cols)] for i in range(rows)]
    pivnum = []
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if sp.simplify(mat[i][c]) != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        p = sp.cancel(mat[r][c])
        pivnum.append(sp.factor(sp.numer(sp.together(p))))
        for i in range(r + 1, rows):
            f = sp.cancel(mat[i][c] / p)
            if f == 0:
                continue
            for j in range(c, cols):
                mat[i][j] = sp.cancel(sp.expand(mat[i][j] - f * mat[r][j]))
        r += 1
        if r == rows:
            break
    return r, pivnum


def candidates(pivnum):
    out = set()
    for p in pivnum:
        for f, _ in sp.factor_list(p)[1]:
            if not f.free_symbols:
                continue
            for root in sp.solve(sp.Eq(f, 0), a):
                if root.is_number:
                    out.add(sp.nsimplify(root))
    return out


def rank_at(M, val):
    try:
        Mv = M.applyfunc(lambda e: sp.cancel(sp.together(e)).subs(a, val))
    except ZeroDivisionError:
        return None
    if any(e.has(sp.zoo) or e.has(sp.nan) or e.has(sp.oo) for e in Mv):
        return None
    return Mv.rank()


def run(par, name, out):
    def P(s):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    P('')
    P('=' * 66)
    P('--- %s ---' % name)
    P('=' * 66)
    L = free_lie_superalgebra(N, MAXH, par=par)
    P('  free dims by height: %s'
      % {h: sum(len(v) for v in L[h].values()) for h in sorted(L) if h >= 1})

    bydeg = defaultdict(int)
    special = defaultdict(set)
    for h in range(1, MAXH + 1):
        by_alpha = L[h]
        for alpha in sorted(by_alpha):
            basis = by_alpha[alpha]
            t0 = time.time()
            M, words = phi_matrix(basis, A_SYM, N, alpha, par)
            r, pivnum = pivots_over_Qa(M)
            bydeg[h] += r
            cands = candidates(pivnum)
            hits = []
            for c in sorted(cands, key=lambda x: sp.re(sp.N(x))):
                rc = rank_at(M, c)
                if rc is not None and rc < r:
                    hits.append((c, rc))
                    special[h].add(c)
            P('    alpha=%-10s dim=%-3d %s   [%.0fs]'
              % (str(alpha), r,
                 ('DROPS at ' + ', '.join('a=%s (dim %d)' % (c, rc)
                                          for c, rc in hits)) if hits else '',
                 time.time() - t0))
    P('')
    P('  generic dim n_+ by height: %s' % dict(bydeg))
    P('  the text prints  deg 1..8 : 3, 3, 8, 18, 46, 110, 294, 756')
    allsp = set()
    for h in special:
        allsp |= special[h]
    P('')
    P('  *** the particular values of a at which the dimensions are smaller,')
    P('      through height %d: %s'
      % (MAXH, ', '.join(str(x) for x in sorted(allsp, key=lambda x: sp.re(sp.N(x))))
         or 'none'))
    for h in sorted(special):
        P('      first seen in height %d: %s'
          % (h, ', '.join(str(x) for x in sorted(special[h],
                                                 key=lambda x: sp.re(sp.N(x))))))
    # the five values of \eqref{2types} are irrational; report whether any is hit
    return allsp


def main():
    out = open('log-sec11walls.txt', 'w')
    out.write('=== the particular values of a, maxheight = %d ===\n' % MAXH)
    out.write('A = %s\n' % A_SYM)
    odd = run([1, 1, 1], 'all three generators ODD', out)
    even = run([0, 0, 0], 'all three generators EVEN', out)
    out.write('\n=== summary ===\n')
    out.write('odd  reading: %s\n' % sorted(map(str, odd)))
    out.write('even reading: %s\n' % sorted(map(str, even)))
    out.close()


if __name__ == '__main__':
    main()
