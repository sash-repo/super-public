"""
Remark 12.1.12 (Sect. 12, the matrices (2types)): the two things about them
that a computation can settle.

(1) The Cartan matrices \eqref{2types} are NOT symmetrizable (except at a=1),
    so no contravariant form exists and the Shapovalov-determinant
    stratification used in the rest of the paper does not transfer.  But
    dimensions still can be computed: n_+^alpha = L^alpha / ker Phi, where
    Phi(x)_w is the iterated lowering D_{i_1}...D_{i_{k-1}}(x) read off at the
    word w.  ker Phi = r_+ by the very definition of r_+, whatever A is; so

            dim n_+^alpha = rank Phi^alpha,

    and this needs no symmetry of A at all.

(2) The dimension table printed before Statement 12.1.11 reads
        deg: 0 1 2 3  4  5   6   7   8
        dim: 3 3 3 8 18 46 110 294 756
    We recompute it, both for the reading in which the three generators are
    odd (which is what a zero main diagonal means) and for the reading in
    which they are even, and we locate the "certain particular values of a"
    at which the dimensions drop.
"""

import sys
import time
from collections import defaultdict

import sympy as sp

from shapovalov import free_lie_superalgebra, phi, perms_of

a = sp.Symbol('a')
MAXH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N = 3

# A == type a) of (2types)
A_SYM = [[sp.Integer(0), sp.Integer(1), a],
         [sp.Integer(-1), sp.Integer(0), 2 + a],
         [sp.Integer(-1), 2 + sp.Rational(1, 1) / a, sp.Integer(0)]]


def symmetrizable(A, n):
    d = sp.symbols('d1:%d' % (n + 1))
    eqs = [sp.together(d[i] * A[i][j] - d[j] * A[j][i])
           for i in range(n) for j in range(i + 1, n)]
    sol = sp.solve([sp.numer(e) for e in eqs], d, dict=True)
    return sol


def rank_phi(basis, A, n, alpha, par):
    """dim n_+^alpha = rank of the matrix ( Phi(u)_w )_{u in basis, w}."""
    words = perms_of(alpha, n)
    rows = []
    for u in basis:
        p = phi(u, A, n, alpha, par)
        rows.append([sp.expand(p.get(w, sp.Integer(0))) for w in words])
    return sp.Matrix(rows).rank()


def table(A, par, maxh, L):
    out = {}
    for k in range(1, maxh + 1):
        tot = 0
        for alpha in sorted(L.get(k, {})):
            basis = L[k][alpha]
            tot += len(basis) if k == 1 else rank_phi(basis, A, N, alpha, par)
        out[k] = tot
    return out


def main():
    out = open('log-sec11.txt', 'w')

    def P(s=''):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    P('=== Section "g as a g_ev-module": A of \\eqref{2types} a) ===')
    P('A = %s' % sp.Matrix(A_SYM).tolist())
    P('')
    P('--- is A symmetrizable? ---')
    sol = symmetrizable(A_SYM, N)
    P('  solutions of d_i A_ij = d_j A_ji : %s' % sol)
    d1, d2, d3 = sp.symbols('d1 d2 d3')
    # the three conditions, solved by hand for the record
    P('  (1,2): d2 = -d1;  (1,3): d3 = -a*d1;')
    P('  (2,3): -d1*(2+a) = -a*d1*(2+1/a) = -d1*(2a+1)  =>  2+a = 2a+1  =>  a = 1.')
    P('  So A is symmetrizable ONLY for a = 1; the five values of a in')
    P('  \\eqref{2types} are irrational, hence A is NOT symmetrizable there.')
    P('  Consequence: there is no contravariant (Shapovalov) form, and the')
    P('  stratification method of Section "Relations" does not apply.  The')
    P('  dimensions below are therefore computed as rank of the iterated-')
    P('  lowering matrix Phi, which needs no symmetry.')
    P('')

    for name, par in [('all three generators ODD (what a zero diagonal means)',
                       [1, 1, 1]),
                      ('all three generators EVEN', [0, 0, 0])]:
        P('-' * 66)
        P('--- %s ---' % name)
        P('-' * 66)
        L = free_lie_superalgebra(N, MAXH, par=par)
        P('  free dims by height: %s'
          % {k: sum(len(v) for v in L[k].values()) for k in sorted(L)})
        for av in [sp.Integer(3), sp.Rational(5, 7), sp.Rational(-11, 3)]:
            A = [[sp.nsimplify(x.subs(a, av)) for x in row] for row in A_SYM]
            t0 = time.time()
            t = table(A, par, MAXH, L)
            P('  a=%-8s dim n_+ by degree: %s   [%.0fs]'
              % (av, t, time.time() - t0))
        P('  the text prints  deg 1..8 : 3, 3, 8, 18, 46, 110, 294, 756')
        P('')

        # symbolic: where do the dimensions drop?
        P('  --- the "certain particular values of a" (symbolic rank) ---')
        for k in range(2, min(MAXH, 5) + 1):
            for alpha in sorted(L.get(k, {})):
                basis = L[k][alpha]
                words = perms_of(alpha, N)
                rows = []
                for u in basis:
                    p = phi(u, A_SYM, N, alpha, par)
                    rows.append([sp.cancel(sp.together(p.get(w, sp.Integer(0))))
                                 for w in words])
                M = sp.Matrix(rows)
                r = M.rank()
                if r == 0:
                    continue
                # minors of size r whose vanishing drops the rank
                mins = []
                import itertools
                rr = list(itertools.combinations(range(M.rows), r))
                cc = list(itertools.combinations(range(M.cols), r))
                if len(rr) * len(cc) > 4000:
                    P('     alpha=%-9s rank=%d (too many minors to scan)'
                      % (str(alpha), r))
                    continue
                for R_ in rr:
                    for C_ in cc:
                        m = sp.cancel(M[list(R_), list(C_)].det())
                        if m != 0:
                            mins.append(sp.factor(sp.numer(sp.together(m))))
                if not mins:
                    continue
                g = sp.factor(sp.gcd(mins))
                facs = [f for f, _ in sp.factor_list(g)[1] if f.free_symbols]
                if facs:
                    P('     alpha=%-9s rank=%d  drops when %s'
                      % (str(alpha), r, ' or '.join('%s=0' % f for f in facs)))
        P('')

    out.close()


if __name__ == '__main__':
    main()
