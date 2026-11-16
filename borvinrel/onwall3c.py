"""
Statement 10.1.1 on its wall, continued:

(A) identify the algebra in the case that matters for this paper (zero main
    diagonal, all generators odd): dimension, superdimension, root heights;
(B) the other two main diagonals of Section 11 ("Relations"): does the
    stratification produce further relations there?
(C) the same dimension count for the 4x4 family of Statement 10.1.10.
"""

import sys
import time

import sympy as sp

import driver
from driver import Stratum, analyse
from shapovalov import free_lie_superalgebra, gram

a = driver.a
MAXH = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def banner(P, s):
    P('-' * 70)
    P(s)
    P('-' * 70)


def main():
    out = open('log-3x3-onwall-others.txt', 'w')

    def P(s=''):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    # ---------------------------------------------------------------- (A)
    banner(P, '(A) zero diagonal, all odd, ON the wall: what is the algebra?')
    par = [1, 1, 1]
    L = free_lie_superalgebra(3, 6, par=par)
    for (av, bv) in [(2, 1), (5, 3), (sp.Rational(1, 2), 1), (sp.Rational(-7, 4), 1)]:
        cv = -av - bv
        A = [[sp.Integer(0), sp.Integer(av), sp.Integer(bv)] if False else
             [sp.Integer(0), sp.nsimplify(av), sp.nsimplify(bv)],
             [sp.nsimplify(av), sp.Integer(0), sp.nsimplify(cv)],
             [sp.nsimplify(bv), sp.nsimplify(cv), sp.Integer(0)]]
        ev = od = 0
        heights = {}
        for k in range(1, 7):
            tot = 0
            for alpha in sorted(L.get(k, {})):
                basis = L[k][alpha]
                r = len(basis) if k == 1 else gram(basis, A, 3, alpha, par).rank()
                tot += r
            heights[k] = tot
            if k % 2:
                od += tot
            else:
                ev += tot
        P('  (a,b,c)=(%s,%s,%s): n_+ heights %s ; dim n_+ = %d'
          % (av, bv, cv, heights, sum(heights.values())))
        P('        n_+ : %d even, %d odd   =>  g = n_- + h + n_+ has'
          % (ev, od))
        P('        dim g = %d,  sdim g = %d|%d'
          % (2 * (ev + od) + 3, 2 * ev + 3, 2 * od))
    P('')
    P('  For comparison, the exceptional Lie superalgebra D(2,1;alpha) has')
    P('  sdim 9|8: even part sl(2)+sl(2)+sl(2), odd part 2 (x) 2 (x) 2, and')
    P('  its Cartan matrix in the all-isotropic normalization is exactly')
    P('  [[0,a,b],[a,0,c],[b,c,0]] with a+b+c=0, the parameter being a:b:c.')
    P('  Positive roots: 3 odd of height 1, 3 even of height 2 (a_i+a_j),')
    P('  1 odd of height 3 (a_1+a_2+a_3)  --  exactly the profile above.')
    P('')

    # ---------------------------------------------------------------- (B)
    for name, diag, par in [('zero diagonal, all even', sp.Integer(0), [0, 0, 0]),
                            ('main diagonal 2, all even', sp.Integer(2), [0, 0, 0])]:
        banner(P, '(B) %s, ON the wall (b normalized to 1)' % name)
        A_SYM = [[diag, a, sp.Integer(1)],
                 [a, diag, -a - 1],
                 [sp.Integer(1), -a - 1, diag]]
        L2 = free_lie_superalgebra(3, MAXH, par=par)
        st = Stratum({}, '', [a])
        t0 = time.time()
        analyse(A_SYM, 3, MAXH, st, L2, maxdepth=0, par=par, log=P)
        P('  relations found, by multidegree:')
        for alpha, rel in st.rels:
            P('     %s  (height %d)' % (alpha, sum(alpha)))
        byh = {}
        for alpha, _ in st.rels:
            byh[sum(alpha)] = byh.get(sum(alpha), 0) + 1
        P('  relations per height: %s   [%.0f s]'
          % (dict(sorted(byh.items())), time.time() - t0))
        P('')

    out.close()


if __name__ == '__main__':
    main()
