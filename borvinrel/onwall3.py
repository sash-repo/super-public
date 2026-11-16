"""
Statement 10.1.1 ("On 3x3 matrices"): are there OTHER relations ON ITS WALL?

Statement 10.1.1 says: for a symmetric Cartan matrix of size 3 with

        A_12 + A_13 + A_23 = 0,

there is the relation (66), for any diagonal and any parity.  One asks
whether there are others, and on what a conjecture "there are none" would
rest.

Section 11 ("Relations") of the paper already answers this for the whole family
(80) through height 7: the strata are exactly the lines a+b+kc=0 and
their S_3-images, and the wall of Statement 10.1.1 is the member k=1.  Note
that a point of the k=1 wall can lie on a second wall a+b+kc=0 only if
(k-1)c=0, i.e. only on the degenerate locus c=0.  So through height 7 the
relation of Statement 10.1.1 is the whole story on its own wall.

Here we push that verification further, cheaply: we sit ON the wall from the
start.  Rescaling A by a nonzero scalar is an isomorphism of the presentation
(it rescales the h_i), so the wall is the projective line a:b, and we may put
b = 1:

        A = ( 0     a     1   )
            ( a     0   -a-1  )
            ( 1   -a-1    0   )     one parameter a, off  a(a+1) = 0.

One symbolic parameter instead of three makes heights 8-10 reachable.  We run
the stratification driver and ask: after (82), does ANY new relation
appear, at any multidegree, up to MAXHEIGHT?
"""

import sys
import time

import sympy as sp

import driver
from driver import Stratum, analyse
from shapovalov import free_lie_superalgebra

a = driver.a
MAXHEIGHT = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PAR = [1, 1, 1]
N = 3

A_SYM = [[sp.Integer(0), a, sp.Integer(1)],
         [a, sp.Integer(0), -a - 1],
         [sp.Integer(1), -a - 1, sp.Integer(0)]]


def main():
    out = open('log-3x3-onwall-h%d.txt' % MAXHEIGHT, 'w')

    def log(s):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    log('=== Statement 10.1.1: ON the wall a+b+c=0, normalized to b=1 ===')
    log('A = %s' % sp.Matrix(A_SYM).tolist())
    log('all three generators odd; maxheight = %d' % MAXHEIGHT)
    t0 = time.time()
    L = free_lie_superalgebra(N, MAXHEIGHT, par=PAR)
    log('free Lie superalgebra built in %.1f s' % (time.time() - t0))
    log('')

    st = Stratum({}, '', [a])
    lines = []
    analyse(A_SYM, N, MAXHEIGHT, st, L, maxdepth=0, par=PAR,
            log=lambda s: (lines.append(s), log(s)))

    log('')
    log('--- every relation found, by multidegree ---')
    for alpha, rel in st.rels:
        log('   %s: %s' % (alpha, driver.pretty(rel)))

    log('')
    log('--- walls (factors of the Shapovalov determinants) on this line ---')
    walls = set()
    for ln in lines:
        if ' det=' in ln:
            d = sp.sympify(ln.split(' det=')[1])
            if d == 0:
                continue
            for f, _ in sp.factor_list(d)[1]:
                if f.free_symbols:
                    walls.add(sp.factor(f))
    for w in sorted(walls, key=str):
        log('   %-16s  a = %s' % (w, sp.solve(sp.Eq(w, 0), a)))

    log('')
    heights = {}
    for alpha, _ in st.rels:
        heights.setdefault(sum(alpha), 0)
        heights[sum(alpha)] += 1
    log('relations per height: %s' % dict(sorted(heights.items())))
    log('elapsed %.1f s' % (time.time() - t0))
    out.close()


if __name__ == '__main__':
    main()
