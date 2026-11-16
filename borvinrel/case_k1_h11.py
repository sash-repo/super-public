"""
Statement 4.3.1 of the paper: the walls of the family with the Cartan matrix
below, computed over Q(a).  The question is whether the cases of various positive
integer n are all alike, or there are singular values; 2/n is replaced by a
symbolic a and the singular values of a are read off.

The second of the two k=1 matrices is

        A = ( 2    2/n )
            ( -1    0  ),      n in Z_{>0}.

A is not symmetric; it is symmetrizable, D A is symmetric for D = diag(1,-a)
with a = 2/n, giving the symmetric form used by the contravariant (Shapovalov)
form -- which is what decides the defining relations:

        B = ( 2  a )
            ( a  0 ),          a = 2/n.

Simple root 1 is even (B_11 = 2 != 0), simple root 2 is odd isotropic
(B_22 = 0), so par = [0, 1].

Strategy: keep a symbolic, run the
stratification driver, and read off the walls -- the irreducible factors of the
Shapovalov determinant in each multidegree.  The singular values of a are the
roots of those factors.  Then ask which of a = 2/n, n = 1..20, are singular.
"""

import sys

import sympy as sp

import driver
from driver import Stratum, analyse
from shapovalov import free_lie_superalgebra

a = driver.a
MAXHEIGHT = int(sys.argv[1]) if len(sys.argv) > 1 else 6

A_SYM = [[sp.Integer(2), a],
         [a, sp.Integer(0)]]
PAR = [0, 1]          # x1 even, x2 odd isotropic
N = 2


def walls_of(det, free):
    """Non-constant factors of a Shapovalov determinant, as a set."""
    out = set()
    if det == 0:
        return out
    for f, _ in sp.factor_list(det)[1]:
        if f.free_symbols & set(free):
            out.add(sp.factor(f))
    return out


def generic_run(maxheight, log):
    """Generic stratum only: collect every wall the determinants produce."""
    L = free_lie_superalgebra(N, maxheight, par=PAR)
    st = Stratum({}, '', [a])
    lines = []
    analyse(A_SYM, N, maxheight, st, L, maxdepth=0, par=PAR,
            log=lambda s: (lines.append(s), log(s)))
    return st, lines


def main():
    out = open('log-case-k1-h11.txt', 'w')

    def log(s):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    log('=== case k=1:  B = [[2,a],[a,0]],  par = [even, odd],  a = 2/n ===')
    log('maxheight = %d' % MAXHEIGHT)
    log('')

    st, lines = generic_run(MAXHEIGHT, log)

    # every wall printed by the driver, re-collected from its det= lines
    walls = set()
    for ln in lines:
        if ' det=' in ln:
            det = sp.sympify(ln.split(' det=')[1])
            walls |= walls_of(det, [a])

    log('')
    log('--- walls (irreducible factors of the Shapovalov determinants) ---')
    roots = set()
    for w in sorted(walls, key=str):
        rs = sp.solve(sp.Eq(w, 0), a)
        log('  %-24s  a = %s' % (w, ', '.join(str(r) for r in rs)))
        roots |= set(rs)

    log('')
    log('--- singular values of a, and which n = 2/a they correspond to ---')
    for r in sorted(roots, key=lambda x: sp.re(sp.N(x)) if x.is_number else 0):
        if r == 0:
            log('  a = 0        -> n = infinity, not in the family')
            continue
        n = sp.simplify(2 / r)
        infam = n.is_integer and n > 0
        log('  a = %-10s -> n = %-8s %s' %
            (r, n, 'IN the family n in Z_{>0}' if infam else 'not in the family'))

    log('')
    log('--- the family a = 2/n, n = 1..20 ---')
    hits = []
    for n in range(1, 21):
        av = sp.Rational(2, n)
        bad = [w for w in walls if sp.simplify(w.subs(a, av)) == 0]
        if bad:
            hits.append(n)
        log('  n=%-3d a=%-8s %s' %
            (n, av, 'SINGULAR: ' + ', '.join(str(w) for w in bad) if bad
             else 'generic'))

    log('')
    log('=== verdict ===')
    if hits:
        log('singular n in 1..20 (up to height %d): %s' % (MAXHEIGHT, hits))
    else:
        log('no n in 1..20 is singular up to height %d: all alike.' % MAXHEIGHT)

    log('')
    log('--- defining relations on the generic stratum ---')
    for alpha, rel in st.rels:
        log('  degree %s: %s' % (alpha, driver.pretty(rel)))

    out.close()


if __name__ == '__main__':
    main()
