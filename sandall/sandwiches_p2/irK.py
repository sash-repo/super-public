#!/usr/bin/env python3
r"""
irK.py --- the sandwich variety of ir(9; \One) over the algebraically closed
field K = \bar F_2, by the method of Subsection "The sandwich variety over K"
of the text (the script sandK.py), applied to the 26-dimensional exceptional
simple Lie algebra ir(9; \One) of [BGLLS], eq. (73).

irsand.py has already scanned all 2^26 - 1 elements over F_2 and found no
sandwich at all, so the F_2-span S of the rational sandwiches is 0 and the
whole projective space P^25 must be covered by the 26 coordinate charts
a_i = 1.  In each chart the quadrics (san) and (san2) become inhomogeneous
polynomials of degree <= 2 in 25 variables, and "no K-point" is certified by
exhibiting 1 in the ideal they generate (weak Nullstellensatz), which is exact
linear algebra over F_2 on the Macaulay matrix -- and the rank of a matrix over
F_2 does not change under field extension, so working over the prime field
loses nothing.

Output goes to runs/irK.log.

Usage:  python3 irK.py [maxdeg]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from css import LieF2                                                # noqa: E402
from sandK import (quadrics, reduce_generators, substitute,          # noqa: E402
                   one_in_ideal, annihilator_basis)
from ir import (G0T, bracket, echelon, flatten, minimal_ideal,       # noqa: E402
                prolong, bits)

LOG = os.path.join(HERE, 'runs', 'irK.log')


def main():
    maxdeg = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, 'w', buffering=1)

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')

    part = prolong(G0T)
    idl, _, _ = minimal_ideal(flatten(part))
    ir = [v for _, v in idl]
    n = len(ir)

    rec = []
    for j, u in enumerate(ir):
        v, cm = u, 1 << j
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            rec.append((v.bit_length() - 1, v, cm))
            rec.sort(key=lambda t: -t[0])

    def coords(v):
        cm = 0
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        return None if v else cm

    tab = [[coords(bracket(ir[i], ir[j])) for j in range(n)] for i in range(n)]
    L = LieF2(tab, 'ir(9;One)')
    ok, msg = L.check_jacobi()
    say('ir(9; \\One): dim %d, Lie algebra axioms over F_2: %s' % (n, msg))
    if not ok:
        return 1

    t0 = time.time()
    polys = reduce_generators(quadrics(L))
    say('%d independent quadrics cut out the sandwich cone (%.0f s)'
        % (len(polys), time.time() - t0))

    # irsand.py: no F_2-sandwich at all, so S = 0 and every coordinate
    # functional is a chart.
    charts = annihilator_basis([], n)
    say('charts to kill: %d' % len(charts))
    allgood = True
    for ell in charts:
        t1 = time.time()
        aff, m = substitute(polys, n, ell)
        aff = reduce_generators(aff)
        proved, deg = False, None
        for D in (2, 3, maxdeg):
            if D > maxdeg:
                break
            if one_in_ideal(aff, m, D):
                proved, deg = True, D
                break
        names = '+'.join('a%d' % i for i in range(n) if ell >> i & 1)
        say('  chart %-8s: %s  (%.0f s)'
            % (names, 'EMPTY over K, degree <= %d' % deg if proved
               else 'NOT decided up to degree %d' % maxdeg, time.time() - t1))
        allgood &= proved
    if allgood:
        say('==> the sandwich variety of ir(9; \\One) is EMPTY over K = \\bar F_2:')
        say('    s(ir) = 0 and n_ir(s) = ir, over every field of characteristic 2.')
    else:
        say('==> INCONCLUSIVE in some chart; raise the degree.')
    log.close()
    return 0 if allgood else 1


if __name__ == '__main__':
    sys.exit(main())
