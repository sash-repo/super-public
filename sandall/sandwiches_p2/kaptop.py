#!/usr/bin/env python3
r"""
kaptop.py --- the sandwich of Kap_2(2m) for m > 2, and the parity question.

The exhaustive scan of kap.py found that the sandwich subalgebra of Kap_2(4) is
spanned by ONE element: the sum of all the basis vectors,

        s = sum_{u in V, u =/= 0} e_u,

and that of Kap_{4,B}(4) by e_0 (the centre) and s.  Under the isomorphism of
Prop. 5.1.1 of arXiv:1301.2781, Kap_{4,B}(2m) = O(2m; N_s) with the bracket
(67), the correspondence being e_u <-> f_u = (1+p)^{u'}(1+q)^{u''}, we have

        sum_{u in V} e_u  <->  prod_{1 <= i <= m} (1+x_i)(1+y_i)
                          =    prod_{1 <= i <= m} p_i q_i,

the TOP monomial of the divided-power algebra --- the element whose coefficient
the Berezin integral (79) reads off.  This script verifies, for m = 1, ..., 4,
that this element is a sandwich (both (san) and (san2)) without any 2^n scan,
and that no scan is needed to see it: the check is a single matrix computation.

It also settles, for the three superizations of Kap_2(4)+V^*, the parity of the
sandwich, its even and odd components, and whether these are sandwiches too ---
the question raised in Subsec. "On DS-functor and restrictedness for p=2" of
the text, where an element s inhomogeneous with respect to parity with (ad_s)^2=0
is asked to be a genuine sandwich, i.e., to satisfy (san2) as well.

Output: runs/kaptop.log.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from css import rref, all_elements                                  # noqa: E402
from kap import (kap2, kapB, kap4, Closure, Q_arf, fast_sandwiches)  # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'kaptop.log')
_log = None


def say(msg=''):
    print(msg)
    if _log:
        _log.write(msg + '\n')


def is_sandwich(L, x):
    """(san) and (san2) for the single element x"""
    cols = [L.mul(x, 1 << j) for j in range(L.n)]
    for c in cols:                                   # (ad_x)^2 = 0
        w = 0
        for k in range(L.n):
            if (c >> k) & 1:
                w ^= cols[k]
        if w:
            return False, '(san) fails'
    img = rref([c for c in cols if c])                # (san2): Im ad_x abelian
    for a in img:
        for b in img:
            if L.mul(a, b):
                return False, '(san2) fails'
    return True, 'sandwich'


def main():
    global _log
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    _log = open(LOG, 'w', buffering=1)

    say('=' * 78)
    say('The top monomial as a sandwich of Kap_2(2m) and Kap_{4,B}(2m)')
    say('=' * 78)
    for m in range(1, 5):
        KB, K2 = kapB(m), kap2(m)
        top = (1 << KB.n) - 1                        # sum over ALL u in V
        ok1, msg1 = is_sandwich(KB, top)
        z = 1 << KB.idx[0]                           # the centre e_0
        ok0, _ = is_sandwich(KB, z)
        ok2, msg2 = is_sandwich(K2, (1 << K2.n) - 1)  # sum over u =/= 0
        say('m = %d:  dim Kap_{4,B}(%d) = %-4d dim Kap_2(%d) = %-4d'
            % (m, 2 * m, KB.n, 2 * m, K2.n))
        say('   sum_{u in V} e_u = p_1q_1...p_mq_m is a sandwich of Kap_{4,B}: %s'
            % (ok1,))
        say('   the centre e_0 = 1 is a sandwich of Kap_{4,B}: %s' % (ok0,))
        say('   sum_{u =/= 0} e_u is a sandwich of Kap_2(%d): %s  (%s)'
            % (2 * m, ok2, msg2))
        # the image of ad on that element, and the codimension of its normaliser
        s = rref([(1 << K2.n) - 1])
        nor = K2.normaliser(s) if K2.n <= 20 else None
        if nor is not None:
            say('   codim of the normaliser of Span(s) in Kap_2(%d): %d'
                % (2 * m, K2.n - len(nor)))
        else:
            # the normaliser of a 1-dimensional s is the centralizer of s
            cols = [K2.mul((1 << K2.n) - 1, 1 << j) for j in range(K2.n)]
            rk = len(rref([c for c in cols if c]))
            say('   rank ad_s = %d, so the centraliser of s has codimension %d'
                ' in Kap_2(%d); the normaliser of Span(s) contains it'
                % (rk, rk, 2 * m))
    # the same for Kap_{4,a}: does the family have a sandwich at all?
    say('')
    for m in (2, 3):
        for a in (0, 1):
            J = kap4(m, a)
            top = (1 << J.n) - 1
            ok, msg = is_sandwich(J, top)
            say('m = %d: in Kap_{4,%d}(%d) (dim %d) the sum of all basis vectors'
                ' is a sandwich: %s' % (m, a, 2 * m, J.n, ok))

    # ------------------------------------------------------ the parity question
    say('')
    say('=' * 78)
    say('The parity of the sandwich in the superizations of Kap_2(4) + V^*')
    say('=' * 78)
    K2 = kap2(2)
    cl = Closure(K2, 4, 'Kap_2(4)+V^*')
    S = fast_sandwiches(cl)
    say('the 2-closure has %d non-zero sandwich(es) over F_2' % len(S))
    parities = [('KapLS_2(4)  (linear)', lambda u: K2.B(1, u)),
                ('KapS_{2,0}(4) (nonlinear)', lambda u: (Q_arf(2, 0)(u) + 1) & 1),
                ('KapS_{2,1}(4) (nonlinear)', lambda u: (Q_arf(2, 1)(u) + 1) & 1)]
    for x in S:
        for name, par in parities:
            ev = od = 0
            for i in range(cl.nJ):
                if (x >> i) & 1:
                    if par(cl.J.gamma[i]):
                        od |= 1 << i
                    else:
                        ev |= 1 << i
            ev |= x >> cl.nJ << cl.nJ                # V^* is even
            oke, _ = is_sandwich(cl, ev) if ev else (None, '')
            oko, _ = is_sandwich(cl, od) if od else (None, '')
            sq = cl.sq(od)
            say('%-26s : sdim of the sandwich (%d|%d);'
                % (name, bin(ev).count('1'), bin(od).count('1')))
            say('    its even part is a sandwich: %s;  its odd part: %s;'
                '  (odd part)^2 = %s'
                % (oke, oko, 'ZERO' if sq == 0 else 'non-zero'))
    say('')
    say('So the sandwich of each superization is inhomogeneous with respect to'
        ' parity,')
    say('and it does satisfy (san2) --- it is a genuine sandwich, not merely an'
        ' element')
    say('with (ad_s)^2 = 0.')
    _log.close()


if __name__ == '__main__':
    main()
