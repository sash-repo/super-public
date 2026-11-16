"""
twoeq.py -- write out the two equations of the reduction lemma explicitly.

For X in L_1 and u in L_{-1} one has  X.(u (x) v) = [X,u].v  in I(V), because L_1 kills V.
So for a degree-1 vector  z = sum_i partial_i (x) v_i + sum_j delta_j (x) w_j  the condition
X z = 0 is  sum_i [X,partial_i] v_i + sum_j [X,delta_j] w_j = 0.

This script prints [X_0,-] and [Y_0,-] on the basis of L_{-1}, in the basis of L_0 = pe(n),
for n = 3..6, so that the two equations can be read off and checked against the hand
computation in the text.
"""

import sys
from fractions import Fraction as F
from le import abr
from pemod import Pe
from imod import Le
from reduce_test import gens


def bname(pe, k, n):
    f = pe.basis[k]
    (a, S), c = next(iter(f.items()))
    if len(S) == 2 and not any(a):
        return 'b_{%d%d}' % (S[0] + 1, S[1] + 1)
    if len(S) == 1:
        i = a.index(1)
        return ('e_%d' % (i + 1)) if i == S[0] else ('a_{%d%d}' % (i + 1, S[0] + 1))
    idx = [i for i in range(n) for _ in range(a[i])]
    return 'c_{%d%d}' % (idx[0] + 1, idx[1] + 1)


def show(pe, f, n):
    if not f:
        return '0'
    v = pe.as_vector(f)
    out = []
    for k, c in sorted(v.items()):
        s = '' if c == 1 else ('-' if c == -1 else str(c))
        out.append(s + bname(pe, k, n))
    return ' + '.join(out).replace('+ -', '- ')


def run(n):
    le = Le(n, odd_up=False)
    pe = Pe(n, odd_up=False)
    g = gens(n)
    names = ['X_0'] + (['Y_0'] if len(g) > 1 else [])
    print('n = %d' % n)
    for nm, f in zip(names, g):
        for k in range(2 * n):
            br = abr(f, le.neg[k], n)
            if br:
                print('   [%s, %-4s] = %s' % (nm, le.negname[k], show(pe, br, n)))
    sys.stdout.flush()


if __name__ == '__main__':
    for n in (2, 3, 4, 5, 6):
        run(n)
