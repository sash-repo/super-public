"""
Statement 11.2.1b of the paper: the relation on the k-th wall in closed form.

For the family (80) (main diagonal 0, all three generators odd) put

    R_1 = b [x_3,[x_1,x_2]] - a [x_2,[x_1,x_3]],     R_k = (ad_{[x_2,x_3]})^{k-1} (R_1).

For each k the script checks that R_k lies in the radical r_+ of the contravariant form
on the wall a + b + k c = 0 (c eliminated), and does NOT lie in it on the neighboring
wall a + b + (k+1) c = 0.  That R_k spans the new part of r_+ in multidegree (1,k,k)
is the output of `strata3.py odd0 H 1` (one new relation on each wall).

Usage:  python closedform.py [KMAX]     (KMAX defaults to 6; k = 6 takes about a minute)
"""

import sys
import time

import sympy as sp

from driver import a, b, c
from check_paper import T, add, scale, mdeg, spec, in_radical, X1, X2, X3

A_SYM = [[sp.Integer(0), a, b], [a, sp.Integer(0), c], [b, c, sp.Integer(0)]]


def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    R = add(scale(T((X3, (X1, X2))), b), scale(T((X2, (X1, X3))), -a))
    y = T((X2, X3))
    from driver import sbracket
    for k in range(1, kmax + 1):
        t0 = time.time()
        alpha = mdeg(R)
        on = in_radical({w: sp.expand(v.subs(c, -(a + b) / k)) for w, v in R.items()},
                        spec(A_SYM, {c: -(a + b) / k}), 3, alpha)
        off = in_radical({w: sp.expand(v.subs(c, -(a + b) / (k + 1))) for w, v in R.items()},
                         spec(A_SYM, {c: -(a + b) / (k + 1)}), 3, alpha)
        print('k=%d  multidegree %s   on a+b+%dc=0: %s   on a+b+%dc=0: %s   [%.1fs]'
              % (k, alpha, k, 'IN r_+' if on else 'NOT in r_+',
                 k + 1, 'IN r_+' if off else 'NOT in r_+', time.time() - t0), flush=True)
        R = sbracket(y, R, [1, 1, 1])


if __name__ == '__main__':
    main()
