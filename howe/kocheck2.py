"""kocheck.py restricted to the families that are new for sle: 2b,2c,3a,3b,4 (+1c lam=0)."""
import sys
from kocheck import om, scal, check
import kocheck

def families(n, lmax=3):
    return [('1c lam=0', om(n, 0), 1, om(n, n - 1)),
            ('2a', om(n, 0), 2, om(n, 0)),
            ('2b', om(n, n - 1), 2, om(n, n - 1)),
            ('2c printed', scal(om(n, n - 1), 2), 2, om(n, n - 2)),
            ('2c swapped', om(n, n - 2), 2, scal(om(n, n - 1), 2)),
            ('3b', om(n, 0), 3, om(n, n - 1)),
            ('3a', om(n, n - 1), 3, om(n, 0)),
            ('4', om(n, n - 1), 4, om(n, n - 1))]

kocheck.families = families
n = int(sys.argv[1]); ht = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print("SLe(%d): the families that are new for sle(n)" % n)
check(n, 3, ht)
