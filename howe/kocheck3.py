"""n=4: the remaining families, cheapest first, skipping the (expensive, and empty) printed 2c."""
import sys
from kocheck import om, scal, check
import kocheck

def families(n, lmax=3):
    return [('3b', om(n, 0), 3, om(n, n - 1)),
            ('2c swapped', om(n, n - 2), 2, scal(om(n, n - 1), 2)),
            ('3a', om(n, n - 1), 3, om(n, 0)),
            ('4', om(n, n - 1), 4, om(n, n - 1))]

kocheck.families = families
n = int(sys.argv[1]); ht = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print("SLe(%d): remaining families" % n)
check(n, 3, ht)
