"""
mgencheck.py -- does L_1 generate L_{>=1}?  For M(n) and for M_{a,b}(n).

If it does, "singular" = killed by n_+ (in L_0) and by L_1, a finite test.
"""

import sys
from fractions import Fraction as F
from le import rref
from mn import mbr, layer, mdeg
from mimod import Mn, Mab


def span_dim(fs, amb):
    idx = {m: i for i, m in enumerate(amb)}
    rows = [{idx[m]: c for m, c in f.items()} for f in fs]
    piv, red = rref([r for r in rows if r], len(amb))
    return len(piv)


def check(alg_layer, n, jmax):
    """alg_layer(j) -> basis of the j-th component"""
    cur = list(alg_layer(1))
    for j in range(2, jmax + 1):
        amb = [next(iter(f)) for f in layer(n, j)]
        # everything reachable as [L_1, (already generated in degree j-1)]
        new = []
        for f in alg_layer(1):
            for g in cur:
                b = mbr(f, g, n)
                if b and mdeg(b, n) == j:
                    new.append(b)
        full = len(alg_layer(j))
        got = span_dim(new, amb)
        print("    degree %d: generated %d of %d%s" % (j, got, full, "" if got == full else "   <-- GAP"))
        cur = alg_layer(j) if got == full else new
        if got != full:
            # report what is missing
            return False
    return True


if __name__ == '__main__':
    ns = [int(a) for a in sys.argv[1:]] or [2, 3, 4]
    for n in ns:
        mn = Mn(n)
        print("=== n = %d ===" % n)
        print("  M(n):")
        check(lambda j: layer(n, j), n, 5 if n <= 3 else 4)
        for (a, b) in [(1, 0), (0, 1), (1, 1), (1, 2), (n, -1), (2, -n)]:
            ab = Mab(mn, a, b)
            print("  M_{%s,%s}(n):" % (a, b))
            check(ab.layer, n, 4)
