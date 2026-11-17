"""The free Volichenko algebra on ONE generator: dim of the degree-n part.

Exact over Q (FLINT rationals) as far as it goes, and over two different primes
beyond, a rank mod p being only a lower bound for the rank over Q.
"""
import sys
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rank as ex_rank

import mdeg
import volid


def dims(n, p=None):
    trees = mdeg.monos((n,))
    ev = mdeg.Evaluator()
    vals = [ev(t) for t in trees]
    keys = sorted({w for v in vals for w in v})
    pos = {k: i for i, k in enumerate(keys)}
    rows = []
    for v in vals:
        r = [0] * len(keys)
        for w, c in v.items():
            r[pos[w]] = int(c) % p if p else Fr(c)
        rows.append(r)
    return len(trees), ex_rank(rows, p=p) if p else ex_rank(rows)


if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    print("  n   dim M_(n)   dim V over Q   mod 2^19-1   mod 2^31-1")
    for n in range(1, NMAX + 1):
        dM, dQ = dims(n)
        _, d1 = dims(n, (1 << 19) - 1)
        _, d2 = dims(n, (1 << 31) - 1)
        print("%3d %10d %14d %12d %12d   %s"
              % (n, dM, dQ, d1, d2, "OK" if dQ == d1 == d2 else "DIFFER"),
              flush=True)
