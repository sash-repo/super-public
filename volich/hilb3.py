"""The free Volichenko algebra on ONE generator, by the streaming rank.

hilb.py builds the whole matrix and hands it to an exact solver, which is what
stops it: in degree 12 that is Cat_11 = 58 786 rows of 4096 entries at once.
Here the rows are fed in blocks to the same echelon basis the multidegree runs
use, whose stored part is (rank x 2^n) and so stays small -- the rank is the
answer, and the answer is under a hundred.

Two primes; a rank mod p is a lower bound for the rank over Q, and hilb.py has
the exact rational value to compare against as far as it reaches.
"""
import sys
import time

import numpy as np

import mdeg
import volid


def dim_one(n, p):
    trees = mdeg.monos((n,))
    ev = mdeg.Evaluator()
    keys = mdeg.word_index((n,))
    e = mdeg.Ech(len(keys), p=p)
    for s in range(0, len(trees), 64):
        X = np.zeros((min(64, len(trees) - s), len(keys)))
        for i, t in enumerate(trees[s:s + 64]):
            for w, c in ev(t).items():
                X[i, keys[w]] = float(int(c) % p)
        e.add_block(X)
    return len(trees), e.rank()


if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    P1, P2 = (1 << 19) - 1, (1 << 31) - 1
    print("  n   dim M_(n)   mod 2^19-1   mod 2^31-1   verdict")
    for n in range(1, NMAX + 1):
        t0 = time.time()
        dM, d1 = dim_one(n, P1)
        # p = 2^31-1 would break the float64 exactness bound of Ech for the
        # ambients here, so the second prime is taken by the exact solver while
        # it still fits, and dropped -- with a word -- when it does not.
        d2 = None
        if dM <= 20000:
            _, d2 = __import__("hilb").dims(n, P2)
        print("%3d %10d %12d %12s   %s  [%.0fs]"
              % (n, dM, d1, d2 if d2 is not None else "-",
                 "OK" if d2 in (None, d1) else "DIFFER", time.time() - t0),
              flush=True)
