#!/usr/bin/env python3
"""H^q(L_k; witt), L_k = <E_i | i >= k> (E_i = L_{-i}), by the truncation device of
witt2.py (k=2 is the symbol of Kontsevich's M_g, whose stabilizer is
C[[t]]d/dt = witt_{>=-1}; k=1 is witt_-, the symbol of M_{g,1}; Statement A.16.4).
Usage: wittk.py K module wlo whi q N1 N2 ..."""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import witt2
K = int(sys.argv[1])

def _subsets(q, N):
    res = []
    def rec(start, left, budget, acc):
        if left == 0:
            res.append(tuple(acc)); return
        i = start
        while i + sum(range(i + 1, i + left)) <= budget:
            acc.append(i); rec(i + 1, left - 1, budget - i, acc); acc.pop(); i += 1
    rec(K, q, N, [])
    return sorted(res)
witt2._subsets = _subsets
mod = witt2.make_module(sys.argv[2])
wlo, whi, q = map(int, sys.argv[3:6]); Ns = [int(x) for x in sys.argv[6:]]
print("k=%d module=%s q=%d backend=%s" % (K, mod.name, q, witt2.backend()))
for w in range(wlo, whi + 1):
    row = [witt2.cohomology(q, N, w, mod)[0] for N in Ns]
    print("%4d" % w, row, "dimC=%d" % witt2.cohomology(q, Ns[-1], w, mod)[1])
