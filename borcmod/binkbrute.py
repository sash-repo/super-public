"""Brute-force census of k(1;N|1)-invariant bilinear maps F_a x F_b -> F_c over all bilinear maps (no ansatz),
a,b,c in F_p; cross-check of the dimensions printed by binkfull.py.
Usage: python3 binkbrute.py p N"""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from bin1 import invariant_bilinear
from osp1 import setup
from kfull import kgens
p, N = int(sys.argv[1]), int(sys.argv[2])
S = setup(p, N); kk = kgens(S); add = S['add']
reps = {a: [(add(K, m, coef=[1, a % p]), par) for K, m, par in kk] for a in range(p)}
for a in range(p):
  for b in range(p):
    for c in range(p):
      for pi in (0, 1):
        ns, unk = invariant_bilinear(S, reps[a], reps[b], reps[c], pi)
        if ns: print(f"p={p} N={N} a={a} b={b} c={c} {'odd' if pi else 'even'}: dim {len(ns)}", flush=True)
