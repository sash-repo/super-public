"""osp(1|2)-invariant bilinear maps F_a x F_b -> F_c, a,b,c in F_p, computed inside the span of the
K_theta-invariant maps B_ij (Lemma 7.2), using only K_{t theta}; output format as bin1.py, to be compared with it.
Usage: python3 binred.py p N"""
import sys, re
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace
from binnum import basics, bilin, flat
from bindef_lib import make_defect
from osp1 import generators
p, N = int(sys.argv[1]), int(sys.argv[2])
S, D, pows, prod = basics(p, N); n, P = S['n'], S['P']
dfn = make_defect(S, generators(S, -1), p, n)
ops = {(i, j): bilin(S, pows, prod, 0, i, j) for i in range(2*P) for j in range(2*P)}
for a in range(p):
  for b in range(p):
    for c in range(p):
      for pi in (0, 1):
        keys = [k for k in ops if sum(k) % 2 == pi]
        cols = [flat(dfn(ops[k], a, b, c, pi), n) for k in keys]
        rows = [list(r) for r in zip(*cols)]
        rows = [r for r in rows if any(r)]
        d = len(keys) if not rows else len(nullspace(rows, p))
        if d: print(f"p={p} N={N} a={a} b={b} c={c} {'odd' if pi else 'even'}: dim {d}", flush=True)
