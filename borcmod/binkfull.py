"""Binary maps F_a x F_b -> F_c invariant under the whole k(1;N|1): within span m(D1^i D2^j) (the K_theta-invariants),
impose invariance under all odd K_{g theta}, g = t^{(r)} (they generate k(1;N|1)).
Prints a basis of each space, one map per line, as {(i,j): c_ij}.  Usage: python3 binkfull.py p N"""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace
from binnum import basics, bilin, flat
from kfull import kgens
p, N = int(sys.argv[1]), int(sys.argv[2])
S, D, pows, prod = basics(p, N); n, P = S['n'], S['P']
odd = [(K, m) for K, m, par in kgens(S) if par == 1]
add = S['add']
def X(r, lam): K, m = odd[r]; return add(K, m, coef=[1, lam % p])
Xc = {}
def getX(r, lam):
    if (r, lam) not in Xc: Xc[(r, lam)] = X(r, lam)
    return Xc[(r, lam)]
def defect(T, r, a, b, c, pi):
    out = {}
    s = -1 if pi else 1
    Xa, Xb, Xcc = getX(r, a), getX(r, b), getX(r, c)
    def addto(k, w): out[k] = (out.get(k, 0) + w) % p
    for (z, u, v), w in T.items():
        for y in range(n):
            if Xcc[y][z]: addto((y, u, v), Xcc[y][z]*w)
        for u2 in range(n):
            if Xa[u][u2]: addto((z, u2, v), -s*Xa[u][u2]*w)
        for v2 in range(n):
            if Xb[v][v2]: addto((z, u, v2), -s*(-1 if u % 2 else 1)*Xb[v][v2]*w)
    return {k: w for k, w in out.items() if w}
ops = {(i, j): bilin(S, pows, prod, 0, i, j) for i in range(2*P) for j in range(2*P)}
rs = range(P) if len(sys.argv) < 4 else [int(x) for x in sys.argv[3].split(',')]
for a in range(p):
  for b in range(p):
    for c in range(p):
      for pi in (0, 1):
        keys = [k for k in ops if sum(k) % 2 == pi]
        rows = []
        for r in rs:
            cols = [flat(defect(ops[k], r, a, b, c, pi), n) for k in keys]
            rows += [list(x) for x in zip(*cols) if any(x)]
        ns = [[1 if x == y else 0 for x in range(len(keys))] for y in range(len(keys))] if not rows else nullspace(rows, p)
        for v in ns:
            terms = {keys[i]: x for i, x in enumerate(v) if x}
            print(f"p={p} N={N} a={a} b={b} c={c} {'odd' if pi else 'even'}: {terms}", flush=True)
