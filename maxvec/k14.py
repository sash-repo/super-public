import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from kcase1 import *
from exactlin import rank, nullspace
n = 2
Xi = {(0, (0, 1, 2, 3)): 1}
def Wp(p, q):  # p*t*xi + q*d Xi
    out = []
    for i in range(2):
        out.append(add(scal(p, {(1, (i,)): 1}), scal(q, dxi(n+i, Xi))))
        out.append(add(scal(p, {(1, (n+i,)): 1}), scal(q, dxi(i, Xi))))
    return out
gm1 = [{(0, (i,)): 1} for i in range(4)]
for name, (p, q) in {'a=1 (q=0)': (1, 0), 'a=-1 (p=0)': (0, 1), 'a=0': (1, -1), 'a=inf': (1, 1), 'a=2': (3, 1), 'a=-2': (-1, -3), 'a=1/2': (F(3,2), F(-1,2))}.items():
    B3 = monos(3, n); U = [vec(f, B3) for f in Wp(p, q)]; dims = [len(U)]
    for k in range(2, 6):
        Bk = monos(k+2, n); Bp = monos(k+1, n)
        if not U: dims.append(0); continue
        phis = nullspace(U); rows = []
        for e in gm1:
            cols = [vec(br(e, {m: 1}, n), Bp) for m in Bk]
            for phi in phis:
                rows.append([sum(F(phi[i])*c[i] for i in range(len(Bp))) for c in cols])
        U = nullspace(rows); dims.append(len(U))
    print(name, dims, ' total sdim of u: even', 1 + 7 + sum(d for k, d in enumerate(dims, 1) if k % 2 == 0), 'odd', 4 + sum(d for k, d in enumerate(dims, 1) if k % 2))
