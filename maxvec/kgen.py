import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from kcase1 import *
from exactlin import rank
for n in (2, 3, 4):
    g = {k: [{m: 1} for m in monos(k+2, n)] for k in range(1, 5)}
    for k in (2, 3, 4):
        B = monos(k+2, n)
        vs = [vec(br(x, y, n), B) for x in g[1] for y in g[k-1]]
        print(f"k(1|{2*n}): rank [g_1,g_{k-1}] = {rank(vs)} of dim g_{k} = {len(B)}")
# k(1|4): diagonal submodules W_a spanned by (1+a) t xi_i + (a-1) d_{eta_i} Xi etc.
n = 2
Xi = {(0, (0, 1, 2, 3)): 1}
def Wa(a):
    out = []
    for i in range(2):
        out.append(add(scal(1+a, {(1, (i,)): 1}), scal(a-1, dxi(n+i, Xi))))
        out.append(add(scal(1+a, {(1, (n+i,)): 1}), scal(a-1, dxi(i, Xi))))
    return out
B3 = monos(3, n); B4 = monos(4, n)
g0 = [{(1, ()): 1}] + [{(0, S): 1} for S in itertools.combinations(range(4), 2)]
for a in (F(1), F(2), F(-1), F(1, 3)):
    W = [vec(f, B3) for f in Wa(a)]
    stable = all(rank(W + [vec(br(x, f, n), B3)]) == rank(W) for x in g0 for f in Wa(a))
    S = [vec(br(f, h, n), B4) for f in Wa(a) for h in Wa(a)]
    print(f"a={a}: W_a g_0-stable: {stable}; rank [W_a,W_a] in g_2 = {rank(S)}")
