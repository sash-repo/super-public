import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); D_ = HERE
sys.path.insert(0, D_)
from gvect import P
import dy3
from negcube import lin
V = dy3.V; Z = tuple([0]*10)
H = [dy3.h1, dy3.h2, dy3.h3]
def eig(F):
    out = []
    for h in H:
        b = V.bracket(h, F)
        # b == lam F ?
        lam = None
        for l in range(P):
            if lin(V, [b, F], (1, (-l) % P)) == {}:
                lam = l
        out.append(lam)
    return tuple(out)
def unit(a): return tuple(1 if t == a else 0 for t in range(10))
fields = {}
for a in range(10):
    for b in range(10):
        if dy3.DEG[b] - dy3.DEG[a] == 3:
            fields["u%d d%d" % (a+1, b+1)] = {(unit(a), b): 1}
fields["d7"] = {(Z, 6): 1}
print("degree -3 fields of vect(10;1), eigenvalues under (h1,h2,h3):")
for k, F in fields.items(): print("  ", k, eig(F))
print("g_-1:", [eig(X) for X in dy3.GM[:3]], " g_-3:", eig(dy3.GM[6]))
X = dy3.GM
for (i, j, k) in ((0,1,2),(1,2,0),(2,0,1)):
    print("[X%d,[X%d,X%d]] =" % (i+1,j+1,k+1), V.bracket(X[i], V.bracket(X[j], X[k])))
print("[Xi,[Xi,Xj]]:", [(i+1,j+1, V.bracket(X[i], V.bracket(X[i], X[j]))) for i in range(3) for j in range(3)])
print("[X1,X2],[X2,X3],[X3,X1] =", V.bracket(X[0],X[1]), V.bracket(X[1],X[2]), V.bracket(X[2],X[0]))
