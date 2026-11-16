import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); D_ = HERE
sys.path.insert(0, D_)
from gvect import P
from skry3 import cube
import dy3
from negcube import lin
V, GM = dy3.V, dy3.GM
def cnt(idx):
    nz = 0; ins = 0; N = 0
    for co in itertools.product(range(P), repeat=len(idx)):
        D = lin(V, [GM[i] for i in idx], co); N += 1
        C = cube(V, D)
        if C:
            nz += 1
            # is C in span of g_-4 (X8..X10)?
            if all(k[0] == tuple([0]*10) and k[1] in (7, 8, 9) for k in C):
                ins += 1
    return N, nz, ins
print("g_-1 only:", cnt([0,1,2]))
print("g_-1 + g_-2:", cnt([0,1,2,3,4,5]))
print("g_-2 + g_-3:", cnt([3,4,5,6]))
# Jacobson [x,[x,y]]+[y,[y,x]] for x in g_-1, y in g_-2
bad = 0
for i in range(3):
    for j in range(3,6):
        x, y = GM[i], GM[j]
        s = lin(V, [V.bracket(x, V.bracket(x, y)), V.bracket(y, V.bracket(y, x))], (1, 1))
        if s: bad += 1; print("  s(X%d,X%d) =" % (i+1, j+1), s)
print("nonzero s(g_-1,g_-2) basis pairs:", bad)
