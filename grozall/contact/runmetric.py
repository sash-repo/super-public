"""The contact analog of Grozman's setting: the arguments are 'metrics' on the
contact distribution, i.e. sections of S^2 D^* (x) L^lam = T(2,lam)."""
import time, sys
from fractions import Fraction as Fr
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from affine import pencil, dim_at, ops_at
from sweep import grid_jumps, minor_gcd, points_on
from gen import term_basis

CONFIGS = []
for k3 in (0,2,4):
    for W in (0,1,2,3,4):
        CONFIGS.append((2,2,k3,W))

for (k1,k2,k3,W) in CONFIGS:
    t0=time.time()
    nt = len(term_basis(k1,k2,k3,W))
    try:
        terms,V0,N0,N1,N2 = pencil(k1,k2,k3,W,nfield=1)
    except Exception as e:
        print("T(%d)xT(%d)->T(%d) W=%d : FAILED %s"%(k1,k2,k3,W,e)); sys.stdout.flush(); continue
    if not V0:
        print("T(%d)xT(%d)->T(%d) W=%d : terms=%d, dim V_0=0  [%.0fs]"%(k1,k2,k3,W,nt,time.time()-t0))
        sys.stdout.flush(); continue
    gdim = dim_at(V0,N0,N1,N2,Fr(17,13),Fr(31,11))
    print("T(%d)xT(%d)->T(%d) W=%d : terms=%d dim V_0=%d generic dim=%d"%(k1,k2,k3,W,nt,len(V0),gdim))
    js = grid_jumps(V0,N0,N1,N2,gdim,den=6,bound=3)
    if js:
        print("     JUMPS:", js[:30], "..." if len(js)>30 else "")
    else:
        print("     dim constant on the grid (den<=6, |.|<=3)")
    print("     [%.0fs]"%(time.time()-t0)); sys.stdout.flush()
