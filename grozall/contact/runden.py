"""Census: bilinear k(3)-invariant operators on weighted densities, F_lam x F_mu -> F_nu."""
import time, sys
from fractions import Fraction as Fr
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from affine import pencil, dim_at, ops_at
from sweep import grid_jumps, minor_gcd, points_on

for W in (0,1,2,3,4,5,6,8,10):
    t0=time.time()
    terms,V0,N0,N1,N2 = pencil(0,0,0,W)
    if not V0:
        print("W=%2d: no sp(2)-invariant scheme at all -> dim 0 for every weight  [%.1fs]"
              %(W,time.time()-t0)); sys.stdout.flush(); continue
    gdim = dim_at(V0,N0,N1,N2,Fr(17,13),Fr(31,11))
    r = len(V0)-gdim
    print("W=%2d: dim V_0=%2d  generic dim=%d  (nu = lam+mu+%s)"%(W,len(V0),gdim,Fr(W,2)))
    js = grid_jumps(V0,N0,N1,N2,gdim)
    if js:
        print("     grid jumps (den<=6, |.|<=4):", js[:40], "..." if len(js)>40 else "")
    else:
        print("     grid: dim is constant = %d on the whole grid (2401 points)"%gdim)
    if r>0 and len(V0)<=10:
        g,facs = minor_gcd(V0,N0,N1,N2,r)
        print("     gcd of 24 random maximal minors:", g)
        for f in facs:
            pts = points_on(f)
            vals = [(str(a),str(b),dim_at(V0,N0,N1,N2,a,b)) for a,b in pts]
            print("       factor %s :"%f, vals)
    print("     [%.1fs]"%(time.time()-t0)); sys.stdout.flush()
