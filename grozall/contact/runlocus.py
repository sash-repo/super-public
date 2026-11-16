"""1) the jump locus of T(2,lam) x T(2,mu) -> T(k3,nu) proved by the minors;
   2) the unary census for every source fiber up to k=4."""
import os, sys, time
from fractions import Fraction as Fr
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from affine import pencil, dim_at
from sweep import minor_gcd, points_on
from unary import u_census

print("=== the exact jump locus (gcd of 24 random maximal minors of the pencil) ===")
for (k1,k2,k3,W) in ((2,2,0,2),(2,2,4,2),(2,2,2,2)):
    t0=time.time()
    terms,V0,N0,N1,N2 = pencil(k1,k2,k3,W,nfield=1)
    gdim = dim_at(V0,N0,N1,N2,Fr(17,13),Fr(31,11))
    r = len(V0)-gdim
    g,facs = minor_gcd(V0,N0,N1,N2,r)
    print("T(2)xT(2)->T(%d) W=%d : dim V_0=%d generic dim=%d ; gcd = %s  [%.0fs]"
          %(k3,W,len(V0),gdim,g,time.time()-t0))
    for f in facs:
        pts = points_on(f)
        print("    on %s :"%f, [(str(a),str(b),dim_at(V0,N0,N1,N2,a,b)) for a,b in pts])
    sys.stdout.flush()

print()
print("=== unary census: T(k,lam) -> T(k3,lam3), k,k3 <= 4, W <= 4 ===")
WEIGHTS = [Fr(17,13)]+[Fr(n,6) for n in range(-18,19)]
for k in range(5):
    for k3 in range(5):
        for W in range(5):
            found=[]
            for lam in WEIGHTS:
                l3,tt,ops = u_census(k,lam,k3,W)
                if ops: found.append((lam,l3,len(ops)))
            if found:
                generic = [f for f in found if f[0]==Fr(17,13)]
                if generic:
                    print("  T(%d,lam) -> T(%d,lam+%s)  W=%d : dim %d for EVERY weight"
                          %(k,k3,Fr(W+k-k3,2),W,generic[0][2]))
                else:
                    print("  T(%d,lam) -> T(%d,lam+%s)  W=%d : only at lam = %s  (dim %d)"
                          %(k,k3,Fr(W+k-k3,2),W,[str(f[0]) for f in found],found[0][2]))
            sys.stdout.flush()
