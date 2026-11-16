"""The exceptional operator T(2,lam) x T(2,-lam) -> F_3 found at W=2: explicit shape,
and the unary census needed to decide whether it is a composition."""
import os, sys
from fractions import Fraction as Fr
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from affine import pencil, dim_at, ops_at
from gen import term_basis, term_latex
from unary import u_census

print("=== the operator T(2,lam) x T(2,-lam) -> T(0,3), W=2 ===")
terms,V0,N0,N1,N2 = pencil(2,2,0,2,nfield=1)
for lam in (Fr(1), Fr(0), Fr(1,2), Fr(-3)):
    print("  lam=%s mu=%s : dim = %d"%(lam,-lam,dim_at(V0,N0,N1,N2,lam,-lam)))
ops = ops_at(terms,V0,N0,N1,N2,Fr(1),Fr(-1))
print("  number of operators at (1,-1):", len(ops))
def show(vec, k1,k2,k3):
    out=[]
    for cf,t in zip(vec,terms):
        if cf: out.append("%s*%s"%(cf,term_latex(t,k1,k2,k3)))
    return "  +  ".join(out)
for v in ops:
    print("  ", show(v,2,2,0))

print()
print("=== unary operators T(k,lam) -> T(k',lam'), all weights (the pencil is not")
print("    needed: a unary census at a generic lam plus a scan of special ones) ===")
for k in (0,2):
    for k3 in (0,1,2,3,4):
        for W in (0,1,2,3,4):
            found=[]
            for lam in [Fr(17,13)]+[Fr(n,6) for n in range(-18,19)]:
                l3,tt,ops = u_census(k,lam,k3,W)
                if ops:
                    found.append((str(lam),str(l3),len(ops)))
            if found:
                gen = [f for f in found if f[0]=="17/13"]
                print("  T(%d,lam)->T(%d,lam+%s)  W=%d : nonzero for %d weights%s"
                      %(k,k3,Fr(W+k-k3,2),W,len(found),
                        "  (generic: dim %d)"%gen[0][2] if gen else "  (special weights only: %s)"%[f[0] for f in found][:8]))
