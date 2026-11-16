import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from hwmod import irrep
from fractions import Fraction as Fr
# the family p=q=r=k, i.e. m=(k+1,k-1,k-1): atypical for EVERY a
for a in [Fr(1), Fr(2), Fr(5,3)]:
    A=[[0,1,a],[-1,2,0],[-1,0,2]]; par=[1,0,0]
    out=[]
    for k in range(1,8):
        m=(k+1,k-1,k-1)
        lam=[Fr(m[0]*(1+a)+m[1]+m[2]*a,1)/2, Fr(m[1]), Fr(m[2])]
        try:
            M=irrep(A,par,0,lam,maxdim=20000); sd=M.sdim()
            out.append("k=%d m=%s (%d|%d) dim=%d 16k^3=%d"%(k,m,sd[0],sd[1],sum(sd),16*k**3))
        except Exception as e:
            out.append("k=%d FAILED"%k)
    print("a=%s"%a); [print("   ",o) for o in out]
