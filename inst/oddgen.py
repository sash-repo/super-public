"""Statement 7.2.1 of the text for GENERAL (non-'t Hooft) ADHM data of charge k=2,3:
a=(lambda;Y), b=(0;-1), Y symmetric quaternionic, a^+a real (lambda solved for).
Also checks the duality criterion: the linear conditions (clin) are independent iff
the space of pairs (P Hermitian, Q anti-Hermitian) with bP + aQ = 0 is zero; its dimension
is the defect delta. Printed for general charge-2 data (delta = 0) and for the degenerate
charge-2 data with Y = 0 (delta = 3).
Exact rational arithmetic (fractions.Fraction, ranks via ../lib/exactlin.py)."""
import random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as F
from oddmod import qmul, qconj, qadd, Z, dagger, mm
from exactlin import rank
def qinv(p):
    n=sum(x*x for x in p); c=qconj(p); return tuple(x/n for x in c)
def rq(r): return tuple(F(r.randint(-5,5)) for _ in range(4))
def data(k, r):
    Y=[[None]*k for _ in range(k)]
    for i in range(k):
        for j in range(i,k):
            Y[i][j]=Y[j][i]=rq(r)
    lam=[None]*k
    lam[0]=(F(r.randint(1,5)),0,0,0)
    # off-diagonal (0,j) of a^+a must be real: conj(l0) lj + sum_i conj(Y_i0) Y_ij
    for j in range(1,k):
        s=Z
        for i in range(k): s=qadd(s, qmul(qconj(Y[i][0]),Y[i][j]))
        imag=(0,s[1],s[2],s[3])
        lam[j]=qadd((F(r.randint(-5,5)),0,0,0), tuple(-x for x in qmul(qinv(qconj(lam[0])),imag)))
    a=[lam]+Y
    ad=dagger(a); G=mm(ad,a)
    ok=all(G[i][j][1]==G[i][j][2]==G[i][j][3]==0 for i in range(k) for j in range(k))
    return a, ok
def count(k,a,s1,s2):
    b=[[Z]*k]+[[(F(-1),0,0,0) if i==j else Z for j in range(k)] for i in range(k)]
    ad,bd=dagger(a),dagger(b); n=(k+1)*k*4; cols=[]
    for v in range(n):
        e=[0]*n; e[v]=1
        c=[[tuple(F(e[((i*k)+j)*4+t]) for t in range(4)) for j in range(k)] for i in range(k+1)]
        X=mm(ad,c); Xd=dagger(X); W=mm(bd,c); Wd=dagger(W); col=[]
        for i in range(k):
            for j in range(k):
                col+=[x-s1*y for x,y in zip(X[i][j],Xd[i][j])]
                col+=[x+s2*y for x,y in zip(W[i][j],Wd[i][j])]
        cols.append(col)
    return n-rank([list(t) for t in zip(*cols)])
r=random.Random(1913)
for k in (2,3):
    for trial in range(3):
        a,ok=data(k,r)
        print(k, "a^+a real:",ok, [count(k,a,s1,s2) for s1 in (1,-1) for s2 in (1,-1)], "expected",4*k)

# duality criterion: coker of (clin) = {(P Herm, Q antiHerm) : bP + aQ = 0}
def herm_basis(k, anti):
    B=[]
    for i in range(k):
        for t in ((1,2,3) if anti else (0,)):
            M=[[Z]*k for _ in range(k)]; e=[0]*4; e[t]=F(1); M[i][i]=tuple(e); B.append(M)
    for i in range(k):
        for j in range(i+1,k):
            for t in range(4):
                M=[[Z]*k for _ in range(k)]; e=[0]*4; e[t]=F(1); M[i][j]=tuple(e)
                ce=qconj(tuple(e)); M[j][i]=tuple(-x for x in ce) if anti else ce
                B.append(M)
    return B
def coker(k,a):
    b=[[Z]*k]+[[(F(-1),0,0,0) if i==j else Z for j in range(k)] for i in range(k)]
    cols=[]
    for P in herm_basis(k,False):
        cols.append([x for row in mm(b,P) for q in row for x in q])
    for Q in herm_basis(k,True):
        cols.append([x for row in mm(a,Q) for q in row for x in q])
    return len(cols)-rank([list(t) for t in zip(*cols)])
r=random.Random(7)
for t in range(3):
    a,ok=data(2,r); print("k=2 general: coker dim", coker(2,a), "kernel", count(2,a,1,1))
lam=[(F(1),0,0,0),(F(2),0,0,0)]; a=[lam,[Z,Z],[Z,Z]]
print("degenerate (coincident centres at 0): coker", coker(2,a), "kernel", count(2,a,1,1))
