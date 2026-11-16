"""Where the extra relations of g(A(t)) sit: minimal multidegrees where dim g(A(t))_beta is below
the Gabber-Kac prediction, versus the lattice points of the sphere (beta|beta)=2(rho|beta) in Q_+."""
import sys, itertools
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE, os.path.join(_HERE, "..", "evenpart")]
from eng import G
from chk import FIN, At, predict
import chk

def symmetrize(A):
    # A Fin (DA symmetric): find d_i>0 with d_i A_ij = d_j A_ji; (alpha_i|alpha_j)=d_i A_ij
    n=len(A); d=[None]*n; d[0]=F(1); ch=True
    while ch:
        ch=False
        for i in range(n):
            for j in range(n):
                if d[i] is not None and d[j] is None and A[i][j]!=0:
                    d[j]=d[i]*A[i][j]/A[j][i]; ch=True
    return [[d[i]*A[i][j] for j in range(n)] for i in range(n)], d

def ip(B,a,b): return sum(a[i]*B[i][j]*b[j] for i in range(len(a)) for j in range(len(a)))
def sphere_points(A, H):
    B,d=symmetrize(A); n=len(A); pts=[]
    for h in range(2,H+1):
        for a in itertools.product(range(h+1),repeat=n):
            if sum(a)!=h: continue
            # 2(rho|beta)=sum_i beta_i (alpha_i|alpha_i)
            if ip(B,a,a)==sum(a[i]*B[i][i] for i in range(n)): pts.append(a)
    return pts

CASES=[('A2',F(3,2),13),('A3',F(4,3),9),('D4',F(2,3),8),('B2',F(4,3),13),
       ('A2',F(1,2),13),('B2',F(1,2),13),('G2',F(1,3),13),('G2',F(2,3),13),('D4',F(1,2),8),('A3',F(1,2),9)]
p=1000003
for name,t,H in CASES:
    A=At(FIN[name],t); n=len(A); pred=predict(A,H)
    Am=[[x.numerator*pow(x.denominator,p-2,p)%p for x in r] for r in A]
    g=G(Am,[0]*n,p); g.build(H)
    mism={a:(pred[a]-g.dim.get(a,0)) for a in pred if g.dim.get(a,0)!=pred[a]}
    assert all(v>0 for v in mism.values())
    le=lambda a,b: a!=b and all(x<=y for x,y in zip(a,b))
    minimal={a:v for a,v in mism.items() if not any(le(b,a) for b in mism)}
    sp=sphere_points(A,H)
    print(name,'t=%s'%t,'H=%d'%H,'| sphere points in Q_+:',sp)
    print('   minimal extra-relation multidegrees (number of new relations):',minimal,
          ' all on sphere:',all(a in sp for a in minimal),flush=True)
