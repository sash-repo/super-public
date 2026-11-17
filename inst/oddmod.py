"""Odd moduli of Semikhatov's N-extended ADHM data, gauge group Sp(1)=SU(2), charge k
(Statement 7.2.1 of the text, system (9)).
Per odd index j: c in H^{(k+1)x k}, linear conditions a^+c - c^+a = 0 and b^+c + c^+b = 0
(and the three sign-swapped variants), at 't Hooft data a=(lambda; Y), b=(0;-1), k = 1..5.
Prints the real dimension of the solution space. Expected 4k.
Exact rational arithmetic (fractions.Fraction, ranks via ../lib/exactlin.py)."""
import random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction as F
from exactlin import rank

def qmul(p, q):
    a1,b1,c1,d1 = p; a2,b2,c2,d2 = q
    return (a1*a2-b1*b2-c1*c2-d1*d2, a1*b2+b1*a2+c1*d2-d1*c2,
            a1*c2-b1*d2+c1*a2+d1*b2, a1*d2+b1*c2-c1*b2+d1*a2)
def qconj(p): return (p[0],-p[1],-p[2],-p[3])
def qadd(p,q): return tuple(x+y for x,y in zip(p,q))
Z=(0,0,0,0)

def dagger(M): return [[qconj(M[i][j]) for i in range(len(M))] for j in range(len(M[0]))]
def mm(A,B):
    return [[ (lambda s: s)( __import__('functools').reduce(qadd,[qmul(A[i][l],B[l][j]) for l in range(len(B))],Z)) for j in range(len(B[0]))] for i in range(len(A))]

def count(k, s1, s2, rnd):
    lam=[ (F(rnd.randint(1,9)),0,0,0) for _ in range(k)]  # 't Hooft: real lambda_i
    Y=[[Z]*k for _ in range(k)]
    for i in range(k): Y[i][i]=tuple(F(rnd.randint(-9,9)) for _ in range(4))
    a=[lam]+Y
    b=[[Z]*k]+[[ (F(-1),0,0,0) if i==j else Z for j in range(k)] for i in range(k)]
    ad, bd = dagger(a), dagger(b)
    n=(k+1)*k*4
    rows=None; cols=[]
    for v in range(n):
        e=[0]*n; e[v]=1
        c=[[tuple(F(e[((i*k)+j)*4+t]) for t in range(4)) for j in range(k)] for i in range(k+1)]
        X=mm(ad,c); Xd=dagger(X); W=mm(bd,c); Wd=dagger(W)
        col=[]
        for i in range(k):
            for j in range(k):
                col+=[x - s1*y for x,y in zip(X[i][j],Xd[i][j])]
                col+=[x + s2*y for x,y in zip(W[i][j],Wd[i][j])]
        cols.append(col)
    M=[list(r) for r in zip(*cols)]
    return n - rank(M)

rnd=random.Random(1904)
for k in range(1,6):
    res=[count(k,s1,s2,rnd) for s1 in (1,-1) for s2 in (1,-1)]
    print(f"k={k}: real dim of c per odd index, sign variants {res}; expected 4k={4*k}")
