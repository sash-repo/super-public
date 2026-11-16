"""Contact bracket on k^L(1|N), N=2,3: Laurent polynomials in t times Lambda(xi,eta[,theta]).
Odd labels: 0=xi, 1=eta, 2=theta. Element = {(a,S): Fraction}."""
from fractions import Fraction as F
import itertools, random, sys

def srt(S):
    S=list(S); s=1
    for i in range(len(S)):
        for j in range(len(S)-1-i):
            if S[j]==S[j+1]: return None,0
            if S[j]>S[j+1]: S[j],S[j+1]=S[j+1],S[j]; s=-s
    for i in range(len(S)-1):
        if S[i]==S[i+1]: return None,0
    return tuple(S),s
def add(*fs):
    o={}
    for f in fs:
        for k,v in f.items(): o[k]=o.get(k,0)+v
    return {k:v for k,v in o.items() if v!=0}
def sm(c,f): return {k:c*v for k,v in f.items() if c*v!=0}
def mul(f,g):
    o={}
    for (a,S),c in f.items():
        for (b,T),d in g.items():
            U,s=srt(S+T)
            if s: o[(a+b,U)]=o.get((a+b,U),0)+s*c*d
    return {k:v for k,v in o.items() if v!=0}
def dt(f): return {(a-1,S):a*c for (a,S),c in f.items() if a!=0}
def dodd(i,f):   # left derivative
    o={}
    for (a,S),c in f.items():
        if i in S:
            p=S.index(i); o[(a,S[:p]+S[p+1:])]=o.get((a,S[:p]+S[p+1:]),0)+(-1)**p*c
    return {k:v for k,v in o.items() if v!=0}
def E(f): return {(a,S):c*len(S) for (a,S),c in f.items()}
def Dl(f): return add(sm(2,f),sm(-1,E(f)))
def par(f):
    ps={len(S)%2 for (a,S) in f}
    assert len(ps)<=1, f
    return ps.pop() if ps else 0
PAIRS=[(0,1),(1,0),(2,2)]
def br(f,g,N):
    if not f or not g: return {}
    pf=par(f)
    P={}
    for (i,j) in PAIRS:
        if i>=N or j>=N: continue
        P=add(P,mul(dodd(i,f),dodd(j,g)))
    return add(mul(Dl(f),dt(g)),sm(-1,mul(dt(f),Dl(g))),sm(-SIGN*(-1)**pf*CPB,P))
SIGN=1; CPB=1
def m(a,*S,c=1):
    U,s=srt(S); return {(a,U):F(c)*s} if s else {}
def rnd(N,par_):
    f={}
    for a in range(-2,3):
        for r in range(N+1):
            if r%2!=par_: continue
            for S in itertools.combinations(range(N),r):
                f=add(f,{(a,S):F(random.randint(-3,3))})
    return f
def jac(N,trials=30):
    for _ in range(trials):
        ps=[random.randint(0,1) for _ in range(3)]
        f,g,h=[rnd(N,p) for p in ps]
        l=br(f,br(g,h,N),N); r=add(br(br(f,g,N),h,N),sm((-1)**(ps[0]*ps[1]),br(g,br(f,h,N),N)))
        if add(l,sm(-1,r)): return False
    return True
if __name__=="__main__":
    for N in (2,3):
        for s in (1,-1):
            for c in (F(1),F(1,2),F(2)):
                SIGN=s; CPB=c
                globals().update(SIGN=s,CPB=c)
                print(N,s,c,jac(N))
