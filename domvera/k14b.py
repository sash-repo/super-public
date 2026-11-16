"""k^L(1|4), Subsecs. 8.3.25-8.3.29 of the text: the triples alpha, beta, gamma of formula (51),
the brackets of Lemma 8.3.26, the generation of n_+, Statement 8.3.27 (the systems of simple roots)
and its converse. Run: python3 k14b.py"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, itertools, random
import kL; kL.SIGN=-1; kL.PAIRS=[(0,1),(1,0),(2,3),(3,2)]
from kL import br, m, add, sm
from fractions import Fraction as F
from exactlin import rank
N=4; X1,E1,X2,E2=0,1,2,3
B=lambda f,g: br(f,g,N)
cnt=[0,0]
def ok(c,msg):
    cnt[0]+=1
    if not c: cnt[1]+=1; print("FAIL",msg)
def eq(a,b,msg): ok(add(a,sm(-1,b))=={},f"{msg}: {a} vs {b}")
def root(a,S):
    return (2*a-2+len(S),(E1 in S)-(X1 in S),(E2 in S)-(X2 in S))
def rootspace(r):
    e,d1,d2=r; L=[]
    for k in range(5):
        for S in itertools.combinations(range(4),k):
            a=F(e+2-k,2)
            if a.denominator==1 and root(int(a),S)==r: L.append((int(a),S))
    return L
# ---- the brackets of Lemma 8.3.26, for all n,k with N=2-n-2k
for n in range(-4,5):
  for k in range(-4,5):
    Nn=2-n-2*k
    Xa,Xma=m(n,X1,E2),m(-n,E1,X2)
    Xb1,Xb2=m(k,E1),m(k-1,E1,X2,E2)
    Xmb1,Xmb2=m(1-k,X1),m(-k,X1,X2,E2)
    Xg=m(Nn,X1,X2)
    ok(root(n,(X1,E2))==(2*n,-1,1),"alpha"); ok(root(k,(E1,))==(2*k-1,1,0),"beta1"); ok(root(k-1,(E1,X2,E2))==(2*k-1,1,0),"beta2")
    ok(root(1-k,(X1,))==(1-2*k,-1,0) and root(-k,(X1,X2,E2))==(1-2*k,-1,0),"-beta")
    ok(root(Nn,(X1,X2))==(2*Nn,-1,-1),"gamma")
    eq(B(Xa,Xb1),add(sm(n,m(n+k-1,X1,E1,E2)),m(n+k,E2)),"[Xa,Xb1]")
    eq(B(Xa,Xb2),m(n+k-1,X1,E1,E2),"[Xa,Xb2]")
    Y=B(B(Xa,Xb2),Xb1); eq(Y,sm(-1,m(n+2*k-1,E1,E2)),"Y")
    eq(B(Xg,Y),sm(-1,add(m(1,X1,E1),m(1,X2,E2))),"[Xg,Y]")
    eq(B(Xg,Xb1),add(sm(Nn,m(Nn+k-1,X1,E1,X2)),m(Nn+k,X2)),"[Xg,Xb1]")
    eq(B(Xg,Xb2),sm(-1,m(Nn+k-1,X1,E1,X2)),"[Xg,Xb2]")
    A1,A2=m(n+k-1,X1,E1,E2),m(n+k,E2); G1,G2=m(Nn+k-1,X1,E1,X2),m(Nn+k,X2)
    eq(B(A2,G2),add(sm(-1,m(2)),sm(n-Nn,m(1,X2,E2))),"[eta2 t^{n+k}, xi2 t^{N+k}]")
    eq(B(A2,G1),add(sm(-1,m(1,X1,E1)),sm(-1,m(0,X1,E1,X2,E2))),"[A2,G1]")
    eq(B(A1,G2),add(sm(-1,m(1,X1,E1)),m(0,X1,E1,X2,E2)),"[A1,G2]")
    eq(B(A1,G1),{},"[A1,G1]=0")
print(f"brackets: {cnt[0]} checks, {cnt[1]} failures")

# ---- B2 bases (alpha0 long, beta0 short), gamma0 = -(alpha0+2beta0)
short=[(1,0),(-1,0),(0,1),(0,-1)]
bases=[]
for b0 in short:
    for a0 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        g0=(-(a0[0]+2*b0[0]),-(a0[1]+2*b0[1]))
        if g0 in [(1,1),(1,-1),(-1,1),(-1,-1)]: bases.append((a0,b0,g0))
print("B2 bases with alpha0 long, beta0 short:",len(bases))
ALL=[(e,d1,d2) for e in range(-60,61) for d1 in (-1,0,1) for d2 in (-1,0,1)
     if (e-abs(d1)-abs(d2))%2==0 and (e,d1,d2)!=(0,0,0)]
def coeffs(r,tri):
    import fractions
    M=[[F(x) for x in t] for t in tri]  # rows alpha,beta,gamma in (e,d1,d2)
    # solve p*a+q*b+s*g=r
    import itertools
    a,b,c=M
    det=lambda u,v,w: u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
    D=det(a,b,c); r=[F(x) for x in r]
    return det(r,b,c)/D, det(a,r,c)/D, det(a,b,r)/D
def triple(base,n,k):
    (a0,b0,g0)=base; Nn=2-n-2*k
    return ((2*n,)+a0,(2*k-1,)+b0,(2*Nn,)+g0)
for base in bases:
  for n in range(-3,4):
    for k in range(-3,4):
      tri=triple(base,n,k)
      ok(tuple(sum(x) for x in zip(tri[0],tri[1],tri[1],tri[2]))==(2,0,0),"a+2b+g=2e")
      good=True
      for r in ALL:
        if abs(r[0])>40: continue
        c=coeffs(r,tri)
        if not all(x.denominator==1 for x in c) or not (all(x>=0 for x in c) or all(x<=0 for x in c)): good=False;break
      ok(good,f"base property {base} n={n} k={k}")
print(f"base property: {cnt[0]} checks, {cnt[1]} failures")

# ---- converse: random generic Borels (eps>0)
random.seed(3)
for trial in range(1500):
    x,y=random.uniform(-9,9),random.uniform(-9,9)
    ell=lambda r: r[0]+r[1]*x+r[2]*y
    pos=[r for r in ALL if ell(r)>0 and ell(r)<45]
    S=set(pos)
    simple=[r for r in pos if not any(tuple(a-b for a,b in zip(r,p)) in S for p in pos if ell(p)<ell(r))]
    found=False
    for base in bases:
        for r in simple:
            if r[1:]==base[0]:
                n=r[0]//2
                for s in simple:
                    if s[1:]==base[1]:
                        k=(s[0]+1)//2; tri=triple(base,n,k)
                        if sorted(tri)==sorted(simple): found=True
    ok(len(simple)==3 and found,f"converse x={x} y={y} simple={simple}")
print(f"converse: {cnt[0]} checks, {cnt[1]} failures")
