"""Subsecs. 8.3.4-8.3.12 of the text: roots, systems of simple roots and odd reflections of
k^L(1|2) and k^L(1|3) (formulas (28)-(35), Lemmas 8.3.6, 8.3.11, Corollaries 8.3.7, 8.3.12). Uses kL.py."""
import kL
from kL import m, br, add, sm
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank
import sys
cnt=0; bad=[]
def ok(c,msg):
    global cnt; cnt+=1
    if not c: bad.append(msg)
for s in (1,-1):
    kL.SIGN=s
    if br(m(2,1),m(0,0),2)==add(m(1,0,1,c=1),m(2,c=-1)): break   # n=1,k=0: (2)xi eta t - t^2
print("SIGN",kL.SIGN)
def root(f,N):
    for (a,S),c in f.items(): break
    g=f
    e=br(m(1),g,N); d=br(m(0,0,1),g,N)
    le=e.get((a,S),0)/c; ld=d.get((a,S),0)/c
    assert add(e,sm(-le,g))=={} and add(d,sm(-ld,g))=={}, (f,e,d)
    return (le,ld)
# --- the root table (28), k^L(1|2)
for k in range(-5,6):
    ok(root(m(k+1),2)==(2*k,0),"t^{k+1}")
    ok(root(m(k,0),2)==(2*k-1,-1),"xi t^k")
    ok(root(m(k,1),2)==(2*k-1,1),"eta t^k")
    ok(root(m(k,0,1),2)==(2*k,0),"xi eta t^k")
# --- two brackets
for n in range(-4,5):
    for k in range(-4,5):
        ok(br(m(n+1,1),m(k,0),2)==add(m(n+k,0,1,c=n+1-k),m(n+k+1,c=-1)),"br1 %d %d"%(n,k))
        ok(br(m(n,1),m(k+1,0),2)==add(m(n+k,0,1,c=n-1-k),m(n+k+1,c=-1)),"br2")
# Lemma 1),2): ad(xi eta t) iso g_{+-d+(2k-1)e} -> g_{+-d+(2k+1)e}
for k in range(-6,7):
    for i in (0,1):
        ok(br(m(1,0,1),m(k,i),2)!={} and root(br(m(1,0,1),m(k,i),2),2)==root(m(k+1,i),2),"L1")
# Lemma 3: span = g_{2(n+k)e}
def vec(f,keys): return [f.get(kk,0) for kk in keys]
for n in range(-5,6):
    for k in range(-5,6):
        keys=[(n+k+1,()),(n+k,(0,1))]
        M=[vec(br(m(n+1,1),m(k,0),2),keys),vec(br(m(n,1),m(k+1,0),2),keys)]
        ok(rank(M)==2,"L3 k^L(1|2)")
# b_+ simple roots (29) alpha=d+(2n+1)e (eta t^{n+1}), beta=-d-(2n-1)e (xi t^{1-n})
for n in range(-4,5):
    a=m(n+1,1); b=m(1-n,0)
    ok(root(a,2)==(2*n+1,1) and root(b,2)==(-(2*n-1),-1),"simple roots")
    c=br(a,b,2); ok(c=={**({(2,()):F(-1)}), **({(1,(0,1)):F(2*n)} if n else {})},"[Xa,Xb]")
    # X_gamma = xi eta t: nonzero brackets with Xa, Xb, independent of [Xa,Xb]
    g=m(1,0,1)
    ok(br(g,a,2)!={} and br(g,b,2)!={},"Xg")
    ok(rank([vec(c,[(2,()),(1,(0,1))]),vec(g,[(2,()),(1,(0,1))])])==2,"Xg indep")
# formula (30): s_alpha(b_+) = b_+ with n -> n+1 : simple roots {-alpha, alpha+2e}
# the shear delta -> delta+2e, e -> e maps roots to roots (multiplicities preserved by table)
# k^L(1|3), formulas (33)-(35)
for k in range(-5,6):
    ok(root(m(k,2),3)==(2*k-1,0),"theta t^k")
    ok(root(m(k,0,2),3)==(2*k,-1),"xi theta")
    ok(root(m(k,1,2),3)==(2*k,1),"eta theta")
    ok(root(m(k,0,1,2),3)==(2*k+1,0),"xi eta theta")
# root vectors, alpha=d+ne, beta=-d+(1-n)e
for k in range(-4,5):
    n=2*k
    ok(root(m(k,1,2),3)==(n,1),"Xa n even = eta theta t^k")
    ok(root(m(k+1,1,2),3)!=(n,1),"printed Xa eta theta t^{k+1} wrong")
    ok(root(m(1-k,0),3)==(1-n,-1),"Xb n even")
    ok(root(m(-k,0,2),3)==(-n,-1),"X-a n even")
    ok(root(m(k,1),3)==(n-1,1),"X-b n even")
    n=2*k+1
    ok(root(m(k+1,1),3)==(n,1),"Xa n odd")
    ok(root(m(-k,0),3)==(-n,-1),"X-a n odd = xi t^{-k}")
    ok(k==0 or root(m(k,0),3)!=(-n,-1),"printed X-a xi t^k wrong unless k=0")
    ok(root(m(-k,0,2),3)==(1-n,-1),"Xb n odd")
    ok(root(m(k,1,2),3)==(n-1,1),"X-b n odd")
# Y1=theta t, Y2=xi eta theta ; brackets
Y1=m(1,2); Y2=m(0,0,1,2)
for k in range(-4,5):
    r=br(Y1,m(k,1,2),3); print_=r
    PRINT1=(r==m(k+1,1))
    r2=br(Y2,m(k,1,2),3)
    r3=br(Y1,m(k+1,1),3); r4=br(Y2,m(k+1,1),3)
    if k in (0,1,2): print("k",k,"[Y1,eta theta t^k]=",r,"[Y2,.]=",r2,"[Y1,eta t^{k+1}]=",r3,"[Y2,eta t^{k+1}]=",r4)
# Lemma 8.3.11: 1) [g_e, g_{+-d+ne}] = g_{+-d+(n+1)e}; 2) [g_{-e}, g_{+-d+ne}] = g_{+-d+(n-1)e}
def oddvec(n,sgn):   # basis of g_{sgn d + n e} in k^L(1|3)
    i = 1 if sgn>0 else 0
    return m((n+1)//2,i) if n%2 else m(n//2,i,2)
ge=[m(1,2),m(0,0,1,2)]; gme=[m(0,2),m(-1,0,1,2)]
for n in range(-7,8):
    for sg in (1,-1):
        x=oddvec(n,sg)
        ok(any(br(y,x,3)!={} for y in ge),"L5.1 %d %d"%(n,sg))
        ok(any(br(y,x,3)!={} for y in gme),"L5.2 %d %d"%(n,sg))
# Lemma 8.3.11, item 3)
fails=[]
for n in range(-6,7):
    for mm in range(-6,7):
        s=m+0 if False else None
        tot=n+mm+1
        keys=[((tot+1)//2,(2,)),((tot-1)//2,(0,1,2))] if tot%2 else [(tot//2+1,()),(tot//2,(0,1))]
        vs=[br(oddvec(n+1,1),oddvec(mm,-1),3),br(oddvec(n,1),oddvec(mm+1,-1),3)]
        for v in vs: assert set(v)<=set(keys),(n,mm,v,keys)
        r=rank([vec(v,keys) for v in vs])
        if r<2: fails.append((n,mm,r))
print("L5.3 rank<2 at (n,m):",fails)
print("checks",cnt,"failed",len(bad)); print(bad)
# k^L(1|3): [X_a,X_b] and generation of n_+ by X_a, X_b, X_g
print("--- k^L(1|3) [Xa,Xb]")
for n in range(-4,5):
    k=n//2
    Xa = m(k,1,2) if n%2==0 else m(k+1,1)
    Xb = m(1-k,0) if n%2==0 else m(-k,0,2)
    assert root(Xa,3)==(n,1) and root(Xb,3)==(1-n,-1)
    print(n, br(Xa,Xb,3))
