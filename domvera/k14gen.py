"""k^L(1|4), Lemma 8.3.26 and Statement 8.3.27 of the text: do X_alpha, X_beta^1, X_beta^2, X_gamma generate n_+ of k^L(1|4)?
Height-by-height ranks per root space vs multiplicities, all 8 B2 bases.
Run: python3 k14gen.py"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, itertools
import kL; kL.SIGN=-1; kL.PAIRS=[(0,1),(1,0),(2,3),(3,2)]
from kL import br, m, add, sm
from fractions import Fraction as F
from exactlin import rank, rref
N=4; X1,E1,X2,E2=0,1,2,3
B=lambda f,g: br(f,g,N)
def root(a,S): return (2*a-2+len(S),(E1 in S)-(X1 in S),(E2 in S)-(X2 in S))
def rootspace(r):
    L=[]
    for k in range(5):
        for S in itertools.combinations(range(4),k):
            a=F(r[0]+2-k,2)
            if a.denominator==1 and root(int(a),S)==r: L.append(m(int(a),*S))
    return L
LONG=[(1,1),(1,-1),(-1,1),(-1,-1)]; SHORT=[(1,0),(-1,0),(0,1),(0,-1)]
bases=[(a0,b0,(-(a0[0]+2*b0[0]),-(a0[1]+2*b0[1]))) for b0 in SHORT for a0 in LONG
       if (-(a0[0]+2*b0[0]),-(a0[1]+2*b0[1])) in LONG]
def triple(base,n,k,sgn=1):
    a0,b0,g0=base; Nn=2-n-2*k
    return [tuple(sgn*x for x in (2*n,)+a0),tuple(sgn*x for x in (2*k-1,)+b0),tuple(sgn*x for x in (2*Nn,)+g0)]
def span_basis(vecs):
    keys=sorted({k for v in vecs for k in v}); 
    if not keys: return []
    R=rref([[v.get(k,0) for k in keys] for v in vecs])
    rows=R[0] if isinstance(R,tuple) else R
    out=[]
    for row in rows:
        d={k:c for k,c in zip(keys,row) if c!=0}
        if d: out.append(d)
    return out
def test(base,n,k,sgn,maxh,gens=None):
    tri=triple(base,n,k,sgn)
    G={}  # coefficient tuple -> basis list
    gl=gens if gens is not None else [rootspace(r) for r in tri]
    for i,r in enumerate(tri):
        c=[0,0,0]; c[i]=1; G[tuple(c)]=gl[i]
    level={c:v for c,v in G.items()}
    bad=[]
    for h in range(2,maxh+1):
        new={}
        for c1,v1 in G.items():
            if sum(c1)!=1: continue
            for c2,v2 in level.items():
                c=tuple(a+b for a,b in zip(c1,c2))
                for x in v1:
                    for y in v2:
                        z=B(x,y)
                        if z: new.setdefault(c,[]).append(z)
        level={}
        for c in set(new)|set():
            level[c]=span_basis(new[c])
        # compare with multiplicity
        for c in [c for c in itertools.product(range(h+1),repeat=3) if sum(c)==h]:
            r=tuple(sum(ci*t[j] for ci,t in zip(c,tri)) for j in range(3))
            mult=len(rootspace(r)) if abs(r[1])<=1 and abs(r[2])<=1 and r!=(0,0,0) else 0
            got=len(level.get(c,[]))
            if got!=mult: bad.append((h,c,r,got,mult))
    return bad
if __name__=="__main__":
    maxh=int(sys.argv[1]) if len(sys.argv)>1 else 6
    tot=0; nb=0
    for base in bases:
        for n in range(-2,3):
            for k in range(-2,3):
                for sgn in (1,-1):
                    b=test(base,n,k,sgn,maxh); tot+=1
                    if b: nb+=1; print("NOT GEN",base,n,k,sgn,b[:3])
    print(f"{tot} systems, height <= {maxh}: {nb} fail")
