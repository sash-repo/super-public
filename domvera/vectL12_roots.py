"""Subsecs. 8.3.19-8.3.21 of the text: systems of simple roots of vect^L(1|2). Roots mu+k eps, mu in
D={+-d1,+-d2,+-(d1-d2),d1+d2} or 0. Checks the six bases of table (46) and Statement 8.3.21:
the indecomposable positive roots of every standard Borel are {a1+n1 e, a2+n2 e, l+n e},
with n+b1 n1+b2 n2 = +1 (eps positive) or -1."""
import random, itertools
from fractions import Fraction as F
D=[(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(1,1)]
mult={(0,0):3,(1,0):2,(-1,0):1,(0,1):2,(0,-1):1,(1,1):1,(1,-1):1,(-1,1):1}
# multiplicities of k eps + mu from the basis t^a xi^S d
cnt=dict()
for a in range(-3,4):
    for S in [(),(1,),(2,),(1,2)]:
        dS=(int(1 in S),int(2 in S))
        r=((a-1),dS); cnt[r]=cnt.get(r,0)+1
        for i in (1,2):
            ei=(int(i==1),int(i==2)); r=(a,(dS[0]-ei[0],dS[1]-ei[1])); cnt[r]=cnt.get(r,0)+1
ok_mult=all(cnt.get((0,mu),0)==mult[mu] for mu in mult)
TABLE=[(((1,0),(-1,1)),(0,-1),(1,1)),(((1,1),(-1,0)),(1,-1),(1,2)),
       (((-1,1),(0,-1)),(1,1),(1,2)),(((-1,0),(1,-1)),(1,1),(2,1)),
       (((0,-1),(1,1)),(-1,1),(2,1)),(((1,-1),(0,1)),(-1,0),(1,1))]
fails=0; checks=0
def coords(mu,a1,a2):
    det=a1[0]*a2[1]-a1[1]*a2[0]
    return (F(mu[0]*a2[1]-mu[1]*a2[0],det),F(a1[0]*mu[1]-a1[1]*mu[0],det))
for (a1,a2),lam,b in TABLE:   # each row: SSR of D, lowest root, b
    cs={mu:coords(mu,a1,a2) for mu in D}
    checks+=1
    good=all(c[0].denominator==1 and c[1].denominator==1 and (c[0]>=0)==(c[1]>=0) or (c[0]<=0 and c[1]<=0) for c in cs.values())
    pos=[mu for mu in D if cs[mu][0]>=0 and cs[mu][1]>=0]
    indec=[mu for mu in pos if not any(tuple(x+y for x,y in zip(p,q))==mu for p in pos for q in pos)]
    good=good and sorted(indec)==sorted([a1,a2]) and cs[lam]==(-b[0],-b[1]) and all(c[0]>=-b[0] and c[1]>=-b[1] for c in cs.values())
    if not good: fails+=1; print("table row fails",a1,a2,lam,b)
random.seed(1); seen=set()
for trial in range(4000):
    phi=(random.uniform(-3,3),random.uniform(-3,3)); s=random.choice((1,-1))
    f=lambda mu,k: s*k+phi[0]*mu[0]+phi[1]*mu[1]
    roots=[(mu,k) for mu in D+[(0,0)] for k in range(-40,41) if (mu,k)!=((0,0),0)]
    P=[r for r in roots if f(*r)>0]; Pset=set(P)
    small=[r for r in P if abs(r[1])<=12]
    ind=[r for r in small if not any(((r[0][0]-q[0][0],r[0][1]-q[0][1]),r[1]-q[1]) in Pset for q in P if q!=r and ((r[0][0]-q[0][0],r[0][1]-q[0][1]),r[1]-q[1]) in Pset)]
    checks+=1
    # match with a table row
    hit=False
    for (a1,a2),lam,b in TABLE:
        dp=sorted(r[0] for r in ind)
        if dp==sorted([a1,a2,lam]) and len(ind)==3:
            n={r[0]:r[1] for r in ind}
            hit = n[lam]+b[0]*n[a1]+b[1]*n[a2]==s; seen.add((a1,a2))
            break
    if not hit: fails+=1; print("Borel fails",phi,s,ind)
print("multiplicities",ok_mult,"; bases realized",len(seen),"of",len(TABLE),";",checks,"checks,",fails,"failures")
print("realized:",sorted(seen))
