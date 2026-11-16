"""Roots of vect(0|n) wrt h=span(xi_i d_i), multiplicities with parity, check-Delta (Subsec. 8.2 of the text)."""
from itertools import combinations
import sys
fails=0
for n in range(2,7):
    mult={}
    for k in range(n+1):
        for S in combinations(range(n),k):
            for i in range(n):
                w=[0]*n
                for s in S: w[s]+=1
                w[i]-=1
                w=tuple(w); p=(k+1)%2
                if any(w):
                    e,o=mult.get(w,(0,0)); mult[w]=(e+(p==0),o+(p==1))
    R=set(mult)
    chk={a for a in R if tuple(-x for x in a) not in R}
    # predicted check-Delta: eps_S - eps_i (|S|>=2, i not in S) and eps_T (|T|>=2)
    pred=set()
    for w in R:
        pos=sum(x for x in w if x>0); neg=[x for x in w if x<0]
        if (neg and pos>=2) or (not neg and pos>=2): pred.add(w)
    ok1= chk==pred
    # predicted multiplicities
    ok2=True
    for w,(e,o) in mult.items():
        neg=[x for x in w if x<0]; T=sum(w) if not neg else None
        if not neg:
            t=sum(w); exp=((n-t),0) if t%2==0 else (0,n-t)
        else:
            s=sum(x for x in w if x>0); exp=(1,0) if s%2==1 else (0,1)
        ok2&= (e,o)==exp
    # distinguished Borel: gl-Borel + degree>=1
    deg=lambda w: sum(w)
    P={w for w in R if deg(w)>=1 or (deg(w)==0 and w.index(1)<w.index(-1))}
    ok3 = all((a in P)!=(tuple(-x for x in a) in P) for a in R if tuple(-x for x in a) in R) and chk<=P
    closed=all(tuple(x+y for x,y in zip(a,b)) in P for a in P for b in P if tuple(x+y for x,y in zip(a,b)) in R)
    print(n,'roots',len(R),'checkDelta',len(chk),ok1,ok2,ok3,closed)
    fails+= not(ok1 and ok2 and ok3 and closed)
sys.exit(fails)
