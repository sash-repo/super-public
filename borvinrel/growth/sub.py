import sys
from gA import build
from orbit import orbit
from fam import Bk
def comps(B,nodes):
    nodes=set(nodes); out=[]
    while nodes:
        s=[nodes.pop()]; c=set(s)
        while s:
            x=s.pop()
            for y in list(nodes):
                if B[x][y]!=0: nodes.discard(y); c.add(y); s.append(y)
        out.append(sorted(c))
    return out
def growth(B,par,H):
    d,l=build(B,par,H)
    tot=[sum(d[b] for b in l[h]) for h in sorted(l)]
    mx=max([d[b] for h in l for b in l[h]] or [0])
    return tot,mx
k=int(sys.argv[1]); H=int(sys.argv[2]); allorb=len(sys.argv)>3
B0,p0=Bk(k)
mats=list(orbit(B0,p0).values()) if allorb else [(B0,p0)]
for B,par in mats:
    n=len(B)
    print("par",par,"diag",[B[i][i] for i in range(n)])
    for r in range(n):
        for c in comps(B,[i for i in range(n) if i!=r]):
            Bs=[[B[i][j] for j in c] for i in c]; ps=[par[i] for i in c]
            tot,mx=growth(Bs,ps,H)
            kind="FINITE" if tot[-1]==0 and tot[-2]==0 else ("bounded mult %d"%mx)
            print("  remove",r,"comp",c,kind,"tail",tot[-6:],flush=True)
