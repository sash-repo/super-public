"""Odd-reflection orbit of (B, par): all Cartan matrices of g(B)."""
import sys, itertools
from fractions import Fraction
from fam import Bk
def reflect(B, par, i):
    n=len(B); assert par[i]==1 and B[i][i]==0
    # new basis vectors as integer combos of old
    M=[[1 if s==t else 0 for t in range(n)] for s in range(n)]
    newpar=list(par)
    for j in range(n):
        if j==i: M[j]=[-1 if t==i else 0 for t in range(n)]
        elif B[i][j]!=0:
            M[j][i]+=1; newpar[j]=1-par[j]
    NB=[[sum(M[a][s]*B[s][t]*M[b][t] for s in range(n) for t in range(n)) for b in range(n)] for a in range(n)]
    return NB,newpar
def cartan(B,par):
    n=len(B); A=[]
    for i in range(n):
        r=[Fraction(x) for x in B[i]]
        if r[i]!=0: c=r[i]/2
        else:
            nz=[x for j,x in enumerate(r) if x!=0]
            c=-nz[0] if nz else 1  # first nonzero off-diag -> -1 (only canonical up to row order; refined below)
        A.append([x/c for x in r])
    return A
def canon(B,par):
    n=len(B)
    best=None
    # isotropic rows: scale so the row is lexicographically minimal among +-; we use: canonical up to perm by brute force
    inv=[(par[i], B[i][i]==0, tuple(sorted(abs(x) for x in B[i])), tuple(sorted((par[j],B[j][j]==0) for j in range(n) if j!=i and B[i][j]!=0))) for i in range(n)]
    classes={}
    for i in range(n): classes.setdefault(inv[i],[]).append(i)
    keys=sorted(classes)
    def perms():
        for combo in itertools.product(*[itertools.permutations(classes[c]) for c in keys]):
            yield [x for part in combo for x in part]
    for p in perms():
        Bp=[[B[p[a]][p[b]] for b in range(n)] for a in range(n)]
        pp=[par[p[a]] for a in range(n)]
        A=[]
        for i in range(n):
            r=[Fraction(x) for x in Bp[i]]
            if r[i]!=0: r=[x/(r[i]/2) for x in r]
            else:
                nz=[x for x in r if x!=0]; r=[x/(-nz[0]) for x in r]  # first nonzero entry = -1
            A.append(tuple(r))
        key=(tuple(pp),tuple(A))
        if best is None or key<best: best=key
    return best
def orbit(B,par):
    seen={}; todo=[(B,par)]
    while todo:
        B,par=todo.pop()
        c=canon(B,par)
        if c in seen: continue
        seen[c]=(B,par)
        for i in range(len(B)):
            if par[i]==1 and B[i][i]==0:
                todo.append(reflect(B,par,i))
    return seen
if __name__=="__main__":
    k=int(sys.argv[1]); B,par=Bk(k)
    S=orbit(B,par)
    print("k",k,"number of inequivalent Cartan matrices",len(S))
    for c,(B,par) in S.items():
        print("par",par); [print("  ",r) for r in B]
