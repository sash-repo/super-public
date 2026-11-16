import sys, itertools, pickle
from fractions import Fraction as F
from symz import census, symmetrizer, connected, det

def roots(A):
    n=len(A); start=[tuple(1 if k==j else 0 for k in range(n)) for j in range(n)]
    seen=set(start); st=list(start)
    while st:
        x=st.pop()
        for i in range(n):
            c=sum(A[i][j]*x[j] for j in range(n))
            y=tuple(x[k]-(c if k==i else 0) for k in range(n))
            if y not in seen:
                seen.add(y); st.append(y)
                if len(seen)>5000: return None
    return seen

def norm(A,d,x):
    n=len(A)
    # B_ij = A_ij/d_i
    return sum(x[i]*x[j]*F(A[i][j])/d[i] for i in range(n) for j in range(n))

def typ(A):
    n=len(A); R=roots(A)
    if R is None: return 'infinite'
    d=symmetrizer(A)
    ns=sorted(set(norm(A,d,x) for x in R))
    cnt=[sum(1 for x in R if norm(A,d,x)==v) for v in ns]
    N=len(R)
    red = all(tuple(2*c for c in x) not in R for x in R)
    if not red: return f'BC{n}' if N==2*n*n+2*n else f'nonreduced{n}:{N}'
    if len(ns)==1:
        tab={n*(n+1):f'A{n}',2*n*(n-1):f'D{n}',72:'E6',126:'E7',240:'E8'}
        if n==6 and N==72: return 'E6'
        if n==7 and N==126: return 'E7'
        if n==8 and N==240: return 'E8'
        if N==n*(n+1): return f'A{n}'
        if N==2*n*(n-1): return f'D{n}'
        return f'?{N}'
    sh=cnt[0]; r=ns[-1]/ns[0]
    if r==3: return 'G2'
    if N==48 and n==4 and sh==24: return 'F4'
    if n==2: return 'B2'
    if sh==2*n: return f'B{n}'
    if sh==2*n*(n-1): return f'C{n}'
    return f'?{N},{cnt}'

if __name__=='__main__':
    nmax=int(sys.argv[1])
    res=census(nmax)
    pickle.dump(res,open(f'coh{nmax}.pkl','wb'))
    for n in range(2,nmax+1):
        tally={}; simplylaced=0
        for t in res[n]:
            A=[list(t[i*n:(i+1)*n]) for i in range(n)]
            if not connected(A): continue
            ty=typ(A); tally[ty]=tally.get(ty,0)+1
        print(n, sorted(tally.items()))
