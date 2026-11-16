"""Check: main 2-partitions -> super partitions a_j=(i_j+j)/2; weights per q; and Gauss identities."""
from fractions import Fraction as F
from itertools import combinations
def main2(q):
    out=[]
    top=4+3*(q-1)
    for I in combinations(range(2,top+1),q):
        if any(I[r+1]-I[r]<3 for r in range(q-1)): continue
        if I[0]==2 and I[-1]>=top: continue
        out.append(I)
    return out
for q in range(1,7):
    S=[tuple(F(i+j+1,2) for j,i in enumerate(I)) for I in main2(q)]
    ws=sorted(sum(a) for a in S)
    exp=[F(q*(2*q+1),2)+F(j,2) for j in range(2*q+1)]
    print(q,len(S),[str(w) for w in ws]==[str(w) for w in exp], [ ' '.join(('L' if a.denominator==1 else 'G')+str(a) for a in s) for s in S][:5])
# Gauss: NS  prod(1-x^{2n})/prod(1+x^{2n-1}) = sum (-1)^n x^{n(2n+1)}, x=t^{1/2}
N=80
def mul(a,b): 
    c=[0]*N
    for i,x in enumerate(a):
        if x:
            for j,y in enumerate(b[:N-i]): c[i+j]+=x*y
    return c
P=[1]+[0]*(N-1)
for n in range(1,N):
    if 2*n<N: f=[0]*N; f[0]=1; f[2*n]=-1; P=mul(P,f)
    if 2*n-1<N: g=[0]*N  # 1/(1+x^m) series
    if 2*n-1<N:
        m=2*n-1; g=[0]*N
        for k in range(0,N,m): g[k]=(-1)**(k//m)
        P=mul(P,g)
R=[0]*N
for n in range(-10,11):
    e=n*(2*n+1)
    if 0<=e<N: R[e]+=(-1)**n
print('NS Gauss', P==R)
P=[1]+[0]*(N-1)
for n in range(1,N):
    f=[0]*N; f[0]=1; f[n]=-1; P=mul(P,f)
    g=[0]*N
    for k in range(0,N,n): g[k]=(-1)**(k//n)
    P=mul(P,g)
R=[0]*N
for n in range(-10,11):
    if n*n<N: R[n*n]+=(-1)**n
print('Ramond Gauss', P==R)
