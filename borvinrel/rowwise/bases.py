# Statement 4.1.4 of the paper: Gram matrices of all bases reachable by odd reflections; which satisfy
#  (A) condition B2) of Sect. 4: all off-diagonal (sigma_i,sigma_j) <=0 (one global sign),
#  (R) relaxed: the same only in rows with (sigma_i,sigma_i)!=0, gray rows free,
#  plus b) integrality on rows with positive norm.
from fractions import Fraction as F
from itertools import product
def refl(G,p,k):
    n=len(G); E=[[int(i==j) for j in range(n)] for i in range(n)]
    new=[]
    for j in range(n):
        if j==k: new.append([-x for x in E[k]])
        elif G[k][j]!=0: new.append([E[j][i]+E[k][i] for i in range(n)])
        else: new.append(E[j])
    g=lambda u,v: sum(u[i]*G[i][j]*v[j] for i in range(n) for j in range(n))
    G2=tuple(tuple(g(new[i],new[j]) for j in range(n)) for i in range(n))
    p2=tuple(sum(p[i]*new[j][i] for i in range(n))%2 for j in range(n))
    return G2,p2
def orbit(G,p,cap=2000):
    G=tuple(map(tuple,G)); p=tuple(p); seen={(G,p)}; todo=[(G,p)]
    while todo and len(seen)<cap:
        G,p=todo.pop()
        for k in range(len(G)):
            if p[k]==1 and G[k][k]==0:
                x=refl(G,p,k)
                if x not in seen: seen.add(x); todo.append(x)
    return seen
def good(G,p,relaxed):
    n=len(G)
    for s in (1,-1):
        B=[[s*G[i][j] for j in range(n)] for i in range(n)]
        ok=all(B[i][j]<=0 for i in range(n) for j in range(n) if i!=j and not(relaxed and B[i][i]==0))
        for i in range(n):
            if B[i][i]>0:
                for j in range(n):
                    r=F(2*B[i][j],B[i][i])
                    if r.denominator!=1 or (p[i]==1 and r.numerator%2): ok=False
        if ok: return True
    return False
def gl(seq):  # seq of 'e'/'d'; (e,e)=1,(d,d)=-1
    v=lambda c:1 if c=='e' else -1
    N=len(seq); n=N-1; G=[[0]*n for _ in range(n)]
    for i in range(n):
        G[i][i]=v(seq[i])+v(seq[i+1])
        if i+1<n: G[i][i+1]=G[i+1][i]=-v(seq[i+1])
    p=[int(seq[i]!=seq[i+1]) for i in range(n)]
    return G,p
def osp(seq,last):  # last: 'B' (short root x_N) or 'D' (x_{N-1}+x_N) or 'C' (2x_N)
    v=lambda c:1 if c=='e' else -1
    N=len(seq); vec=[]
    for i in range(N-1):
        u=[0]*N; u[i]=1; u[i+1]=-1; vec.append(u)
    u=[0]*N
    if last=='B': u[N-1]=1
    elif last=='D': u[N-2]=1;u[N-1]=1
    else: u[N-1]=2
    vec.append(u)
    g=lambda a,b: sum(a[i]*b[i]*v(seq[i]) for i in range(N))
    G=[[g(a,b) for b in vec] for a in vec]
    par=lambda u: sum(abs(u[i]) for i in range(N) if seq[i]=='d')%2
    return G,[par(u) for u in vec]
cases={
 'sl(2|2)':gl('eedd'),'sl(2|3)':gl('eeddd'),'sl(3|3)':gl('eeeddd'),'sl(3|4)':gl('eeedddd'),
 'osp(3|2)':osp('ed','B'),'osp(3|4)':osp('edd','B'),'osp(5|4)':osp('eedd','B'),
 'osp(4|2)':osp('dee','D'),'osp(4|4)':osp('ddee','D'),'osp(6|4)':osp('ddeee','D'),
 'osp(2|4)':osp('edd','C'),'osp(4|4)C':osp('eedd','C'),
 'G(3)':([[0,-1,0],[-1,2,-3],[0,-3,6]],[1,0,0]),
 'F(4)':([[0,-1,0,0],[-1,2,-2,0],[0,-2,4,-2],[0,0,-2,4]],[1,0,0,0]),
 'Prop4.1.5':([[2,-1,0],[-1,2,-2],[0,-2,0]],[0,0,1]),
}
def ospa(a):  # osp_a(4|2): sigma_1 odd isotropic, sigma_2, sigma_3 even (the form scaled by -1/2)
    a=F(a); return ([[F(0),F(1),a],[F(1),F(-2),F(0)],[a,F(0),-2*a]],[1,0,0])
cases.update({'sl(2|4)':gl('eedddd'),'sl(2|5)':gl('eeddddd')})
for a in ('2','-3','1/2','-1/3','7/5'):
    cases[f'osp_{a}(4|2)']=ospa(a)

for name,(G,p) in cases.items():
    O=orbit(G,p)
    a=sum(good(G2,p2,False) for G2,p2 in O); r=sum(good(G2,p2,True) for G2,p2 in O)
    print(f"{name:10s} bases(Gram,parity) {len(O):4d}   Def a)+b) holds in {a:4d}   relaxed in {r:4d}")
