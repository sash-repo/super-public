"""The r-matrices with values in sl(2|1) of Section 3, checked with sympy (exact, symbolic in s):
  * Omega is invariant and super-symmetric;
  * Proposition 3.9: for P(s), P_A, P_B, the equality P + P^21 = Omega and the CYBE with no
    spectral parameter (for P(s) symbolically in s, i.e., for all s at once); the CYBE with the
    spectral parameter and the unitarity of r(z) = P + Omega/(z-1);
  * Remark 3.6 (Karaali's r): r + r^21 = Omega and the CYBE;
  * Example 3.4: the toral parts r_0 for which r solves the CYBE, and those allowed by
    condition (8) together with r_0 + r_0^21 = toral part of Omega;
  * formula (10) of Proposition 3.10: det r(z) for the family P(s);
  * the rational identity used in the proof of Proposition 3.9.
Indices 0, 1, 2 stand for 1, 2, 3; E(i,j) is the matrix unit, U1 = E11 + E33, U2 = E22 + E33.
Run:  python3 sl21.py
"""
import sympy as sp
from itertools import product
par=[0,0,1]  # indices 1,2 even, 3 odd
B=[(i,j) for i in range(3) for j in range(3)]
def p(e): return (par[e[0]]+par[e[1]])%2
def mul(e,f):
    return {(e[0],f[1]):1} if e[1]==f[0] else {}
def br(e,f):
    out={}
    for k,v in mul(e,f).items(): out[k]=out.get(k,0)+v
    s=(-1)**(p(e)*p(f))
    for k,v in mul(f,e).items(): out[k]=out.get(k,0)-s*v
    return {k:v for k,v in out.items() if v!=0}
def E(i,j): return {(i-1,j-1):1}
def lin(*pairs):
    out={}
    for c,x in pairs:
        for k,v in x.items(): out[k]=out.get(k,0)+c*v
    return {k:v for k,v in out.items() if v!=0}
U1=lin((1,E(1,1)),(1,E(3,3))); U2=lin((1,E(2,2)),(1,E(3,3)))
def T(c,x,y):  # c x(x)y as dict over basis pairs
    out={}
    for a,u in x.items():
        for b,v in y.items(): out[(a,b)]=out.get((a,b),0)+c*u*v
    return out
def tadd(*ts):
    out={}
    for t in ts:
        for k,v in t.items(): out[k]=out.get(k,0)+v
    return {k:sp.simplify(v) for k,v in out.items() if sp.simplify(v)!=0}
def flip(r):  # super flip tau
    return {(b,a):(-1)**(p(a)*p(b))*v for (a,b),v in r.items()}
def schouten(r):
    out={}
    def acc(k,c):
        out[k]=out.get(k,0)+c
    items=list(r.items())
    for (ua,va),ca in items:
        for (ub,vb),cb in items:
            pa=p(ua); pb=p(ub); s=(-1)**(pa*pb)
            for k,v in br(ua,ub).items(): acc((k,va,vb),s*ca*cb*v)
            for k,v in br(va,ub).items(): acc((ua,k,vb),ca*cb*v)
            for k,v in br(va,vb).items(): acc((ua,ub,k),s*ca*cb*v)
    return {k:sp.simplify(v) for k,v in out.items() if sp.simplify(v)!=0}
Om=tadd(T(-1,U1,U2),T(-1,U2,U1),T(1,E(1,2),E(2,1)),T(1,E(2,1),E(1,2)),
        T(-1,E(1,3),E(3,1)),T(1,E(3,1),E(1,3)),T(-1,E(2,3),E(3,2)),T(1,E(3,2),E(2,3)))
# invariance of Omega: [x(x)1+1(x)x, Om]=0 with Koszul signs
def act(x,r):
    out={}
    for (a,b),c in r.items():
        for k,v in br(x,a).items(): out[(k,b)]=out.get((k,b),0)+c*v
        s=(-1)**(p(x)*p(a))
        for k,v in br(x,b).items(): out[(a,k)]=out.get((a,k),0)+s*c*v
    return {k:v for k,v in out.items() if v!=0}
print("Omega invariant:",all(not act(e,Om) for e in B))
print("Omega supersymmetric tau(Om)=Om:",tadd(flip(Om),{k:-v for k,v in Om.items()})=={})
s=sp.Symbol('s')
D=tadd(T(1,E(1,2),E(2,1)),T(-1,E(1,3),E(3,1)),T(1,E(3,2),E(2,3)))
Ps=tadd(T(-(s+1),U1,U2),T(s,U2,U1),D)
PA=tadd(T(-1,U2,U1),D,T(1,E(1,3),E(2,3)),T(1,E(2,3),E(1,3)))
PB=tadd(T(-1,U1,U2),D,T(1,E(3,1),E(3,2)),T(1,E(3,2),E(3,1)))
for name,P in [("P(s)",Ps),("P_A",PA),("P_B",PB)]:
    print(name,"P+tau(P)=Omega:",tadd(P,flip(P),{k:-v for k,v in Om.items()})=={},"CYBE:",schouten(P)=={})
K=tadd(T(-1,U2,U1),T(1,E(1,2),E(2,1)),T(-1,E(1,3),E(3,1)),T(1,E(3,2),E(2,3)),T(-1,E(1,3),E(1,3)),T(1,E(2,3),E(2,3)))
print("Remark 3.6, Karaali's r: r+tau r=Om:",tadd(K,flip(K),{k:-v for k,v in Om.items()})=={},"CYBE:",schouten(K)=={})
# Example 3.4: r0 unknown in t(x)t
a,b,c,d=sp.symbols('a b c d')
r0=tadd(T(a,U1,U1),T(b,U1,U2),T(c,U2,U1),T(d,U2,U2))
R=tadd(r0,T(1,E(2,1),E(1,2)),T(-1,E(2,3),E(3,2)),T(1,E(3,1),E(1,3)),T(1,E(3,1),E(3,2)),T(1,E(3,2),E(3,1)))
eqs=[]
for k,v in tadd(R,flip(R),{k:-v for k,v in Om.items()}).items(): eqs.append(v)
for k,v in schouten(R).items(): eqs.append(v)
print("Example 3.4, r0 solving the CYBE:",sp.solve(eqs,[a,b,c,d],dict=True))
# condition (8) for Example 3.4: alpha=eps1-lam1, tau(alpha)=lam1-eps2 ; weights on U1,U2
def wt(e1,e2,l1): return {'U1':e1*1+l1*1, 'U2':e2*1+l1*1}  # eps1(U1)=1,eps2(U2)=1,lam1(U_i)=1
alpha={'U1':1-1,'U2':0-1}; talpha={'U1':1-0,'U2':1-1}
# (talpha (x)1)(r0)+(1(x)alpha)(r0): r0=a U1U1+b U1U2+c U2U1+d U2U2
cond_U1 = talpha['U1']*a + talpha['U2']*c + a*alpha['U1'] + b*alpha['U2']
cond_U2 = talpha['U1']*b + talpha['U2']*d + c*alpha['U1'] + d*alpha['U2']
symm=[2*a,2*d,b+c+1]
print("Example 3.4, r0 allowed by (8) and the symmetric part:",sp.solve([cond_U1,cond_U2]+symm,[a,b,c,d],dict=True))
# spectral CYBE
z1,z2,z3=sp.symbols('z1 z2 z3')
def rz(P,z): return tadd(P,{k:v/(z-1) for k,v in Om.items()})
def emb(r,i,j):
    # place r in factors i,j of triple; returns dict triple->coef with signs (for r even)
    out={}
    for (u,v),c in r.items():
        key=[None,None,None]
        if (i,j) in [(0,1),(0,2),(1,2)]:
            key[i]=u; key[j]=v; out[tuple(key)]=c
    return out
def specCYBE(P):
    r12=rz(P,z1/z2); r13=rz(P,z1/z3); r23=rz(P,z2/z3)
    out={}
    def acc(k,c): out[k]=out.get(k,0)+c
    for (ua,va),ca in r12.items():
        for (ub,vb),cb in r13.items():
            s=(-1)**(p(ua)*p(ub))
            for k,v in br(ua,ub).items(): acc((k,va,vb),s*ca*cb*v)
    for (ua,va),ca in r12.items():
        for (ub,vb),cb in r23.items():
            for k,v in br(va,ub).items(): acc((ua,k,vb),ca*cb*v)
    for (ua,va),ca in r13.items():
        for (ub,vb),cb in r23.items():
            s=(-1)**(p(ua)*p(ub))
            for k,v in br(va,vb).items(): acc((ua,ub,k),s*ca*cb*v)
    return {k:sp.simplify(sp.together(v)) for k,v in out.items() if sp.simplify(sp.together(v))!=0}
for name,P in [("P(s)",Ps),("P_A",PA),("P_B",PB)]:
    print(name,"spectral CYBE:",specCYBE(P)=={})
    z=sp.Symbol('z'); 
    # unitarity r(z)=-tau(r)(1/z)
    u=tadd(rz(P,z),{k:v for k,v in flip(rz(P,1/z)).items()})
    print(name,"unitarity:",u=={})
# formula (10): det r(z) for P(s)
z=sp.Symbol('z')
basis=[(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]
rr=rz(Ps,z)
# coordinates in basis E12,E21,E13,E31,E23,E32,U1,U2 : express U-part
def coords(r):
    M=sp.zeros(8,8)
    idx={b:i for i,b in enumerate(basis)}
    # U1=E11+E33,U2=E22+E33 ; diagonal part h(x)h: solve
    for (u,v),c in r.items():
        if u in idx and v in idx: M[idx[u],idx[v]]+=c
    return M
M=coords(rr)
# toral block: read coefficients of U_i(x)U_j from P(s) and Omega
M[6,7]=-(s+1)-1/(z-1); M[7,6]=s-1/(z-1)
print("det:",sp.factor(M.det()), " vs text:", sp.factor(-z**3*(s*z-s-1)*(s*z-s+z)/(z-1)**8))
x,w=sp.symbols('x w')
print("identity:",sp.simplify(1/((x-1)*(x*w-1))-1/((x-1)*(w-1))+1/((x*w-1)*(w-1))+1/(x*w-1)))
