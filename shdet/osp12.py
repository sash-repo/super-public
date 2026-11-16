# osp(1|2) via the operators (these) of aIvan, acting on C[x,theta]; check brackets and
# the Shapovalov anti-involution sigma (Sign Rule) with sigma^2 = delta_{-1}.
import sympy as sp
x,l=sp.symbols('x lambda')
# element f0(x)+theta f1(x) as tuple
def dx(F): return (sp.diff(F[0],x), sp.diff(F[1],x))
def dth(F): return (F[1],0)            # d/dtheta(f0+th f1)=f1 (even part), odd part 0
def thm(F): return (0,F[0])            # theta*(f0+th f1)=th f0
def mul(c,F): return (sp.expand(c*F[0]),sp.expand(c*F[1]))
def add(*Fs): return (sp.expand(sum(F[0] for F in Fs)), sp.expand(sum(F[1] for F in Fs)))
Xm=lambda F: mul(-1,dx(F))
H =lambda F: add(mul(2*x,dx(F)), thm(dth(F)), mul(-l,F))
Xp=lambda F: add(mul(x**2,dx(F)), mul(x,thm(dth(F))), mul(-l*x,F))
Nm=lambda F: add(dth(F), mul(-1,thm(dx(F))))
Np=lambda F: add(mul(-x,dth(F)), mul(x,thm(dx(F))), mul(-l,thm(F)))
ops={'Xm':(Xm,0),'H':(H,0),'Xp':(Xp,0),'Nm':(Nm,1),'Np':(Np,1)}
tests=[(x**k,0) for k in range(5)]+[(0,x**k) for k in range(5)]
def br(a,b):
    A,pa=ops[a];B,pb=ops[b]; s=(-1)**(pa*pb)
    return lambda F: add(A(B(F)), mul(-s,B(A(F))))
def ident(f):
    # find linear combination of basis ops equal to f on tests
    cs=sp.symbols('c0:5'); eqs=[]
    for T in tests:
        r=f(T); comb=add(*[mul(cs[i],ops[n][0](T)) for i,n in enumerate(ops)])
        eqs+= [sp.expand(r[0]-comb[0]), sp.expand(r[1]-comb[1])]
    sol=sp.solve([e for E in eqs for e in sp.Poly(E,x).coeffs()] if True else eqs, cs, dict=True)
    return {n:sol[0][cs[i]] for i,n in enumerate(ops) if sol[0][cs[i]]!=0} if sol else None
names=list(ops)
table={}
for i,a in enumerate(names):
    for b in names[i:]:
        table[(a,b)]=ident(br(a,b)); print(a,b,table[(a,b)])
# sigma (Sign Rule): Xp<->Xm, H->H, Np->k1*Nm, Nm->k2*Np
k1,k2=sp.symbols('k1 k2')
sig={'Xp':{'Xm':1},'Xm':{'Xp':1},'H':{'H':1},'Np':{'Nm':k1},'Nm':{'Np':k2}}
par={'Xm':0,'H':0,'Xp':0,'Nm':1,'Np':1}
def lin(d,f):
    out={}
    for n,c in d.items():
        for m,e in f(n).items(): out[m]=out.get(m,0)+c*e
    return out
def brl(da,db):
    out={}
    for a,ca in da.items():
        for b,cb in db.items():
            key=(a,b) if names.index(a)<=names.index(b) else (b,a)
            r=table[key]
            if names.index(a)>names.index(b): r={k:-(-1)**(par[a]*par[b])*v for k,v in r.items()}
            for k,v in r.items(): out[k]=out.get(k,0)+ca*cb*v
    return out
eqs=[]
for i,a in enumerate(names):
    for b in names:
        lhs=lin(table[(a,b)] if names.index(a)<=names.index(b) else {k:-(-1)**(par[a]*par[b])*v for k,v in table[(b,a)].items()}, lambda n: sig[n])
        rhs={k:(-1)**(par[a]*par[b])*v for k,v in brl(sig[b],sig[a]).items()}
        for k in set(lhs)|set(rhs): eqs.append(sp.expand(lhs.get(k,0)-rhs.get(k,0)))
print(sp.solve([e for e in eqs if e!=0],[k1,k2],dict=True))
Om=lambda F: add(Np(Nm(F)), mul(-1,Nm(Np(F))))
for T in [(1,0),(x,0),(x**2,0),(0,1),(0,x),(0,x**2)]:
    r=Om(T); print(T, r)
