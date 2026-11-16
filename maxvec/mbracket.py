"""[M_f, M_g] = M_{f,g}_{m.b.} (formulas (22), (23), (26), (28)), [Le_f, Le_g] = Le_{f,g}_{B.b.}
and the divergence of M_f, formula (33), on m(2) (even q1, q2; odd x1, x2, tau), for all pairs of monomials of degree <= 3.  The sign of the term
(-1)^{p(f)} df/dtau (2-E)(g) in (28) is tested both ways: sgn = 1 is formula (28)."""
import sys, itertools
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact import *
setup(['q1','q2'],['x1','x2','tau'])
N=2
def tmE(f): return add(sc(2,f),sc(-1,E_of(f,'tau')))
def Le(f):
    s=(-1)**par(f); X={}
    for i in (1,2):
        X=vadd(X,{f'x{i}':d(f'q{i}',f), f'q{i}':sc(s,d(f'x{i}',f))})
    return X
def M(f):
    s=(-1)**par(f)
    X={'tau':tmE(f)}; X=vadd(X,vsc(-1,Le(f)))
    ft=d('tau',f)
    X=vadd(X,{v:sc(-s,mul(ft,var(v))) for v in ALL() if v!='tau'})
    return X
def bb(f,g):
    s=(-1)**par(f)
    return add(*[add(mul(d(f'q{i}',f),d(f'x{i}',g)),sc(s,mul(d(f'x{i}',f),d(f'q{i}',g)))) for i in (1,2)])
def mb(f,g,sgn=1):
    s=(-1)**par(f)
    return add(mul(tmE(f),d('tau',g)),sc(sgn*s,mul(d('tau',f),tmE(g))),sc(-1,bb(f,g)))
mons=[]
for e in itertools.product(range(2),repeat=2):
    for r in range(4):
        for S in itertools.combinations(range(3),r):
            if sum(e)+r<=3: mons.append(mono(1,e,S))
for sgn in (1,-1):
    bad=n=0
    for f in mons:
        for g in mons:
            n+=1
            if not veq(comm(M(f),M(g)),M(mb(f,g,sgn))): bad+=1
    print("sgn",sgn,"M checks",n,"bad",bad)
bad=n=0
m2=[m for m in mons if all(2 not in S for (e,S) in m)]
for f in m2:
    for g in m2:
        n+=1
        if not veq(comm(Le(f),Le(g)),Le(bb(f,g))): bad+=1
print("Le checks",n,"bad",bad)
import contact as C
def Div(X):
    px=vpar(X); r={}
    for v,c in X.items():
        pv=1 if v in C.OD else 0
        r=add(r,sc((-1)**(pv*(px+1)),d(v,c)))
    return r
bad=0
for f in mons:
    s=(-1)**par(f)
    ft=d('tau',f)
    rhs=sc(2*s, add(add(ft,sc(-1,E_of(ft,'tau'))), sc(-1,add(*[d(f'q{i}',d(f'x{i}',f)) for i in (1,2)]))))
    if add(Div(M(f)),sc(-1,rhs)): bad+=1
print("Div M_f formula bad:",bad,"of",len(mons))
