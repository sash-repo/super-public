"""[K_f, K_g] = K_{f,g}_{k.b.} (formulas (20), (21), (25), (27)) and [H_f, H_g] = H_{f,g}_{P.b.}
on k(3|3) (variables t, p, q | xi, eta, theta), for all pairs of monomials of degree <= 3."""
import sys, itertools, random
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact import *
setup(['t','p','q'],['xi','eta','th'])
def twoMinusE(f,skip): return add(sc(2,f),sc(-1,E_of(f,skip)))
def H(f):
    s=(-1)**par(f)
    return {k:v for k,v in {'q':d('p',f),'p':sc(-1,d('q',f)),'eta':sc(-s,d('xi',f)),'xi':sc(-s,d('eta',f)),'th':sc(-s,d('th',f))}.items() if v}
def K(f):
    X={'t':twoMinusE(f,'t')}
    X=vadd(X,vsc(-1,H(f)))
    ft=d('t',f)
    X=vadd(X,{v:mul(ft,var(v)) for v in ALL() if v!='t'})
    return X
def pb(f,g):
    s=(-1)**par(f)
    a=add(mul(d('p',f),d('q',g)),sc(-1,mul(d('q',f),d('p',g))))
    b=add(mul(d('xi',f),d('eta',g)),mul(d('eta',f),d('xi',g)),mul(d('th',f),d('th',g)))
    return add(a,sc(-s,b))
def kb(f,g):
    return add(mul(twoMinusE(f,'t'),d('t',g)),sc(-1,mul(d('t',f),twoMinusE(g,'t'))),sc(-1,pb(f,g)))
mons=[]
for e in itertools.product(range(2),repeat=3):
    for r in range(4):
        for S in itertools.combinations(range(3),r):
            if sum(e)+r<=3: mons.append(mono(1,e,S))
bad=0;n=0
for f in mons:
    for g in mons:
        n+=1
        if not veq(comm(K(f),K(g)),K(kb(f,g))): bad+=1
print("K bracket checks",n,"bad",bad)
# also H: [H_f,H_g] vs H_{pb}
mons2=[m for m in mons if list(m)[0][0][0]==0]
bad=0;n=0
for f in mons2:
    for g in mons2:
        n+=1
        if not veq(comm(H(f),H(g)),H(pb(f,g))): bad+=1
print("H bracket checks",n,"bad",bad)
