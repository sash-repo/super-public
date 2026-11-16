"""Polynomial superalgebra arithmetic (exact, over Q) used to check the formulas of
Section 2 of the text: the vector fields K_f, H_f, M_f, Le_f, the Poisson, Buttin and
contact brackets, and the divergences of K_f and M_f."""
from fractions import Fraction as Fr
import itertools, random, sys
EV, OD = None, None
def setup(ev, od):
    global EV, OD; EV, OD = ev, od
def mono(c, e=None, S=()):
    e = e or (0,)*len(EV)
    S=list(S); sg=1
    for i in range(len(S)):
        for j in range(len(S)-1-i):
            if S[j]==S[j+1]: return {}
            if S[j]>S[j+1]: S[j],S[j+1]=S[j+1],S[j]; sg=-sg
    if len(set(S))<len(S): return {}
    return {(tuple(e),tuple(S)):Fr(c)*sg}
def add(*ps):
    r={}
    for p in ps:
        for k,v in p.items():
            r[k]=r.get(k,0)+v
            if r[k]==0: del r[k]
    return r
def sc(c,p): return {k:c*v for k,v in p.items() if c*v!=0}
def mul(p,q):
    r={}
    for (e1,S1),a in p.items():
        for (e2,S2),b in q.items():
            m=mono(a*b,[x+y for x,y in zip(e1,e2)],S1+S2)
            r=add(r,m)
    return r
def par(p):
    ps={len(S)%2 for (e,S) in p}
    assert len(ps)<=1, p
    return ps.pop() if ps else 0
def var(name):
    if name in EV:
        e=[0]*len(EV); e[EV.index(name)]=1; return mono(1,e)
    return mono(1,None,(OD.index(name),))
def d(name,p):
    r={}
    if name in EV:
        i=EV.index(name)
        for (e,S),a in p.items():
            if e[i]:
                e2=list(e); e2[i]-=1; r=add(r,{(tuple(e2),S):a*e[i]})
    else:
        i=OD.index(name)
        for (e,S),a in p.items():
            if i in S:
                pos=S.index(i); r=add(r,{(e,S[:pos]+S[pos+1:]):a*(-1)**pos})
    return r
ALL=lambda: EV+OD
def apply(X,f):
    return add(*[mul(X.get(v,{}),d(v,f)) for v in ALL()])
def vpar(X):
    for v,c in X.items():
        if c: return (par(c)+(v in OD))%2
    return 0
def comm(X,Y):
    s=(-1)**(vpar(X)*vpar(Y))
    R={v:add(apply(X,Y.get(v,{})),sc(-s,apply(Y,X.get(v,{})))) for v in ALL()}
    return {v:c for v,c in R.items() if c}
def veq(X,Y):
    return all(add(X.get(v,{}),sc(-1,Y.get(v,{})))=={} for v in ALL())
def deg(p):  # total degree except t / tau (homogeneous assumed)
    return None
def E_of(f, skip):
    r={}
    for v in ALL():
        if v==skip: continue
        r=add(r,mul(var(v),d(v,f)))
    return r
def vadd(*Xs):
    R={}
    for X in Xs:
        for v,c in X.items(): R[v]=add(R.get(v,{}),c)
    return {v:c for v,c in R.items() if c}
def vsc(c,X): return {v:sc(c,p) for v,p in X.items()}
