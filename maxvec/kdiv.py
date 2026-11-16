"""Div K_f = (2n+2-m) df/dt, formula (32), for k(2n+1|m), all monomials of degree <= 2 in
the even indeterminates, in four cases (n|m) = (1|2), (1|1), (2|3), (0|2)."""
import sys, itertools; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact as C
from contact import *
def run(ev,od,n,m):
    setup(ev,od)
    def tmE(f): return add(sc(2,f),sc(-1,E_of(f,'t')))
    def H(f):
        s=(-1)**par(f); X={}
        for i in range(1,n+1): X=vadd(X,{f'q{i}':d(f'p{i}',f), f'p{i}':sc(-1,d(f'q{i}',f))})
        for v,w in [('xi','eta'),('eta','xi'),('th','th')]:
            if v in od: X=vadd(X,{w:sc(-s,d(v,f))})
        return X
    def K(f):
        X={'t':tmE(f)}; X=vadd(X,vsc(-1,H(f))); ft=d('t',f)
        return vadd(X,{v:mul(ft,var(v)) for v in ALL() if v!='t'})
    def Div(X):
        px=vpar(X); r={}
        for v,c in X.items(): r=add(r,sc((-1)**((v in C.OD)*(px+1)),d(v,c)))
        return r
    bad=0;cnt=0
    for e in itertools.product(range(3),repeat=len(ev)):
        for r in range(len(od)+1):
            for S in itertools.combinations(range(len(od)),r):
                f=mono(1,e,S); cnt+=1
                if add(Div(K(f)),sc(-(2*n+2-m),d('t',f))): bad+=1
    print(n,m,"Div K_f=(2n+2-m) f_t: bad",bad,"of",cnt)
run(['t','p1','q1'],['xi','eta'],1,2)
run(['t','p1','q1'],['th'],1,1)
run(['t','p1','q1','p2','q2'],['xi','eta','th'],2,3)
run(['t'],['xi','eta'],0,2)
