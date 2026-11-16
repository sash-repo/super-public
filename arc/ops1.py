"""Explicit invariant operators of low order, with symbolic weights."""
import sympy as sp
from sl12 import hom_dim, ansatz_terms, invariance_matrix, nullspace

a1,a2,b1,b2=sp.symbols('a1 a2 b1 b2')
wa,wb=(a1,a2),(b1,b2)

def name(term):
    s,u,m1,p1,r1,m2,p2,r2=term
    def D(m,p,r,v):
        out=v
        if m: out="d_t^%d %s"%(m,out) if m>1 else "d_t %s"%out
        if p: out="d_xi(%s)"%out
        if r: out="d_eta(%s)"%out
        return out
    pre="".join(["xi " if s else "", "eta " if u else ""])
    return pre+D(m1,p1,r1,"f")+" * "+D(m2,p2,r2,"g")

def show(N,q,dmax=5):
    terms,mat=invariance_matrix(N,q,wa,wb,dmax)
    ns=nullspace(mat,len(terms))
    print("=== N=%d q=%+d : dim %d  (%d terms, dmax=%d)"%(N,q,len(ns),len(terms),dmax))
    for k,v in enumerate(ns):
        v=sp.Matrix(v)
        dens=[sp.denom(sp.together(x)) for x in v]
        L=sp.lcm([sp.Integer(1)]+dens)
        v=sp.simplify(L*v)
        cont=sp.gcd([sp.factor(x) for x in v if x!=0])
        if cont not in (0,):
            v=sp.Matrix([sp.cancel(x/cont) for x in v])
        print("  basis vector %d:"%(k+1))
        for t,c in zip(terms,v):
            c=sp.factor(sp.simplify(c))
            if c!=0:
                print("     %-34s : %s"%(name(t),c))
    print()

if __name__=="__main__":
    import sys
    for N,q in [(1,1),(1,-1),(2,0)]:
        show(N,q)
