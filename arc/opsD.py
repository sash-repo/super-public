"""The low-order invariant operators in the D-basis (Y-type derivatives)."""
import itertools
import sympy as sp
from sl12 import (add, smul, mul, mono, d_t, d_xi, d_eta, rho, parity,
                  lf_basis, ODD_GENS, nullspace)

def D_XI(f):  return add(d_xi(f),  mul(mono(0,0,1), d_t(f)))
def D_ETA(f): return add(d_eta(f), mul(mono(0,1,0), d_t(f)))

def terms_D(N,q):
    out=[]
    for s,u,p1,r1,p2,r2 in itertools.product((0,1),repeat=6):
        if s-u-p1-p2+r1+r2!=q: continue
        rest=N+s+u-(p1+r1+p2+r2)
        if rest<0 or rest%2: continue
        h=rest//2
        for m1 in range(h+1):
            out.append((s,u,m1,p1,r1,h-m1,p2,r2))
    return out

def apply_D(term,phi,psi):
    s,u,m1,p1,r1,m2,p2,r2=term
    a=phi
    for _ in range(m1): a=d_t(a)
    if p1: a=D_XI(a)
    if r1: a=D_ETA(a)
    b=psi
    for _ in range(m2): b=d_t(b)
    if p2: b=D_XI(b)
    if r2: b=D_ETA(b)
    out=mul(a,b)
    if u: out=mul(mono(0,0,1),out)
    if s: out=mul(mono(0,1,0),out)
    return out

def name(term):
    s,u,m1,p1,r1,m2,p2,r2=term
    def D(m,p,r,v):
        if m: v=("d_t^%d "%m if m>1 else "d_t ")+v
        if p: v="D_xi(%s)"%v
        if r: v="D_eta(%s)"%v
        return v
    return ("xi " if s else "")+("eta " if u else "")+D(m1,p1,r1,"f")+" * "+D(m2,p2,r2,"g")

def run(N,q,wa,wb,dmax=5):
    terms=terms_D(N,q)
    a1,a2=wa; b1,b2=wb
    c1,c2=a1+b1+N+q, a2+b2-q
    lf=dict(lf_basis())
    mons=[mono(m,e,g) for m in range(dmax//2+1) for e in (0,1) for g in (0,1)
          if 2*m+e+g<=dmax]
    rows={}
    for gname in ODD_GENS:
        f=lf[gname]; pf=parity(f)
        for phi in mons:
            pphi=parity(phi)
            for psi in mons:
                for j,t in enumerate(terms):
                    lhs=rho(c1,c2,f,apply_D(t,phi,psi))
                    r1_=apply_D(t,rho(a1,a2,f,phi),psi)
                    sgn=-1 if (pf and pphi) else 1
                    r2_=apply_D(t,phi,rho(b1,b2,f,psi))
                    res=add(lhs,smul(-1,r1_),smul(-sgn,r2_))
                    for key,val in res.items():
                        rows.setdefault((gname,tuple(sorted(phi)),tuple(sorted(psi)),key),
                                        [0]*len(terms))[j]+=val
    mat=[r for r in rows.values() if any(x!=0 for x in r)]
    ns=nullspace(mat,len(terms))
    print("=== N=%d q=%+d : dim %d (%d D-terms)"%(N,q,len(ns),len(terms)))
    for k,v in enumerate(ns):
        v=sp.Matrix(v)
        L=sp.lcm([sp.Integer(1)]+[sp.denom(sp.together(x)) for x in v])
        v=sp.Matrix([sp.simplify(L*x) for x in v])
        nz=[x for x in v if x!=0]
        if nz:
            g=sp.gcd([sp.factor(x) for x in nz])
            if g not in (0,1): v=sp.Matrix([sp.cancel(x/g) for x in v])
        print("  vector %d:"%(k+1))
        for t,c in zip(terms,v):
            if c!=0: print("     %-32s : %s"%(name(t), sp.factor(sp.simplify(c))))
    print()

if __name__=="__main__":
    a1,a2,b1,b2=sp.symbols('a1 a2 b1 b2')
    for N,q in [(1,-1),(1,1),(2,0)]:
        run(N,q,(a1,a2),(b1,b2))
