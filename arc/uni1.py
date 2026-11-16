"""Is the order-1 bracket  kappa-correction = X applied to the naive part?"""
import sympy as sp
from sl12 import *
from sl12 import _EX

a1,a2,b1,b2=sp.symbols('a1 a2 b1 b2')

def D_XI(f):  return add(d_xi(f), mul(mono(0,0,1), d_t(f)))
def D_ETA(f): return add(d_eta(f), mul(mono(0,1,0), d_t(f)))

def B0(f,g):
    return add(smul(-a1, mul(f, D_XI(g))), smul(b1, mul(D_XI(f), g)))

# the true operator, from ops1.py output (N=1, q=-1)
TERMS=[((0,0,0,0,0,0,1,0), -a1),((0,0,0,1,0,0,0,0), b1),
       ((0,1,0,0,0,1,0,0), -a1),((0,1,1,0,0,0,0,0), b1),
       ((0,1,0,0,0,0,1,1), 2*a1),((0,1,0,1,0,0,0,1), -2*b1),
       ((0,1,0,1,1,0,0,0), -2*b1),((1,0,0,1,0,0,1,0), -2*b1),
       ((1,1,0,0,0,1,1,0), -2*a1),((1,1,1,0,0,0,1,0), 2*b1),
       ((1,1,0,1,0,1,0,0), -2*a1),((1,1,1,1,0,0,0,0), 2*b1),
       ((1,1,0,1,0,0,1,1), 4*(a1+b1))]
def Btrue(f,g):
    out={}
    for term,c in TERMS:
        out=add(out, smul(c, apply_term(term,f,g)))
    return out

mons=[mono(m,e,g) for m in range(3) for e in (0,1) for g in (0,1)]

def diff(F,G):
    bad=0
    for f in mons:
        for g in mons:
            if add(F(f,g), smul(-1,G(f,g))): bad+=1
    return bad

print("B0 vs Btrue          :", diff(B0,Btrue))
cands={
 "(1-2 xi D_xi) B0":   lambda f,g: add(B0(f,g), smul(-2, mul(mono(0,1,0), D_XI(B0(f,g))))),
 "(1-2 eta D_eta) B0": lambda f,g: add(B0(f,g), smul(-2, mul(mono(0,0,1), D_ETA(B0(f,g))))),
 "(1-2 xi D_xi-2 eta D_eta) B0": lambda f,g: add(B0(f,g),
        smul(-2, mul(mono(0,1,0), D_XI(B0(f,g)))), smul(-2, mul(mono(0,0,1), D_ETA(B0(f,g))))),
 "(1+2 eta D_eta) B0": lambda f,g: add(B0(f,g), smul(2, mul(mono(0,0,1), D_ETA(B0(f,g))))),
 "(1+2 xi D_xi) B0":   lambda f,g: add(B0(f,g), smul(2, mul(mono(0,1,0), D_XI(B0(f,g))))),
}
for nm,F in cands.items():
    print("%-32s: %d"%(nm, diff(F,Btrue)))

# --- solve for the correction as a unary operator applied to B0
c=sp.symbols('c1:6')
def M(f,g):
    v=B0(f,g)
    out=v
    out=add(out, smul(c[0], mul(mono(0,1,0), D_XI(v))))
    out=add(out, smul(c[1], mul(mono(0,0,1), D_ETA(v))))
    out=add(out, smul(c[2], mul(mono(0,1,1), D_XI(D_ETA(v)))))
    out=add(out, smul(c[3], mul(mono(0,1,1), d_t(v))))
    return out
eqs=set()
for f in mons:
    for g in mons:
        r=add(M(f,g), smul(-1,Btrue(f,g)))
        for k,v in r.items(): eqs.add(sp.expand(v))
eqs={e for e in eqs if e!=0}
sol=sp.solve(list(eqs), c[:4], dict=True)
print("correction ansatz solution:", sol)
