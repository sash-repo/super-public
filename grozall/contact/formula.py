"""Explicit shapes: the first-order family on densities, and kGz at lam=0."""
import os, sys
from fractions import Fraction as Fr
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import solve as lsolve
from affine import pencil, ops_at
from gen import apply_term, rand_field, term_latex
from contact import madd, mscal, e1, e2, reeb, pmul, pscal, padd

print("=== the W=2 family on densities, in the basis (Pi, R_F, R_G) ===")
terms,V0,N0,N1,N2 = pencil(0,0,0,2)
Fs=[rand_field(0,3,seed=41+i) for i in range(2)]
Gs=[rand_field(0,3,seed=77+i) for i in range(2)]
def scal(d): return {k[:3]:v for k,v in d.items()}
def ev(vec,F,G):
    acc={}
    for cf,t in zip(vec,terms):
        if cf: acc=madd(acc,mscal(cf,apply_term(t,F,G)))
    return scal(acc)
def PI(f,g):
    f={k[:3]:v for k,v in f.items()}; g={k[:3]:v for k,v in g.items()}
    return padd(pscal(-1,pmul(e1(f),e2(g))), pmul(e2(f),e1(g)))
def RF(f,g):
    f={k[:3]:v for k,v in f.items()}; g={k[:3]:v for k,v in g.items()}
    return pmul(reeb(f),g)
def RG(f,g):
    f={k[:3]:v for k,v in f.items()}; g={k[:3]:v for k,v in g.items()}
    return pmul(f,reeb(g))
base=[PI,RF,RG]
for (lam,mu) in ((Fr(-1),Fr(-1)),(Fr(0),Fr(0)),(Fr(1),Fr(0)),(Fr(0),Fr(1)),(Fr(2),Fr(-3))):
    op=ops_at(terms,V0,N0,N1,N2,lam,mu)[0]
    rows,b=[],[]
    for F in Fs:
        for G in Gs:
            tgt=ev(op,F,G); cols=[bb(F,G) for bb in base]
            ks=sorted(set(tgt)|set().union(*[set(c) for c in cols]))
            for k in ks:
                rows.append([c.get(k,Fr(0)) for c in cols]); b.append(tgt.get(k,Fr(0)))
    sol=lsolve(rows,b)
    if sol and sol[0]:
        sol=[x/sol[0] for x in sol]
    print("  lam=%-4s mu=%-4s :  Pi + (%s) R_F + (%s) R_G"%(lam,mu,sol[1],sol[2]))

print()
print("=== kGz at lam=mu=0, in components ===")
terms2,V02,A0,A1,A2 = pencil(2,2,0,2,nfield=1)
op=ops_at(terms2,V02,A0,A1,A2,Fr(0),Fr(0))[0]
piv=[c for c in op if c]
op=[c/piv[0] for c in op]
den=1
for c in op:
    den = den*c.denominator//__import__("math").gcd(den,c.denominator)
op=[c*den for c in op]
g=0
for c in op: g=__import__("math").gcd(g,int(c))
op=[c//g for c in op]
out=[]
for cf,t in zip(op,terms2):
    if cf: out.append("%+d\\,%s"%(cf,term_latex(t,2,2,0)))
print("  " + " ".join(out))
print("  (%d terms)"%len(out))
