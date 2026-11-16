"""Calibration of the CONTRACTION convention, in both forms.

The Poisson bracket (even form) and Buttin's antibracket (odd form) are
themselves one-pair contraction schemes with scalar fiber:

        {f,g} = sum_{k,l} omega^{kl} (d_k f)(d_l g)  up to a factor,

so evaluating that scheme with the machinery of sgen.py and comparing with the
bracket the algebra already knows tests exactly the sign that the scheme
evaluation adds by hand -- the Koszul sign of moving an operator on the second
factor past the first one.  A scheme evaluation that gets it wrong produces a
map that is not invariant, which is what a wrong answer would look like.
"""
import random, sys
from fractions import Fraction
from ssuper import SSetup

def onepair(S, f, g, twist=0):
    """sum_{k,l} (-1)^{twist*p(k)} omega^{kl} (d_k f)(d_l g), Koszul included."""
    A=S.A; out=[]
    pf=A.parity(f)
    for (k,l,w) in S.Omnz:
        P=S.dz(f,k)
        if not P: continue
        c=Fraction(w)
        if twist and S.par[k]: c=-c
        if S.par[l] and (pf+S.par[k])%2: c=-c
        Q=S.dz(g,l)
        if not Q: continue
        out.append(A.scale(A.mul(P,Q),c))
    return A.summ(out)

def report(S, name):
    A=S.A
    mons=[p for d in range(1,4) for p in S.monomials(d)]
    rng=random.Random(2); sel=rng.sample(mons,min(10,len(mons)))
    for twist in (0,1):
        ratios=set(); ok=True
        for f in sel:
            for g in sel:
                b=S.bracket(f,g); t=onepair(S,f,g,twist)
                if not b and not t: continue
                if not b or not t: ok=False; break
                # is t a constant multiple of b?
                k0=next(iter(t)); 
                if k0 not in b: ok=False; break
                r=t[k0]/b[k0]
                if A.scale(b,r)!=t: ok=False; break
                ratios.add(r)
            if not ok: break
        print(f"  {name}, twist={twist}: "
              + (f"equals {sorted(ratios)} x the bracket" if ok and len(ratios)==1
                 else "NOT proportional to the bracket"))

print("=== the one-pair contraction against the bracket")
for (m,n) in [(1,1),(1,2),(2,2)]:
    report(SSetup(m,n), f"even form (2m|n)=({2*m}|{n})")
for k in (2,3):
    for oc in (1,-1):
        report(SSetup(k,k,form="odd",odd_coef=oc), f"odd form k={k}, omega sign {oc:+d}")

print("\n=== the one-pair contraction as an INVARIANT operator on functions")
def inv_test(S, twist, pT):
    A=S.A
    mons=[p for d in range(0,4) for p in S.monomials(d)]
    ev=[p for p in mons if A.parity(p)==0]
    hams=[p for d in range(1,4) for p in S.monomials(d)]
    rng=random.Random(11)
    ev=rng.sample(ev,min(6,len(ev)))
    for f in ev:
        for g in ev:
            T=onepair(S,f,g,twist)
            for h in hams:
                pX=(A.parity(h)+(1 if S.form=="odd" else 0))%2
                e=-1 if (pT*pX)%2 else 1
                lhs=S.bracket(h,T)
                rhs=A.scale(A.add(onepair(S,S.bracket(h,f),g,twist),
                                  onepair(S,f,S.bracket(h,g),twist)), e)
                if lhs!=rhs: return False
    return True

for (m,n) in [(1,1),(1,2),(2,2)]:
    S=SSetup(m,n)
    print(f"  even form ({2*m}|{n}): invariant (pT=0) = {inv_test(S,0,0)}")
for k in (2,3):
    for oc in (1,-1):
        S=SSetup(k,k,form='odd',odd_coef=oc)
        print(f"  odd form k={k}, omega sign {oc:+d}: invariant (pT=1) = {inv_test(S,0,1)}, "
              f"(pT=0) = {inv_test(S,0,0)}")
