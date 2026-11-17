"""Statement 16.2, formula (30): the class z, written in the basis of [Ba5],
is proportional to the invariant found by the engine; [e',z] ~ 0, [f',z] ~ 0;
and the five brackets printed in the proof in [Ba5] ("his" below), one of
which carries a double minus where a single one is right.
"""
import quot, sl2inv as S, bashkin as B
from fractions import Fraction
ks=(1,1,1,1,1); m=5
Q=quot.Quot(ks,2)
e,f=S.s_prime(5,m)
inv=Q.invariants(e,f)
zeng=Q.as_cocycle(inv[0])
# candidate written in HIS basis
zpap=B.add(B.w((1,3,2),2,-1,-1), B.w((1,3,5),5,-1,-2),
           B.w((1,2,5),4,-1,-2), B.w((2,3,4),5,-1,1))
ce, cp = Q.coords(zeng), Q.coords(zpap)
lam=None
for a,b in zip(cp,ce):
    if b: lam=a/b; break
print("candidate is proportional to the engine's invariant:",
      all(a==lam*b for a,b in zip(cp,ce)), " factor",lam)
print("candidate is non-zero in H^1:", any(cp))
for nm,X in (("e'",e),("f'",f)):
    Z=S.bracket_even(m,X,S.coc_to_field(m,zpap))
    print(f"   [{nm}, z] ~ 0 :", not Q.E.reduce(S.field_to_coc(m,2,Z)))

print("\n--- his five printed brackets, checked as classes in H^1:")
def cls(z): return Q.coords(z)
def eq(a,b): return all(x==y for x,y in zip(a,b))
tests=[
 ("(i)", B.w((1,3,2),2,-1), [B.w((2,3,1),1,-1,1)]),
 ("(ii) as printed (double minus = plus)", B.w((1,3,5),5,-1),
      [B.w((2,3,1),1,-1,-1), B.w((2,3,4),4,-1,-1), B.w((1,3,5),4,-1,1)]),
 ("(ii) with a single minus on the third term", B.w((1,3,5),5,-1),
      [B.w((2,3,1),1,-1,-1), B.w((2,3,4),4,-1,-1), B.w((1,3,5),4,-1,-1)]),
 ("(iii)", B.w((4,5,1),1,-1), [B.w((2,4,5),1,-1,1)]),
 ("(iv)", B.w((4,5,2),2,-1), [B.w((3,4,5),2,-1,1), B.w((2,4,5),1,-1,-1)]),
 ("(v)", B.w((2,3,4),5,-1), [B.w((2,3,1),1,-1,-1), B.w((2,3,4),4,-1,-2)]),
]
for nm,src,rhs in tests:
    Z=S.bracket_even(m,e,S.coc_to_field(m,src))
    lhs=cls(S.field_to_coc(m,2,Z))
    r=cls(B.add(*rhs))
    print(f"   {nm}: [e',v] equals the printed right-hand side: {eq(lhs,r)}")
