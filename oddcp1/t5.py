"""Statement 16.2: dim H^1(T_2)^{s'} = 1 for the copy s' of case 5) of
[Ba5, Thm. 2.2] over CP^{1|5}; the invariant class, [e',z] ~ 0, [f',z] ~ 0.
"""
import quot, sl2inv as S
from fractions import Fraction
ks=(1,1,1,1,1); m=5
Q=quot.Quot(ks,2)
e,f=S.s_prime(5,m)
inv=Q.invariants(e,f)
print("dim H^1(T_2)^{s'} for case 5 =",len(inv))
def show(el):
    out=[]
    for (g,ex),c in sorted(el.items(), key=lambda kv:(str(kv[0][0]),kv[0][1])):
        if g[0]=='u': nm="xi_%s d/dx"%("".join(map(str,g[1])))
        else: nm="xi_%s d/dxi_%d"%("".join(map(str,g[1])),g[2])
        out.append(f"{c}*x^{ex} {nm}")
    return "  +  ".join(out)
for v in inv:
    z=Q.as_cocycle(v)
    print("\ninvariant class:", show(z))
    # independent certificate: are [e',z] and [f',z] coboundaries?
    for nm,X in (("e'",e),("f'",f)):
        Z=S.bracket_even(m,X,S.coc_to_field(m,z))
        r=Q.E.reduce(S.field_to_coc(m,2,Z))
        print(f"   [{nm},z] reduces to zero mod B^1: {not r}")
    # and is z itself non-zero in H^1?
    print("   z itself is NOT a coboundary:", bool(Q.E.reduce(z)))
