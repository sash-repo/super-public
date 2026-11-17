"""Statement 16.2, formula (30): with v_5 in the list, the kernel of ad_{e'}
on the span of the six weight-0 cocycles v_1, ..., v_6 is one-dimensional.
"""
import quot, sl2inv as S, bashkin as B
from fractions import Fraction
ks=(1,1,1,1,1); m=5
Q=quot.Quot(ks,2); e,f=S.s_prime(5,m)
def nm(g,ex):
    if g[0]=='u': return f"x^{{{ex}}}\\xi_{{{''.join(map(str,g[1]))}}}\\partial_x"
    return f"x^{{{ex}}}\\xi_{{{''.join(map(str,g[1]))}}}\\partial_{{\\xi_{g[2]}}}"
def show(z):
    z=Q.E.reduce(z) if False else z
    return " + ".join(f"({c}){nm(g,ex)}" for (g,ex),c in sorted(z.items(),key=lambda kv:str(kv[0])))
# the six weight-0 cocycles
w0=[("v1",B.w((1,3,2),2,-1)),("v2",B.w((1,3,5),5,-1)),("v3",B.w((4,5,1),1,-1)),
    ("v4",B.w((4,5,2),2,-1)),("v5",B.w((1,2,5),4,-1)),("v6",B.w((2,3,4),5,-1))]
print("brackets [e', v_i], as classes expressed in the engine's basis:")
cols=[]
for n,z in w0:
    Z=S.bracket_even(m,e,S.coc_to_field(m,z))
    c=Q.coords(S.field_to_coc(m,2,Z)); cols.append(c)
    nz={k:v for k,v in Q.as_cocycle(c).items()}
    print(f"  {n}: {show(nz) if nz else '~ 0'}")
# kernel of the 6 x dim matrix
from exactlin import nullspace
rows=[[cols[j][i] for j in range(6)] for i in range(Q.dim)]
ker=nullspace(rows)
print("\nkernel of ad_{e'} on span(v_1..v_6):",len(ker),"dimensional")
for v in ker: print("   coefficients (v1..v6):",v)
# the missing bracket alone, in a readable form
Z=S.bracket_even(m,e,S.coc_to_field(m,B.w((1,2,5),4,-1)))
print("\n[e', x^-1 xi_1xi_2xi_5 d/dxi_4] as a cocycle (before reduction):")
print("   ",show(S.field_to_coc(m,2,Z)))
