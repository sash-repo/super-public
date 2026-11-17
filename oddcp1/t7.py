"""Statement 16.2: the printed basis cocycles of h'-weight 0 for the case 5)
of [Ba5, Thm. 2.2]; there are six, of which [Ba5] lists five ("the paper"
below is [Ba5]); the one omitted is v_5 = x^{-1} xi_1 xi_2 xi_5 d/dxi_4.
"""
import quot, sl2inv as S, bashkin as B
from fractions import Fraction
from itertools import combinations
ks=(1,1,1,1,1); m=5; ALL=(1,2,3,4,5)
Q=quot.Quot(ks,2)
e,f=S.s_prime(5,m)
h=S.bracket_even(m,e,f)

# --- 1) the h'-weight of every printed basis cocycle of cp15t2
vlist=[(1,2,4),(1,2,5),(1,3,2),(1,3,5),(1,4,3),(1,4,5),(1,5,2),(1,5,3),
       (2,3,1),(2,3,4),(2,4,1),(2,4,3),(2,5,3),(2,5,4),(3,4,2),(3,4,5),
       (3,5,1),(3,5,4),(4,5,1),(4,5,2)]
named=[(f"x^-1 xi_{i}xi_{j}xi_{t} d/dxi_{t}", B.w((i,j,t),t,-1)) for (i,j,t) in vlist]
named+=[(f"x^-1 xi_{J[0]}xi_{J[1]}xi_{J[2]} d/dxi_{r}", B.w(J,r,-1))
        for J in combinations(ALL,3) for r in ALL if r not in J]
print("weight-0 printed basis cocycles for case 5 (h'-eigenvalue 0):")
w0=[]
for nm,z in named:
    Z=S.bracket_even(m,h,S.coc_to_field(m,z))
    c=Q.coords(S.field_to_coc(m,2,Z))
    z0=Q.coords(z)
    # eigenvalue: c = lam * z0
    lam=None
    for a,b in zip(c,z0):
        if b: lam=a/b; break
    if all(a==(lam or 0)*b for a,b in zip(c,z0)):
        if lam==0: w0.append(nm); print("   ",nm)
    else:
        print("    NOT an eigenvector:",nm)
print("  -> engine finds",len(w0),"; the paper lists 5")
