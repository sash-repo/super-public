import os, pickle, random, sys
from fractions import Fraction
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank
from ssuper import SSetup
from sgen import enumerate_schemes, evaluate
from suniv import zero_subspace, rref

N=2; schemes=enumerate_schemes(N)
def flat(S,v,probes,swap=False):
    A=S.A; out=[]
    for (P,Q) in probes:
        if swap: P,Q=Q,P
        out.append(A.summ([A.scale(evaluate(S,sch,P,Q),c) for c,sch in zip(v,schemes) if c]))
    return out
for k in (2,3):
    D=pickle.load(open(f"sgen_odd_m{k}_n{k}_N{N}.pkl","rb"))
    kernel=[[Fraction(c) for c in v] for v in D["kernel"]]
    S=SSetup(k,k,form="odd"); A=S.A
    Z=zero_subspace(S,schemes,kernel,random.Random(31337))
    ZR,piv=rref(Z); reps=[]
    for v in kernel:
        w=list(v)
        for row,c in zip(ZR,piv):
            if w[c]: f=w[c]; w=[a-f*b for a,b in zip(w,row)]
        if any(w): reps.append(w)
    R,_=rref(reps)
    rng=random.Random(555+k)
    probes=[(S.random_field(3,rng,nterms=3),S.random_field(3,rng,nterms=3)) for _ in range(3)]
    vals=[flat(S,v,probes) for v in R]
    sw  =[flat(S,v,probes,swap=True) for v in R]
    keys=[sorted(set().union(*[set(x[i]) for x in vals+sw])) for i in range(len(probes))]
    def vec(x): return [x[i].get(kk,Fraction(0)) for i in range(len(probes)) for kk in keys[i]]
    V=[vec(x) for x in vals]; Sw=[vec(x) for x in sw]
    # dimension of the symmetric part = dim ker(swap - 1) on the span
    import itertools
    # express: find coefficients c with sum c_i (V_i - Sw_i) = 0  -> symmetric combos
    from exactlin import nullspace
    Msym=[[V[i][j]-Sw[i][j] for i in range(len(V))] for j in range(len(V[0]))]
    Manti=[[V[i][j]+Sw[i][j] for i in range(len(V))] for j in range(len(V[0]))]
    ds=len(nullspace(Msym)); da=len(nullspace(Manti))
    print(f"k={k}: order-2 space of dimension {len(R)} = "
          f"{ds} symmetric + {da} antisymmetric")
