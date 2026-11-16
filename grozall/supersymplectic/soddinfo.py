"""The three order-2 invariant operators over an ODD symplectic form:
are they symmetric or antisymmetric in the two arguments, and is the space
spanned by one and the same combination of schemes for every k?
"""
import os, pickle, random, sys
from fractions import Fraction
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank
from ssuper import SSetup
from sgen import Scheme, enumerate_schemes, evaluate
from suniv import zero_subspace, rref
from srun import op_parity

N=int(sys.argv[1]) if len(sys.argv)>1 else 2
schemes=enumerate_schemes(N)

def opvec(S, v, probes):
    A=S.A; out=[]
    for (P,Q) in probes:
        out.append(A.summ([A.scale(evaluate(S,sch,P,Q),c) for c,sch in zip(v,schemes) if c]))
    return out

def basis(k):
    D=pickle.load(open(f"sgen_odd_m{k}_n{k}_N{N}.pkl","rb"))
    kernel=[[Fraction(c) for c in v] for v in D["kernel"]]
    S=SSetup(k,k,form="odd")
    Z=zero_subspace(S,schemes,kernel,random.Random(31337))
    ZR,piv=rref(Z)
    reps=[]
    for v in kernel:
        w=list(v)
        for row,c in zip(ZR,piv):
            if w[c]:
                f=w[c]; w=[a-f*b for a,b in zip(w,row)]
        if any(w): reps.append(w)
    # independent representatives
    R,_=rref(reps)
    return S, R, D["dim"]

for k in (2,3):
    S,R,dim=basis(k)
    print(f"k={k}: operator space of dimension {dim}, {len(R)} independent representatives")
    A=S.A; rng=random.Random(555+k)
    probes=[(S.random_field(3,rng,nterms=3),S.random_field(3,rng,nterms=3)) for _ in range(2)]
    # symmetric / antisymmetric split
    sym=anti=0
    for v in R:
        vals=opvec(S,v,probes)
        swapped=opvec(S,v,[(Q,P) for (P,Q) in probes])
        if all(a==b for a,b in zip(vals,swapped)): sym+=1
        elif all(a==A.scale(b,-1) for a,b in zip(vals,swapped)): anti+=1
    print(f"    of the {len(R)} basis vectors: {sym} symmetric, {anti} antisymmetric, "
          f"{len(R)-sym-anti} of mixed type")

# universality: do the k=3 representatives stay invariant at k=2 and k=4?
S3,R3,_=basis(3)
print("\nthe same three combinations, tested in other superdimensions:")
for k in (2,3,4):
    S=SSetup(k,k,form="odd"); A=S.A; pT=op_parity(S,N)
    hams=[p for d in (1,2,3) for p in S.monomials(d)]
    rng=random.Random(88+k)
    good=0; nz=0
    for v in R3:
        P=S.random_field(N+2,rng,nterms=3); Q=S.random_field(N+2,rng,nterms=3)
        def T(X,Y): return A.summ([A.scale(evaluate(S,sch,X,Y),c) for c,sch in zip(v,schemes) if c])
        val=T(P,Q)
        if val: nz+=1
        ok=True
        for f in hams:
            pX=(A.parity(f)+1)%2
            e=-1 if (pT*pX)%2 else 1
            if S.lie(f,val)!=A.scale(A.add(T(S.lie(f,P),Q),T(P,S.lie(f,Q))),e): ok=False; break
        if ok: good+=1
    print(f"  k={k} (sdim ({k}|{k})): {good} of {len(R3)} still invariant, {nz} non-zero")
