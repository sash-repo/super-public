"""Diagnostic: how many of the contraction schemes are non-zero AS OPERATORS?

A dimension 0 for the invariant operators is only meaningful if the ansatz is
alive in the first place.  This measures the dimension of the span of all the
schemes of a given order, invariant or not.

Usage:  python3 sspan.py m n N [form]
"""
import os, random, sys
from fractions import Fraction
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank
from ssuper import SSetup
from sgen import enumerate_schemes, evaluate

m,n,N = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
form = sys.argv[4] if len(sys.argv)>4 else "even"
S=SSetup(m,n,form=form)
schemes=enumerate_schemes(N)
rng=random.Random(8080)
probes=[(S.random_field(N+1,rng,nterms=3),S.random_field(N+1,rng,nterms=3)) for _ in range(2)]
vals=[[evaluate(S,sch,P,Q) for sch in schemes] for (P,Q) in probes]
alive=sum(1 for i in range(len(schemes)) if any(vals[p][i] for p in range(len(probes))))
keys=[sorted(set().union(*[set(x) for x in v])) if any(v) else [] for v in vals]
M=[]
for i in range(len(schemes)):
    row=[]
    for p in range(len(probes)):
        for k in keys[p]: row.append(vals[p][i].get(k,Fraction(0)))
    M.append(row)
r=rank(M) if M and M[0] else 0
lab = f"periplectic k={m}" if form=="odd" else f"(2m|n)=({2*m}|{n})"
print(f"{lab}, order {N}: {len(schemes)} schemes, {alive} non-zero individually, "
      f"span of dimension {r} as operators into the fiber")
