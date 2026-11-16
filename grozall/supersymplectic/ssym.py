"""Is the superized Grozman operator (super)commutative on metrics?

Both fields are even, so a symmetric operator satisfies T(A,B) = T(B,A) on the
nose.  The universal coefficient vector of suniv.py is evaluated both ways.
"""
import os, pickle, random, sys
from fractions import Fraction
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from ssuper import SSetup
from sgen import Scheme, enumerate_schemes, evaluate
from suniv import zero_subspace, canonical

N=2
schemes=enumerate_schemes(N)
D=pickle.load(open(f"sgen_even_m1_n3_N{N}.pkl","rb"))
kernel=[[Fraction(c) for c in v] for v in D["kernel"]]
S=SSetup(1,3); rng=random.Random(31337)
v=canonical(kernel, zero_subspace(S,schemes,kernel,rng))

for (m,n) in [(1,0),(1,1),(1,2),(1,3),(2,1),(2,2)]:
    S=SSetup(m,n); A=S.A; rng=random.Random(2024+m*10+n)
    def T(X,Y): return A.summ([A.scale(evaluate(S,sch,X,Y),c) for c,sch in zip(v,schemes) if c])
    same=sym=anti=0
    for _ in range(3):
        P=S.random_field(3,rng,nterms=3); Q=S.random_field(3,rng,nterms=3)
        a,b=T(P,Q),T(Q,P)
        if a==b: sym+=1
        elif a==A.scale(b,-1): anti+=1
        same+=1
    print(f"(2m|n)=({2*m}|{n}): symmetric on {sym}/{same} random pairs, antisymmetric on {anti}/{same}")
