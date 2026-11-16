"""k^L(1|4), Subsec. 8.3.24 of the text: the substitutions xi_i<->eta_i and (xi1,eta1)<->(xi2,eta2) are automorphisms of k^L(1|4);
and the sign of [H, xi1 t^k]. Run: python3 k14aut.py"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys, itertools, random
import kL; kL.SIGN=-1; kL.PAIRS=[(0,1),(1,0),(2,3),(3,2)]
from kL import br, m, add, sm
from fractions import Fraction as F
B=lambda f,g: br(f,g,4)
def subst(f,perm):
    o={}
    for (a,S),c in f.items():
        o=add(o,m(a,*[perm[i] for i in S],c=c))
    return o
random.seed(5); bad=0; n=0
def rnd(p):
    f={}
    for a in range(-2,3):
        for r in range(5):
            if r%2!=p: continue
            for S in itertools.combinations(range(4),r): f=add(f,{(a,S):F(random.randint(-3,3))})
    return f
for perm in ([1,0,2,3],[0,1,3,2],[2,3,0,1]):
    for _ in range(30):
        p,q=random.randint(0,1),random.randint(0,1); f,g=rnd(p),rnd(q); n+=1
        if add(subst(B(f,g),perm),sm(-1,B(subst(f,perm),subst(g,perm)))): bad+=1
print(n,"automorphism checks,",bad,"failures")
H=m(-1,0,1,2,3)
for k in (-2,0,3): print("[H,xi1 t^k]",k,B(H,m(k,0)),"  [H,eta1 t^k]",B(H,m(k,1)))
