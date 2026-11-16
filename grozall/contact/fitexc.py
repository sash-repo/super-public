"""Write the exceptional operator invariantly: fit it in the span of named schemes."""
import os, sys
from fractions import Fraction as Fr
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import solve as lsolve, rank
from affine import pencil, ops_at, dim_at
from gen import apply_term, rand_field
from contact import madd, mscal
from named import words, apply_word

terms,V0,N0,N1,N2 = pencil(2,2,0,2,nfield=1)
LAM = Fr(0)
ops = ops_at(terms,V0,N0,N1,N2,LAM,-LAM)
assert len(ops)==1, len(ops)
op = ops[0]

WS = words(2,2,0,2)
print("candidate schemes (%d):"%len(WS), WS)

Fs=[rand_field(2,3,seed=41+i) for i in range(3)]
Gs=[rand_field(2,3,seed=77+i) for i in range(3)]

def opvals(vec):
    out=[]
    for F in Fs:
        for G in Gs:
            acc={}
            for cf,t in zip(vec,terms):
                if cf: acc=madd(acc,mscal(cf,apply_term(t,F,G)))
            out.append({k[:3]: v for k,v in acc.items()})   # k3=0: drop the fiber slot
    return out

def wordvals(w):
    return [apply_word(w,F,G,2,2) for F in Fs for G in Gs]

target = opvals(op)
cand = [wordvals(w) for w in WS]
keys = sorted(set(k for lst in cand+[target] for d in lst for k in d))
def flat(lst):
    return [d.get(k,Fr(0)) for i,d in enumerate(lst) for k in keys]
A = list(zip(*[flat(c) for c in cand]))
A = [list(r) for r in A]
b = flat(target)
print("span of the schemes has rank", rank([list(r) for r in A]), "; dim V_0 =", len(V0))
sol = lsolve(A,b)
print("fit:", "FAILED" if sol is None else
      " + ".join("%s*%s"%(c,"".join(w)) for c,w in zip(sol,WS) if c))

print("--- diagnosis ---")
V0vals=[flat(opvals(v)) for v in V0]
print("value-rank of V_0 :", rank(V0vals))
print("value-rank of words:", rank([list(r) for r in zip(*[flat(c) for c in cand])] ))
Ab=[list(r)+[bb] for r,bb in zip(A,b)]
print("rank A =",rank(A)," rank [A|b] =",rank(Ab))
