"""H^2(mb(4|5); mb(4|5))_0 -- depth 2."""
import sys, time
from fractions import Fraction
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace
from mb import build, prolong_gen
from run2 import h2_zero

t=time.time()
Sp, gm, g0 = build(1, verbose=False)
comp = prolong_gen(Sp, gm, g0, 1, gm[-1])
print('mb(4|5): sdim', {k:(sum(1 for D in comp[k] if Sp.vf_par(D)==0), sum(1 for D in comp[k] if Sp.vf_par(D)==1)) for k in sorted(comp)})
allowed = {(-2,-2),(-2,-1),(-1,-1),(-2,0),(-1,0),(-2,1),(-1,1),(0,0)}
det     = {(-1,-1),(-2,0),(-1,0),(-2,1)}
degs=[-2,-1,0,1]
triples=[]
for i in degs:
    for j in degs:
        if j<i: continue
        for k in degs:
            if k<j: continue
            if i+j+k not in comp: continue
            if [i,j,k].count(1)>1: continue
            triples.append((i,j,k))
print('triples:', triples)
h2_zero(Sp, comp, allowed, det, triples, degs, 'mb(4|5)')
print('%.1fs'%(time.time()-t))
