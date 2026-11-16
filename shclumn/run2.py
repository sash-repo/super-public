"""H^2(g;g)_0 through the generic solver; cross-check of run1.py."""
import sys, time
from fractions import Fraction
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace
from models import build_vle, build_kas, build_vas
from excalg import prolong
from cocyc import Graded, Cocycle, equations, coboundaries

def h2_zero(Sp, comp, allowed, det, triples, phidegs, name):
    G = Graded(Sp, comp)
    res=[]
    for pc in (0,1):
        C = Cocycle(G, allowed, pc)
        if C.n == 0:
            res.append((0,0,0,True)); continue
        rows = equations(G, C, triples)
        ker = nullspace(rows) if rows else [[Fraction(1) if i==j else Fraction(0) for j in range(C.n)] for i in range(C.n)]
        detcols = [C.var[k] for k in C.var if (min(k[0],k[2]),max(k[0],k[2])) in det]
        Zp = rank([[v[c] for c in detcols] for v in ker]) if ker else 0
        cob = coboundaries(G, C, det, phidegs)
        B = rank([[v[c] for c in detcols] for v in cob]) if cob else 0
        ok = all(not any(sum(a*b for a,b in zip(r,v)) for r in rows) for v in cob)
        print('  %-12s parity %d: unknowns %4d  equations %5d  dim Z|det <= %3d  dim B = %3d  B in Z: %s'
              % (name, pc, C.n, len(rows), Zp, B, ok))
        res.append((C.n,Zp,B,ok))
    return res

if __name__ == '__main__':
    allowed = {(-1,-1),(-1,0),(0,0)}
    det = {(-1,0)}
    triples = [(-1,0,0),(0,0,0)]
    for name, builder in (('vle(4|3)',build_vle),('kas(1|6)',build_kas),('vas(4|4)',build_vas)):
        t=time.time()
        Sp,gm,g0,top = builder()
        comp = prolong(Sp,gm,g0,1)
        h2_zero(Sp, comp, allowed, det, triples, [-1,0], name)
        print('   %.1fs'%(time.time()-t))
