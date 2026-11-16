import sys, random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from eng import G, inv
from run1 import sqrtmod, prime_for, mat, SPEC
from run2 import span_rank
H=int(sys.argv[1]) if len(sys.argv)>1 else 16
p=prime_for([21,15,13,6,5])
def mingens(g,H):
    """minimal generators of n_+ even part as Lie algebra: per even multidegree D,
    dim g_D - dim sum_{D1+D2=D, both even height>0} [g_D1,g_D2]"""
    out={}
    degs=[d for d,v in g.dim.items() if v and sum(d)%2==0 and sum(d)<=H]
    for D in sorted(degs,key=lambda d:(sum(d),d)):
        vs=[]
        for D1 in degs:
            D2=tuple(x-y for x,y in zip(D,D1))
            if min(D2)<0 or sum(D2)==0 or sum(D1)==0 or g.dim.get(D2,0)==0: continue
            if sum(D2)%2: continue
            for b1 in range(g.dim[D1]):
                for b2 in range(g.dim[D2]):
                    vs.append(g.br(D1,b1,D2,b2))
        r=len(span_rank(vs,p)) if vs else 0
        if g.dim[D]-r: out[D]=g.dim[D]-r
    return out
if __name__=="__main__":
  for kind in 'ab':
     vals=[('generic',random.Random(5).randrange(2,p))]
     for (u,v,d,w) in SPEC[kind]:
         vals.append(('(%d+%d*sqrt%d)/%d'%(u,v,d,w),(u+v*sqrtmod(d,p))*inv(w,p)%p))
     for name,a in vals:
         g=G(mat(kind,a,p),[1,1,1],p); g.build(H)
         print(kind,name,mingens(g,H),flush=True)
