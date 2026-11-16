import sys, random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from eng import G, inv
from run1 import sqrtmod, prime_for, mat, SPEC
import run2
from run3 import mingens
H=int(sys.argv[1]); start=int(sys.argv[2])
p=prime_for([21,15,13,6,5],start=start); run2.p=p
import run3; run3.p=p
print("p",p)
for kind in 'ab':
    for (u,v,d,w) in SPEC[kind]:
        a=(u+v*sqrtmod(d,p))*inv(w,p)%p; name='(%d+%d*sqrt%d)/%d'%(u,v,d,w)
        g,B,S=run2.analyse(kind,name,a)
        # independent g(B): even generators, matrix B (unnormalized: rows scaled ok)
        gb=G(B,[0,0,0],p); gb.build(H//2)
        print("  g(B) by height:",gb.totals(H//2))
        Hs=None
        print("  min generators of g_ev^+ :",mingens(g,H),flush=True)
        # Jacobi spot check
        els=[(D,b) for D,dd in g.dim.items() for b in range(dd) if dd and 2<=sum(D)<=3]
        bad=0;cnt=0
        for x in els:
          for y in els:
            for z in els:
              if sum(x[0])+sum(y[0])+sum(z[0])>H: continue
              def br(u,vec):  # [u, vec] u basis, vec=(D,coords)
                  D,c=vec; E=g.add(u[0],D); o=[0]*g.dim.get(E,0)
                  for bb,cc in enumerate(c):
                      if cc:
                          for t,q in enumerate(g.br(u[0],u[1],D,bb)): o[t]=(o[t]+cc*q)%p
                  return (E,o)
              def bas(u): v=[0]*g.dim[u[0]]; v[u[1]]=1; return (u[0],v)
              l=br(x,br(y,bas(z))); r1=br(x,bas(y)); 
              # [[x,y],z] with sign: [x,[y,z]] = [[x,y],z] + (-1)^{p x p y}[y,[x,z]]
              t1=(0,0)
              D,c=r1; E=g.add(D,z[0]); o=[0]*g.dim.get(E,0)
              for bb,cc in enumerate(c):
                  if cc:
                      for t,q in enumerate(g.br(D,bb,z[0],z[1])): o[t]=(o[t]+cc*q)%p
              sg=-1 if (g.pa(x[0]) and g.pa(y[0])) else 1
              r2=br(y,br(x,bas(z)))[1]
              cnt+=1
              if any((A-B_-sg*C)%p for A,B_,C in zip(l[1],o,r2)): bad+=1
        print("  Jacobi checks",cnt,"bad",bad,flush=True)
