import sys, random, time
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from eng import G, inv
def sqrtmod(d,p):
    assert p%4==3; r=pow(d%p,(p+1)//4,p); assert r*r%p==d%p; return r
def prime_for(ds, start=10**6):
    p=start
    while True:
        p+=1
        if p%4!=3 or any(p%q==0 for q in range(2,int(p**.5)+1)): continue
        if all(pow(d%p,(p-1)//2,p)==1 for d in ds): return p
def mat(kind,a,p):
    c=2 if kind=='a' else 3
    return [[0,1,a],[-1,0,(c+a)%p],[-1,(c*a+1)*inv(a,p)%p,0]]
SPEC={'a':[(-11,1,21,10),(-8,1,15,7),(-7,1,13,6)],'b':[(-7,2,6,5),(-3,1,5,2)]}
if __name__=='__main__':
  H=int(sys.argv[1]) if len(sys.argv)>1 else 12
  p=prime_for([21,15,13,6,5])
  print("p",p)
  for kind in 'ab':
      vals=[('generic',random.Random(5).randrange(2,p))]
      for (u,v,d,w) in SPEC[kind]:
          vals.append(('(%d+%d*sqrt%d)/%d'%(u,v,d,w),(u+v*sqrtmod(d,p))*inv(w,p)%p))
      for name,a in vals:
          t=time.time(); g=G(mat(kind,a,p),[1,1,1],p); g.build(H)
          print(kind,name,g.totals(H),'%.1fs'%(time.time()-t),flush=True)
