# minimal Grassmann-valued differential algebra (used by kupereuler2.py)
import sympy as sp
x=sp.Symbol('x')
def canon(atoms):
    a=list(atoms); sign=1
    for i in range(len(a)):
        for j in range(len(a)-1-i):
            if a[j]>a[j+1]: a[j],a[j+1]=a[j+1],a[j]; sign=-sign
    for i in range(len(a)-1):
        if a[i]==a[i+1]: return None,0
    return tuple(a),sign
class G:
    def __init__(s,d=None): s.d={k:v for k,v in (d or {}).items() if sp.expand(v)!=0}
    @staticmethod
    def even(e): return G({():e})
    @staticmethod
    def odd(name,k=0,c=1): return G({((name,k),):c})
    def __add__(s,o):
        o=o if isinstance(o,G) else G.even(o); r=dict(s.d)
        for k,v in o.d.items(): r[k]=r.get(k,0)+v
        return G(r)
    __radd__=__add__
    def __neg__(s): return G({k:-v for k,v in s.d.items()})
    def __sub__(s,o): return s+(-(o if isinstance(o,G) else G.even(o)))
    def __rsub__(s,o): return (-s)+o
    def __mul__(s,o):
        if not isinstance(o,G): return G({k:v*o for k,v in s.d.items()})
        r={}
        for k1,v1 in s.d.items():
            for k2,v2 in o.d.items():
                k,sg=canon(k1+k2)
                if k is None: continue
                r[k]=r.get(k,0)+sg*v1*v2
        return G(r)
    __rmul__=lambda s,o: s*o
    def dx(s):
        r=G()
        for k,v in s.d.items():
            r=r+G({k:sp.diff(v,x)})
            for i,(n,j) in enumerate(k):
                if n.startswith('th'): continue
                kk=k[:i]+((n,j+1),)+k[i+1:]
                kk2,sg=canon(kk)
                if kk2 is None: continue
                r=r+G({kk2:sg*v})
        return r
    def dtheta(s,name):
        r={}
        for k,v in s.d.items():
            for i,a in enumerate(k):
                if a==(name,0):
                    kk=k[:i]+k[i+1:]; r[kk]=r.get(kk,0)+(-1)**i*v
        return G(r)
    def simp(s): return G({k:sp.expand(v) for k,v in s.d.items()})
    def comp(s,thetas):
        # component: coefficient of theta-monomial, returned as G w/o thetas, theta placed LEFT
        out={}
        for k,v in s.d.items():
            th=tuple(a for a in k if a[0].startswith('th'))
            rest=tuple(a for a in k if not a[0].startswith('th'))
            kk,sg=canon(th+rest)  # move thetas to the left
            out.setdefault(th,{}); out[th][rest]=out[th].get(rest,0)+sg*v
        return {th:G(d) for th,d in out.items()}
    def __repr__(s): return ' + '.join('(%s)*%s'%(sp.factor(v),''.join('%s%d'%a for a in k)) for k,v in s.simp().d.items()) or '0'
def D(F,th): return G.odd(th)*F.dx()+F.dtheta(th)
