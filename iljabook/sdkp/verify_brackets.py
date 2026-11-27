"""Exact checks of the brackets of the odd symmetries Xi_h, Xi'_g (closed formulas in T_f, S_f),
and of the symmetries A_a, B_b, T_f, S_f for arbitrary functions a, b, f of t."""
import sys; sys.path.insert(0,'.')
from general import *
report()
h,g,hp,gp = hk(0),hk(0,'g'),hk(1),hk(1,'g')
def Ex(f,fp):
    p={c:f*V(c,1) for c in COMPS}; p['U31']=p['U31']+fp*F(1,3); return p
def Es(f): return {'U31':f*2,'U40':-f}
def add(*ps):
    r={}
    for c0,p in ps:
        for k,q in p.items(): r[k]=r.get(k,Q())+q*c0
    return r
def eq(a,b): return all((a.get(c,Q())-b.get(c,Q())).iszero() for c in COMPS)
print("[Xi_h,Xi_g] = T_{2hg} - 1/3 S_{(hg)'}:", eq(bracket(Xi('h'),1,Xi('g'),1), add((1,Ex(h*g*2,(hp*g+h*gp)*2)),(F(-1,3),Es(hp*g+h*gp)))))
print("[Xi_h,Xi'_g] = T_{hg} - 1/3 S_{h'g}:", eq(bracket(Xi('h'),1,Xip('g'),1), add((1,Ex(h*g,hp*g+h*gp)),(F(-1,3),Es(hp*g)))))
print("[Xi'_h,Xi'_g] = 0:", eq(bracket(Xip('h'),1,Xip('g'),1), {}))
A={'U10':h,'U30':hp*Q.var(('y',))*F(2,3)}; B={'U30':h}
print('A_a symmetry:', len(condition(A,1))==0, ' B_b symmetry:', len(condition(B,1))==0)
f, fp = hk(0), hk(1)   # f(t) and f'(t), as free jet variables (named h in general.py)
print('T_f symmetry:', len(condition(Ex(f, fp), 0)) == 0, ' S_f symmetry:', len(condition(Es(f), 0)) == 0)
