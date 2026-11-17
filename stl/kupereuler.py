"""Statement 9.4.1: the Lie-Poisson flow on ns(1)^* of H = int(u^2/2 + 2 xi' xi), the bracket of
formula (265) with c_2 = 0, and the cocycle c * int(f' g'' + phi' psi') (odd coefficient 1, the one
for which Jacobi holds; the run also shows the coefficient 1/2 for comparison).  The flow printed
under "odd cocycle coefficient 1" is system (268):
    u_t = 3 u u_x - c u_xxx - 6 xi xi_xx,   xi_t = 3 u xi_x + (3/2) u_x xi - 4 c xi_xxx
(xi = e1 a + e2 b, so xi xi_xx = e1 e2 (a b'' - b a'')).  Run:  python3 kupereuler.py
"""
import sympy as sp
from sympy.calculus.euler import euler_equations
import os, sys; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from nsgrass import *
c,kk=sp.symbols('c k')
u=sp.Function('u')(x); a=sp.Function('a')(x); b=sp.Function('b')(x)
g=sp.Function('g')(x); q=sp.Function('q')(x)
xi=G.g('e1',a)+G.g('e2',b); U=G.c(u)
def vard(expr, fn):
    res=euler_equations(expr,[fn],x); return sp.simplify(res[0].lhs-res[0].rhs) if res and res[0] is not True else 0
def flow(kodd):
    f_=U; phi=-4*xi.dx()   # dH = (u, delta H/delta xi) ; delta int(-2 xi' xi) = int dxi * (-4 xi')? sign fixed below by matching
    psi=G.g('e4',q); G_=G.c(g)
    be = f_*G_.dx()-G_*f_.dx()+G.c(sp.Rational(1,2))*phi*psi
    bo = f_*psi.dx()-G.c(sp.Rational(1,2))*psi*f_.dx()-G_*phi.dx()+G.c(sp.Rational(1,2))*phi*G_.dx()
    pair = U*be + xi*bo + G.c(c)*(f_.dx()*G_.dx(2)+G.c(kodd)*phi.dx()*psi.dx())
    du=G({k:-vard(v,g) for k,v in pair.d.items() if v.has(g)})
    dxi={}
    for k,v in pair.d.items():
        if 'e4' in k:
            i=k.index('e4'); rest=k[:i]+k[i+1:]; sign=(-1)**(len(k)-1-i)
            dxi[rest]=dxi.get(rest,0)-sign*vard(v,q)
    return du, G(dxi)
for kodd in (sp.Rational(1,2),1):
    du,dxi=flow(kodd)
    print('odd cocycle coefficient',kodd)
    print('  u_t  =', sp.expand(du.d.get((),0)), ' + e1e2*(', sp.factor(du.d.get(('e1','e2'),0)),')')
    print('  xi_t(e1-part) =', sp.expand(dxi.d.get(('e1',),0)))
# which sign of delta H/delta xi is the Lie-Poisson one?  test conservation of H itself
def flow2(sgn,kodd=1):
    f_=U; phi=G.c(4*sgn)*xi.dx()
    psi=G.g('e4',q); G_=G.c(g)
    be = f_*G_.dx()-G_*f_.dx()+G.c(sp.Rational(1,2))*phi*psi
    bo = f_*psi.dx()-G.c(sp.Rational(1,2))*psi*f_.dx()-G_*phi.dx()+G.c(sp.Rational(1,2))*phi*G_.dx()
    pair = U*be + xi*bo + G.c(c)*(f_.dx()*G_.dx(2)+G.c(kodd)*phi.dx()*psi.dx())
    du=G({k:-vard(v,g) for k,v in pair.d.items() if v.has(g)})
    dxi={}
    for k,v in pair.d.items():
        if 'e4' in k:
            i=k.index('e4'); rest=k[:i]+k[i+1:]; sign=(-1)**(len(k)-1-i)
            dxi[rest]=dxi.get(rest,0)-sign*vard(v,q)
    return du,G(dxi)
for sgn in (1,-1):
    du,dxi=flow2(sgn)
    dH = U*du - G.c(2)*(dxi.dx()*xi + xi.dx()*dxi)
    tot=True
    for kk_,v in dH.d.items():
        for fn in (u,a,b):
            if v.has(fn):
                for e in euler_equations(sp.expand(v),[fn],x):
                    if e is not True and sp.simplify(e.lhs-e.rhs)!=0: tot=False
    print('phi = %+d*4 xi\': H conserved:'%sgn, tot, ' xi_t:', sp.expand(dxi.d.get(('e1',),0)))
du,dxi=flow2(1)
print('honest flow: u_t =', sp.expand(du.d.get((),0)), '+ e1e2*(', sp.factor(du.d.get(('e1','e2'),0)), ')')
du,dxi=flow2(-1)
dH = U*du + G.c(2)*(dxi.dx()*xi + xi.dx()*dxi)   # H~ = int(u^2/2 + 2 xi' xi)
tot=True
for kk_,v in dH.d.items():
    for fn in (u,a,b):
        if v.has(fn):
            for e in euler_equations(sp.expand(v),[fn],x):
                if e is not True and sp.simplify(e.lhs-e.rhs)!=0: tot=False
print('flow with phi=-4xi\' conserves H~=int(u^2/2+2xi\'xi):', tot)
