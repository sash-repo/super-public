"""Statement 7.1.1: I(u, xi) = int( sqrt(u) - xi' xi / (2 u^{3/2}) ) dx is invariant under the coadjoint
action of k^L(1|1) on the hyperplane c = 0, for the bracket
    [(f,phi),(g,psi)] = (f g' - g f' + (1/2) phi psi,  f psi' - (1/2) psi f' - g phi' + (1/2) phi g')
and the pairing int(u f + xi phi) dx of the Statement.  Over C-points (odd constants e1..e4).
The variation of I along the coadjoint vector field is tested for being a total derivative,
for the two possible signs of each component; the coadjoint action itself is the pair of signs
(1, 1).  Expected: "signs 1 1 variation is a total derivative: True".  Run:  python3 casimir.py
"""
import sympy as sp
from sympy.calculus.euler import euler_equations
import os, sys; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from nsgrass import *
u=sp.Function('u')(x); a=sp.Function('a')(x); b=sp.Function('b')(x); fx=sp.Function('fF')(x); p=sp.Function('p')(x)
xi=G.g('e1',a)+G.g('e2',b); phi=G.g('e3',p); f_=G.c(fx); U=G.c(u)
# bracket [(f,phi),(g,psi)] = (f g'-g f' + 1/2 phi psi, f psi' - 1/2 psi f' - g phi' + 1/2 phi g')
# coadjoint (c=0): <ad*_X F, Y> = -<F,[X,Y]>; components derived by integration by parts:
# even comp:  -( -(u f)' - u f' ... ) -> compute generically below via Euler operator on g and psi
g=sp.Function('g')(x); q=sp.Function('q')(x); psi=G.g('e4',q); G_=G.c(g)
br_even = f_*G_.dx() - G_*f_.dx() + G.c(sp.Rational(1,2))*phi*psi
br_odd  = f_*psi.dx() - G.c(sp.Rational(1,2))*psi*f_.dx() - G_*phi.dx() + G.c(sp.Rational(1,2))*phi*G_.dx()
pair = U*br_even + xi*br_odd   # integrand of <F,[X,Y]>
# extract delta u = coefficient of g (variational), delta xi = coefficient of psi (psi on the right)
def vard(expr, fn):
    res=euler_equations(expr,[fn],x)
    return sp.simplify(res[0].lhs-res[0].rhs) if res and res[0] is not True else 0
du = G({k: -vard(v, g) for k, v in pair.d.items() if v.has(g)})
# psi = e4 q: term  T * e4 q ; move e4 to the right: monomial key contains 'e4'
dxi = {}
for k, v in pair.d.items():
    if 'e4' in k:
        i=k.index('e4'); rest=k[:i]+k[i+1:]; sign=(-1)**(len(k)-1-i)   # move e4 to the end
        dxi[rest]=dxi.get(rest,0) - sign*vard(v, q)
dxi=G(dxi)
for s in (1,-1):
    for s2 in (1,-1):
        DU=G.c(s)*du; DX=G.c(s2)*dxi
        # variation of the integrand sqrt(u) - xi' xi /(2 u^{3/2})
        I_var = DU*G.c(1/(2*sp.sqrt(u))) - (DX.dx()*xi + xi.dx()*DX)*G.c(1/(2*u**sp.Rational(3,2))) \
                + xi.dx()*xi*DU*G.c(sp.Rational(3,4)/u**sp.Rational(5,2))
        good=True
        for k,v in I_var.d.items():
            for fn in (u,a,b,fx,p):
                if v.has(fn):
                    res=euler_equations(sp.expand(v),[fn],x)
                    for e in res:
                        if e is True or e==True: continue
                        if sp.simplify(e.lhs-e.rhs)!=0: good=False
        print('signs',s,s2,'variation is a total derivative:',good)
