"""Statement 9.4.1, a second computation with an independent Grassmann algebra (grass2.py): the
Lie-Poisson flow of H = Res(u^2/2 + 2 xi_x xi) for the bracket (265) with c_2 = 0, c_1 = 1, the
pairing Res(u f + xi phi) + a c, C the central coordinate; xi_k denotes the k-th derivative of xi.
It prints  u_t = (C u_xxx - 3 u u_x) + 6 xi xi_xx,  xi_t = -3 u xi_x - (3/2) u_x xi + 4 C xi_xxx,
i.e. system (268) with t -> -t (the sign of time depends only on the convention ad^*_F = -ad_F^T vs
ad_F^T, here <m_t, G> = <m, [dH, G]>).  Run:  python3 kupereuler2.py
"""
import sympy as sp; from grass2 import *
u=sp.Function('u')(x); g=sp.Function('g')(x)
C=sp.Symbol('C'); h=sp.Rational(1,2)
xi=G.odd('xi'); chi=G.odd('chi')
def extract(E):
    A=G(); B=G()
    for k,v in E.simp().d.items():
        chis=[a for a in k if a[0]=='chi']
        if not chis:
            vv=sp.expand(v)
            for j in range(0,6):
                dv=sp.diff(vv, sp.diff(g,x,j)) if j>0 else sp.diff(vv,g)
                if dv!=0:
                    t=G({k:dv})
                    for _ in range(j): t=-t.dx()
                    A=A+t
        else:
            i=k.index(chis[0]); kk=k[:i]+k[i+1:]
            t=G({kk:(-1)**i*v})
            for _ in range(chis[0][1]): t=-t.dx()
            B=B+t
    return A.simp(),B.simp()
# dH: even part u, odd part = left variational derivative of 2 xi' xi
# compute it: E_H = 2 xi' xi with xi -> xi + chi : linear part
EH=(2*(xi+chi).dx()*(xi+chi))
lin=G({k:v for k,v in EH.d.items() if sum(1 for a in k if a[0]=='chi')==1})
_,dHxi=extract(lin); print('dH/dxi (left) =',dHxi)
f=u; ph=dHxi
gg=g; ps=chi
e=f*sp.diff(gg,x)-sp.diff(f,x)*gg+h*ph*ps
o=f*ps.dx()-h*sp.diff(f,x)*ps-ph.dx()*gg+h*ph*sp.diff(gg,x)
coc=sp.diff(f,x)*sp.diff(gg,x,2)+ph.dx()*ps.dx()
E=u*e+xi*o+C*coc
A,B=extract(E)
# <m_t,G> = int u_t g + xi_t chi = int g u_t - chi xi_t  => u_t=A, xi_t=-B  (convention <ad*_F m,G>=<m,[F,G]>)
print('u_t =',A); print('xi_t =',-B)
