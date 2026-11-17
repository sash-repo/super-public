"""Statement 5.10.2: for n = 1 the super Miura map U = g_1(V) = -V' - V D(V) is a Poisson map,
(1/2) A o D o A^* = ad^*_{(., a)}(U, -1/2), where A(w) = -w' - w D(V) - V D(w) is its linearization
and A^*(y) = y' - D(V) y - D(V y) its adjoint, and the coadjoint action of ns(1) is the one recalled
before the Statement: ad^*_{(F,a)}(U,c) = -U'F - (3/2) U F' - (1/2) D(U) D(F) + c D^5(F).
V is a general odd superfield over C-points (odd constants e1, e2), h an even or odd test superfield.
Expected: "even h diff zero: True", "odd h diff zero: True".  Run:  python3 miura1.py
"""
import sympy as sp
import os, sys; sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from nsgrass import *
t='th'
DV=lambda F: D(F,t)
Vs = G.g('e1', f[0]) + G.g('e2', f[1]) + G.g(t) * (G.c(f[2]) + G.g('e1') * G.g('e2', f[3]))
Us = -Vs.dx() - Vs*DV(Vs)
A  = lambda w: -w.dx() - w*DV(Vs) - Vs*DV(w)
As = lambda y: y.dx() - DV(Vs)*y - DV(Vs*y)
for name,h in [('even h',G.c(f[4])+G.g(t)*G.g('e3',f[5])),('odd h',G.g('e3',f[4])+G.g(t,f[5]))]:
    lhs = G.c(sp.Rational(1,2))*A(DV(As(h)))
    rhs = -G.c(sp.Rational(1,2))*DV(h.dx(2)) - G.c(sp.Rational(1,2))*Us*h.dx() - (Us*h).dx() - G.c(sp.Rational(1,2))*DV(Us)*DV(h)
    print(name, 'diff zero:', (lhs-rhs).iszero())
    for sg in [1,-1]:
      for s2 in [1,-1]:
        rhs2 = -G.c(sp.Rational(1,2))*DV(h.dx(2)) - G.c(sp.Rational(1,2))*Us*h.dx() - (Us*h).dx() - G.c(sp.Rational(s2,2))*DV(Us)*DV(h)
        if (G.c(sg)*lhs-rhs2).iszero(): print('  matches with overall sign',sg,'and sign of D(U)D term',s2)
