"""Conformal behaviour of the three harmonics c_5, c_3, c_1 of Statement 3.14.

Metric g = exp(2 u0) |dz|^2.  Spin-s covariant Wirtinger derivatives in the orthonormal frame:
   d eta  = e^{-u0} (d_z eta - s d_z u0 eta),   dbar eta = e^{-u0} (dbar_z eta + s dbar_z u0 eta).
Spins: a,b: 0;  alpha,beta: 2;  conjugates: -2.
Change: u0 -> u0+u,  alpha -> e^{-w u} alpha, beta -> e^{-w u} beta,  a,b -> e^{-2u} a,b
(w = 2: surface-conformal change with II fixed; w = 1: ambient Moebius change, where a,b are
 in addition inhomogeneous -- not modelled, since c_5 does not contain them).
"""
import sympy as sp
x,y,w = sp.symbols('x y w', real=True)
u0,u,u2 = [sp.Function(n)(x,y) for n in ('u0','u','u2')]
a,b = [sp.Function(n)(x,y) for n in ('a','b')]
al = sp.Function('al')(x,y); be = sp.Function('be')(x,y)   # complex-valued
dz  = lambda f: (sp.diff(f,x) - sp.I*sp.diff(f,y))/2
dzb = lambda f: (sp.diff(f,x) + sp.I*sp.diff(f,y))/2
def D(f,s,U):  return sp.exp(-U)*(dz(f)  - s*dz(U)*f)
def Db(f,s,U): return sp.exp(-U)*(dzb(f) + s*dzb(U)*f)
def harmonics(a,al,b,be,U,alb,beb):
    # alb, beb: the conjugate fields (spin -2), passed explicitly
    c5 = sp.Rational(3,4)*sp.I*(al*D(be,2,U) - be*D(al,2,U))
    c3 = sp.Rational(7,4)*sp.I*(al*D(b,0,U) - be*D(a,0,U)) + sp.I/2*(a*D(be,2,U) - b*D(al,2,U))
    c1 = (sp.I/2*(a*D(b,0,U) - b*D(a,0,U)) + sp.Rational(5,4)*sp.I*(al*Db(b,0,U) - be*Db(a,0,U))
          - sp.I/4*(alb*D(be,2,U) - beb*D(al,2,U)))
    return c5,c3,c1
alb = sp.Function('alb')(x,y); beb = sp.Function('beb')(x,y)
c5,c3,c1 = harmonics(a,al,b,be,u0,alb,beb)
# same factor on both surfaces
t5,t3,t1 = harmonics(sp.exp(-2*u)*a, sp.exp(-w*u)*al, sp.exp(-2*u)*b, sp.exp(-w*u)*be, u0+u,
                     sp.exp(-w*u)*alb, sp.exp(-w*u)*beb)
r5 = sp.simplify(t5 - sp.exp(-(2*w+1)*u)*c5)
r3 = sp.simplify(t3 - sp.exp(-(w+3)*u)*c3)
r1 = sp.simplify(t1 - sp.exp(-(w+3)*u)*c1)
print("common conformal factor u on both surfaces, alpha,beta of weight w, a,b of weight 2:")
print("  c5: residual after removing e^{-(2w+1)u} =", r5)
print("  c3: residual after removing e^{-(w+3)u}  =", sp.factor(r3))
print("  c1: residual after removing e^{-(w+3)u}  =", sp.factor(r1))
# different factors on the two surfaces (Moebius with non-constant factor: u at x, u2 at phi(x))
t5b,_,_ = harmonics(a, sp.exp(-w*u)*al, b, sp.exp(-w*u2)*be, u0+u, sp.exp(-w*u)*alb, sp.exp(-w*u2)*beb)
r5b = sp.simplify(t5b - sp.exp(-(w+1)*u - w*u2)*c5)
print("different factors u (visit 1) and u2 (visit 2):")
print("  c5: residual =", sp.factor(r5b))
