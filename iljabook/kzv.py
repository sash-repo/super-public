"""The solutions invariant under Xi'_1: with u=phi_x, v=(phi_y-phi_x^2)/2,
(i) the second reduced equation is -2/3 times the dmKP residual R;
(ii) the KZv residual equals 1/2 R_y - (phi_x R)_x, so dmKP implies KZv;
(iii) the first reduced equation holds identically;
(iv) item 3): psi-equation, d/dx, = linearization of KZv at v applied to psi_x."""
import sympy as sp
x, y, t = sp.symbols('x y t'); f = sp.Function('phi')(x, y, t)
d = sp.diff
u = d(f, x); v = (d(f, y) - d(f, x)**2) / 2
R = d(f, x, t) - sp.Rational(3, 4)*d(f, y, 2) - sp.Rational(3, 2)*d(f, y)*d(f, x, 2) + sp.Rational(3, 2)*d(f, x)**2*d(f, x, 2)
e1 = d(u, y) - 2*u*d(u, x) - 2*d(v, x)
e2 = d(v, y) - sp.Rational(2, 3)*d(u, t) + 2*d(u*v, x) + 2*u**2*d(u, x)
K = d(v, x, t) - sp.Rational(3, 4)*d(v, y, 2) - 3*d(v*d(v, x), x)
print('(iii) first equation identically:', sp.simplify(e1) == 0)
print('(i) second eq = -2/3 R:', sp.simplify(e2 + sp.Rational(2, 3)*R) == 0)
print('(ii) KZv residual = R_y/2 - (phi_x R)_x:', sp.simplify(K - (d(R, y)/2 - d(d(f, x)*R, x))) == 0)
psi = sp.Function('psi')(x, y, t); V = sp.Function('V')(x, y, t)
P3 = d(psi, x, t) - sp.Rational(3, 4)*d(psi, y, 2) - 3*d(V*d(psi, x), x)
delta = d(psi, x)
lin = d(delta, x, t) - sp.Rational(3, 4)*d(delta, y, 2) - 3*d(V*delta, x, 2)
print('(iv) d/dx(psi eq) = linearized KZv on psi_x:', sp.simplify(d(P3, x) - lin) == 0)
eps = sp.Symbol('e'); W = sp.Function('W')(x, y, t)
KZ = lambda w: d(w, x, t) - sp.Rational(3, 4)*d(w, y, 2) - 3*d(w*d(w, x), x)
print('    linearization of KZv at V is w_xt - 3/4 w_yy - 3(Vw)_xx:',
      sp.simplify(d(KZ(V + eps*W), eps).subs(eps, 0) - (d(W, x, t) - sp.Rational(3, 4)*d(W, y, 2) - 3*d(V*W, x, 2))) == 0)
print('checks run: 5')
