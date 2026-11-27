"""The time reparametrization of the solutions invariant under Xi'_h.

Let tau(t) be arbitrary, s := tau', h := s^(-1/6) (so tau' = h^(-6)) and k := h'/h.  Let
(U, V, P, C) be a solution, in the variables X = s^(1/3) x + 2 s'/(9 s^(2/3)) y^2,
Y = s^(2/3) y, T = tau(t), of the system

    U_Y = 2 U U_X + 2 V_X,   V_Y = 2/3 U_T - 2 (U V)_X - 2 U^2 U_X,
    P_Y = 2 C_X,             C_Y = 2/3 P_T - 2 V P_X.

Then u = s^(1/3) U + s'/(9 s) y, v = s^(2/3) V + s'/(18 s) x + {tau;t} y^2/27,
psi = s^(1/6) P, chi = s^(1/2) C + s'/(9 s^(5/6)) y P solve the reduced system of the
solutions invariant under Xi'_h (printed by invariant.py, second case).  The jets of U, V, P, C
are symbols; their derivatives in Y are replaced by the system above.  The system is linear in
the odd unknowns P, C, so commuting symbols suffice."""
import sympy as sp
x, y, t = sp.symbols('x y t')
s = sp.Function('s')(t)
names = ['U', 'V', 'P', 'C']
J = {n: sp.symbols(f'{n} {n}_X {n}_Y {n}_T') for n in names}
U, V, P, C = (J[n][0] for n in names)
Xv = s**sp.Rational(1, 3)*x + 2*s.diff(t)/(9*s**sp.Rational(2, 3))*y**2
Yv = s**sp.Rational(2, 3)*y
Tt = s                                    # dT/dt = tau' = s; T itself never enters
grad = {'x': (Xv.diff(x), Yv.diff(x), 0), 'y': (Xv.diff(y), Yv.diff(y), 0), 't': (Xv.diff(t), Yv.diff(t), Tt)}
var = {'x': x, 'y': y, 't': t}
def D(e, w):
    """total derivative in w of an expression in x, y, t, s(t) and the zeroth jets of U, V, P, C"""
    r = sp.diff(e, var[w])
    gX, gY, gT = grad[w]
    for n in names:
        f, fX, fY, fT = J[n]
        r += sp.diff(e, f)*(fX*gX + fY*gY + fT*gT)
    return r
# the system for (U, V, P, C), solved for the derivatives in Y
UY = 2*U*J['U'][1] + 2*J['V'][1]
VY = sp.Rational(2, 3)*J['U'][3] - 2*(J['U'][1]*V + U*J['V'][1]) - 2*U**2*J['U'][1]
PY = 2*J['C'][1]
CY = sp.Rational(2, 3)*J['P'][3] - 2*V*J['P'][1]
onY = {J['U'][2]: UY, J['V'][2]: VY, J['P'][2]: PY, J['C'][2]: CY}
sp1, sp2 = s.diff(t), s.diff(t, 2)
schw = sp2/s - sp.Rational(3, 2)*(sp1/s)**2           # {tau;t}, since tau' = s
u = s**sp.Rational(1, 3)*U + sp1/(9*s)*y
v = s**sp.Rational(2, 3)*V + sp1/(18*s)*x + schw*y**2/27
psi = s**sp.Rational(1, 6)*P
chi = s**sp.Rational(1, 2)*C + sp1/(9*s**sp.Rational(5, 6))*y*P
h = s**sp.Rational(-1, 6)
k = sp.simplify(h.diff(t)/h); kp = k.diff(t)
ux, uy, ut, vx, vy = D(u, 'x'), D(u, 'y'), D(u, 't'), D(v, 'x'), D(v, 'y')
px, py, pt, cx, cy = D(psi, 'x'), D(psi, 'y'), D(psi, 't'), D(chi, 'x'), D(chi, 'y')
R = sp.Rational
res = {
    'u_y': uy - (2*u*ux + 2*vx - R(4, 3)*k*y*ux),
    'v_y': vy - (R(2, 3)*ut - 2*(ux*v + u*vx) - 2*u**2*ux + R(4, 3)*k*y*(vx + 2*u*ux)
                 + R(2, 3)*k*(u + x*ux) - R(4, 9)*(k**2 - kp)*y**2*ux),
    'psi_y': py - (2*cx - R(4, 3)*k*y*px),
    'chi_y': cy - (R(2, 3)*pt - 2*v*px + R(4, 3)*k*y*cx + R(2, 3)*k*x*px - R(4, 9)*(k**2 - kp)*y**2*px),
}
ok = True
for nm, e in res.items():
    z = sp.simplify(sp.expand(e.subs(onY)))
    print(f'{nm}: residual = {z}')
    ok &= z == 0
print('the functions solve the reduced system for every tau:', ok)
# control: without the Schwarzian term in v the residual is not zero
v_bad = v - schw*y**2/27
e = D(v_bad, 'y') - (R(2, 3)*ut - 2*(ux*v_bad + u*D(v_bad, 'x')) - 2*u**2*ux + R(4, 3)*k*y*(D(v_bad, 'x') + 2*u*ux)
                     + R(2, 3)*k*(u + x*ux) - R(4, 9)*(k**2 - kp)*y**2*ux)
print('control (v without the Schwarzian term) fails:', sp.simplify(sp.expand(e.subs(onY))) != 0)
