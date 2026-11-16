"""
What Radul's Sect. 6 says, checked: the printed S_W(F) of p. 54 IS the
obstruction cocycle of the extension of his own p. 54, its linearisation is the
cocycle of Table (61) on vect^L(1|2), and the kernel of that linearisation
is eq. (97).

Nothing is expanded in a parameter: f, phi, psi are the general even
superdiffeomorphism of S^{1|2}, the six coefficients being arbitrary functions
of y -- so every "ok" below is an identity, not a sample.

Run:  ../venv/bin/python runW.py
"""
import sympy as sp

from schwW import (y, E1, E2, D1, D2, ETA, DETA, mono, add, sub, mul, neg,
                   is_zero, clean, d_y, d_eta, generic_F, berezinian,
                   pullback_theta, radul_SW)

OK = FAIL = 0


def check(name, cond):
    global OK, FAIL
    if cond:
        OK += 1
        print(f'  ok   {name}')
    else:
        FAIL += 1
        print(f'  FAIL {name}')


def with_eps(eps, A):
    return clean({k: eps * c for k, c in A.items()})


f, phi, psi = generic_F()
P = pullback_theta(f, phi, psi)          # Ber * F^*(dx)^{-1} F^*(d z_1) F^*(d z_2)
S = radul_SW(f, phi, psi)                # the formula printed on p. 54

print('1) where the pull-back of  v (x) dx^{-1} d zeta_1 d zeta_2  lands')
basis = {(-1, (D1, D2)), (1, ()), (0, (D1,)), (0, (D2,))}
supp = {(k, tuple(g for g in Sm if g in DETA)) for (k, Sm) in P}
check('its support is the printed basis  dy^{-1} d eta_1 d eta_2, dy, d eta_1, d eta_2',
      supp <= basis)

print('2) the component  pi  sees')
pi_part = clean({key: c for key, c in P.items() if key[0] == -1})
check('pi(F^* Theta) = 1: the pull-back stays in Phi_{-1,0} and pi is Diff-invariant',
      is_zero(sub(pi_part, mono(1, -1, (D1, D2)))))

print('3) the rest IS the printed formula')
rest = clean({key: c for key, c in P.items() if key[0] != -1})
check('F^*(Theta) - Theta = S_W(F) as printed on p. 54', is_zero(sub(rest, S)))

print('4) the cocycle identity on an explicit product')
# G: x = c y, zeta_1 = g11 eta_1 + g12 eta_2, zeta_2 = g21 eta_1 + g22 eta_2
# H: x = h(y), zeta_i = eta_i;   G o H means "apply H first".
c, g11, g12, g21, g22 = sp.symbols('c g11 g12 g21 g22')
h = sp.Function('h')(y)
G = (mono(c * y), add(mono(g11, 0, (E1,)), mono(g12, 0, (E2,))),
     add(mono(g21, 0, (E1,)), mono(g22, 0, (E2,))))
H = (mono(h), mono(1, 0, (E1,)), mono(1, 0, (E2,)))
GH = (mono(c * h), G[1], G[2])


def H_star(A, hfun):
    """Pull-back along  y -> h(y), eta_i -> eta_i  in  v (x) Omega^1:
    u(y) -> u(h),  dy -> h' dy,  d eta_i -> d eta_i,  v -> Ber(H) v = h' v."""
    out = {}
    for (k, Sm), coef in A.items():
        newc = coef.subs(y, hfun).doit()
        if k == 1:
            newc = newc * sp.diff(hfun, y)
        newc = newc * sp.diff(hfun, y)
        out[(k, Sm)] = out.get((k, Sm), 0) + newc
    return clean(out)


SG, SH, SGH = radul_SW(*G), radul_SW(*H), radul_SW(*GH)
check('S_W(G o H) = S_W(H) + H^*(S_W(G))',
      is_zero(sub(SGH, add(SH, H_star(SG, h)))))
check('S_W(H) = 0 for a change of the even variable alone', is_zero(SH))
check('S_W(G) = 0 for a change of the odd ones with constant coefficients',
      is_zero(SG))
# something outside the annihilator: zeta_1 = y eta_1
K = (mono(y), mono(y, 0, (E1,)), mono(1, 0, (E2,)))
check('S_W != 0 as soon as the odd images depend on the even variable',
      not is_zero(radul_SW(*K)))

print('4a) the annihilator at the level of the group')
# the general F whose odd images do not involve the even variable
c11, c12, c21, c22 = sp.symbols('c11 c12 c21 c22')
f_any = add(mono(sp.Function('F0')(y)), mono(sp.Function('F12')(y), 0, (E1, E2)))
p_const = add(mono(c11, 0, (E1,)), mono(c12, 0, (E2,)))
q_const = add(mono(c21, 0, (E1,)), mono(c22, 0, (E2,)))
check('S_W(F) = 0 whenever d_y(phi) = d_y(psi) = 0, the even image being arbitrary',
      is_zero(radul_SW(f_any, p_const, q_const)))

print('5) the linearisation at the identity')
eps = sp.Symbol('eps')
u0, u12 = sp.Function('u0')(y), sp.Function('u12')(y)
v11, v12 = sp.Function('v11')(y), sp.Function('v12')(y)
v21, v22 = sp.Function('v21')(y), sp.Function('v22')(y)
g0 = add(mono(u0), mono(u12, 0, (E1, E2)))              # the d_y-component of X
g1 = add(mono(v11, 0, (E1,)), mono(v12, 0, (E2,)))      # the d_{eta_1}-component
g2 = add(mono(v21, 0, (E1,)), mono(v22, 0, (E2,)))      # the d_{eta_2}-component
fe = add(mono(y), with_eps(eps, g0))
pe = add(mono(1, 0, (E1,)), with_eps(eps, g1))
qe = add(mono(1, 0, (E2,)), with_eps(eps, g2))
Se = radul_SW(fe, pe, qe)
const = clean({key: sp.expand(c.subs(eps, 0)) for key, c in Se.items()})
lin = clean({key: sp.expand(sp.diff(c, eps).subs(eps, 0)) for key, c in Se.items()})
check('S_W(id) = 0', is_zero(const))
expected = add(mul(mono(1, 0, (D1,)), d_y(g2)),
               neg(mul(mono(1, 0, (D2,)), d_y(g1))))
check('the first order is  d eta_1 (d_y g_2) - d eta_2 (d_y g_1)', is_zero(sub(lin, expected)))
check('it does not involve the d_y-component of X at all',
      all(u0 not in sp.sympify(cc).atoms(sp.Function) and
          u12 not in sp.sympify(cc).atoms(sp.Function) for cc in lin.values()))

print('6) the pairing: the cocycle of Table (61)')
# < s(X_1), X_2 > = Res of the eta-free part of the product of the components.
w11, w12 = sp.Function('w11')(y), sp.Function('w12')(y)
w21, w22 = sp.Function('w21')(y), sp.Function('w22')(y)
t1 = add(mono(w11, 0, (E1,)), mono(w12, 0, (E2,)))     # the d_{eta_1}-component of X_2
t2 = add(mono(w21, 0, (E1,)), mono(w22, 0, (E2,)))     # the d_{eta_2}-component of X_2
pairing = sub(mul(d_y(g2), t1), mul(d_y(g1), t2))       # < s(X_1), X_2 >
maint = sub(mul(g1, d_y(t2)), mul(g2, d_y(t1)))        # g_1 g~_2' - g_2 g~_1'
check('the pairing differs from  g_1 g~_2\' - g_2 g~_1\'  by a total derivative,'
      ' which  Res  kills',
      is_zero(sub(sub(pairing, maint),
                  d_y(sub(mul(g2, t1), mul(g1, t2))))))

print('7) the kernel of the linearisation = eq. (97)')
ker, nonker = [], []
for comp in ('x', 'th1', 'th2'):
    for m in range(-2, 3):
        for Sm in ((), (E1,), (E2,), (E1, E2)):
            coef = mono(y ** m, 0, Sm)
            gg1 = coef if comp == 'th1' else {}
            gg2 = coef if comp == 'th2' else {}
            val = add(mul(mono(1, 0, (D1,)), d_y(gg2)),
                      neg(mul(mono(1, 0, (D2,)), d_y(gg1))))
            (ker if is_zero(val) else nonker).append((comp, m, Sm))
inker = set(ker)
check('every  f(x,theta) d_x  is in the kernel',
      all((('x', m, Sm) in inker) for m in range(-2, 3)
          for Sm in ((), (E1,), (E2,), (E1, E2))))
check('a  d_theta-field is in the kernel iff its coefficient is free of x',
      all((((c, m, Sm) in inker) == (m == 0))
          for c in ('th1', 'th2') for m in range(-2, 3)
          for Sm in ((), (E1,), (E2,), (E1, E2))))
print(f'     kernel: {len(ker)} of the {len(ker) + len(nonker)} basis fields tried'
      f' (5 x 4 fields f d_x, and 2 x 4 fields psi(theta) d_theta)')

print('8) and the converse, at the level of the group')
# For the general F, S_W(F) = 0 is a LINEAR system in the four derivatives
# a_1', a_2', b_1', b_2' of the coefficients of phi = a_1 eta_1 + a_2 eta_2,
# psi = b_1 eta_1 + b_2 eta_2 (the d eta_j-components), the dy-component being
# quadratic in them and hence implied.  Its matrix is invertible exactly when
# the odd-odd block of the Jacobian is.
a1, a2, b1, b2 = (sp.Function(n)(y) for n in ('a1', 'a2', 'b1', 'b2'))
A1, A2, B1, B2 = sp.symbols('A1 A2 B1 B2')
Fgen = (add(mono(sp.Function('f0')(y)), mono(sp.Function('f12')(y), 0, (E1, E2))),
        add(mono(a1, 0, (E1,)), mono(a2, 0, (E2,))),
        add(mono(b1, 0, (E1,)), mono(b2, 0, (E2,))))
Sgen = radul_SW(*Fgen)
rep = {sp.diff(a1, y): A1, sp.diff(a2, y): A2,
       sp.diff(b1, y): B1, sp.diff(b2, y): B2}
# every coefficient of S_W, with the derivatives of the odd images named
raw = [sp.expand(sp.together(c).subs(rep)) for c in Sgen.values()]
eqs = [sp.numer(sp.cancel(r)) for r in raw]
lin = [e for e in eqs if sp.Poly(e, A1, A2, B1, B2).total_degree() == 1]
print(f'     {len(eqs)} coefficient equations, {len(lin)} of them linear'
      f' in the four derivatives')
M, _ = sp.linear_eq_to_matrix(lin[:4], [A1, A2, B1, B2])
check('S_W(F) = 0 is linear in the four derivatives and its 4x4 matrix has'
      ' determinant  det(odd-odd block)^2',
      sp.simplify(M.det() - (a1 * b2 - a2 * b1) ** 2) == 0)
check('hence, F being a diffeomorphism, S_W(F) = 0 forces d_y(phi) = d_y(psi) = 0',
      sp.simplify(M.det()) != 0)

print(f'\n{OK} ok, {FAIL} failed')
raise SystemExit(1 if FAIL else 0)
