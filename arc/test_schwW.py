"""Calibration of schwW.py: the algebra, the Berezinian, the inverse."""
import sympy as sp

from schwW import (y, E1, E2, D1, D2, mono, add, sub, mul, inv, neg, is_zero,
                   d_y, d_eta, generic_F, pullback_dx, berezinian, ONE, clean)

OK = FAIL = 0


def check(name, cond):
    global OK, FAIL
    if cond:
        OK += 1
        print(f'  ok   {name}')
    else:
        FAIL += 1
        print(f'  FAIL {name}')


print('algebra')
e1 = mono(1, 0, (E1,))
e2 = mono(1, 0, (E2,))
d1 = mono(1, 0, (D1,))
check('eta_1 eta_2 = - eta_2 eta_1', is_zero(add(mul(e1, e2), mul(e2, e1))))
check('eta_1^2 = 0', is_zero(mul(e1, e1)))
check('d eta_1^2 = 0', is_zero(mul(d1, d1)))
check('eta_1 d eta_1 = - d eta_1 eta_1', is_zero(add(mul(e1, d1), mul(d1, e1))))
check('dy is central', is_zero(sub(mul(mono(1, 1), e1), mul(e1, mono(1, 1)))))
check('dy dy^{-1} = 1', is_zero(sub(mul(mono(1, 1), mono(1, -1)), ONE)))

print('derivatives')
g = sp.Function('g')(y)
h = sp.Function('h')(y)
A = add(mono(g), mul(mono(h), mul(e1, e2)))
check('d_y is a derivation',
      is_zero(sub(d_y(mul(A, A)), add(mul(d_y(A), A), mul(A, d_y(A))))))
check('d_eta_1 (eta_1 eta_2) = eta_2', is_zero(sub(d_eta(1, mul(e1, e2)), e2)))
check('d_eta_2 (eta_1 eta_2) = - eta_1', is_zero(add(d_eta(2, mul(e1, e2)), e1)))
check('d_eta_1^2 = 0', is_zero(d_eta(1, d_eta(1, mul(e1, e2)))))

print('inverse')
u = sp.Function('u')(y)
X = add(mono(u, 1), mul(mono(sp.Function('v')(y)), d1))
check('inv(X) X = 1', is_zero(sub(mul(inv(X), X), ONE)))
check('X inv(X) = 1', is_zero(sub(mul(X, inv(X)), ONE)))

print('the Berezinian')
f, phi, psi = generic_F()
Ber = berezinian(f, phi, psi)
# 1) the identity map has Ber = 1
idf = mono(y)
idp = mono(1, 0, (E1,))
idq = mono(1, 0, (E2,))
check('Ber(id) = 1', is_zero(sub(berezinian(idf, idp, idq), ONE)))
# 2) a purely even change  x = F(y),  zeta_i = zeta_i  has Ber = F'
F0 = sp.Function('F0')(y)
check('Ber(x = F(y)) = F\'(y)',
      is_zero(sub(berezinian(mono(F0), idp, idq), mono(sp.diff(F0, y)))))
# 3) a linear change of the odd coordinates has Ber = 1/det
c11, c12, c21, c22 = sp.symbols('c11 c12 c21 c22')
lin_p = add(mono(c11, 0, (E1,)), mono(c12, 0, (E2,)))
lin_q = add(mono(c21, 0, (E1,)), mono(c22, 0, (E2,)))
check('Ber(linear in the etas) = 1/det',
      is_zero(sub(berezinian(idf, lin_p, lin_q),
                  mono(1 / (c11 * c22 - c12 * c21)))))
# 4) Ber is multiplicative on the two commuting changes above
check('Ber of the product of the two = the product of the Bers',
      is_zero(sub(berezinian(mono(F0), lin_p, lin_q),
                  mul(mono(sp.diff(F0, y)), mono(1 / (c11 * c22 - c12 * c21))))))

print('pullback of the differentials')
check('F^*(dx) has the printed components',
      is_zero(sub(pullback_dx(f),
                  add(mul(d_y(f), mono(1, 1)),
                      mul(d_eta(1, f), mono(1, 0, (D1,))),
                      mul(d_eta(2, f), mono(1, 0, (D2,)))))))
check('F^*(dx) is even', all(len(S) % 2 == 0 for (k, S) in pullback_dx(f)))
check('F^*(d zeta_1) is odd', all(len(S) % 2 == 1 for (k, S) in pullback_dx(phi)))
# the inverse printed at the top of p. 54
a = d_y(f)
b1, b2 = d_eta(1, f), d_eta(2, f)
ai = inv(a)
ai2 = mul(ai, ai)
ai3 = mul(ai2, ai)
printed = add(mul(ai, mono(1, -1)),
              neg(mul(mul(ai2, b1), mono(1, -2, (D1,)))),
              neg(mul(mul(ai2, b2), mono(1, -2, (D2,)))),
              neg(mul(mono(2), mul(mul(ai3, mul(b1, b2)), mono(1, -3, (D1, D2))))))
check('the inverse of F^*(dx) printed on p. 54',
      is_zero(sub(inv(pullback_dx(f)), printed)))

print(f'\n{OK} ok, {FAIL} failed')
raise SystemExit(1 if FAIL else 0)
