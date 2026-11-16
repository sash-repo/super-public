"""Validation of the engine against the two classical answers of the paper:

  (a) flat hyper-Kaehler C^4, three even J's:  hei(0|8) +) sp(4), sdim (11|8);
  (b) special almost periplectic C^{n|n}:      the closure of {X^+, X^-, d}.
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction
from flatmodel import Model, closure, d_operator, form_to_multiplication, \
    form_to_convolution

# ---------------------------------------------------------------- (a)
M = Model([0, 0, 0, 0])
d = d_operator(M)

wi = [(1, (0, 1)), (1, (2, 3))]
wj = [(1, (0, 2)), (-1, (1, 3))]
wk = [(1, (0, 3)), (1, (1, 2))]

E = [form_to_multiplication(M, w) for w in (wi, wj, wk)]
F = [form_to_convolution(M, w) for w in (wi, wj, wk)]

basis, sd = closure(M, E + F)
print('hyper-Kaehler C^4, {E_l, F_l} only          : sdim', sd)

basis, sd = closure(M, E + F + [d])
print('hyper-Kaehler C^4, {E_l, F_l, d}            : sdim', sd,
      '  (expected (11|8) = hei(0|8) +) sp(4))')

# even part should be sp(4) + centre of hei(0|8)
M2 = Model([0, 0])
d2 = d_operator(M2)
E1 = form_to_multiplication(M2, [(1, (0, 1))])
F1 = form_to_convolution(M2, [(1, (0, 1))])
print('Kaehler C^2 (one even J), {E, F, d}         : sdim',
      closure(M2, [E1, F1, d2])[1], '  (expected (4|4) = hei(0|4) +) sp(2))')

# ---------------------------------------------------------------- (b)
for n in (1, 2, 3):
    # coordinates q_1..q_n (even), pi_1..pi_n (odd)
    P = Model([0] * n + [1] * n)
    d = d_operator(P)
    Xp = P.add(*[P.mul(P.gen('c', i), P.gen('c', n + i)) for i in range(n)])
    Xm = P.add(*[P.mul(P.gen('b', n + i), P.gen('b', i)) for i in range(n)])
    delta = P.br(d, Xm)
    S = P.br(d, delta)
    print('special periplectic C^{%d|%d}: sdim of <X^+,X^-,d> = %s ; '
          '[d,delta] = %s' % (n, n, closure(P, [Xp, Xm, d])[1],
                              'ZERO' if not S else S))
    print('    sdim of <X^+, X^-> = %s ; sdim of <d, delta> = %s'
          % (closure(P, [Xp, Xm])[1], closure(P, [d, delta])[1]))
