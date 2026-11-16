"""The cases with odd J's / odd h, and Open Problem 7."""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction
from flatmodel import Model, closure, d_operator

# --------------------------------------------------------------------
# 0.  Kaehler, both J and h even: R^2 with J: \hat y_1 -> \hat y_2
# --------------------------------------------------------------------
M = Model([0, 0])
d = d_operator(M)
dc = M.add(M.mul(M.gen('c', 1), M.gen('t', 0)),
           M.smul(-1, M.mul(M.gen('c', 0), M.gen('t', 1))))
E = M.mul(M.gen('c', 0), M.gen('c', 1))
F = M.mul(M.gen('b', 1), M.gen('b', 0))
print('1) p(J)=p(h)=ev, R^2:  <E,F> =', closure(M, [E, F])[1],
      ' ;  <E,F,d,d_c> =', closure(M, [E, F, d, dc])[1],
      ' (expected (4|4) = hei(0|4) +) sl(2))')

# --------------------------------------------------------------------
# 1.  p(J) = p(h) = odd  (item "both J and h odd"): C^{n|n},
#     the 2-form avatar is supported on one parity block only
# --------------------------------------------------------------------
for n in (1, 2, 3):
    P = Model([0] * n + [1] * n)
    d = d_operator(P)
    # avatar supported on the \hat\pi (even fibre) block
    E = P.add(*[P.smul(Fraction(1, 2), P.mul(P.gen('c', n + i), P.gen('c', n + i)))
                for i in range(n)])
    F = P.add(*[P.smul(Fraction(1, 2), P.mul(P.gen('b', n + i), P.gen('b', n + i)))
                for i in range(n)])
    delta = P.br(d, F)
    lap = P.br(d, delta)
    print('2) p(J)=p(h)=od, C^{%d|%d}: <E,F> = %s ; <E,F,d> = %s ; [d,delta] = %s'
          % (n, n, closure(P, [E, F])[1], closure(P, [E, F, d])[1],
             'ZERO' if not lap else 'nonzero'))

# --------------------------------------------------------------------
# 2.  hyper-Kaehler, two odd J's, EVEN h  (Subsec. 3.3.4):
#     periplectic Darboux pair of odd 2-forms on C^{2|2}
#     y1,y2 = \hat q_1,\hat q_2 (odd);  y3,y4 = \hat\pi_1,\hat\pi_2 (even)
# --------------------------------------------------------------------
def periplectic_model(n):
    return Model([0] * n + [1] * n)

Q = periplectic_model(2)
q1, q2, p1, p2 = 0, 1, 2, 3          # indices of q_1,q_2,pi_1,pi_2
c = lambda a: Q.gen('c', a)
b = lambda a: Q.gen('b', a)
d = d_operator(Q)

Ei = Q.add(Q.mul(c(q1), c(p1)), Q.mul(c(q2), c(p2)))          # y1y3 + y2y4
Ej = Q.add(Q.mul(c(q1), c(p2)), Q.smul(-1, Q.mul(c(q2), c(p1))))  # y1y4 - y2y3
Fi = Q.add(Q.mul(b(p1), b(q1)), Q.mul(b(p2), b(q2)))
Fj = Q.add(Q.mul(b(p2), b(q1)), Q.smul(-1, Q.mul(b(p1), b(q2))))

print('3) two odd Js, even h, C^{2|2}: <E_i,E_j,F_i,F_j> =',
      closure(Q, [Ei, Ej, Fi, Fj])[1],
      '; with d =', closure(Q, [Ei, Ej, Fi, Fj, d])[1])

H = Q.br(Ei, Fi)
K = Q.br(Ei, Fj)
print('   [E_i,F_i] - [E_j,F_j] =', 'ZERO' if not Q.add(H, Q.smul(-1, Q.br(Ej, Fj))) else 'nonzero',
      '; [E_i,F_j] + [E_j,F_i] =',
      'ZERO' if not Q.add(K, Q.br(Ej, Fi)) else 'nonzero')
alg = closure(Q, [Ei, Ej, Fi, Fj])[0]
print('   H central in <E,F>? ',
      all(not Q.br(H, a) for a in alg),
      '; K central? ', all(not Q.br(K, a) for a in alg))

# the same on C^{4|4} (n = 4 -- two Darboux pairs)
R = periplectic_model(4)
dR = d_operator(R)
cR = lambda a: R.gen('c', a)
bR = lambda a: R.gen('b', a)
# q_1..q_4 -> 0..3, pi_1..pi_4 -> 4..7 ; pair (1,2) and (3,4)
EiR = R.add(*[R.mul(cR(i), cR(4 + i)) for i in range(4)])
EjR = R.add(R.mul(cR(0), cR(5)), R.smul(-1, R.mul(cR(1), cR(4))),
            R.mul(cR(2), cR(7)), R.smul(-1, R.mul(cR(3), cR(6))))
FiR = R.add(*[R.mul(bR(4 + i), bR(i)) for i in range(4)])
FjR = R.add(R.mul(bR(5), bR(0)), R.smul(-1, R.mul(bR(4), bR(1))),
            R.mul(bR(7), bR(2)), R.smul(-1, R.mul(bR(6), bR(3))))
print('   the same on C^{4|4}: <E,F> =', closure(R, [EiR, EjR, FiR, FjR])[1],
      '; with d =', closure(R, [EiR, EjR, FiR, FjR, dR])[1])

# --------------------------------------------------------------------
# 3.  hyper-Kaehler, two odd J's, ODD h  (item "two odd Js and odd h")
# --------------------------------------------------------------------
S = periplectic_model(2)
c = lambda a: S.gen('c', a)
b = lambda a: S.gen('b', a)
dS = d_operator(S)
half = Fraction(1, 2)
Ei = S.smul(-half, S.add(S.mul(c(2), c(2)), S.mul(c(3), c(3))))    # -(y3^2+y4^2)/2
Ej = S.smul(-1, S.mul(c(0), c(1)))                                 # -y1y2
Ek = S.add(S.mul(c(0), c(3)), S.smul(-1, S.mul(c(1), c(2))))       # y1y4 - y2y3
Eh = S.add(S.mul(c(0), c(2)), S.mul(c(1), c(3)))                   # y1y3 + y2y4
Fi = S.smul(-half, S.add(S.mul(b(2), b(2)), S.mul(b(3), b(3))))
Fj = S.smul(-1, S.mul(b(1), b(0)))
Fk = S.add(S.mul(b(3), b(0)), S.smul(-1, S.mul(b(2), b(1))))
Fh = S.add(S.mul(b(2), b(0)), S.mul(b(3), b(1)))

g0 = closure(S, [Ei, Ej, Ek, Fi, Fj, Fk])
print('4) two odd Js, odd h, C^{2|2}: <E_l,F_l> =', g0[1])
print('   E_h in it? ', not Echelon_test(S, g0[0], Eh) if False else '(see below)')

# is E_h in the closure?
from flatmodel import Echelon
ech = {0: Echelon(), 1: Echelon()}
for a in g0[0]:
    ech[S.parity(a)].add(a)
print('   E_h in <E_l,F_l>? ', not ech[S.parity(Eh)].reduce(Eh)[0],
      '; F_h in it? ', not ech[S.parity(Fh)].reduce(Fh)[0])
print('   with d:', closure(S, [Ei, Ej, Ek, Fi, Fj, Fk, dS], limit=200)[1])
