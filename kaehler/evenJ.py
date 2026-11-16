r"""Theorem 1.9.3, the even J.  The omega-compatible even J on the flat periplectic
C^{n|n} with two independent parameters,

    J(hat q_1) = a hat q_2,  J(hat q_2) = -a^{-1} hat q_1,
    J(hat pi_1) = b hat pi_2, J(hat pi_2) = -b^{-1} hat pi_1,

swept over a grid of rationals: which pairs (a,b) are omega-compatible
(J(omega) = kappa*omega), what kappa and J^2 are, and the superdimension of the
closure of {X^+, X^-, d, d_c} for each of them (the hypothesis of Theorem 1.9.3
that J is not scalar on the even part).

Run (from this folder):  ../venv/bin/python evenJ.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
from fractions import Fraction
from itertools import product


from flatmodel import closure, d_operator                      # noqa: E402
from mixed import periplectic, Xplus, Xminus, dc_from_J        # noqa: E402
from mixed2 import compatible                                  # noqa: E402


def J_even_ab(a, b):
    """J even on C^{2|2}, with independent parameters on the q's and the pi's."""
    a, b = Fraction(a), Fraction(b)
    return {0: [(a, 1)], 1: [(-1 / a, 0)],
            2: [(b, 3)], 3: [(-1 / b, 2)]}


def J_square(J):
    """J^2 as a dict index -> list of (coeff, index); returns kappa if J^2=kappa*id."""
    out = {}
    for i, image in J.items():
        acc = {}
        for co, j in image:
            for co2, k in J[j]:
                acc[k] = acc.get(k, Fraction(0)) + co * co2
        out[i] = acc
    scalars = set()
    for i, acc in out.items():
        nz = {k: v for k, v in acc.items() if v}
        if list(nz) != [i]:
            return None
        scalars.add(nz[i])
    return scalars.pop() if len(scalars) == 1 else None


GRID = [1, -1, 2, -2, 3, Fraction(1, 2), Fraction(-1, 3), Fraction(2, 3)]

n = 2
M = periplectic(n)
d, Xp, Xm = d_operator(M), Xplus(M, n), Xminus(M, n)

print('J even on C^{2|2}:  J(q1)=a q2, J(q2)=-a^{-1}q1, J(p1)=b p2, J(p2)=-b^{-1}p1')
print()
print('  a       b       omega-compatible?   kappa    J^2      closure')
print('  ' + '-' * 68)

good, bad, closures = 0, 0, {}
for a, b in product(GRID, repeat=2):
    J = J_even_ab(a, b)
    kappa = compatible(M, n, J)
    if kappa is None:
        bad += 1
        continue
    good += 1
    sq = J_square(J)
    s = closure(M, [Xp, Xm, d, dc_from_J(M, J)])[1]
    closures.setdefault(s, 0)
    closures[s] += 1
    print('  %-7s %-7s %-19s %-8s %-8s (%d|%d)'
          % (a, b, 'yes', kappa, sq, s[0], s[1]))

print()
print('  %d of the %d pairs (a,b) are omega-compatible, %d are not.'
      % (good, good + bad, bad))
print('  closures: %s'
      % ', '.join('(%d|%d) x %d' % (k[0], k[1], v) for k, v in closures.items()))
print()

# the general statement, checked symbolically on the grid:
#   J(omega) = ab * q2 p2 + (ab)^{-1} q1 p1, so J(omega) = kappa*omega iff (ab)^2 = 1.
print('the rule:  J(omega) = ab*(q2 p2) + (ab)^{-1}*(q1 p1), hence J(omega)=kappa*omega')
print('           iff (ab)^2 = 1, i.e. b = a^{-1} (kappa=+1) or b = -a^{-1} (kappa=-1);')
print('           and then J^2 = -id in both cases.')
viol = [(a, b) for a, b in product(GRID, repeat=2)
        if (compatible(M, n, J_even_ab(a, b)) is not None)
        != ((Fraction(a) * Fraction(b)) ** 2 == 1)]
print('  grid check of that rule: %d disagreements' % len(viol))
sq = {J_square(J_even_ab(a, b)) for a, b in product(GRID, repeat=2)
      if compatible(M, n, J_even_ab(a, b)) is not None}
print('  J^2 over the compatible pairs: %s' % sq)
