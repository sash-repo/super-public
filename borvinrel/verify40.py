"""
Checks of Statements 10.1.11 (item 5) and 12.1.11.

(1) The second relation (73), exactly as it will be printed, is in
    the radical of the multilinear degree of (70), and is independent
    of (71).
(2) The four height-3 walls of the 4x4 family are the four 3-subset sums
    A_ij + A_ik + A_jk of Statement 10.1.1.
(3) The "certain particular values of a" of Statement 12.1.11: scan a.
"""

from fractions import Fraction

import sympy as sp

from shapovalov import bracket, free_lie_superalgebra, phi, perms_of
from fourbyfour import A_SYM, N, a, b, gram_of

out = open('log-verify40.txt', 'w')


def P(s=''):
    print(s)
    out.write(str(s) + '\n')
    out.flush()


def br(i, j, par):
    return bracket({(i,): Fraction(1)}, {(j,): Fraction(1)}, par)


def sym(v):
    return {w: sp.Rational(c.numerator, c.denominator) for w, c in v.items()}


def scale(v, s):
    return {w: sp.expand(s * c) for w, c in v.items()}


def add(*vs):
    o = {}
    for v in vs:
        for w, c in v.items():
            o[w] = sp.expand(o.get(w, sp.Integer(0)) + c)
    return {w: c for w, c in o.items() if c != 0}


# ------------------------------------------------------------------- (1) ---
P('=== (1) the printed second relation, all four generators odd ===')
par = [1, 1, 1, 1]
T1 = sym(bracket(br(0, 1, par), br(2, 3, par), par))
T2 = sym(bracket(br(0, 2, par), br(1, 3, par), par))
T3 = sym(bracket(br(0, 3, par), br(1, 2, par), par))
S1 = sym(bracket(bracket(br(0, 1, par), {(2,): Fraction(1)}, par),
                 {(3,): Fraction(1)}, par))
S2 = sym(bracket(bracket(br(0, 2, par), {(1,): Fraction(1)}, par),
                 {(3,): Fraction(1)}, par))
S3 = sym(bracket(bracket(br(0, 3, par), {(1,): Fraction(1)}, par),
                 {(2,): Fraction(1)}, par))

REL2 = add(scale(T1, a**2 * b**2 * (a * b + a + 1)),
           scale(T3, -(a * b + a - 1) * (a * b + a + 1)),
           scale(S1, -a * b * (a * b - a + 1)),
           scale(S2, a * b * (a * b + a - 1)),
           scale(S3, a * b * (a * b + a + 1)))
REL1 = add(scale(T1, a * b), scale(T2, b), scale(T3, sp.Integer(-1)))

words = perms_of((1, 1, 1, 1), 4)
for name, R in (('rel4x4b  (ab*T1 + b*T2 - T3)', REL1),
                ('rel4x4c  (the second one)', REL2)):
    p = phi(R, A_SYM, N, (1, 1, 1, 1), par)
    bad = [(w, sp.simplify(v)) for w, v in p.items() if sp.simplify(v) != 0]
    P('  %-32s in the radical: %s' % (name, not bad))
    if bad:
        P('      nonzero: %s' % bad[:3])

M = sp.Matrix([[sp.expand(R.get(w, sp.Integer(0))) for w in words]
               for R in (REL1, REL2)])
P('  the two are independent: %s (rank %d)' % (M.rank() == 2, M.rank()))
G = gram_of([T1, T2, T3, S1, S2, S3], A_SYM, par)
P('  radical is exactly 2-dimensional: %s' % (6 - G.rank() == 2))
P('')

# ------------------------------------------------------------------- (2) ---
P('=== (2) the four height-3 walls are the four 3-subset sums ===')
A0 = [[sp.Integer(0), sp.Integer(1), a, -a * b],
      [sp.Integer(1), sp.Integer(0), a * b, -a],
      [a, a * b, sp.Integer(0), sp.Integer(-1)],
      [-a * b, -a, sp.Integer(-1), sp.Integer(0)]]
import itertools
for s in itertools.combinations(range(4), 3):
    tot = sp.expand(sum(A0[i][j] for i, j in itertools.combinations(s, 2)))
    P('  subset %s : A_sum = %s' % (tuple(i + 1 for i in s), sp.factor(tot)))
P('  walls printed by the stratification at height 3:')
P('    ab+a+1, ab+a-1, ab-a+1, ab-a-1   (up to sign, the same four)')
P('')

# ------------------------------------------------------------------- (3) ---
P('=== (3) Statement 12.1.11: scan of a, EVEN reading (which reproduces the table) ===')
av = sp.Symbol('a')
A11 = [[sp.Integer(0), sp.Integer(1), av],
       [sp.Integer(-1), sp.Integer(0), 2 + av],
       [sp.Integer(-1), 2 + sp.Integer(1) / av, sp.Integer(0)]]


def rank_phi(basis, A, n, alpha, p):
    ws = perms_of(alpha, n)
    return sp.Matrix([[sp.expand(phi(u, A, n, alpha, p).get(w, sp.Integer(0)))
                       for w in ws] for u in basis]).rank()


def dims(A, p, L, maxh):
    d = {}
    for k in range(1, maxh + 1):
        d[k] = sum(len(L[k][al]) if k == 1 else rank_phi(L[k][al], A, 3, al, p)
                   for al in sorted(L[k]))
    return d


MAXH = 5
for p, tag in (([0, 0, 0], 'EVEN'), ([1, 1, 1], 'ODD')):
    L = free_lie_superalgebra(3, MAXH, par=p)
    generic = None
    special = []
    vals = [sp.Rational(n_, d_) for d_ in (1, 2, 3) for n_ in range(-9, 10)
            if n_ != 0]
    vals = sorted(set(vals), key=lambda x: (sp.denom(x), x))
    for x in vals:
        A = [[sp.nsimplify(e.subs(av, x)) for e in row] for row in A11]
        d = dims(A, p, L, MAXH)
        t = tuple(d[k] for k in sorted(d))
        if generic is None:
            counts = {}
        counts = None
        special.append((x, t))
    from collections import Counter
    c = Counter(t for _, t in special)
    gen = c.most_common(1)[0][0]
    P('  %s reading: generic profile (deg 1..%d) = %s  [%d of %d values]'
      % (tag, MAXH, list(gen), c[gen], len(special)))
    odd_ones = [(x, list(t)) for x, t in special if t != gen]
    P('     values of a with a SMALLER profile: %s'
      % (odd_ones if odd_ones else 'none in the scan'))
P('')
P('  entries of A that can vanish: A_13 = a  (a=0);  A_23 = 2+a  (a=-2);')
P('                                A_32 = 2+1/a  (a=-1/2).')
out.close()
