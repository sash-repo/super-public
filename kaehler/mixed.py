r"""The mixed-parity case p(J) != p(h) (Lemma 1.9.1a and Theorem 1.9.3).

On a special almost periplectic C^{n|n}, the operators X^+, X^- and d generate
ba(2) (+) sl(1|1) of sdim (3|5).  Does d_c := J d J^{-1} lie in that closure?
Decided in the flat model for both mixed cases:

    case A:  J odd,  h even   (J: hat q_i <-> hat pi_i, J^2 = kappa)
    case B:  J even, h odd    (J: hat q_1 -> hat q_2 -> -hat q_1, and on hat pi)

Run (from this folder):  ../venv/bin/python mixed.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
from fractions import Fraction


from flatmodel import Model, Echelon, closure, d_operator      # noqa: E402


def periplectic(n):
    """C^{n|n}: indices 0..n-1 = q_i (even), n..2n-1 = pi_i (odd)."""
    return Model([0] * n + [1] * n)


def Xplus(M, n):
    r"""left multiplication by the odd 2-form avatar omega = sum \hat q_i \hat pi_i."""
    return M.add(*[M.mul(M.gen('c', i), M.gen('c', n + i)) for i in range(n)])


def Xminus(M, n):
    """convolution with the dual bivector, eq. (few)."""
    return M.add(*[M.mul(M.gen('b', n + i), M.gen('b', i)) for i in range(n)])


def dc_from_J(M, J):
    r"""d_c = sum_a (J \hat z)_a \partial_{z_a}, the flat reading of J d J^{-1}.

    J is a dict: source fibre index -> list of (coeff, target fibre index),
    i.e. J(\hat z_a) = sum coeff * \hat z_b.
    """
    terms = []
    for a, image in J.items():
        for coeff, b in image:
            terms.append(M.smul(coeff, M.mul(M.gen('c', b), M.gen('t', a))))
    return M.add(*terms)


def J_odd(n, kappa):
    r"""case A: J odd, \hat q_i -> \hat pi_i, \hat pi_i -> kappa \hat q_i."""
    J = {}
    for i in range(n):
        J[i] = [(Fraction(1), n + i)]
        J[n + i] = [(Fraction(kappa), i)]
    return J


def J_even(n):
    r"""case B: J even, J^2 = -1, rotating q's among themselves and pi's likewise.

    Needs n even.  \hat q_{2k} -> \hat q_{2k+1} -> -\hat q_{2k}, same on \hat pi.
    """
    assert n % 2 == 0
    J = {}
    for k in range(0, n, 2):
        J[k] = [(Fraction(1), k + 1)]
        J[k + 1] = [(Fraction(-1), k)]
        J[n + k] = [(Fraction(1), n + k + 1)]
        J[n + k + 1] = [(Fraction(-1), n + k)]
    return J


def echelon_of(M, basis):
    ech = {0: Echelon(), 1: Echelon()}
    for a in basis:
        ech[M.parity(a)].add(a)
    return ech


def member(M, ech, e):
    """is e in the span recorded by ech?"""
    if not e:
        return True
    return not ech[M.parity(e)].reduce(e)[0]


def square(M, e):
    """the square of e:  e^2 = [e,e]/2 if e is even, [e,e]/2 if odd -- report [e,e]."""
    return M.br(e, e)


def report(tag, n, J, kappa=None):
    M = periplectic(n)
    d = d_operator(M)
    Xp, Xm = Xplus(M, n), Xminus(M, n)

    base, sd_base = closure(M, [Xp, Xm, d])
    dc = dc_from_J(M, J)
    ech = echelon_of(M, base)

    with_dc, sd_dc = closure(M, [Xp, Xm, d, dc])

    k = '' if kappa is None else ', kappa=%+d' % kappa
    print('  %s  C^{%d|%d}%s' % (tag, n, n, k))
    print('     <X^+, X^-, d>            = sdim (%d|%d)%s'
          % (sd_base[0], sd_base[1],
             '   [= ba(2) (+) sl(1|1)]' if sd_base == (3, 5) else '   <-- NOT (3|5)'))
    print('     p(d) = %d, p(d_c) = %d, [d_c,d_c] = %s'
          % (M.parity(d), M.parity(dc), 'ZERO' if not square(M, dc) else 'NONZERO'))
    print('     d_c in <X^+, X^-, d>?      %s' % member(M, ech, dc))
    print('     <X^+, X^-, d, d_c>       = sdim (%d|%d)%s'
          % (sd_dc[0], sd_dc[1],
             '   <-- same, so d_c adds nothing'
             if sd_dc == sd_base else '   <-- LARGER than (3|5)'))
    # what the extra even/odd elements are, if any
    if sd_dc != sd_base:
        extra = [a for a in with_dc if not member(M, ech, a)]
        print('     new elements: %d (parities %s)'
              % (len(extra), sorted(set(M.parity(a) for a in extra))))
    print('     [d, d_c] = %s ; [X^+, d_c] = %s ; [X^-, d_c] = %s'
          % tuple('ZERO' if not x else 'nonzero'
                  for x in (M.br(d, dc), M.br(Xp, dc), M.br(Xm, dc))))
    return sd_base, sd_dc, member(M, ech, dc)


if __name__ == '__main__':
    print(__doc__.splitlines()[0])
    print()
    print('Lemma 1.9.1a and Theorem 1.9.3, case A: J odd, h even')
    resA = [report('A', n, J_odd(n, kap), kap)
            for n in (1, 2, 3) for kap in (1, -1)]
    print()
    print('Lemma 1.9.1a and Theorem 1.9.3, case B: J even, h odd')
    resB = [report('B', n, J_even(n)) for n in (2, 4)]
    print()
    print('VERDICT')
    allres = resA + resB
    if all(base == dc and inside for base, dc, inside in allres):
        print('  d_c adds nothing in either mixed case: the closure of {X^+, X^-, d, d_c}')
        print('  is that of {X^+, X^-, d}.')
    else:
        print('  d_c enlarges the closure of {X^+, X^-, d}:')
        for tag, (base, dc, inside) in zip(
                ['A n=1 k=+1', 'A n=1 k=-1', 'A n=2 k=+1', 'A n=2 k=-1',
                 'A n=3 k=+1', 'A n=3 k=-1', 'B n=2', 'B n=4'], allres):
            print('    %-12s (%d|%d) -> (%d|%d)%s'
                  % (tag, base[0], base[1], dc[0], dc[1],
                     '' if base == dc else '   d_c is new'))
