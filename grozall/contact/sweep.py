"""The jump locus of dim Hom_{k(3)}(T x T, T) in the weights.

Two independent searches, because either alone can miss something:

  (a) a grid of rational weights, dim computed exactly at each point -- this finds every
      jump whose weights have denominator at most 6 and modulus at most 4;
  (b) the gcd of many random maximal minors of the pencil, factored -- its zero set is
      the codimension-one part of the locus, whatever the denominators, and each factor
      found is then confirmed by evaluating dim at rational points on it.

Nothing is reported as a jump that dim_at has not confirmed at a point.
"""

import sys
from fractions import Fraction as Fr
from affine import pencil, dim_at, at, _minors
from exactlin import rank


def grid(den=6, bound=4):
    return [Fr(n, den) for n in range(-bound * den, bound * den + 1)]


def grid_jumps(V0, N0, N1, N2, gdim, den=6, bound=4):
    out = []
    vals = grid(den, bound)
    for lam in vals:
        for mu in vals:
            d = dim_at(V0, N0, N1, N2, lam, mu)
            if d != gdim:
                out.append((lam, mu, d))
    return out


def minor_gcd(V0, N0, N1, N2, r, ntrials=24, seed=5):
    import sympy as sp
    mins = _minors(V0, N0, N1, N2, r, ntrials, seed)
    if not mins:
        return None, []
    g = sp.gcd_list(mins)
    if g.is_number:
        return sp.Integer(1), []
    return g, [f for f, _ in sp.factor_list(g)[1] if f.free_symbols]


def points_on(factor, ndim=6):
    """A few rational points on the curve factor = 0 (factor is affine or quadratic)."""
    import sympy as sp
    lam, mu = sp.symbols("lam mu")
    out = []
    for v in (Fr(0), Fr(1), Fr(-1), Fr(2), Fr(1, 2), Fr(-3), Fr(5, 3)):
        for free, other in ((lam, mu), (mu, lam)):
            sol = sp.solve(factor.subs(free, sp.Rational(v.numerator, v.denominator)),
                           other)
            for s in sol:
                if s.is_rational:
                    a = Fr(int(sp.Rational(s).p), int(sp.Rational(s).q))
                    out.append((v, a) if free is lam else (a, v))
    return out[:ndim]
