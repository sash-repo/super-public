#!/usr/bin/env python3
r"""
sumderiv.py -- The questionon eq. equation 227:

    "please write explicitly with more details how this formula is obtained",

  eq. equation 227:   [e_i, e_{-i}] = -i (2 e_0 + sum_{j >= 1} j . z).

The formula is the *formal* (unordered) computation of the bracket in the
oscillator representation.  This script does the same computation exactly, on
the free-boson Fock space, and separates the three statements the text has to
make.

Setting (the case F_0/C.1 of Statement Proposition 7.3, i.e. one free boson, c = 1).
The Fock space is C[a_{-1}, a_{-2}, ...] . |0>, with

    [a_m, a_n] = m delta_{m+n,0},        a_k |0> = 0 for k >= 0,

a state being a multiset of positive integers.  Put (physics normalisation,
annihilators to the right)

    Lp_m = (1/2) sum_k : a_{m-k} a_k :,

so that Lp_0 = sum_{k >= 1} a_{-k} a_k counts the level.  What is checked:

 (1) the honest bracket:   [Lp_m, Lp_n] = (m-n) Lp_{m+n} + ((m^3-m)/12)
     delta_{m+n,0},   i.e. c = 1 -- exactly, on every state of level <= LMAX
     and all |m|, |n| <= MMAX;

 (2) the reordering identity behind eq. equation 227: with the cut-off N,

     Lt_0(N) := (1/2) sum_{|k| <= N} a_{-k} a_k  =  Lp_0 + (1/2) sum_{1<=j<=N} j

     as operators -- exactly, for every N and every state;

 (3) what the formal index shift of the derivation gives:

     [Lp_m, Lp_{-m}] = 2 m Lt_0(N) + ( (m^3-m)/12 - m sum_{1<=j<=N} j ) . id,

     i.e. the shift replaces the true (cubic) anomaly by m . sum_{j>=1} j.
     Hence eq. equation 227 -- read with e_0 the ordered L_0 and z the central element
     -- is exactly the statement "the anomaly is m times the reordering
     constant of L_0"; after zeta-regularisation, (1/2) sum_{j>=1} j =
     (1/2) zeta(-1) = -1/24, so the constant it fixes is the vacuum value
     -c/24 of the symmetrised L_0, NOT the cubic term of eq. equation 148.

In the convention of eq. equation 148, [L_a, L_b] = (b-a) L_{a+b} + ((a^3-a)/12)
delta_{a,-b} z, the paper's generators are the negatives of the physics ones,
L_m = -Lp_m, and (1) reads [L_i, L_{-i}] = -2i L_0 + ((i^3-i)/12) z, while (3)
reads [L_i, L_{-i}] = -i (2 Lt_0) = -i (2 L_0 - sum_{1<=j<=N} j . z), which is
eq. equation 227 up to the sign of z.

Run:  python3 sumderiv.py
"""

from fractions import Fraction as Q
from itertools import combinations_with_replacement

LMAX = 5          # states of level <= LMAX
MMAX = 4          # |m|, |n| <= MMAX


# ------------------------------------------------------------------- states
# A state is a sorted tuple of positive integers (the modes a_{-k} applied to
# |0>); a vector is a dict {state: coefficient}.

def states_upto(level):
    out = [()]
    for L in range(1, level + 1):
        for k in range(1, L + 1):
            pass
        # all multisets of positive integers of total sum L
        def parts(n, mx):
            if n == 0:
                yield ()
                return
            for k in range(min(n, mx), 0, -1):
                for rest in parts(n - k, k):
                    yield (k,) + rest
        for p in parts(L, L):
            out.append(tuple(sorted(p)))
    return out


def vadd(u, v, c=Q(1)):
    out = dict(u)
    for s, x in v.items():
        out[s] = out.get(s, Q(0)) + c * x
        if out[s] == 0:
            del out[s]
    return out


def alpha(k, vec):
    """the mode a_k on a vector; a_0 = 0 (zero momentum)."""
    out = {}
    if k == 0:
        return out
    if k < 0:                                     # creation
        for s, x in vec.items():
            t = tuple(sorted(s + (-k,)))
            out[t] = out.get(t, Q(0)) + x
    else:                                         # annihilation
        for s, x in vec.items():
            nu = s.count(k)
            if nu:
                t = list(s)
                t.remove(k)
                t = tuple(t)
                out[t] = out.get(t, Q(0)) + Q(k * nu) * x
    return {s: x for s, x in out.items() if x}


def Lp(m, vec, N=None):
    """the normal-ordered Lp_m = (1/2) sum_k : a_{m-k} a_k :."""
    if N is None:
        top = max((max(st) for st in vec if st), default=0)
        N = top + abs(m) + 2
    out = {}
    for k in range(-N, N + 1):
        a, b = m - k, k
        if a == 0 or b == 0:
            continue
        # normal order: the creation operator (negative index) acts first
        if b < 0 <= a or (a < 0 and b < 0 and False):
            a, b = b, a
        if a > 0 and b < 0:
            a, b = b, a
        part = alpha(a, alpha(b, vec))
        out = vadd(out, part, Q(1, 2))
    return out


def Lt0(N, vec):
    """the unordered Lt_0(N) = (1/2) sum_{|k| <= N} a_{-k} a_k, as written."""
    out = {}
    for k in range(-N, N + 1):
        if k == 0:
            continue
        out = vadd(out, alpha(-k, alpha(k, vec)), Q(1, 2))
    return out


def level(s):
    return sum(s)


# ------------------------------------------------------------------- checks

def check_virasoro():
    bad = 0
    tested = 0
    for s in states_upto(LMAX):
        vec = {s: Q(1)}
        for m in range(-MMAX, MMAX + 1):
            for n in range(-MMAX, MMAX + 1):
                if m == 0 or n == 0:
                    continue
                lhs = vadd(Lp(m, Lp(n, vec)), Lp(n, Lp(m, vec)), Q(-1))
                rhs = {}
                if m + n != 0:
                    rhs = vadd(rhs, Lp(m + n, vec), Q(m - n))
                else:
                    rhs = vadd(rhs, Lp(0, vec), Q(m - n))
                    rhs = vadd(rhs, vec, Q(m ** 3 - m, 12))
                tested += 1
                if vadd(lhs, rhs, Q(-1)):
                    bad += 1
    return tested, bad


def check_reordering():
    """Lt_0(N) - Lp_0 = (1/2) sum_{1<=j<=N} j on every state."""
    bad = 0
    tested = 0
    for N in range(1, 9):
        const = Q(sum(range(1, N + 1)), 2)
        for s in states_upto(LMAX):
            if s and max(s) > N:
                continue          # the cut-off must cover the modes present
            vec = {s: Q(1)}
            d = vadd(Lt0(N, vec), Lp(0, vec), Q(-1))
            tested += 1
            if vadd(d, vec, -const):
                bad += 1
    return tested, bad


def check_formal_shift():
    """[Lp_m, Lp_{-m}] = 2 m Lt_0(N) + ((m^3-m)/12 - m sum_{1<=j<=N} j) id."""
    bad = 0
    tested = 0
    for N in range(2 * MMAX + LMAX, 2 * MMAX + LMAX + 4):
        S = sum(range(1, N + 1))
        for m in range(1, MMAX + 1):
            for s in states_upto(LMAX):
                if s and max(s) > N:
                    continue
                vec = {s: Q(1)}
                lhs = vadd(Lp(m, Lp(-m, vec)), Lp(-m, Lp(m, vec)), Q(-1))
                rhs = vadd({}, Lt0(N, vec), Q(2 * m))
                rhs = vadd(rhs, vec, Q(m ** 3 - m, 12) - Q(m * S))
                tested += 1
                if vadd(lhs, rhs, Q(-1)):
                    bad += 1
    return tested, bad


def zeta_check():
    """the two regularised constants."""
    print('  (1/2) sum_{j>=1} j  "=" (1/2) zeta(-1) = -1/24 = -c/24 for c = 1')
    print('        sum_{j>=1} j  "=" zeta(-1)       = -1/12')


# --------------------------------------------------------------------------
# The same in the paper's own conventions, on the paper's own engine fock.py:
# for each of the four modules of Statement Proposition 7.3, the operator (150) with
# j = 0 read literally, i.e. e0~ = (1/2) sum_i phi_{i+j} phi_{sigma(i)}, differs
# from the normally ordered e_0 by the reordering constant
#
#     e0~ = e_0 + eps . (1/2) sum_{a in I, a > 0} a . z,
#
# eps = +1 in the two spinor (Clifford) cases and -1 in the two oscillator
# (Weyl) ones, I being the mode set of eq. (178).  The check: e0~ applied to
# the vacuum is the constant, and raising the window W by 1 changes it by
# eps . (1/2) . (the new mode).

def check_reordering_fock():
    import fock
    from fractions import Fraction as F
    one = {((), 0): F(1)}
    out = []
    for c, eps, half in ((fock.CASES[0], 1, True), (fock.CASES[1], 1, False),
                         (fock.CASES[2], -1, False), (fock.CASES[3], -1, True)):
        bad = 0
        incr = []
        prev = None
        for W in range(12, 30):
            v = fock.L(c, 0, one, window=W).get(((), 0), F(0))
            if prev is not None:
                incr.append(v - prev)
            prev = v
        # each increment is eps . (1/2) . (the new mode), the modes advancing
        # by 1, and lying in I: integer resp. 1/2 + integer
        for k, d in enumerate(incr):
            mode = eps * 2 * d
            if mode <= 0:
                bad += 1
                continue
            if half and (mode - F(1, 2)) != int(mode - F(1, 2)):
                bad += 1
            if (not half) and mode != int(mode):
                bad += 1
            if k and incr[k] - incr[k - 1] != eps * F(1, 2):
                bad += 1
        out.append((c.name.strip(), eps, 'half-integer' if half else 'integer',
                    bad, str(incr[0]), str(incr[-1])))
    return out


def report_fock():
    print('=' * 72)
    print('(4) the same identity in the conventions of eq. (150), on fock.py:')
    for name, eps, modes, bad, i0, i1 in check_reordering_fock():
        print('    %-44s eps = %+d, %s modes: %d failures'
              ' (increments %s ... %s)' % (name, eps, modes, bad, i0, i1))
    print('    regularised: sum_{a in Z_{>0}} a = zeta(-1) = -1/12,')
    print('                 sum_{a in 1/2 + Z_{>=0}} a = (2^{-1}-1) zeta(-1) = 1/24')


if __name__ == '__main__':
    print(__doc__.split('Run:')[0])
    print('=' * 72)
    t, b = check_virasoro()
    print('(1) Virasoro with c = 1, normal-ordered: %d triples tested, %d failures'
          % (t, b))
    t, b = check_reordering()
    print('(2) reordering identity Lt_0(N) = Lp_0 + (1/2)sum_{1<=j<=N} j: '
          '%d cases, %d failures' % (t, b))
    t, b = check_formal_shift()
    print('(3) the formal shift, [Lp_m,Lp_{-m}] = 2m Lt_0(N) + '
          '((m^3-m)/12 - m sum_{1<=j<=N} j): %d cases, %d failures' % (t, b))
    report_fock()
    print('=' * 72)
    zeta_check()
    # a control: the naive answer without the cubic term is wrong
    print('  control: the "naive" central term m . sum_{j>=1} j regularises to')
    for m in range(1, 5):
        print('     m = %d:  m . zeta(-1) = %-6s whereas (m^3-m)/12 = %s'
              % (m, Q(-m, 12), Q(m ** 3 - m, 12)))