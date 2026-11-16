"""invhom.py -- is there a periplectic X^-?

Over a symplectic manifold the Hodge--Lepage decomposition rests on TWO
operators: X^+ = om ^ .  and  X^- = the convolution with the bivector dual
to om.  Over an ODD (periplectic) om the first one survives, with X_+^2 = 0.
This file settles the second one, in two independent ways.

 A) By the definition.  Writing om in the basis (d/dx_1..n, d/dxi_1..n) of the
    tangent space, its Gram matrix is  G = antidiag(I,-I)  and the inverse is
    G^{-1} = antidiag(-I,I); the corresponding second-order operator is
        sum_{a,b} (G^{-1})^{ab} iota_a iota_b,   iota_{d/dy} = d/d(dy),
    and over an odd om the two summands CANCEL: half of the iota's are even,
    so the operator is identically ZERO.  Over an even om the same iota's are
    all odd, they anticommute, and the operator is 2 sum_i d/d(dp_i) d/d(dq_i).

 B) By representation theory, which does not depend on any normalization:
        Hom_{pe(n)}(Omega^i, Omega^{i-2}) = 0     in both parities,
    whereas the same space in the other direction is not zero (it contains the
    multiplication by om) and End_{pe(n)}(Omega^i) = C.  So there is no
    invariant lowering operator whatever, not merely no obvious one; the
    filtration I^i <= K^i <= Omega^i of [L3] therefore admits no invariant
    splitting, which is why [L3] prints a graph and not a decomposition.

pe(n) preserves the total form degree, not the bidegree, so the Hom spaces are
computed degree by degree; inside one degree an invariant map preserves the
weight of the diagonal torus of gl(n), and the system splits into small blocks
along it -- that is what makes n = 3 feasible.

Run:  python3 invhom.py
"""
import sys
from fractions import Fraction
from collections import defaultdict

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from periprim import (basis, mono, add, Xp, Xm, pe_basis, act_gen, say, check,
                      CHECKS)
from exactlin import rank as _rank
import periprim
periprim.LOG = open(os.devnull, 'w')


def deg_basis(n, i):
    """the monomials of total form degree i; pe(n) preserves this degree"""
    out = []
    for p in range(min(i, n) + 1):
        out += basis(n, p, i - p)
    return out


def weight(n, m):
    """the weight of the diagonal torus of gl(n): [i in I] - a_i"""
    I, a = m
    return tuple((1 if i in I else 0) - a[i] for i in range(n))


def blocks(n, src, dst):
    """(src, dst) cut into the blocks of one weight"""
    S, D = defaultdict(list), defaultdict(list)
    for m in src:
        S[weight(n, m)].append(m)
    for m in dst:
        D[weight(n, m)].append(m)
    return [(S[w], D[w]) for w in set(S) | set(D) if S[w] and D[w]]


def rho(n, act, par, src, dst):
    """the matrix of L_D from span(src) to span(dst)"""
    idx = dict((m, k) for k, m in enumerate(dst))
    M = [[Fraction(0)] * len(src) for _ in dst]
    for j, m in enumerate(src):
        for mm, c in act_gen(n, act, m, par).items():
            M[idx[mm]][j] = c
    return M


def dim_hom(n, src, dst, parity):
    """dim Hom_{pe(n)}(span src, span dst) in the given parity.

    An invariant map commutes with the diagonal torus of gl(n), hence preserves
    the weight: only the weight-matched entries M[a][b] are unknowns.  The odd
    generators of pe(n) shift the weight, so they couple the blocks; the system
    is therefore set up globally on the reduced set of unknowns."""
    col = {}
    for a, ma in enumerate(dst):
        wa = weight(n, ma)
        for b, mb in enumerate(src):
            if weight(n, mb) == wa:
                col[(a, b)] = len(col)
    if not col:
        return 0
    rows = []
    for par, act in pe_basis(n):
        A = rho(n, act, par, src, src)
        B = rho(n, act, par, dst, dst)
        s = (-1) ** (par * parity)
        for a in range(len(dst)):
            for b in range(len(src)):
                row = {}
                for k in range(len(src)):            # (M L_D)[a][b]
                    if A[k][b] and (a, k) in col:
                        row[col[(a, k)]] = row.get(col[(a, k)], 0) + A[k][b]
                for k in range(len(dst)):            # (L_D M)[a][b]
                    if B[a][k] and (k, b) in col:
                        row[col[(k, b)]] = row.get(col[(k, b)], 0) - s * B[a][k]
                row = dict((c, v) for c, v in row.items() if v)
                if row:
                    dense = [Fraction(0)] * len(col)
                    for c, v in row.items():
                        dense[c] = v
                    rows.append(dense)
    return len(col) - (_rank(rows) if rows else 0)


def gram_argument(nmax=4):
    """A) the dual bivector, written as an operator, is zero over an odd form"""
    say('\nA)  the operator dual to om, from the Gram matrix')
    say('    om = sum dx_i d xi_i:  G = [[0,I],[-I,0]] on (d/dx, d/dxi),')
    say('    G^{-1} = [[0,-I],[I,0]], so the operator is')
    say('      sum_i ( -d/d(dx_i) d/d(dxi_i) + d/d(dxi_i) d/d(dx_i) ).')
    say('    d/d(dxi_i) is EVEN and d/d(dx_i) is ODD, hence they commute and')
    say('    the two summands cancel: the operator is identically 0.')
    say('    Over an EVEN om both are odd, they anticommute, and the operator')
    say('    is -2 sum_i d/d(dp_i) d/d(dq_i) -- the classical X^-.')
    # the same statement as a computation: the two orders agree on every monomial
    ok = True
    for n in range(1, nmax + 1):
        for p in range(n + 1):
            for q in range(4):
                for m in basis(n, p, q):
                    a = Xm(n, mono(m))
                    b = Xm_rev(n, mono(m))
                    if a != b:
                        ok = False
    check('d/d(dx) d/d(dxi) = d/d(dxi) d/d(dx) on every monomial, n<=%d' % nmax, ok)


def Xm_rev(n, f):
    """the same two derivatives in the other order, d/d(dxi) then d/d(dx)"""
    out = {}
    for (I, a), c in f.items():
        for i in range(n):
            if i not in I or a[i] == 0:
                continue
            b = list(a)
            b[i] -= 1
            s = (-1) ** sum(1 for j in I if j < i)
            J = tuple(j for j in I if j != i)
            add(out, (J, tuple(b)), s * a[i] * c)
    return out


def hom_argument(nmax=3, imax=6):
    """B) there is no invariant lowering operator at all"""
    say('\nB)  Hom_{pe(n)}(Omega^i, Omega^{i-2}), both parities')
    for n in range(1, nmax + 1):
        okdown = okup = okend = True
        top = imax if n < 3 else 4
        for i in range(2, top + 1):
            src, dst = deg_basis(n, i), deg_basis(n, i - 2)
            d0 = dim_hom(n, src, dst, 0)
            d1 = dim_hom(n, src, dst, 1)
            u1 = dim_hom(n, dst, src, 1)
            if d0 or d1:
                okdown = False
            if u1 < 1:
                okup = False
            say('    n=%d i=%d: dim Om^i=%d, dim Om^{i-2}=%d;  down: %d (even), '
                '%d (odd);  up: %d' % (n, i, len(src), len(dst), d0, d1, u1))
        for i in range(0, top + 1):
            b = deg_basis(n, i)
            if dim_hom(n, b, b, 0) != 1:
                okend = False
        check('n=%d: Hom_{pe}(Om^i, Om^{i-2}) = 0 for 2<=i<=%d, both parities'
              % (n, top), okdown)
        check('n=%d: Hom_{pe}(Om^{i-2}, Om^i) contains the multiplication by om'
              % n, okup)
        check('n=%d: End_{pe}(Om^i) = C for 0<=i<=%d, so Om^i is indecomposable'
              % (n, top), okend)


if __name__ == '__main__':
    say('invhom.py -- there is no periplectic X^-')
    gram_argument()
    hom_argument()
    say('\n%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
