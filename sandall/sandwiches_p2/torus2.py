#!/usr/bin/env python3
r"""
torus2.py --- maximal tori and the root decomposition of a restricted Lie
algebra over F_2 given by a bracket table and a squaring, and the isomorphism
search that the decomposition seeds.

Written for the two 26-dimensional algebras

        L1 = ir(9; \One)   and   L2 = Out ir(9; \One) = Der ir / ad ir,

but nothing here knows about them: the input is (BR, SQ, n).

WHY A TORUS.  In characteristic 2 an element t with t^{[2]} = t has
(ad t)^2 = ad_{t^{[2]}} = ad t, so ad t is a PROJECTION and L = ker(ad t) +
im(ad t).  For a set of commuting such t's the projections commute, so L
splits into the joint eigenspaces

        L_chi = {x | [t_i, x] = chi_i x},   chi in Hom(T, F_2) = F_2^r,

a grading of L by the group F_2^r.  Both the multiset of the dim L_chi and the
whole bracket table of the graded algebra are isomorphism invariants once the
torus is maximal, and -- this is the point -- an isomorphism L1 --> L2 carries
a maximal torus onto a maximal torus, so the search for one can be organized
weight space by weight space instead of element by element.

Usage:  python3 torus2.py            (uses the cached tables `algs.pkl`)
"""

import itertools
import pickle
import random
import sys
import time

import lin2
from lin2 import bits, echelon, in_span

random.seed(20260817)


class Alg:
    """A restricted Lie algebra over F_2 in coordinates."""

    def __init__(self, BR, SQ, name):
        self.BR, self.SQ, self.n, self.name = BR, SQ, len(BR), name

    def br(self, x, y):
        out = 0
        for i in bits(x):
            row = self.BR[i]
            for j in bits(y):
                out ^= row[j]
        return out

    def sq(self, x):
        out, bs = 0, list(bits(x))
        for k, i in enumerate(bs):
            out ^= self.SQ[i]
            row = self.BR[i]
            for j in bs[k + 1:]:
                out ^= row[j]
        return out

    def ad(self, x):
        cols = [0] * self.n
        for i in bits(x):
            row = self.BR[i]
            for j in range(self.n):
                cols[j] ^= row[j]
        return cols

    @staticmethod
    def applym(cols, y):
        out = 0
        for j in bits(y):
            out ^= cols[j]
        return out

    def rank_ad(self, x):
        return len(echelon(self.ad(x)))

    def centralizer(self, xs):
        """The subspace commuting with every x in xs."""
        rows = []
        for x in xs:
            cols = self.ad(x)
            for t in range(self.n):
                e = 0
                for j in range(self.n):
                    if cols[j] >> t & 1:
                        e |= 1 << j
                if e:
                    rows.append(e)
        return lin2.solve_null(rows, self.n)


# --------------------------------------------------------------- maximal tori

def toral_elements(A, limit=None):
    """All non-zero x with x^{[2]} = x, by a Gray-code walk (2^n - 1 steps)."""
    n = A.n
    out = []
    cur, prev, subset = 0, 0, []
    for g in range(1, 1 << n):
        gray = g ^ (g >> 1)
        j = (gray ^ prev).bit_length() - 1
        upd = A.SQ[j]
        for i in subset:
            upd ^= A.BR[i][j]
        cur ^= upd
        if j in subset:
            subset.remove(j)
        else:
            subset.append(j)
        prev = gray
        if cur == gray:                       # x^{[2]} = x
            out.append(gray)
            if limit and len(out) >= limit:
                return out
    return out


def torus_in(A, xs):
    """Extend the commuting toral set `xs` to a maximal one, greedily."""
    T = list(xs)
    while True:
        C = A.centralizer(T)                  # the centralizer of the torus
        span = echelon(T)
        cand = None
        # look for a toral element of the centralizer outside the span
        pool = []
        for k in range(1, 1 << len(C)):
            v = 0
            for i in bits(k):
                v ^= C[i]
            if v and not in_span(v, span) and A.sq(v) == v:
                pool.append(v)
                if len(pool) > 4000:
                    break
        if not pool:
            break
        cand = random.choice(pool)
        T.append(cand)
    return echelon(T)


def maximal_torus(A, tries=40):
    """A maximal torus, by random greedy extension repeated `tries` times."""
    best = []
    tor = toral_elements(A, limit=4000)
    for _ in range(tries):
        t0 = random.choice(tor)
        T = torus_in(A, [t0])
        if len(T) > len(best):
            best = T
    return best


def weights(A, T):
    """The weight decomposition L = + L_chi with respect to the torus T.

    Returns a dict chi (an r-bit int) -> echelon basis of L_chi.
    """
    r = len(T)
    parts = {0: [1 << i for i in range(A.n)]}
    for k, t in enumerate(T):
        new = {}
        cols = A.ad(t)
        for chi, sp in parts.items():
            # ad t is a projection on the span; split into kernel and image
            rows = []
            for v in sp:
                rows.append((v, A.br(t, v)))
            ker, im = [], []
            # ker(ad t) inside sp
            eqs = []
            for j, (v, w) in enumerate(rows):
                pass
            # solve: sum c_j v_j with sum c_j w_j = 0
            mat = []
            for t_bit in range(A.n):
                e = 0
                for j, (v, w) in enumerate(rows):
                    if w >> t_bit & 1:
                        e |= 1 << j
                if e:
                    mat.append(e)
            for sol in lin2.solve_null(mat, len(rows)):
                v = 0
                for j in bits(sol):
                    v ^= rows[j][0]
                if v:
                    ker.append(v)
            ker = echelon(ker)
            im = echelon([A.br(t, v) for v in sp])
            assert len(ker) + len(im) == len(sp), 'ad t is not a projection'
            if ker:
                new[chi] = ker
            if im:
                new[chi | (1 << k)] = im
        parts = new
    tot = sum(len(v) for v in parts.values())
    assert tot == A.n, 'the weight spaces do not exhaust L'
    return parts


def profile(A, T, W):
    """The isomorphism data of a weight decomposition: the multiset of the
    dim L_chi together with the bracket pattern (which chi + chi' hit which)."""
    dims = sorted((len(v) for v in W.values()), reverse=True)
    pat = {}
    for c1 in W:
        for c2 in W:
            d = lin2.dim([A.br(u, v) for u in W[c1] for v in W[c2]])
            pat[(c1, c2)] = d
    return dims, pat


def report(A, verbose=True):
    t0 = time.time()
    T = maximal_torus(A)
    W = weights(A, T)
    dims, pat = profile(A, T, W)
    if verbose:
        print('  %-18s dim %d, maximal torus of dim %d' % (A.name, A.n, len(T)))
        print('     weight spaces: %s   (%d in all)'
              % (dims, len(W)))
        zero = W.get(0, [])
        print('     dim L_0 = %d, the torus sits in it: %s'
              % (len(zero), all(in_span(t, echelon(zero)) for t in T)))
        print('     computed in %.1f s' % (time.time() - t0))
    return T, W, dims, pat


def main():
    with open('algs.pkl', 'rb') as f:
        d = pickle.load(f)
    L1 = Alg(d['BR1'], d['SQ1'], 'ir(9;One)')
    L2 = Alg(d['BR2'], d['SQ2'], 'Out ir(9;One)')
    print('=== maximal tori and root decompositions ===')
    for A in (L1, L2):
        report(A)


if __name__ == '__main__':
    main()
