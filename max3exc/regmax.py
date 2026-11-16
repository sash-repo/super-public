#!/usr/bin/env python3
"""Maximal subalgebras of MAXIMAL RANK (containing a maximal torus h) of
osp_a(4|2), ag(2), ab(3), via closed sets of roots.  Root spaces are 1-dim, so
every subalgebra containing h is h + sum_{S} g_alpha, S closed under the ACTUAL
bracket (pattern of nonzero structure constants of the exact models).
For each symmetric closed S we test maximality: closure(S + beta) = everything
for every root beta outside S.  Non-symmetric maximal S are classified too
(parabolic = S u -S is everything)."""
import sys, os
from fractions import Fraction as F
from itertools import product
HERE = os.path.dirname(os.path.abspath(__file__))
import exc3, d21a


def roots_from_model(name):
    """returns (roots, par, prod) : roots = list of weight tuples, par = parity,
    prod[(i,j)] = k  if [e_i,e_j] has nonzero component on root vector k."""
    if name == "osp":
        c = (F(1), F(2), F(-3))           # generic a
        B = d21a.build(c)
        N = d21a.N
        wt = [None] * N
        for m in range(3):
            for kind, s in (("E", 2), ("H", 0), ("F", -2)):
                v = [0, 0, 0]; v[m] = s
                wt[d21a.sl2(m, kind)] = tuple(v)
        for I in product((0, 1), repeat=3):
            wt[d21a.xi(*I)] = tuple(1 if t == 0 else -1 for t in I)
        par = d21a.PAR
        br = lambda i, j: B[i][j]
    else:
        m = exc3.build_G3() if name == "ag" else exc3.build_F4()
        N = m.N; wt = [tuple(x) for x in m.WT]; par = m.PAR
        def br(i, j):
            v = [F(0)] * N
            for k, c in (m.B.get((i, j)) or {}).items():
                v[k] = c
            return v
    zero = tuple(0 for _ in wt[0])
    R = [i for i in range(N) if wt[i] != zero]
    roots = [wt[i] for i in R]
    assert len(set(roots)) == len(roots), "root space not 1-dim"
    idx = {wt[i]: n for n, i in enumerate(R)}
    prod = {}
    for a, i in enumerate(R):
        for b, j in enumerate(R):
            v = br(i, j)
            s = tuple(x + y for x, y in zip(wt[i], wt[j]))
            if s in idx and v[R[idx[s]]] != 0:
                prod[(a, b)] = idx[s]
            # sanity: nothing else
            for k in range(N):
                if v[k] and wt[k] != s:
                    raise AssertionError("weight mismatch")
    return roots, [par[i] for i in R], prod


class RS:
    def __init__(self, name):
        self.name = name
        self.roots, self.par, self.prod = roots_from_model(name)
        self.n = len(self.roots)
        self.neg = [self.roots.index(tuple(-x for x in r)) for r in self.roots]
        self.out = [[] for _ in range(self.n)]
        for (a, b), k in self.prod.items():
            self.out[a].append((b, k))
        self.ALL = (1 << self.n) - 1

    def closure(self, S):
        changed = True
        while changed:
            changed = False
            for a in range(self.n):
                if S >> a & 1:
                    for b, k in self.out[a]:
                        if S >> b & 1 and not S >> k & 1:
                            S |= 1 << k; changed = True
        return S

    def closed(self, S):
        return self.closure(S) == S

    def is_max(self, S):
        for b in range(self.n):
            if not S >> b & 1 and self.closure(S | 1 << b) != self.ALL:
                return False
        return True

    def sdim(self, S):
        r = len(self.roots[0])
        ev = r + sum(1 for a in range(self.n) if S >> a & 1 and self.par[a] == 0)
        od = sum(1 for a in range(self.n) if S >> a & 1 and self.par[a] == 1)
        return ev, od

    def members(self, S):
        return [a for a in range(self.n) if S >> a & 1]


def symmetric_maximal(rs):
    pairs = sorted({tuple(sorted((a, rs.neg[a]))) for a in range(rs.n)})
    found = []
    for bits in range(1, 1 << len(pairs)):
        S = 0
        for t, (a, b) in enumerate(pairs):
            if bits >> t & 1:
                S |= (1 << a) | (1 << b)
        if S == rs.ALL or not rs.closed(S):
            continue
        if rs.is_max(S):
            found.append(S)
    return found


if __name__ == "__main__":
    for name in sys.argv[1:] or ["osp", "ag", "ab"]:
        rs = RS(name)
        ev = sum(1 for p in rs.par if p == 0); od = rs.n - ev
        print("==", name, "roots:", ev, "even,", od, "odd")
        found = symmetric_maximal(rs)
        print("maximal symmetric closed sets:", len(found))
        for S in found:
            ms = rs.members(S)
            print("  sdim", rs.sdim(S), " even roots:",
                  sorted(rs.roots[a] for a in ms if rs.par[a] == 0))
            print("        odd roots:", sorted(rs.roots[a] for a in ms if rs.par[a] == 1))
