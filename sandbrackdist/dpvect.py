#!/usr/bin/env python3
r"""
dpvect.py --- vect(m; \un) over F_3 for an ARBITRARY shearing vector \un, and the
weight-blocked Cartan prolongation inside it.

O(m; \un) has the basis of divided-power monomials u_1^{(a_1)} ... u_m^{(a_m)}
with 0 <= a_i < p^{N_i}; u^{(a)}u^{(b)} = binom(a+b, a) u^{(a+b)}, and the
distinguished derivative is d_i(u^{(a)}) = u^{(a-1)}.  A vector field is a dict
{(mon, i): c} for sum c*mon*d_i.  gvect.Vect is the case \un = \One.

The prolongation is done one (degree, weight) block at a time: a bracket is
homogeneous in both, so the linear problem splits, and the blocks stay small even
where dim O(m; \un) does not.

Beware: d_i^3 = d_i^{(3)} as an operator on O(m; \un) whenever N_i >= 2 --- which
is the whole point of the Remark on \un =/= \One.  This module therefore computes
p-th powers as OPERATORS on O(m; \un), not by their values on the coordinates
(the latter do not determine a derivation once N_i >= 2).
"""
import numpy as np
from itertools import product as iproduct

P = 3


def binom_mod(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r % P


class DPVect:
    def __init__(self, degs, wts, heights, names=None):
        self.n = len(degs)
        self.degs = list(degs)
        self.wts = [tuple(w) for w in wts]
        self.h = [P ** N for N in heights]          # exponent bound (exclusive)
        self.heights = list(heights)
        self.names = list(names) if names else ["u%d" % (i + 1)
                                                for i in range(self.n)]
        self.mons = [m for m in iproduct(*[range(b) for b in self.h])]
        self.mdeg = {m: sum(a * d for a, d in zip(m, self.degs))
                     for m in self.mons}
        r = len(self.wts[0])
        self.mwt = {m: tuple(sum(a * w[k] for a, w in zip(m, self.wts))
                             for k in range(r)) for m in self.mons}
        self.mpos = {m: i for i, m in enumerate(self.mons)}

    # ---------------------------------------------------------------- O(m; un)
    def pmul(self, f, g):
        out = {}
        for m1, u in f.items():
            for m2, v in g.items():
                co, m, ok = u * v, [], True
                for a, b, bound in zip(m1, m2, self.h):
                    s = a + b
                    if s >= bound:
                        ok = False
                        break
                    co = co * binom_mod(s, a)
                    if co % P == 0:
                        ok = False
                        break
                    m.append(s)
                if not ok:
                    continue
                co %= P
                if co:
                    m = tuple(m)
                    out[m] = (out.get(m, 0) + co) % P
        return {k: v for k, v in out.items() if v}

    def dpart(self, f, a, times=1):
        """the distinguished derivative d_a applied `times` times, i.e. d_a^{(k)}
        when times = k -- NOT the k-th power of d_a."""
        out = {}
        for m, u in f.items():
            if m[a] < times:
                continue
            k = list(m)
            k[a] -= times
            k = tuple(k)
            out[k] = (out.get(k, 0) + u) % P
        return {k: v for k, v in out.items() if v}

    # ------------------------------------------------------------ vector fields
    def deg(self, key):
        m, i = key
        return self.mdeg[m] - self.degs[i]

    def wt(self, key):
        m, i = key
        return tuple(a - b for a, b in zip(self.mwt[m], self.wts[i]))

    def fdeg(self, D):
        ds = {self.deg(k) for k in D}
        assert len(ds) == 1, "field not homogeneous"
        return ds.pop()

    def fwt(self, D):
        ws = {self.wt(k) for k in D}
        assert len(ws) == 1, "field not weight-homogeneous"
        return ws.pop()

    def bracket(self, D, E):
        out = {}

        def add(m, i, co):
            co %= P
            if co:
                k = (m, i)
                v = (out.get(k, 0) + co) % P
                if v:
                    out[k] = v
                else:
                    out.pop(k, None)

        for (m1, i), c1 in D.items():
            for (m2, j), c2 in E.items():
                c = c1 * c2 % P
                if not c:
                    continue
                for mm, cc in self.pmul({m1: 1}, self.dpart({m2: 1}, i)).items():
                    add(mm, j, c * cc)
                for mm, cc in self.pmul({m2: 1}, self.dpart({m1: 1}, j)).items():
                    add(mm, i, -c * cc)
        return {k: v for k, v in out.items() if v}

    def act(self, D, f):
        out = {}
        for (m, i), c in D.items():
            for mm, cc in self.pmul({m: 1}, self.dpart(f, i)).items():
                v = (out.get(mm, 0) + c * cc) % P
                if v:
                    out[mm] = v
                else:
                    out.pop(mm, None)
        return {k: v for k, v in out.items() if v}

    def power_operator(self, D, k=P):
        """D^k as a linear operator on O(m;\\un), given by its values on the
        monomial basis (a dict mon -> poly)."""
        out = {}
        for m in self.mons:
            f = {m: 1}
            for _ in range(k):
                f = self.act(D, f)
                if not f:
                    break
            if f:
                out[m] = f
        return out

    def as_field(self, op):
        """if the operator `op` is f |-> sum g_i d_i(f) for some g_i, return that
        vector field; else None.  (d_i is the distinguished derivative, so this is
        membership in vect(m;\\un) as opposed to der O(m;\\un).)"""
        # g_i is forced: apply op to u_i
        g = []
        for i in range(self.n):
            e = tuple(1 if t == i else 0 for t in range(self.n))
            g.append(op.get(e, {}))
        D = {}
        for i, gi in enumerate(g):
            for m, c in gi.items():
                if c % P:
                    D[(m, i)] = c % P
        # now check the guess on every monomial
        for m in self.mons:
            want = op.get(m, {})
            got = self.act(D, {m: 1})
            if got != want:
                return None
        return D

    def show(self, D):
        terms = []
        for (m, i), c in sorted(D.items()):
            s = "" if c == 1 else str(c)
            for a, e in zip(self.names, m):
                if e == 1:
                    s += a
                elif e > 1:
                    s += "%s^(%d)" % (a, e)
            terms.append("%sd_%s" % (s, self.names[i]))
        return " + ".join(terms) if terms else "0"

    def basis_keys(self):
        return [(m, i) for m in self.mons for i in range(self.n)]
