"""Polynomial vector fields on C^{ne|no}; exact arithmetic. Monomial = (even exps tuple, odd sorted tuple)."""
from fractions import Fraction as F
from itertools import combinations
import sympy as sp

class VF:
    def __init__(s, ne, no):
        s.ne, s.no = ne, no
    def mulmono(s, a, b):
        ea, oa = a; eb, ob = b
        if set(oa) & set(ob): return None, 0
        # sign of merging oa+ob into sorted order
        arr = list(oa) + list(ob); sign = 1
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] > arr[j]: sign = -sign
        return (tuple(x+y for x, y in zip(ea, eb)), tuple(sorted(arr))), sign
    def mul(s, f, g):
        r = {}
        for a, ca in f.items():
            for b, cb in g.items():
                m, sg = s.mulmono(a, b)
                if m is None: continue
                r[m] = r.get(m, 0) + sg*ca*cb
        return {k: v for k, v in r.items() if v != 0}
    def der(s, v, f):
        """left derivative d/d(var v); v<ne even, else odd index v-ne"""
        r = {}
        for (e, o), c in f.items():
            if v < s.ne:
                if e[v] == 0: continue
                e2 = list(e); e2[v] -= 1
                m = (tuple(e2), o); r[m] = r.get(m, 0) + c*e[v]
            else:
                i = v - s.ne
                if i not in o: continue
                pos = o.index(i); o2 = tuple(x for x in o if x != i)
                m = (e, o2); r[m] = r.get(m, 0) + c*(-1)**pos
        return {k: v for k, v in r.items() if v != 0}
    def pvar(s, v): return 0 if v < s.ne else 1
    def par(s, X):
        ps = {(len(m[1]) + s.pvar(v)) % 2 for (m, v) in X}
        assert len(ps) <= 1, X
        return ps.pop() if ps else 0
    def apply(s, X, g):
        """X = dict (mono, var)->c ; X(g)"""
        r = {}
        for (m, v), c in X.items():
            d = s.der(v, g)
            if not d: continue
            p = s.mul({m: c}, d)
            for k, val in p.items(): r[k] = r.get(k, 0) + val
        return {k: v for k, v in r.items() if v != 0}
    def coeffs(s, X):
        out = {}
        for (m, v), c in X.items(): out.setdefault(v, {})[m] = c
        return out
    def br(s, X, Y):
        px, py = s.par(X), s.par(Y)
        cx, cy = s.coeffs(X), s.coeffs(Y)
        r = {}
        for w in range(s.ne + s.no):
            a = s.apply(X, cy.get(w, {}))
            b = s.apply(Y, cx.get(w, {}))
            for m, c in a.items(): r[(m, w)] = r.get((m, w), 0) + c
            for m, c in b.items(): r[(m, w)] = r.get((m, w), 0) - (-1)**(px*py)*c
        return {k: v for k, v in r.items() if v != 0}
    def monos(s, deg_even_max, odd_vars=None):
        pass
