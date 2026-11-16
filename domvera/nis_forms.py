#!/usr/bin/env python3
"""Statement 8.3.2 of the text: the general invariant bilinear form on k^L(1|n)/k^M(1|n), all pieces at once.
Invariance under the grading element K_t forces deg x + deg y = 0 (deg = m - 1 + |S|/2);
invariance under o(n) (resp. o(n-1) on theta_1.. for Ramond, theta_0 = zeta twisted) restricts the
support to Hodge pairs (S',T' complementary) and metric pairs (S' = T'), coefficient free in
(type, m, 0 in S, 0 in T, |S'|).  Equations: B([x,y],z) = B(x,[y,z]) for y of degree in
[-1, 3/2] (a generating set), x, z in the window.  Prints the nullity, split by parity."""
import sys, os
from fractions import Fraction as Fr
from functools import lru_cache
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from nis import kL, _sort_theta
from exactlin import nullspace
deg = lambda b: b[0] - 1 + Fr(len(b[1]), 2)
def run(n, ramond, M, parity):
    alg = kL(n, M, ramond=ramond); bas = alg.basis
    first = 1 if ramond else 0
    rest = set(range(first, n)); idx = {}
    @lru_cache(None)
    def br(x, y):
        r = alg.bracket(x, y); return None if r is None else tuple(r.items())
    def term(x, y):
        (m, S), (k, T) = x, y
        if deg(x) + deg(y) != 0 or (len(S) + len(T)) % 2 != parity: return None
        Sp, Tp = [i for i in S if i >= first], [i for i in T if i >= first]
        z0 = (0 in S, 0 in T) if ramond else ()
        if not set(Sp) & set(Tp) and set(Sp) | set(Tp) == rest:
            key = ('H', m) + z0 + (len(Sp),); _, sg = _sort_theta(tuple(Sp) + tuple(Tp))
        elif Sp == Tp:
            key = ('G', m) + z0 + (len(Sp),); sg = 1
        else: return None
        if key not in idx: idx[key] = len(idx)
        return idx[key], sg
    gens = [b for b in bas if -1 <= deg(b) <= Fr(3, 2)]
    bydeg = {}
    for b in bas: bydeg.setdefault(deg(b), []).append(b)
    rows = set()
    for y in gens:
        for z in bas:
            yz = br(y, z)
            if yz is None: continue
            for x in bydeg.get(-deg(y) - deg(z), []):
                xy = br(x, y)
                if xy is None: continue
                row = {}
                for w, c in xy:
                    b = term(w, z)
                    if b: row[b[0]] = row.get(b[0], 0) + c * b[1]
                for w, c in yz:
                    b = term(x, w)
                    if b: row[b[0]] = row.get(b[0], 0) - c * b[1]
                row = tuple(sorted((k, v) for k, v in row.items() if v))
                if row: rows.add(row)
    nv = len(idx); keys = sorted(idx, key=idx.get)
    ns = nullspace([[dict(r).get(j, 0) for j in range(nv)] for r in rows]) if rows else []
    inner = [v for v in ns if any(v[j] for j in range(nv) if abs(keys[j][1]) <= 1)]
    return keys, len(rows), ns, inner
if __name__ == "__main__":
    cases = [eval(a) for a in sys.argv[1:]] or [(6, False, 2, 0)]
    for n, ram, M, par in cases:
        keys, nr, ns, inner = run(n, ram, M, par)
        print("%s(1|%d) window %d, %s forms: %d unknowns, %d eqs, nullity %d, nonzero on |m|<=1: %d"
              % ("k^M" if ram else "k^L", n, M, "odd" if par else "even", len(keys), nr, len(ns), len(inner)), flush=True)
        for v in inner[:3]:
            d = {keys[j]: v[j] for j in range(len(keys)) if v[j] and abs(keys[j][1]) <= 1}
            print("    ", sorted(d.items(), key=lambda kv: str(kv[0])), flush=True)
