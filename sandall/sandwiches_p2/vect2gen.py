#!/usr/bin/env python3
"""vect(3; N) over F_2 for a general shearing vector: the degree-1 component of
the Cartan prolong (V, sl(V))_* and its sl(3)-submodules of dimension 4 and 5.

Monomials are exponent triples with r_i < 2^{N_i}; the divided-power product is
x^(r) x^(s) = binom(r+s, r) x^(r+s).  With N_i = 2 every monomial of degree <= 3
is present, which is all the prolong of degree <= 1 can see, so N = (2,2,2)
covers every shearing vector with all N_i >= 2 (and the case N = 1 is the
sub-case of square-free monomials).
"""
from itertools import product
from math import comb

NV = 3
NMAX = 2                        # exponents < 2^2 = 4


def binom2(a, b):
    return comb(a, b) & 1


def mon_mul(r, s):
    t = tuple(a + b for a, b in zip(r, s))
    if any(x >= (1 << NMAX) for x in t):
        return None
    c = 1
    for a, b in zip(r, s):
        c &= binom2(a + b, a)
    return t if c else None


def d(r, i):
    if r[i] == 0:
        return None
    return tuple(x - 1 if k == i else x for k, x in enumerate(r))


def bracket(f, g):
    out = set()
    for (r, i) in f:
        for (s, j) in g:
            ds = d(s, i)
            if ds is not None:
                m = mon_mul(r, ds)
                if m is not None:
                    out ^= {(m, j)}
            dr = d(r, j)
            if dr is not None:
                m = mon_mul(s, dr)
                if m is not None:
                    out ^= {(m, i)}
    return frozenset(out)


def show(f):
    if not f:
        return "0"
    out = []
    for (r, i) in sorted(f):
        mon = "".join(("x%d" % (k + 1)) + ("^(%d)" % e if e > 1 else "")
                      for k, e in enumerate(r) if e)
        out.append("%s d%d" % (mon or "1", i + 1))
    return " + ".join(out)


def mons(deg):
    return [r for r in product(range(1 << NMAX), repeat=NV) if sum(r) == deg]


G_m1 = [frozenset({(tuple([0] * NV), i)}) for i in range(NV)]
G_0 = [frozenset({(tuple(1 if k == a else 0 for k in range(NV)), b)})
       for a in range(NV) for b in range(NV)]
G_1_basis = [(r, i) for r in mons(2) for i in range(NV)]

SL3 = ([frozenset({(tuple(1 if k == a else 0 for k in range(NV)), b)})
        for a in range(NV) for b in range(NV) if a != b] +
       [frozenset({(tuple(1 if k == 0 else 0 for k in range(NV)), 0),
                   (tuple(1 if k == 1 else 0 for k in range(NV)), 1)}),
        frozenset({(tuple(1 if k == 1 else 0 for k in range(NV)), 1),
                   (tuple(1 if k == 2 else 0 for k in range(NV)), 2)})])

IDX = {t: n for n, t in enumerate(G_1_basis)}


def tov(f, idx=IDX):
    v = 0
    for t in f:
        v |= 1 << idx[t]
    return v


def tofield(v, basis=G_1_basis):
    return frozenset({t for t in basis if (v >> basis.index(t)) & 1})


def rref(vs):
    piv = {}
    for v in vs:
        for p in sorted(piv, reverse=True):
            if (v >> p) & 1:
                v ^= piv[p]
        if v:
            p = v.bit_length() - 1
            for q in list(piv):
                if (piv[q] >> p) & 1:
                    piv[q] ^= v
            piv[p] = v
    return [piv[p] for p in sorted(piv, reverse=True)]


def in_span(v, rows):
    for r in rows:
        if v ^ r < v:
            v ^= r
    return v == 0


def all_elements(rows):
    els = [0]
    for r in rows:
        els += [e ^ r for e in els]
    return els


def trace_free(f):
    """is the degree-0 field f in sl(3)?"""
    tr = 0
    for (r, i) in f:
        if r[i] == 1 and sum(r) == 1:
            tr ^= 1
    return tr == 0


def main():
    print("dim vect(3;N)_1 =", len(G_1_basis))
    # the prolong: D with [D, d_i] in sl(3) for all i
    good = []
    for v in range(1, 1 << len(G_1_basis)):
        f = frozenset({t for k, t in enumerate(G_1_basis) if (v >> k) & 1})
        if all(trace_free(bracket(f, e)) for e in G_m1):
            good.append(v)
    P = rref(good)
    print("dim (V, sl(3))_1 =", len(P))
    print("  basis:", [show(frozenset({t for k, t in enumerate(G_1_basis)
                                       if (r >> k) & 1})) for r in P])
    # sl(3)-action
    def act(v, X):
        f = frozenset({t for k, t in enumerate(G_1_basis) if (v >> k) & 1})
        return tov(bracket(X, f))

    def closure(vs):
        rows = rref([v for v in vs if v])
        queue = list(rows)
        while queue:
            u = queue.pop()
            for X in SL3:
                w = act(u, X)
                if w and not in_span(w, rows):
                    rows = rref(rows + [w])
                    queue.append(w)
        return tuple(rows)

    cyc = set()
    for v in all_elements(P):
        if v:
            cyc.add(closure([v]))
    print("distinct cyclic submodules of (V,sl(3))_1:",
          sorted(len(c) for c in cyc))
    subs = set(cyc) | {()}
    frontier = set(subs)
    while frontier:
        new = set()
        for a in frontier:
            for b in cyc:
                s = tuple(rref(list(a) + list(b)))
                if len(s) <= 6 and s not in subs:
                    new.add(s)
        subs |= new
        frontier = new
    from collections import Counter
    print("submodules of dim <= 6:", dict(Counter(len(s) for s in subs)))
    for dim in (4, 5):
        ws = [s for s in subs if len(s) == dim]
        print("  dim %d: %d submodule(s)" % (dim, len(ws)))
        for s in ws[:3]:
            print("     ", [show(frozenset({t for k, t in enumerate(G_1_basis)
                                            if (r >> k) & 1})) for r in s])


if __name__ == "__main__":
    main()
