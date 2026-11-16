#!/usr/bin/env python3
"""
Z-gradings of the three exceptional simple finite-dimensional Lie superalgebras
  osp_a(4|2) = D(2,1;a),   ag(2) = G(3),   ab(3) = F(4)
that fit the setting of the Leites--Shchepochkina method, i.e. such that
  (a) the grading is symmetric of depth d <= 3, with no gaps,
  (b) g_{-1} generates g_- (Weisfeiler property),
  (c) g_0 acts irreducibly on every g_{-k}, k>0.

Roots are tuples of rational coordinates in a basis (delta; eps_1, ...);
degree of a root alpha = <alpha, h> (dot product in these coordinates).

All root spaces are 1-dimensional and g_{-k} (k>0) consists of root vectors
only, so a g_0-submodule of g_{-k} is spanned by weight vectors, hence
  g_{-k} is g_0-irreducible  <=>  its root set is connected under adding
  roots of degree 0.
That is the test used for (c).
"""
from fractions import Fraction as F
from itertools import product, combinations
from math import gcd


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- root systems

def roots_D21a():
    """D(2,1;a): even +-2e_i (i=1,2,3); odd +-e_1+-e_2+-e_3.  sdim 9|8."""
    ev, od = [], []
    for i in range(3):
        for s in (2, -2):
            r = [F(0)] * 3
            r[i] = F(s)
            ev.append(tuple(r))
    for s in product((1, -1), repeat=3):
        od.append(tuple(F(x) for x in s))
    return ev, od, 3, "osp_a(4|2) = D(2,1;a)", (9, 8), ("2e1", "2e2", "2e3")


def _g2e(i):
    if i == 0: return (F(0), F(1), F(0))
    if i == 1: return (F(0), F(0), F(1))
    return (F(0), F(-1), F(-1))


def roots_G3():
    """ag(2)=G(3), coords (delta, e_1, e_2), e_3 = -e_1-e_2.
       even: +-2delta  and  g(2)-roots +-e_i, +-(e_i-e_j)
       odd : +-delta, +-delta+-e_i.      sdim 17|14"""
    ev, od = [], []
    ev += [(F(2), F(0), F(0)), (F(-2), F(0), F(0))]
    for i in range(3):
        ei = _g2e(i)
        ev.append(ei); ev.append(tuple(-x for x in ei))
    for i, j in combinations(range(3), 2):
        d = tuple(a - b for a, b in zip(_g2e(i), _g2e(j)))
        ev.append(d); ev.append(tuple(-x for x in d))
    d1 = (F(1), F(0), F(0))
    od += [d1, tuple(-x for x in d1)]
    for i in range(3):
        ei = _g2e(i)
        for sd in (1, -1):
            for se in (1, -1):
                od.append(tuple(sd * a + se * b for a, b in zip(d1, ei)))
    return ev, od, 3, "ag(2) = G(3)", (17, 14), ("delta", "e1", "e2")


def roots_F4():
    """ab(3)=F(4), coords (delta, e_1, e_2, e_3)
       even: +-delta (the sl(2)) and o(7)-roots +-e_i, +-e_i+-e_j
       odd : (+-e_1+-e_2+-e_3+-delta)/2.   sdim 24|16"""
    ev, od = [], []
    ev += [(F(1), F(0), F(0), F(0)), (F(-1), F(0), F(0), F(0))]
    for i in range(3):
        for s in (1, -1):
            r = [F(0)] * 4
            r[i + 1] = F(s)
            ev.append(tuple(r))
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                r = [F(0)] * 4
                r[i + 1] = F(si); r[j + 1] = F(sj)
                ev.append(tuple(r))
    for s in product((1, -1), repeat=4):
        od.append((F(s[3], 2), F(s[0], 2), F(s[1], 2), F(s[2], 2)))
    return ev, od, 4, "ab(3) = F(4)", (24, 16), ("delta", "e1", "e2", "e3")


# ---------------------------------------------------------------- tests

def connected(weights, zero_roots):
    if len(weights) <= 1:
        return True
    ws = set(weights)
    start = next(iter(ws)); seen = {start}; stack = [start]
    while stack:
        w = stack.pop()
        for r in zero_roots:
            v = tuple(a + b for a, b in zip(w, r))
            if v in ws and v not in seen:
                seen.add(v); stack.append(v)
    return seen == ws


def generates(comps, depth):
    """does g_{-1} generate g_{-2}, ..., g_{-d}?  (root-level check)"""
    cur = set(comps[-1][0] + comps[-1][1])
    minus1 = set(cur)
    for k in range(2, depth + 1):
        target = set(comps[-k][0] + comps[-k][1])
        got = set()
        for a in cur:
            for b in minus1:
                s = tuple(x + y for x, y in zip(a, b))
                if s in target:
                    got.add(s)
        if got != target:
            return False
        cur = target
    return True


def analyse(h, ev, od, rank, sdim):
    deg = {}
    for r in ev:
        d = dot(r, h)
        if d.denominator != 1:
            return None
        deg[(r, 0)] = int(d)
    for r in od:
        d = dot(r, h)
        if d.denominator != 1:
            return None
        deg[(r, 1)] = int(d)
    ds = list(deg.values())
    if all(v == 0 for v in ds):
        return None
    depth = -min(ds)
    if depth != max(ds) or depth > 3:
        return None
    g = 0
    for v in ds:
        g = gcd(g, abs(v))
    if g != 1:
        return None
    comps = {}
    for (r, p), v in deg.items():
        comps.setdefault(v, [[], []])[p].append(r)
    for k in range(-depth, depth + 1):
        if k not in comps:
            return None
    zero_roots = comps[0][0] + comps[0][1]
    for k in range(-depth, 0):
        if not connected(comps[k][0] + comps[k][1], zero_roots):
            return None
    if not generates(comps, depth):
        return None
    out = {}
    for k in sorted(comps):
        out[k] = (len(comps[k][0]) + (rank if k == 0 else 0), len(comps[k][1]))
    te = sum(v[0] for v in out.values()); to = sum(v[1] for v in out.values())
    assert (te, to) == sdim
    return {"h": h, "depth": depth, "comps": out, "raw": comps}


def name_g0_even(zero_ev, rank):
    """crude identification of the semisimple type of g_0's even part by
    rank of the span + number of roots."""
    n = len(zero_ev)
    if n == 0:
        return "torus C^%d" % rank
    # rank of the root lattice spanned
    import itertools
    basis = []
    for r in zero_ev:
        cand = basis + [r]
        # gaussian elimination over Q
        m = [list(x) for x in cand]
        rk = 0
        cols = len(m[0])
        row = 0
        for c in range(cols):
            piv = None
            for i in range(row, len(m)):
                if m[i][c] != 0:
                    piv = i; break
            if piv is None:
                continue
            m[row], m[piv] = m[piv], m[row]
            for i in range(len(m)):
                if i != row and m[i][c] != 0:
                    f = m[i][c] / m[row][c]
                    m[i] = [a - f * b for a, b in zip(m[i], m[row])]
            row += 1
        rk = row
        if rk > len(basis):
            basis = cand
    ss = len(basis)
    table = {(2, 1): "sl(2)", (4, 2): "sl(2)+sl(2)", (6, 2): "sl(3) or so(4)+?",
             (8, 2): "sp(4)=o(5)", (12, 2): "g(2)", (6, 3): "sl(2)^3",
             (18, 3): "o(7) or sp(6)", (12, 3): "sl(4)=o(6)? / sl(2)^3+?",
             (10, 3): "?", (2, 1): "sl(2)"}
    nm = table.get((n, ss), "type with %d roots, ss-rank %d" % (n, ss))
    extra = rank - ss
    return nm + (" + C^%d" % extra if extra else "")


def search(rootfun, box=3):
    ev, od, rank, name, sdim, coords = rootfun()
    print("=" * 78)
    print("%s   sdim %d|%d   coords %s" % (name, sdim[0], sdim[1], coords))
    print("=" * 78)
    vals = [F(k, 2) for k in range(0, 2 * box + 1)]
    found, seen = [], set()
    for h in product(vals, repeat=rank):
        r = analyse(list(h), ev, od, rank, sdim)
        if r:
            key = (tuple(sorted(r["comps"].items())),
                   tuple(sorted(r["raw"][0][0])), tuple(sorted(r["raw"][0][1])))
            if key in seen:
                continue
            seen.add(key); found.append(r)
    for r in sorted(found, key=lambda x: (x["depth"], [str(y) for y in x["h"]])):
        cs = "  ".join("g_%+d = %d|%d" % (k, v[0], v[1])
                       for k, v in sorted(r["comps"].items()))
        print("h = (%s)   depth %d" % (", ".join(str(x) for x in r["h"]), r["depth"]))
        print("    " + cs)
        print("    g_0 even part: %s   (+ %d odd)"
              % (name_g0_even(r["raw"][0][0], rank), len(r["raw"][0][1])))
        print("    g_-1 roots: even %s | odd %s"
              % ([",".join(str(x) for x in t) for t in sorted(r["raw"][-1][0])],
                 [",".join(str(x) for x in t) for t in sorted(r["raw"][-1][1])]))
        print()
    print("total: %d gradings (before identifying outer-automorphism twins)\n" % len(found))
    return found


if __name__ == "__main__":
    search(roots_D21a)
    search(roots_G3)
    search(roots_F4)
