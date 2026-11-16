#!/usr/bin/env python3
"""Independent verification of regrade.py.

Four checks, each written so that it shares as little as possible with the
recursion of regrade.act:

  axioms   the module axiom  x.(y.v) - (-1)^{p(x)p(y)} y.(x.v) = [x,y].v  for
           all x, y in g_0 + g_1 and all v of degree <= 2, which is what makes
           regrade.act an action of a Lie superalgebra at all: a wrong sign or
           a wrong bracket breaks it at once;

  deg1     the degree-1 singular vectors recomputed from the closed formula
           x.(e (x) m) = chi([x,e]) m, which uses no recursion whatsoever;

  deg2     the degree-2 singular vectors recomputed from the closed formula
           x.(e f (x) m) = [[x,e],f] (x) m
                         + (-1)^{p([x,e])p(f)} chi([x,e]) f
                         + (-1)^{p(x)p(e)}     chi([x,f]) e,
           written out here by hand and independently of regrade.act;

  scan     dim Z_N = dim {v : g_1.v = 0} evaluated at every half-integer lam
           of [-8, 8] by exact rational linear algebra, so that the values of
           lam reported by the symbolic rank-drop test can be seen directly,
           and any value that test might have missed would show up;

  g2       the singular vectors found are killed by g_2 as well (they must be,
           g_1 generating g_+, but it is a cheap independent confirmation).

Usage:  python3 regcheck.py [axioms|deg1|deg2|scan|g2|all]
"""
import sys
from fractions import Fraction as Fr

import sympy as sp

from kas6 import ALG, XI, ETA
from regrade import (Graded, words, sort_word, act, coords, weight, CARTAN,
                     char_info, sing, show_word, PHI, _pos_part)

LAM = sp.Symbol('lam')


def setup(zero, name, topbr=1, D=None):
    D = D if D is not None else 2 * topbr + 3
    gr = Graded(zero, name, D=D, topbr=topbr)
    ci = char_info(gr)
    chi = ([LAM * c for c in ci[0][0]] if ci
           else [sp.Integer(0)] * len(gr.g0))
    return gr, chi, (ci[0][1] if ci else None)


# ------------------------------------------------------------------- axioms

def gbracket(gr, d1, i1, d2, i2):
    """[x, y] for x in g_{d1}, y in g_{d2}, in the basis of g_{d1+d2}."""
    h = ALG.kb(gr.G[d1][i1], gr.G[d2][i2])
    d = d1 + d2
    if not h:
        return d, [Fr(0)] * len(gr.G[d])
    c = coords(gr.G[d], h)
    assert c is not None, ("the bracket left g_%d" % d)
    return d, c


def actlin(gr, d, coeffs, word, chi):
    """act with the element of g_d whose coordinates are coeffs."""
    out = {}
    for i, c in enumerate(coeffs):
        if not c:
            continue
        for w, v in act(gr, d, i, word, chi).items():
            out[w] = out.get(w, 0) + c * v
    return {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}


def axioms(zero, name, N=2):
    gr, chi, _ = setup(zero, name, topbr=2)
    bad = 0
    tested = 0
    for d1 in (0, 1):
        for d2 in (0, 1):
            for i1 in range(len(gr.G[d1])):
                for i2 in range(len(gr.G[d2])):
                    p1, p2 = gr.par(d1, i1), gr.par(d2, i2)
                    sgn = -1 if (p1 == 1 and p2 == 1) else 1
                    d, br = gbracket(gr, d1, i1, d2, i2)
                    for n in range(0, N + 1):
                        for w in words(gr, n):
                            lhs = {}
                            for u, v in act(gr, d2, i2, w, chi).items():
                                for u2, v2 in act(gr, d1, i1, u, chi).items():
                                    lhs[u2] = lhs.get(u2, 0) + v * v2
                            for u, v in act(gr, d1, i1, w, chi).items():
                                for u2, v2 in act(gr, d2, i2, u, chi).items():
                                    lhs[u2] = lhs.get(u2, 0) - sgn * v * v2
                            rhs = actlin(gr, d, br, w, chi)
                            keys = set(lhs) | set(rhs)
                            for k in keys:
                                if sp.expand(lhs.get(k, 0)
                                             - rhs.get(k, 0)) != 0:
                                    bad += 1
                            tested += 1
    print(f"  {name}: module axiom checked on {tested} (x, y, v) triples "
          f"with x, y in g_0 + g_1 and deg v <= {N}: "
          f"{'ALL HOLD' if bad == 0 else str(bad) + ' FAILURES'}")
    return bad


# ------------------------------------------------------- degree 1, by hand

def deg1(zero, name):
    """v = sum_e c_e (e (x) m) is killed by g_1 iff sum_e c_e chi([x,e]) = 0
    for every x in g_1.  No recursion is involved."""
    gr, chi, cw = setup(zero, name)
    rows = []
    for i in range(len(gr.g1)):
        row = []
        for j in range(len(gr.gm1)):
            row.append(sum(a * b for a, b in zip(gr.br[(1, i, j)], chi)))
        rows.append(row)
    M = sp.Matrix(rows)
    p = sp.expand((M.T * M).det())
    vals = ([None] if p == 0
            else sorted(r for r in sp.roots(sp.Poly(p, LAM)) if r.is_rational))
    print(f"  {name}: degree 1, dim Z_1 nonzero exactly at lam in {vals}")
    for v in vals:
        Mv = M if v is None else M.subs(LAM, v)
        ns = Mv.nullspace()
        print(f"      lam = {v}: dim Z_1 = {len(ns)}, spanned by "
              + ", ".join(show_word(gr, (j,)) for j in range(len(gr.gm1))
                          if any(x[j] != 0 for x in ns)))
    return vals


# ------------------------------------------------------- degree 2, by hand

def deg2(zero, name):
    gr, chi, cw = setup(zero, name)
    ws = words(gr, 2)
    rows = {}
    for i in range(len(gr.g1)):
        px = gr.par(1, i)
        for col, w in enumerate(ws):
            e, f = w
            res = {}
            # [[x,e],f] (x) m
            for k, c in enumerate(gr.br[(1, i, e)]):
                if not c:
                    continue
                for l, c2 in enumerate(gr.br[(0, k, f)]):
                    if c2:
                        res[(l,)] = res.get((l,), 0) + c * c2
            # (-1)^{p([x,e]) p(f)} chi([x,e]) f
            for k, c in enumerate(gr.br[(1, i, e)]):
                if not c or not chi[k]:
                    continue
                s = -1 if (gr.par(0, k) == 1 and gr.par_m1(f) == 1) else 1
                res[(f,)] = res.get((f,), 0) + s * c * chi[k]
            # (-1)^{p(x) p(e)} chi([x,f]) e
            for k, c in enumerate(gr.br[(1, i, f)]):
                if not c or not chi[k]:
                    continue
                s = -1 if (px == 1 and gr.par_m1(e) == 1) else 1
                res[(e,)] = res.get((e,), 0) + s * c * chi[k]
            for u, v in res.items():
                if v:
                    rows.setdefault((i, u), [0] * len(ws))[col] += v
    M = sp.Matrix(list(rows.values()))
    p = sp.expand((M.T * M).det())
    vals = ([None] if p == 0
            else sorted(r for r in sp.roots(sp.Poly(p, LAM)) if r.is_rational))
    print(f"  {name}: degree 2, dim Z_2 nonzero exactly at lam in {vals}")
    for v in vals:
        Mv = M if v is None else M.subs(LAM, v)
        ns = Mv.nullspace()
        print(f"      lam = {v}: dim Z_2 = {len(ns)}")
    return vals


# ------------------------------------------------------------------- scan

def scan(zero, name, N=2, rng=8):
    """dim Z_N at every half-integer of [-rng, rng], by exact arithmetic."""
    gr, chi, cw = setup(zero, name)
    ws, hits = words(gr, N), []
    for num in range(-2 * rng, 2 * rng + 1):
        lv = sp.Rational(num, 2)
        c = [sp.simplify(x.subs(LAM, lv)) if hasattr(x, 'subs') else x
             for x in chi]
        rows = {}
        for i in range(len(gr.g1)):
            for col, w in enumerate(ws):
                for u, v in act(gr, 1, i, w, c).items():
                    rows.setdefault((i, u), [0] * len(ws))[col] += v
        if not rows:
            d = len(ws)
        else:
            d = len(ws) - sp.Matrix(list(rows.values())).rank()
        if d:
            hits.append((lv, d))
    print(f"  {name}: degree {N}, dim Z_{N} > 0 at "
          + (", ".join(f"lam = {a} (dim {b})" for a, b in hits) or "no lam")
          + f"   [scan over the half-integers of [-{rng}, {rng}]]")
    return hits


# --------------------------------------------------------------- g_2 check

def g2(zero, name, N=2):
    gr, chi, cw = setup(zero, name, topbr=2)
    for n in range(1, N + 1):
        for c, wt, vec in sing(gr, n, chi, LAM, hw=True):
            cv = ([sp.simplify(x.subs(LAM, c)) if hasattr(x, 'subs') else x
                   for x in chi] if c is not None else chi)
            bad = 0
            for i in range(len(gr.G[2])):
                tot = {}
                for w, co in vec.items():
                    for u, v in act(gr, 2, i, w, cv).items():
                        tot[u] = tot.get(u, 0) + co * v
                bad += sum(1 for v in tot.values() if sp.expand(v) != 0)
            body = " + ".join(f"({x})*{show_word(gr, w)}"
                              for w, x in sorted(vec.items()))
            at = "every lam" if c is None else f"lam = {c}"
            print(f"  {name}: degree {n}, [{at}] {body}: "
                  f"killed by g_2 as well: {bad == 0}")


if __name__ == '__main__':
    job = sys.argv[1] if len(sys.argv) > 1 else 'all'
    todo = [job] if job != 'all' else ['deg1', 'deg2', 'scan', 'g2', 'axioms']
    for j in todo:
        print(f"--- {j} " + "-" * 60)
        for zero, name in [(XI, '3xi'), (ETA, '3eta')]:
            if j == 'axioms':
                axioms(zero, name,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 2)
            elif j == 'deg1':
                deg1(zero, name)
            elif j == 'deg2':
                deg2(zero, name)
            elif j == 'scan':
                scan(zero, name, 1)
                scan(zero, name, 2)
            elif j == 'g2':
                g2(zero, name)
