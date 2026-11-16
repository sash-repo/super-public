#!/usr/bin/env python3
"""Control: the degree-0, weight-0 derivations of kle(9|6;K) modulo the inner
ones -- expected to be spanned by the grading operator alone."""
import sys
from fractions import Fraction
import kle
import klecoh as K

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace                                  # noqa: E402

DEG = list(range(-2, 4))
comp = {d: kle.comp(d) for d in DEG}
G = K.Graded(kle, comp)

# unknowns: phi(e^i_a) = sum_b c[i,a,b] e^i_b with equal weight and parity
var = {}
for i in DEG:
    for a in range(len(comp[i])):
        for b in range(len(comp[i])):
            if G.wt[i][a] == G.wt[i][b]:
                var[(i, a, b)] = len(var)
n = len(var)
rows = []
for i in DEG:
    for j in DEG:
        if i + j not in comp:
            continue
        for a in range(len(comp[i])):
            for b in range(len(comp[j])):
                v = G.bracket(i, a, j, b)
                if not v:
                    continue
                acc = [dict() for _ in range(len(comp[i + j]))]
                # [phi(x), y] + [x, phi(y)]
                for c in range(len(comp[i])):
                    if (i, a, c) not in var:
                        continue
                    for t, co in enumerate(G.bracket(i, c, j, b)):
                        if co:
                            acc[t][var[(i, a, c)]] = acc[t].get(var[(i, a, c)], Fraction(0)) + co
                for c in range(len(comp[j])):
                    if (j, b, c) not in var:
                        continue
                    for t, co in enumerate(G.bracket(i, a, j, c)):
                        if co:
                            acc[t][var[(j, b, c)]] = acc[t].get(var[(j, b, c)], Fraction(0)) + co
                # - phi([x,y])
                for c, co in enumerate(v):
                    if not co:
                        continue
                    for t in range(len(comp[i + j])):
                        if (i + j, c, t) in var:
                            k2 = var[(i + j, c, t)]
                            acc[t][k2] = acc[t].get(k2, Fraction(0)) - co
                for t in range(len(comp[i + j])):
                    if acc[t]:
                        row = [Fraction(0)] * n
                        for k2, c in acc[t].items():
                            row[k2] += c
                        if any(row):
                            rows.append(row)
ker = nullspace(rows)
print('degree-0 weight-0 derivations (restricted to g_%d..g_%d): %d'
      % (DEG[0], DEG[-1], len(ker)))

# the inner ones: ad_z for z in (g_0) of weight 0
inner = []
for z in range(len(comp[0])):
    if G.wt[0][z] != (0, 0, 0, 0):
        continue
    row = [Fraction(0)] * n
    for i in DEG:
        for a in range(len(comp[i])):
            for t, co in enumerate(G.bracket(0, z, i, a)):
                if co and (i, a, t) in var:
                    row[var[(i, a, t)]] += co
    if any(row):
        inner.append(row)
print('inner (ad of the weight-0 part of g_0):', rank(inner))
print('outer, degree 0, weight 0 =', len(ker) - rank(inner))
