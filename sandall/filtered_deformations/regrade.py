#!/usr/bin/env python3
"""Z-gradings of the 125-dimensional prolong Me = (g_-, cvect(1;1))_{*,1} inside
k(5;1) of the form deg t = s, deg y_j = c_j (c_1+c_4 = c_2+c_3 = s, the condition
for the contact bracket to be homogeneous), field degree = deg f - s.
Reports those for which Me is homogeneous, with the dims of the components."""
from mel5 import *
mats, forms = setup()
K = Contact(forms[0], (5,) * 5)
gm2 = [mono(0, 0, 0, 0, 0)]
gm1 = [{lin_index(n): 1} for n in range(4)]
H = [quad_hamiltonian(K, A) for A in mats]
comps = prolong(K, gm2, gm1, [mono(1, 0, 0, 0, 0)] + H)
basis = [f for d in sorted(comps) for f in comps[d]]
assert len(basis) == 125


def dims_for(s, c):
    fd = lambda m: s * m[0] + sum(ci * b for ci, b in zip(c, m[1:])) - s
    proj = {}
    for f in basis:
        parts = {}
        for m, u in f.items():
            parts.setdefault(fd(m), {})[m] = u
        for d, g in parts.items():
            proj.setdefault(d, []).append(g)
    out, tot = {}, 0
    for d, gs in proj.items():
        ms = sorted({m for g in gs for m in g})
        r = rk([[g.get(m, 0) for m in ms] for g in gs])
        out[d] = r
        tot += r
    return tot, dict(sorted(out.items()))


seen = set()
for s in range(1, 8):
    for c1 in range(-3, s + 4):
        for c2 in range(-3, s + 4):
            c = (c1, c2, s - c2, s - c1)
            tot, dims = dims_for(s, c)
            if tot == 125:
                neg = {d: v for d, v in dims.items() if d < 0}
                key = tuple(sorted(dims.items()))
                if key in seen:
                    continue
                seen.add(key)
                print("s=%d c=%s  g_-=%s  dim g_0=%d" % (s, c, neg, dims.get(0, 0)), flush=True)
