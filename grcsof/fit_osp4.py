#!/usr/bin/env python3
"""The unique i(osp(6|2))-invariant binary operator of order 4 over k(1|6):
solve at several weight pairs and fit every coefficient as a polynomial of
degree <= 2 in (lam1, lam2).  Normalization: the coefficient of
D_xi1 D_eta1 (x) D_xi1 D_eta1 is 1."""
from fractions import Fraction as Fr
from kasbin import build, testmons, ALGS, START
from kasops import blabel, binary_terms
from knclass import nullspace

F = ALGS['osp']()
gens = [F[k] for k in START['osp']]
PTS = [(Fr(3, 7), Fr(-5, 11)), (Fr(1), Fr(2)), (Fr(-1), Fr(3)), (Fr(5), Fr(-2)),
       (Fr(1, 2), Fr(1, 3)), (Fr(-3), Fr(-4)), (Fr(2), Fr(7)), (Fr(-1,2), Fr(5,3)),
       (Fr(4), Fr(1)), (Fr(0), Fr(0))]
ts = binary_terms(4, True)
norm = ts.index((0, (0, 1), 0, (0, 1)))
data = []
for l1, l2 in PTS:
    _, ns = build(gens, 4, l1, l2, testmons(1, 3))
    assert len(ns) == 1, (l1, l2, len(ns))
    v = ns[0]
    assert v[norm] != 0
    data.append(((l1, l2), [Fr(x) / Fr(v[norm]) for x in v]))
    print("solved", l1, l2, flush=True)

mons = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
lab = ['1', 'l1', 'l2', 'l1^2', 'l1 l2', 'l2^2']
M = [[l1 ** a * l2 ** b for (a, b) in mons] for (l1, l2), _ in data]
for i, t in enumerate(ts):
    ys = [v[i] for _, v in data]
    if not any(ys):
        continue
    # solve M c = ys  in the least-squares/exact sense
    aug = [row[:] + [y] for row, y in zip(M, ys)]
    piv, r = [], 0
    for c in range(6):
        p = next((k for k in range(r, len(aug)) if aug[k][c]), None)
        if p is None:
            continue
        aug[r], aug[p] = aug[p], aug[r]
        pv = aug[r][c]
        aug[r] = [x / pv for x in aug[r]]
        for k in range(len(aug)):
            if k != r and aug[k][c]:
                f = aug[k][c]
                aug[k] = [x - f * y for x, y in zip(aug[k], aug[r])]
        piv.append(c)
        r += 1
    ok = all(not any(row[:6]) or True for row in aug[r:]) and \
        all(row[6] == 0 for row in aug[r:])
    c = [Fr(0)] * 6
    for k, cc in enumerate(piv):
        c[cc] = aug[k][6]
    s = " + ".join(f"({c[k]}){lab[k]}" for k in range(6) if c[k])
    print(f"{blabel(t):45s} = {s}   {'' if ok else '*** NO FIT ***'}")
