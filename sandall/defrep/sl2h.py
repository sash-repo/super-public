#!/usr/bin/env python3
"""sl(2), p odd, M = L^{p-1} with basis m_u (u = p-1, p-3, ..., 1-p), m_{u-2} = y m_u.
The weight-0 class of H^1(sl(2); M (x) M): its h-component c(h), which is the
same for every representative (a weight-0 coboundary vanishes on h)."""
import sys, itertools
from hwmod import Fp
from prol import nullspace_sparse
p = int(sys.argv[1]); K = Fp(p)
us = list(range(p - 1, -p, -2)); N = len(us); ix = {u: i for i, u in enumerate(us)}
# y m_u = m_{u-2};  h m_u = u m_u;  x m_u = c_u m_{u+2} with c_u = (p-1-u)(p-1+u+2)/4 ... from [x,y]=h
def X(u):   # x m_u = xc(u) m_{u+2}
    # x m_{u} = x y m_{u+2} = (y x + h) m_{u+2}: xc(u) = xc(u+2) + (u+2)
    s = 0; v = p - 1
    while v > u:
        s += v; v -= 2
    return s % p
ops = {"x": lambda i: {i - 1: X(us[i])} if i > 0 else {},
       "y": lambda i: {i + 1: 1} if i < N - 1 else {},
       "h": lambda i: {i: us[i] % p}}
# check [x,y]=h on M
for i in range(N):
    xy = {}
    for j, a in ops["y"](i).items():
        for k, b in ops["x"](j).items(): xy[k] = (xy.get(k, 0) + a * b) % p
    yx = {}
    for j, a in ops["x"](i).items():
        for k, b in ops["y"](j).items(): yx[k] = (yx.get(k, 0) + a * b) % p
    d = {k: (xy.get(k, 0) - yx.get(k, 0)) % p for k in set(xy) | set(yx)}
    assert {k: v for k, v in d.items() if v} == {k: v for k, v in ops["h"](i).items() if v}
def act(g, t):   # g . (m_a (x) m_b)
    a, b = divmod(t, N); out = {}
    for a2, v in ops[g](a).items(): out[a2 * N + b] = (out.get(a2 * N + b, 0) + v) % p
    for b2, v in ops[g](b).items(): out[a * N + b2] = (out.get(a * N + b2, 0) + v) % p
    return out
wt = lambda t: us[t // N] + us[t % N]
gw = {"x": 2, "y": -2, "h": 0}
# weight-0 cochains: c(g) in (M(x)M)_{wt g}
unk = {}
for g in "xyh":
    for t in range(N * N):
        if wt(t) == gw[g]: unk[(g, t)] = len(unk)
br = {("x", "y"): ("h", 1), ("h", "x"): ("x", 2), ("h", "y"): ("y", -2)}
rows = []
for (g1, g2), (g3, c3) in br.items():
    row = {}
    for (g, t), u in unk.items():
        if g == g2:
            for s, v in act(g1, t).items(): row.setdefault(s, {})[u] = (row.get(s, {}).get(u, 0) + v) % p
        if g == g1:
            for s, v in act(g2, t).items(): row.setdefault(s, {})[u] = (row.get(s, {}).get(u, 0) - v) % p
        if g == g3:
            row.setdefault(t, {})[u] = (row.get(t, {}).get(u, 0) - c3) % p
    rows += [{u: v for u, v in r.items() if v} for r in row.values() if any(r.values())]
Z = nullspace_sparse(K, rows, len(unk))
inv = {u: k for k, u in unk.items()}
hparts = []
for z in Z:
    hp = {inv[u][1]: v for u, v in z.items() if inv[u][0] == "h"}
    if hp: hparts.append(hp)
print("p=%d: %d cocycles of weight 0, %d with a non-zero h-part" % (p, len(Z), len(hparts)))
for hp in hparts[:1]:
    items = sorted(((us[t // N], us[t % N]), v) for t, v in hp.items())
    # normalize so that the coefficient of m_0 (x) m_0 is -1
    c00 = dict(items)[(0, 0)]
    f = (-pow(c00, p - 2, p)) % p
    print("   h-part:", [(uv, (v * f) % p if (v * f) % p <= p // 2 else (v * f) % p - p) for uv, v in items])
    print("   (-1)^{|u|/2+1}:", [((u, -u), (-1) ** (abs(u) // 2 + 1)) for u in us])
