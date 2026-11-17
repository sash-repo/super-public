#!/usr/bin/env python3
"""rscal.py -- which of the two weight-(0,0) degree-2 classes of
Par=(00001) (Riemannian g_0 = sl2_L + sl2_R) carries the usual scalar curvature.

M = g_{-1}^even (4-dim).  A degree-2 cochain restricted to Lambda^2 M -> g_0 is a
Riemann-type tensor R; coboundaries vanish there ([M,M]=0 and 1-cochains of
degree 2 live on g_{-2} only) -- checked below.  From R we form the two
invariant contractions s_L, s_R (Ricci contraction with only the sl2_L, resp.
sl2_R, part of R(x,y), then the invariant form of M).  The usual scalar
curvature is s_L + s_R; s_L - s_R is the pseudoscalar, = 0 for a usual
Riemann tensor (Bianchi).
"""
import sys
from fractions import Fraction
sys.path.insert(0, "../lib")
from exactlin import nullspace, rank
from models import Prolong
from cohom import Complex, c2_items, c1_items

P = Prolong("00001", "riem"); M = P.M; C = Complex(P)
Mi = [a for a in range(P.ngm) if P.gm_deg[a] == -1 and P.gm_par[a] == 0]
units = {a: P.gm[a][2] for a in Mi}
print("M basis:", [units[a] for a in Mi], " sl2 pairs:", M.sl2)
g0 = P.basis[0]

def bracket(X, u):          # [X, E_u], X a dict, all even
    out = {}
    for k, c in X.items():
        for v, c2 in M.unit_bracket(*k, *u).items():
            out[v] = out.get(v, 0) + c * c2
    return {v: c for v, c in out.items() if c}

# which g_0 basis vectors are L / R
side = []
for X in g0:
    k = next(iter(X)); side.append('L' if k[0] in M.sl2[0] else 'R')
# invariant form on M = (E_ij), i in block 2 (rows 2,3), j in block 1 (cols 0,1): det
def gform(u, v):
    (i, j), (k, l) = u, v
    if i == k or j == l: return 0
    s = 1 if (i - 2) == (j - 0) else -1        # E_20,E_31 -> +1 ; E_21,E_30 -> -1
    s2 = 1 if (k - 2) == (l - 0) else -1
    return Fraction(s * 1) if s == s2 else 0
G = [[gform(units[a], units[b]) for b in Mi] for a in Mi]
# inverse form
import itertools
def inv(A):
    n = len(A); B = [list(map(Fraction, r)) + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if B[r][c]); B[c], B[p] = B[p], B[c]
        f = B[c][c]; B[c] = [x / f for x in B[c]]
        for r in range(n):
            if r != c and B[r][c]:
                g = B[r][c]; B[r] = [x - g * y for x, y in zip(B[r], B[c])]
    return [r[n:] for r in B]
Gi = inv(G)
pos = {units[a]: i for i, a in enumerate(Mi)}

def contractions(vals):
    """vals: {(a,b): {(0,idx): coef}} on Lambda^2 M.  Returns (s_L, s_R)."""
    res = {}
    for S in ('L', 'R'):
        Ric = [[Fraction(0)] * 4 for _ in range(4)]
        for yi, y in enumerate(Mi):
            for zi, z in enumerate(Mi):
                tot = Fraction(0)
                for xi, x in enumerate(Mi):
                    Rxy = C.val2(vals, x, y) if x != y else {}
                    for (m, idx), cf in Rxy.items():
                        if side[idx] != S: continue
                        img = bracket(g0[idx], units[z])
                        tot += cf * img.get(units[x], 0)
                Ric[yi][zi] = tot
        res[S] = sum(Gi[i][j] * Ric[i][j] for i in range(4) for j in range(4))
    return res['L'], res['R']

# degree-2 even blocks of sl2-weight (0,0)
blocks = {}
for (a, b, m, idx, deg, w) in c2_items(P):
    if deg == 2 and M.sl2_weight(w) == (0, 0) and M.parity_of_weight(w) == 0:
        blocks.setdefault(w, []).append((a, b, m, idx))
for w, cols in blocks.items():
    colpos = {c: i for i, c in enumerate(cols)}
    rows = {}
    for i, c in enumerate(cols):
        for tri, val in C.d2_column(*c).items():
            for key, cf in val.items():
                rows.setdefault((tri, key), {})[i] = cf
    mat = [[r.get(i, 0) for i in range(len(cols))] for r in rows.values()]
    Z = nullspace(mat, None) if False else None
    from sugra import nullspace as ns
    Z = ns(mat, len(cols)) if mat else [[Fraction(int(i == j)) for j in range(len(cols))] for i in range(len(cols))]
    B = []
    for (a, m, idx, deg, w1) in c1_items(P):
        if deg != 2 or w1 != w: continue
        col = C.d1_column(a, m, idx); vec = [0] * len(cols)
        for (x, y), val in col.items():
            for (mm, t), cf in val.items():
                vec[colpos[(x, y, mm, t)]] += cf
        if any(vec): B.append(vec)
    rB = rank(B) if B else 0
    print(f"\nweight {w}: dim C2={len(cols)} dim Z={len(Z)} rank B={rB} dim H={len(Z)-rB}")
    onM = [i for i, (a, b, m, idx) in enumerate(cols) if a in Mi and b in Mi]
    print("  coboundaries on Lambda^2 M:", [sum(abs(v[i]) for i in onM) for v in B])
    for z in Z:
        vals = {}
        for i, cf in enumerate(z):
            if cf and i in onM:
                a, b, m, idx = cols[i]; vals.setdefault((a, b), {})[(m, idx)] = cf
        sL, sR = contractions(vals)
        print("  cocycle: Lambda^2M part nonzero:", bool(vals), " s_L =", sL, " s_R =", sR,
              " other components:", sum(1 for i, cf in enumerate(z) if cf and i not in onM))
