#!/usr/bin/env python3
r"""Z/2-gradings of Shen's gs(2) (N=(1,1), dim 14) and their "method 2"
superizations [BLLS, Subsec. 3.3].

Over a field of characteristic 2 a Z/2-grading g = g_0 + g_1 of a Lie algebra is
the same thing as a derivation D with D^2 = D (D = 0 on g_0, 1 on g_1): the
Leibniz rule is then exactly [g_i, g_j] in g_{i+j}.  So we compute Der g,
enumerate its F_2-points D with D^2 = D, and for each grading build
L = ad g + Span{(ad x)^2 | x in g_1} inside End(g) (the minimal subalgebra of
the 2-envelope containing g and the squares of the odd part); then
s(g, gr) = L with odd part g_1 has sdim (dim g_0 + dim L/g | dim g_1).
Input: cmp/gs2.json (structure table over F_2, tab[i][j] = bitmask)."""
import json, sys, itertools, os
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f2vf import kernel_bits, BitEchelon

src = sys.argv[1] if len(sys.argv) > 1 else 'cmp/gs2.json'
T = json.load(open(src))
n = T['dim']; tab = T['tab']
C = np.zeros((n, n, n), dtype=np.uint8)          # C[i,j,k]: coefficient of e_k in [e_i,e_j]
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i, j, k] = (tab[i][j] >> k) & 1
assert (C == C.transpose(1, 0, 2)).all() and all(C[i, i].sum() == 0 for i in range(n))


def ad(v):          # matrix M[k, j] = coeff of e_k in [v, e_j]
    return (np.einsum('i,ijk->kj', v, C) % 2).astype(np.uint8)


# --- Der g: unknowns D[l,k] (D e_k = sum_l D[l,k] e_l)
imgs = []
for l in range(n):
    for k in range(n):
        # contribution of unknown (l,k) to the equation D[e_i,e_j] - [De_i,e_j] - [e_i,De_j], coordinate m
        v = 0
        for i in range(n):
            for j in range(n):
                for m in range(n):
                    c = 0
                    if C[i, j, k]:                   # D[e_i,e_j] has sum_k C[i,j,k] D e_k
                        c ^= int(l == m)
                    if i == k:                       # [D e_i, e_j] with D e_k = e_l
                        c ^= int(C[l, j, m])
                    if j == k:
                        c ^= int(C[i, l, m])
                    if c:
                        v |= 1 << ((i * n + j) * n + m)
        imgs.append(v)
ker = kernel_bits(imgs)
DER = []
for kv in ker:
    M = np.zeros((n, n), dtype=np.uint8)
    for idx in range(n * n):
        if (kv >> idx) & 1:
            M[idx // n, idx % n] = 1
    DER.append(M)
d = len(DER)
ADS = [ad(np.eye(n, dtype=np.uint8)[i]) for i in range(n)]
E = BitEchelon()
def flat(M):
    return int(''.join(map(str, M.flatten()[::-1])), 2)
for A in ADS:
    E.add(flat(A))
print('dim g = %d, dim Der g = %d, dim ad g = %d, outer = %d' % (n, d, len(E), d - len(E)))

# --- toral elements D^2 = D: brute force over the F_2-points of Der g (in chunks)
Dstack = np.array(DER, dtype=np.int64)          # d x n x n
res = []
total = 1 << d
chunk = 1 << 14
for start in range(1, total, chunk):
    idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(d)) & 1).astype(np.int64)
    Ms = np.einsum('bd,dij->bij', bits, Dstack) % 2
    Sq = np.einsum('bij,bjk->bik', Ms, Ms) % 2
    ok = (Sq == Ms).all(axis=(1, 2))
    for b in np.nonzero(ok)[0]:
        res.append(Ms[b].astype(np.uint8))
print('toral D (D^2 = D, D != 0) over F_2: %d' % len(res))


def rank2(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def colspace(M):
    """basis of the column space (list of vectors)"""
    E = BitEchelon(); out = []
    for c in range(M.shape[1]):
        v = M[:, c]
        if E.add(int(''.join(map(str, v[::-1])), 2)):
            out.append(v.copy())
    return out


def nullspace(M):
    imgs = [int(''.join(map(str, M[:, c][::-1])), 2) for c in range(M.shape[1])]
    out = []
    for kv in kernel_bits(imgs):
        out.append(np.array([(kv >> c) & 1 for c in range(M.shape[1])], dtype=np.uint8))
    return out


inner = BitEchelon()
for A in ADS:
    inner.add(flat(A))
rows = []
for D in res:
    g1 = colspace(D)                 # image = eigenvalue 1 (D idempotent)
    g0 = nullspace(D)
    # squares of the odd part in End(g)
    Lsp = BitEchelon()
    for A in ADS:
        Lsp.add(flat(A))
    for x in g1:
        A = ad(x).astype(np.int64)
        Lsp.add(flat((A @ A % 2).astype(np.uint8)))
    extra = len(Lsp) - n
    # is g_0 (as Lie algebra) solvable / its derived dim
    B0 = g0
    der0 = BitEchelon()
    for a in B0:
        for b in B0:
            der0.add(int(''.join(map(str, ((np.einsum('i,j,ijk->k', a, b, C) % 2).astype(np.uint8))[::-1])), 2))
    is_inner = inner.reduce(flat(D)) == 0 if hasattr(inner, 'reduce') else None
    rows.append((len(g0), len(g1), extra, len(der0), is_inner))
cnt = Counter(rows)
print('(dim g0, dim g1, dim L/g, dim [g0,g0], D inner?)  number of D over F_2')
for k in sorted(cnt):
    print('  ', k, cnt[k])
json.dump({'dimDer': d, 'toral': len(res), 'classes': [[list(k), v] for k, v in sorted(cnt.items())]},
          open(os.path.splitext(src)[0] + '_grad.json', 'w'))
