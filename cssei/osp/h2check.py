"""Sanity checks of h2.py: dim B^2 = dim(even gl) - dim Der_ev (Der computed independently),
and H^2 of the desuperized Lie algebras (all parities even, no squaring)."""
import json, itertools, sys, os
sys.path.insert(0, '..')
from f2vf import kernel_bits
from galg import GFField
from msa import MSA
from h2 import h2
F = GFField(2)
S = json.load(open('s_gs2.json'))
def der_dim(tab, sq, par):
    d = len(par)
    idx = {}
    for l in range(d):
        for k in range(d):
            if par[l] == par[k]: idx[(l, k)] = len(idx)     # D e_l has e_k-coefficient
    eqs = {}
    def add(key, v):
        eqs[key] = eqs.get(key, 0) ^ (1 << v)
    for a in range(d):
        for b in range(d):
            # D[a,b] = [Da,b] + [a,Db]
            for l in range(d):
                if tab[a][b][l]:
                    for k in range(d):
                        if (l, k) in idx: add((a, b, k), idx[(l, k)])
            for k in range(d):
                if (a, k) in idx:
                    for m in range(d):
                        if tab[k][b][m]: add((a, b, m), idx[(a, k)])
                if (b, k) in idx:
                    for m in range(d):
                        if tab[a][k][m]: add((a, b, m), idx[(b, k)])
        if par[a] and sq[a] is not None:    # D(s(a)) = [a, Da]
            for l in range(d):
                if sq[a][l]:
                    for k in range(d):
                        if (l, k) in idx: add(('s', a, k), idx[(l, k)])
            for k in range(d):
                if (a, k) in idx:
                    for m in range(d):
                        if tab[a][k][m]: add(('s', a, m), idx[(a, k)])
    img = [0] * len(idx)
    for r, (key, m) in enumerate(eqs.items()):
        while m:
            v = m.bit_length() - 1; img[v] |= 1 << r; m ^= 1 << v
    return len(kernel_bits(img)), len(idx)
for t in ('inner', 'outer'):
    A = MSA(F, S[t]['even'], S[t]['odd'])
    tab = [[[x & 1 for x in v] for v in row] for row in A.tab]
    sq = [None if s is None else [x & 1 for x in s] for s in A.sq]
    dd, nall = der_dim(tab, sq, A.par)
    h, Z, B, var = h2(tab, sq, A.par, verbose=False)
    print('%s: Der_ev %d, even gl %d, dim B^2 = %d (expected %d)  H^2_ev %d' % (t, dd, nall, len(B), nall - dd, h))
    par0 = [0] * A.d
    dd0, n0 = der_dim(tab, [None] * A.d, par0)
    h0, Z0, B0, _ = h2(tab, [None] * A.d, par0, verbose=False)
    print('   desuperization (Lie algebra, dim %d): Der %d, H^2 %d  [B^2 %d = %d]' % (A.d, dd0, h0, len(B0), n0 - dd0))
# gs(2) itself
T = json.load(open('../cmp/gs2.json')); n = T['dim']
tab = [[[(T['tab'][i][j] >> k) & 1 for k in range(n)] for j in range(n)] for i in range(n)]
dd0, n0 = der_dim(tab, [None] * n, [0] * n)
h0, _, B0, _ = h2(tab, [None] * n, [0] * n, verbose=False)
print('gs(2): Der %d, H^2 %d  [B^2 %d = %d]' % (dd0, h0, len(B0), n0 - dd0))
