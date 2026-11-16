r"""Export b_w(2;1) (lam = w in F_4\F_2) with the b(2) parity (tau, xi odd; q even) as GF(4)
structure constants in a parity-homogeneous basis -> b_w.npz."""
import sys, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = ['poab4.py', '2', 'w']
src = open(os.path.join(os.path.dirname(here), 'poab4.py')).read().split('PARS = {')[0]
g = {'__name__': 'poab4'}; exec(src, g)
B = g['build']('w'); par5 = [1, 0, 0, 1, 1]
basis = []
for dg in sorted(B):
    for pp in (0, 1):
        E = g['Ech']()
        for v in B[dg]:
            w = g['split'](v, par5).get(pp)
            if w: E.add(w)
        basis += [(dg, pp, v) for v in E.basis()]
basis.sort(key=lambda t: t[1])
keys = sorted({k for _, _, v in basis for k in v})
kpos = {k: i for i, k in enumerate(keys)}
d = len(basis); print('dim', d, 'sdim', sum(1 for b in basis if b[1] == 0), sum(1 for b in basis if b[1] == 1))
from gf4 import rref, MUL, INV
V = np.zeros((d, len(keys)), dtype=np.uint8)
for i, (_, _, v) in enumerate(basis):
    for k, c in v.items(): V[i, kpos[k]] = c
# solve coordinates: augment
def coords(w):
    x = np.zeros(len(keys), dtype=np.uint8)
    for k, c in w.items(): x[kpos[k]] = c
    A = np.hstack([V.T, x[:, None]])
    R, piv = rref(A)
    assert d not in piv, 'not in span'
    out = np.zeros(d, dtype=np.uint8)
    for i, pc in enumerate(piv): out[pc] = R[i, d]
    return out
tab = np.zeros((d, d, d), dtype=np.uint8); sq = np.zeros((d, d), dtype=np.uint8)
for i in range(d):
    for j in range(d):
        w = g['bracket'](basis[i][2], basis[j][2])
        if w: tab[i, j] = coords(w)
    if basis[i][1] == 1:
        w = g['square'](basis[i][2])
        if w: sq[i] = coords(w)
par = np.array([b[1] for b in basis])
np.savez(os.path.join(here, 'b_w.npz'), tab=tab, sq=sq, par=par)
print('saved; tab symmetric:', (tab == tab.transpose(1, 0, 2)).all())
