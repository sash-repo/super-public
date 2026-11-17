#!/usr/bin/env python3
"""Calibration of gen_eqs.py on the 7-dim irrep V_7 (restricted, highest weight (0,1), from
restricted_irreps.pkl, written by superrep.py): diagonalize the torus, read off the weight shape, rewrite the
generators in a weight basis and evaluate every equation of the generated system on them.
A genuine module must violate none; a perturbed one must violate some.
Control for Statement 4.4: the equations of gen_eqs.py are satisfied by V_7."""
import pickle, subprocess, re, sys, numpy as np
from f2lin import rref
z = np.load('shenL.npz'); names = [str(s) for s in z['names'][:14]]; I = {n: k for k, n in enumerate(names)}
mods = pickle.load(open('restricted_irreps.pkl', 'rb'))
M = [np.array(m, dtype=np.int64) % 2 for m in mods[(1, 0)]]   # the 7-dim one (pickle keys use the other labelling)
n = M[0].shape[0]; print('V dim', n)
H1, H2 = M[I['u1d1']], M[I['u2d2']]
# simultaneous eigenvectors over F_2 (H toral: H^2 = H on a restricted module with chi(h)=0)
def kern(A):
    A = A % 2; R, piv = rref(A.astype(np.uint8))[:2] if isinstance(rref(A.astype(np.uint8)), tuple) else (None, None)
    return R, piv
def nullspace(A):
    A = (A % 2).astype(np.uint8); m, k = A.shape; A = A.copy(); piv = []; r = 0
    for c in range(k):
        p = next((i for i in range(r, m) if A[i, c]), None)
        if p is None: continue
        A[[r, p]] = A[[p, r]]
        for i in range(m):
            if i != r and A[i, c]: A[i] ^= A[r]
        piv.append(c); r += 1
    free = [c for c in range(k) if c not in piv]; out = []
    for f in free:
        v = np.zeros(k, dtype=np.uint8); v[f] = 1
        for i, c in enumerate(piv):
            if A[i, f]: v[c] = 1
        out.append(v)
    return out
E = np.eye(n, dtype=np.int64); cols = []; shape = {}
for e in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    ker = nullspace(np.vstack([(H1 - e[0] * E) % 2, (H2 - e[1] * E) % 2]))
    shape[e] = len(ker); cols += ker
P = np.array(cols, dtype=np.int64).T; assert P.shape == (n, n)
# inverse of P over F_2
aug = np.hstack([P % 2, E]).astype(np.uint8)
for c in range(n):
    p = next(i for i in range(c, n) if aug[i, c]); aug[[c, p]] = aug[[p, c]]
    for i in range(n):
        if i != c and aug[i, c]: aug[i] ^= aug[c]
Pi = aug[:, n:].astype(np.int64)
W = {g: (Pi @ M[I[g]] @ P) % 2 for g in names}
print('shape', shape)
s = [shape[e] for e in [(0, 0), (0, 1), (1, 0), (1, 1)]]
fn = 'eqs_calib_%d%d%d%d.m' % tuple(s)
subprocess.run([sys.executable, 'gen_eqs.py'] + [str(x) for x in s] + [fn], check=True)
CL = z['CL']
def modwt(k): return (int(CL[I['u1d1'], k, k]), int(CL[I['u2d2'], k, k]))
basis = [e for e in [(0, 0), (0, 1), (1, 0), (1, 1)] for _ in range(shape[e])]
vals = []; ok = True
for g in ['u1', 'u2', 'u1vol', 'u2vol', 'u1u2']:
    w = modwt(I[g])
    for r in range(n):
        for c in range(n):
            allowed = tuple((a + b) % 2 for a, b in zip(basis[c], w)) == basis[r]
            if allowed: vals.append(int(W[g][r, c]))
            elif W[g][r, c]: ok = False
print('block structure respected:', ok)
eqs = open(fn).read().split('eqs = {', 1)[1].rsplit('}', 1)[0].split(',\n ')
def ev(e, v):
    tot = 0
    for mono in e.split(' + '):
        p = 1
        for f in mono.split('*'):
            f = f.strip()
            if f == '1': continue
            m = re.fullmatch(r'x(\d+)(\^(\d+))?', f); p *= v[int(m.group(1))] ** int(m.group(3) or 1)
        tot += p
    return tot % 2
print('V_7: %d equations, %d violated, %d unknowns' % (len(eqs), sum(ev(e, vals) for e in eqs), len(vals)))
nz = [i for i, x in enumerate(vals) if x]; bad = 0
for i in nz[:10]:
    v2 = list(vals); v2[i] ^= 1; bad += sum(ev(e, v2) for e in eqs) > 0
print('perturbations detected: %d of %d' % (bad, min(10, len(nz))))
