#!/usr/bin/env python3
r"""Subsec. 7.1 of the text: Shen's gs(2) = me^(1)(5;1) in Brown's model (BGLLS Sect. 23), N=(1,1), in the basis of
monomials, which is homogeneous for the Z^2-weight (wrt u1 d1, u2 d2, over Z) and for the Z-degree
deg u^r d_i = 3|r|-3, deg u^r vol = 3|r|-2, deg u^r = 3|r|-4; and its restricted closure
L = [ad gs(2)]_2 in End(gs(2)) (dim 17).  Output: shenL.npz."""
import itertools, numpy as np
from f2lin import rref, rank, solve

MON = [(0, 0), (1, 0), (0, 1), (1, 1)]
def mul(a, b):
    if a[0] & b[0] or a[1] & b[1]: return None
    return (a[0] | b[0], a[1] | b[1])
def d(i, a):
    if a[i] == 0: return None
    b = list(a); b[i] -= 1; return tuple(b)
B = [('v', a, i) for a in MON for i in (0, 1)] + [('w', a) for a in MON] + [('f', a) for a in MON]
IDX = {b: k for k, b in enumerate(B)}
def vf(a, i, f):
    df = d(i, f); return None if df is None else mul(a, df)
def br(X, Y):
    out = {}
    def tog(k): out[k] = out.get(k, 0) ^ 1
    if X[0] == 'v' and Y[0] == 'v':
        (_, a, i), (_, b, j) = X, Y
        m = vf(a, i, b)
        if m is not None: tog(('v', m, j))
        m = vf(b, j, a)
        if m is not None: tog(('v', m, i))
    elif X[0] != 'v' and Y[0] == 'v':
        return br(Y, X)
    elif X[0] == 'v':
        (_, a, i) = X
        if Y[0] == 'f':
            m = vf(a, i, Y[1])
            if m is not None: tog(('f', m))
        else:
            m = vf(a, i, Y[1])
            if m is not None: tog(('w', m))
            da = d(i, a)
            if da is not None:
                m = mul(da, Y[1])
                if m is not None: tog(('w', m))
    elif X[0] == 'w' and Y[0] == 'w':
        pass
    elif X[0] == 'f' and Y[0] == 'w':
        return br(Y, X)
    elif X[0] == 'w' and Y[0] == 'f':
        f, g = X[1], Y[1]
        for i, j in ((0, 1), (1, 0)):
            dg = d(i, g)
            if dg is not None:
                m = mul(f, dg)
                if m is not None: tog(('v', m, j))
    else:
        f, g = X[1], Y[1]
        for i, j in ((0, 1), (1, 0)):
            df, dg = d(i, f), d(j, g)
            if df is not None and dg is not None:
                m = mul(df, dg)
                if m is not None: tog(('w', m))
    return {k for k, v in out.items() if v}

def name(b):
    def mono(a):
        s = ''.join(['u1' if a[0] else '', 'u2' if a[1] else ''])
        return s
    if b[0] == 'v': return (mono(b[1]) + 'd%d' % (b[2] + 1))
    if b[0] == 'w': return (mono(b[1]) + 'vol')
    return mono(b[1]) or '1'
def wt(b):
    """Z^2-grading w (a refinement of the weights wrt u1d1, u2d2: w = weight mod 2);
    deg = w[0] + w[1] is the Z-grading of BGLLS"""
    if b[0] == 'v':
        a = list(b[1]); a[b[2]] -= 1; return (3 * a[0], 3 * a[1])
    if b[0] == 'w':
        return (3 * b[1][0] - 1, 3 * b[1][1] - 1)
    return (3 * b[1][0] - 2, 3 * b[1][1] - 2)

# gs(2): drop the center 1 and u1u2 vol; order by degree
G = [b for b in B if b not in (('f', (0, 0)), ('w', (1, 1)))]
G.sort(key=lambda b: (sum(wt(b)), wt(b)[0] - wt(b)[1]))
n = len(G); GI = {b: k for k, b in enumerate(G)}
assert n == 14
C = np.zeros((n, n, n), dtype=np.uint8)
for i, X in enumerate(G):
    for j, Y in enumerate(G):
        for k in br(X, Y):
            if k == ('f', (0, 0)): continue      # modulo the center
            assert k in GI, (X, Y, k)
            C[i, j, GI[k]] ^= 1
# Jacobi and grading
for i in range(n):
    for j in range(n):
        for k in range(n):
            s = (C[i, j] @ C[:, k] + C[j, k] @ C[:, i] + C[k, i] @ C[:, j]) & 1
            assert not s.any()
        for k in range(n):
            if C[i, j, k]:
                assert tuple(np.add(wt(G[i]), wt(G[j]))) == wt(G[k])
ad = [C[i].T.copy() for i in range(n)]          # ad(e_i) as matrix: column j = [e_i, e_j]
# restricted closure in End: add squares until closed
mats = [a for a in ad]; wts = [wt(b) for b in G]; names = [name(b) for b in G]
def inspan(M, mats):
    A = np.array([m.flatten() for m in mats], dtype=np.uint8)
    return solve(A.T, M.flatten())
changed = True
while changed:
    changed = False
    for i in range(len(mats)):
        S = (mats[i].astype(int) @ mats[i]) & 1
        if inspan(S, mats) is None:
            mats.append(S.astype(np.uint8)); wts.append(tuple(2 * x for x in wts[i]))
            names.append('(' + names[i] + ')^2'); changed = True
    for i in range(len(mats)):
        for j in range(len(mats)):
            Br = (mats[i].astype(int) @ mats[j] + mats[j].astype(int) @ mats[i]) & 1
            if inspan(Br, mats) is None:
                mats.append(Br.astype(np.uint8)); wts.append(tuple(np.add(wts[i], wts[j])))
                names.append('[%s,%s]' % (names[i], names[j])); changed = True
N = len(mats)
A = np.array([m.flatten() for m in mats], dtype=np.uint8).T
CL = np.zeros((N, N, N), dtype=np.uint8); SQ = np.zeros((N, N), dtype=np.uint8)
for i in range(N):
    SQ[i] = inspan((mats[i].astype(int) @ mats[i]) & 1, mats)
    for j in range(N):
        CL[i, j] = inspan((mats[i].astype(int) @ mats[j] + mats[j].astype(int) @ mats[i]) & 1, mats)
print('dim gs(2) =', n, ' dim L =', N)
for k in range(N):
    sq = [names[t] for t in range(N) if SQ[k, t]]
    print('%2d %-14s wt=%-12s  x^[2] = %s' % (k, names[k], wts[k], ' + '.join(sq) or '0'))
np.savez('shenL.npz', CL=CL, SQ=SQ, wts=np.array(wts), names=np.array(names), ng=n,
         mats=np.array(mats))
