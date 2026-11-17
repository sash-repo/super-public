#!/usr/bin/env python3
"""Properties of the restricted irreps: highest weights wrt the BGLLS Borel (X1+ = u1d2, X2+ = u2vol),
invariant bilinear forms, duals.
In the text: Proposition 4.3, 1) (the invariant symmetric non-alternating form on V_7)."""
import numpy as np, pickle, sys
from f2lin import nullspace, rank
from lalg import *
mods = pickle.load(open('restricted_irreps.pkl', 'rb'))
H1 = [NAMES.index('u1d1'), NAMES.index('u2d2')]
phi = (3, 2)
pos = [k for k in range(N) if k not in H1 and phi[0] * WTS[k][0] + phi[1] * WTS[k][1] > 0]
print('BGLLS positive part:', [NAMES[k] for k in pos])
for lam, M in mods.items():
    n = M[0].shape[0]
    S = nullspace(np.vstack([M[k] for k in pos]).astype(np.uint8), n)
    hw = []
    for v in S:
        a = [int(((M[h] @ v) & 1) @ v) if False else None for h in H1]
    # weight of the singular vector(s): eigenvalues of u1d1, u2d2 on span S
    wts = []
    for v in S:
        w = []
        for h in H1:
            hv = (M[h] @ v) & 1
            w.append(1 if (hv == v).all() else (0 if not hv.any() else None))
        wts.append(tuple(w))
    # (lambda(H1), lambda(H2)) with H1 = u1d1+u2d2, H2 = u2d2
    hwH = [((w[0] + w[1]) % 2, w[1]) if None not in w else None for w in wts]
    # invariant bilinear forms: B with M^T B + B M = 0
    rows = []
    for k in range(N):
        X = M[k]
        # vec(B): (X^T B + B X)_{ij} = sum_a X_ai B_aj + B_ia X_aj
        A = np.zeros((n * n, n * n), dtype=np.uint8)
        for i in range(n):
            for j in range(n):
                for a in range(n):
                    if X[a, i]: A[i * n + j, a * n + j] ^= 1
                    if X[a, j]: A[i * n + j, i * n + a] ^= 1
        rows.append(A)
    Bs = nullspace(np.vstack(rows[:NG]), n * n)
    BL = nullspace(np.vstack(rows), n * n)
    desc = []
    for B in Bs:
        B = B.reshape(n, n); desc.append(('sym' if (B == B.T).all() else 'nonsym') + ('/alt' if not np.diag(B).any() and (B == B.T).all() else '') + ' rank %d' % rank(B))
    print('L%s dim %d: singular wrt BGLLS Borel %d, weight wrt (u1d1,u2d2) %s -> (lambda(H1),lambda(H2)) %s; '
          'gs(2)-invariant forms %d %s; L-invariant %d' % (lam, n, len(S), wts, hwH, len(Bs), desc, len(BL)))
