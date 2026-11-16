import numpy as np, search as S, ambients as A
from lie import closure
from gf2 import rref, mm
S.AMS = S.make_ambients()
Ns = (2, 2)
def mono(a):
    e = [(a >> 0) & 3, (a >> 2) & 3]
    s = ''.join('x%d^(%d)' % (i + 1, k) if k > 1 else ('x%d' % (i + 1) if k == 1 else '') for i, k in enumerate(e))
    return s or '1'
def show(v, amb):
    # coordinates in W basis: solve basis^T c = v
    B = amb.basis
    R, piv = rref(np.hstack([B.T, v[:, None]]))
    # simple: least squares over F2 via rref of augmented
    n = len(B)
    M = np.hstack([B.T, v[:, None]]).astype(np.uint8)
    R, piv = rref(M)
    c = np.zeros(n, np.uint8)
    for i, p in enumerate(piv):
        if p < n: c[p] = R[i, n]
    terms = ['%s d%d' % (mono(k // 2), k % 2 + 1) for k in np.nonzero(c)[0]]
    return ' + '.join(terms)
for seed in (1001651, 1033153):
    rng = np.random.default_rng(seed)
    amb = S.AMS[seed % len(S.AMS)]
    assert amb.name == 'W(2;(2,2))'
    mode = 'graded' if (amb.deg is not None and rng.random() < 0.5) else 'sparse'
    if mode == 'graded':
        gens = A.graded_pair(amb, rng, kmax=int(rng.integers(1, 5)))
    else:
        g = 2 if rng.random() < 0.8 else 3
        kmax = int(rng.choice([1, 2, 3, 4, 6]))
        gens = [A.sparse(amb, rng, kmax) for _ in range(g)]
    print('seed', seed, mode)
    for x in gens: print('  gen:', show(x, amb))
    E = closure(amb, gens, 100)
    print('  dim', E.dim())
    for v in E.E: print('   ', show(v, amb))
