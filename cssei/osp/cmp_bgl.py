"""Compare the 16-dim Lie algebras: desuperization of b_w(2;1) (= po_w), wk^(1)(3;w)/c (the
desuperization of bgl^(1)(3;w)/c), and the derived algebras of the simple members of the s_b
families.  Over GF(4), alpha = w."""
import numpy as np, pickle, sys, itertools
from gf4 import SA, MUL, INV, rref, rank, nullspace, matmul
from sconst import load
from galg import generate
from gf2k import GFField
from msa import MSA

def sub_or_quot(tab, sq, par, B, quot=None):
    """structure constants of the subalgebra with basis rows B (GF(4)), optionally modulo the ideal
    with basis rows quot (contained in span B)."""
    B = np.asarray(B, dtype=np.uint8); d = len(par)
    Q = np.asarray(quot, dtype=np.uint8) if quot is not None and len(quot) else np.zeros((0, d), dtype=np.uint8)
    # basis of complement: rows of B not in span Q, echelon
    comp = []; cur = rref(Q)[0] if len(Q) else Q
    for v in B:
        t = rref(np.vstack([cur, v[None]]))[0] if len(cur) else rref(v[None])[0]
        if len(t) > len(cur): comp.append(v); cur = t
    C = np.array(comp, dtype=np.uint8); m = len(C)
    M = np.vstack([C, Q]) if len(Q) else C
    A = SA(tab, sq, par)
    def coords(w):
        aug = np.hstack([M.T, w[:, None]]); R, piv = rref(aug)
        assert M.shape[0] not in piv, 'not closed'
        x = np.zeros(M.shape[0], dtype=np.uint8)
        for i, pc in enumerate(piv): x[pc] = R[i, M.shape[0]]
        return x[:m]
    T = np.zeros((m, m, m), dtype=np.uint8)
    for i in range(m):
        for j in range(m): T[i, j] = coords(A.br(C[i], C[j]))
    ppar = np.array([int((par[np.nonzero(v)[0]] == 1).all()) for v in C])
    return T, np.zeros((m, m), dtype=np.uint8), ppar

def lie_inv(T):
    m = T.shape[0]; A = SA(T, np.zeros((m, m), dtype=np.uint8), np.zeros(m, dtype=int))
    D = A.derived()
    der = A.der_sdim()[0]
    return dict(dim=m, perfect=len(D) == m, center=len(A.center()), assoc=A.assoc_dim(), der=der,
                restricted=A.restricted(), simple=A.is_simple_sample(tries=200))

# 1) po_w = desuperization of b_w
z = np.load('b_w.npz'); print('po_w(5;1) = desuper b_w(2;1):', lie_inv(z['tab'])); sys.stdout.flush()
# 2) bgl^(1)(3;w)/c
F = GFField(2); a = F.exp[(F.q - 1) // 3]
Am = [[0, F.el(a), 0], [F.el(a), 0, 1], [0, 1, 0]]
N, dims, el, central = generate(Am, [1, 0, 0], [0, 0, 1], F)
par3 = [1, 0, 0]
ev = [X for d_, L in el.items() for X in L if sum(x * y for x, y in zip(d_, par3)) % 2 == 0]
od = [X for d_, L in el.items() for X in L if sum(x * y for x, y in zip(d_, par3)) % 2 == 1]
Mb = MSA(F, ev, od)
tabb = np.array([[[int(x) for x in v] for v in row] for row in Mb.tab], dtype=np.uint8)
sqb = np.array([[int(x) for x in s] if s is not None else [0] * Mb.d for s in Mb.sq], dtype=np.uint8)
parb = np.array(Mb.par)
Ab = SA(tabb, sqb, parb); cz = Ab.center(); print('bgl^(1)(3;w): sdim', (int((parb == 0).sum()), int(parb.sum())), 'center', len(cz))
T, _, pp = sub_or_quot(tabb, sqb, parb, np.eye(Mb.d, dtype=np.uint8), cz)
print('wk^(1)(3;w)/c:', lie_inv(T)); sys.stdout.flush()
np.savez('bgl1_w.npz', tab=tabb, sq=sqb, par=parb)
# 3) derived algebras of the simple (9|8) members of s_b's families
res = pickle.load(open('family.pkl', 'rb'))
tab, sq, par, _ = load('outer'); seen = set()
for lam, deg, st, terms in res['outer']:
    if st is not None: continue
    T1 = (sum(x[0] for x in terms) & 1).astype(np.uint8); S1 = (sum(x[1] for x in terms) & 1).astype(np.uint8)
    A1 = SA(T1, S1, par); D = A1.derived()
    if not A1.is_simple_sample(tries=50): continue
    key = (len(D), A1.assoc_dim())
    if key in seen: continue
    seen.add(key)
    TD, _, _ = sub_or_quot(T1, S1, par, D)
    print('s_b member lam=%s deg %d, derived sdim %s:' % (''.join(map(str, lam)), deg, A1.sdim_of(D)), lie_inv(TD)); sys.stdout.flush()
