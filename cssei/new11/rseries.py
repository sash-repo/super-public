"""Is r15 the member N=(2,2) of a series r(N) in vect(2;N)?
Candidate: r(N) = Span{ H_f + f * x1^{(top1)} d2 | f monomial of O(2;N), f != x^{top} },
H_f = d2(f) d1 + d1(f) d2 (char 2).  For N=(2,2) this is exactly the basis of the 15-dimensional algebra r15.
For each N: closure, dim, simplicity (MeatAxe), centroid, restrictedness, dim Der, and the CSS
invariants (2-nilpotent, toral, zero included) when dim <= 31."""
import sys, os, itertools, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ('..', '../rand', '../cmp', '../grad'): sys.path.insert(0, os.path.join(HERE, p))
from gf2 import rref, nullspace
from lie import factors, centroid_dim, restricted

def build(N, variant=0):
    T = [(1 << n) - 1 for n in N]
    monos = list(itertools.product(range(T[0] + 1), range(T[1] + 1)))
    keys = [(a, i) for a in monos for i in (0, 1)]
    kidx = {k: t for t, k in enumerate(keys)}
    def mul(a, b):
        if a[0] & b[0] or a[1] & b[1]: return None
        return (a[0] | b[0], a[1] | b[1])
    def der(i, a):
        if a[i] == 0: return None
        b = list(a); b[i] -= 1; return tuple(b)
    def vec(terms):
        v = np.zeros(len(keys), np.uint8)
        for k in terms:
            if k is not None and k[0] is not None: v[kidx[k]] ^= 1
        return v
    def br(u, v):        # [f d_i, g d_j] = f d_i(g) d_j + g d_j(f) d_i
        out = np.zeros(len(keys), np.uint8)
        for s in np.nonzero(u)[0]:
            fa, i = keys[s]
            for t in np.nonzero(v)[0]:
                gb, j = keys[t]
                dg = der(i, gb)
                if dg is not None:
                    m = mul(fa, dg)
                    if m is not None: out[kidx[(m, j)]] ^= 1
                df = der(j, fa)
                if df is not None:
                    m = mul(gb, df)
                    if m is not None: out[kidx[(m, i)]] ^= 1
        return out
    top = (T[0], T[1]); x1t = (T[0], 0)
    B = []
    for f in monos:
        if f == top: continue
        terms = [(der(1, f), 0), (der(0, f), 1)]
        m = mul(f, x1t)
        if variant == 0: terms.append((m, 1))
        B.append(vec(terms))
    B = np.array(B, np.uint8)
    R, piv = rref(B.copy()); B = R[:len(piv)]
    return B, br

def table(B, br):
    d = len(B)
    aug_basis = B.T.copy()
    ad = np.zeros((d, d, d), np.uint8)
    for i in range(d):
        for j in range(d):
            w = br(B[i], B[j])
            R, piv = rref(np.hstack([aug_basis, w[:, None]]))
            assert d not in piv, 'not closed'
            c = np.zeros(d, np.uint8)
            for r, p in enumerate(piv): c[p] = R[r, d]
            ad[i, :, j] = c
    return ad

def dimder(ad):
    d = ad.shape[0]; rows = []
    for i in range(d):
        for j in range(i + 1, d):
            c = ad[i][:, j]
            for r in range(d):
                row = np.zeros(d * d, np.uint8)
                for k in np.nonzero(c)[0]: row[r * d + k] ^= 1
                for k in range(d):
                    if ad[k][r, j]: row[k * d + i] ^= 1
                    if ad[i][r, k]: row[k * d + j] ^= 1
                if row.any(): rows.append(row)
    return len(nullspace(np.array(rows, np.uint8)))

if __name__ == '__main__':
    Ns = [tuple(map(int, s.split(','))) for s in sys.argv[1:]] or [(1,1),(2,1),(1,2),(2,2),(3,1),(1,3),(3,2),(2,3),(3,3),(4,1),(4,2)]
    res = {}
    for N in Ns:
        B, br = build(N)
        try:
            ad = table(B, br)
        except AssertionError:
            print(N, 'NOT CLOSED', flush=True); res[str(N)] = 'not closed'; continue
        d = ad.shape[0]
        out = []; ab = factors(ad, np.random.default_rng(1), out)
        simp = (ab == 0 and len(out) == 1 and out[0][0].shape[0] == d and out[0][1] == 'simple')
        info = {'dim': d, 'abelian_factors': ab, 'factors': [o[0].shape[0] for o in out], 'simple': simp}
        if simp:
            info['centroid'] = centroid_dim(ad, out[0][2]); info['restricted'] = bool(restricted(ad))
            if d <= 40: info['dimDer'] = dimder(ad)
            if d <= 31:
                from count import invariants
                tab = [[int(sum(int(ad[i][k, j]) << k for k in range(d))) for j in range(d)] for i in range(d)]
                info['inv'] = [int(x) for x in invariants(tab)]
                json.dump({'name': 'r%s' % (N,), 'dim': d, 'tab': tab}, open(os.path.join(HERE, 'r_%d%d.json' % N), 'w'))
        print(N, info, flush=True); res[str(N)] = info
    json.dump(res, open(os.path.join(HERE, 'rseries.json'), 'w'), indent=1)
