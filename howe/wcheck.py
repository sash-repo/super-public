"""wcheck.py -- family 1a), the delta-part w_j = (1/(k-1)) sum_i b_{ji} v_i,
v_k = m_1, v_i = -a_{ik} m_1 (i<k).  Checks  c_pq w_j = -(delta_{jq} v_p + delta_{jp} v_q)
in V(lambda) for every p<=q, j (Kochetkov's Borel, odd_up=False)."""
import sys, signal
from fractions import Fraction as F
from pemod import Pe, Irrep, xxi, xx, xixi
signal.alarm(3000)
def idx(pe, f):
    return next(i for i, g in enumerate(pe.basis) if g == f)
def run(n, k):
    pe = Pe(n, odd_up=False)
    lam = tuple([2]*(k-1) + [1]*(n-k+1))
    V = Irrep(pe, lam)
    def add(u, v, c=1):
        o = dict(u)
        for a, b in v.items():
            o[a] = o.get(a, 0) + c*b
            if o[a] == 0: del o[a]
        return o
    def wshift(mu, b): return tuple(x+y for x, y in zip(mu, pe.bwt[b]))
    m = {0: F(1)}
    v = {}
    for i in range(k):          # 0-based: i = 0..k-1, the k-th index is k-1
        if i == k-1:
            v[i] = (lam, m)
        else:
            b = idx(pe, xxi(n, i, k-1))            # a_{ik} = D_{x_i xi_k}
            mu = wshift(lam, b)
            v[i] = (mu, {a: -c for a, c in V.act(b, lam, m).items()})
    ok = cnt = 0
    for j in range(n):
        # w_j as (weight, coords)
        w = None; wmu = None
        for i in range(k):
            if i == j: continue
            bi = idx(pe, xixi(n, min(i, j), max(i, j)))
            sgn = 1 if j < i else -1             # b_{ji} = -b_{ij}
            mu, vec = v[i]
            img = V.act(bi, mu, vec)
            nu = wshift(mu, bi)
            wmu = nu
            w = add(w or {}, img, F(sgn, k-1))
        for p in range(n):
            for q in range(p, n):
                c = idx(pe, xx(n, p, q))
                lhs = V.act(c, wmu, w) if w else {}
                rhs = {}
                for (a, b) in ((q, p), (p, q)):     # delta_{jq} v_p + delta_{jp} v_q
                    if j == a and b in v:
                        rhs = add(rhs, v[b][1], -1)
                if p == q and j == p and p in v:     # the two terms coincide for p=q
                    pass
                cnt += 1
                if lhs == rhs: ok += 1
                else: print('FAIL n=%d k=%d j=%d p=%d q=%d lhs=%s rhs=%s' % (n, k, j+1, p+1, q+1, lhs, rhs))
    print('n=%d k=%d lambda=%s: %d/%d identities hold' % (n, k, lam, ok, cnt)); sys.stdout.flush()
for n in (3, 4, 5):
    for k in range(2, n+1):
        run(n, k)
