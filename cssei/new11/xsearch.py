"""Search twists X in vect(2;N): L_X = Span{H_f + f X | f != top}.  Keep the closed ones,
analyse them, and record (dim, 2-nilp, toral) when dim <= 31.  X runs over monomial fields
and over sums of two monomial fields."""
import sys, os, itertools, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ('..', '../rand', '../cmp'): sys.path.insert(0, os.path.join(HERE, p))
from gf2 import rref, rank
from lie import factors, centroid_dim, restricted
from rseries import table, dimder
from count import invariants

def space(N):
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
    # precompute bracket of basis keys
    K = len(keys)
    BR = {}
    def brk(s, t):
        if (s, t) in BR: return BR[(s, t)]
        fa, i = keys[s]; gb, j = keys[t]; out = []
        dg = der(i, gb)
        if dg is not None:
            m = mul(fa, dg)
            if m is not None: out.append(kidx[(m, j)])
        df = der(j, fa)
        if df is not None:
            m = mul(gb, df)
            if m is not None: out.append(kidx[(m, i)])
        BR[(s, t)] = out; return out
    def br(u, v):
        out = np.zeros(K, np.uint8)
        for s in np.nonzero(u)[0]:
            for t in np.nonzero(v)[0]:
                for k in brk(s, t): out[k] ^= 1
        return out
    top = (T[0], T[1])
    def H(f):
        v = np.zeros(K, np.uint8)
        a = der(1, f)
        if a is not None: v[kidx[(a, 0)]] ^= 1
        a = der(0, f)
        if a is not None: v[kidx[(a, 1)]] ^= 1
        return v
    def fX(f, X):
        v = np.zeros(K, np.uint8)
        for s in np.nonzero(X)[0]:
            g, i = keys[s]; m = mul(f, g)
            if m is not None: v[kidx[(m, i)]] ^= 1
        return v
    return monos, keys, br, H, fX, top

def run(N, maxterms=2):
    monos, keys, br, H, fX, top = space(N)
    K = len(keys); seen = {}; results = []
    fs = [f for f in monos if f != top]
    cands = [np.eye(K, dtype=np.uint8)[s] for s in range(K)]
    if maxterms >= 2:
        cands += [np.eye(K, dtype=np.uint8)[s] ^ np.eye(K, dtype=np.uint8)[t] for s in range(K) for t in range(s + 1, K)]
    for X in cands:
        B = np.array([H(f) ^ fX(f, X) for f in fs], np.uint8)
        r = rank(B)
        if r < len(fs): continue
        W = [br(B[i], B[j]) for i in range(len(fs)) for j in range(i + 1, len(fs))]
        if rank(np.vstack([B, np.array(W, np.uint8)])) > r: continue
        R, piv = rref(B.copy()); Bb = R[:len(piv)]
        ad = table(Bb, br); d = ad.shape[0]
        out = []; ab = factors(ad, np.random.default_rng(1), out)
        if not (ab == 0 and len(out) == 1 and out[0][0].shape[0] == d and out[0][1] == 'simple'):
            continue
        tab = [[int(sum(int(ad[i][k, j]) << k for k in range(d))) for j in range(d)] for i in range(d)]
        inv = tuple(int(x) for x in invariants(tab)) if d <= 31 else None
        Xs = ' + '.join('x^(%d,%d)d%d' % (keys[s][0] + (keys[s][1] + 1,)) for s in np.nonzero(X)[0])
        key = (d, inv)
        if key not in seen:
            seen[key] = Xs
            json.dump({'name': 'X %s in vect(2;%s)' % (Xs, N), 'dim': d, 'tab': tab},
                      open(os.path.join(HERE, 'xs_%d%d_%d_%d.json' % (N + (inv or (0, 0)))), 'w'))
            print(N, d, inv, 'restricted' if restricted(ad) else 'nr', 'X =', Xs, flush=True)
        results.append((Xs, key))
    return seen

if __name__ == '__main__':
    N = tuple(map(int, sys.argv[1].split(','))); mt = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    s = run(N, mt)
    json.dump({str(k): v for k, v in s.items()}, open(os.path.join(HERE, 'xsearch_%d%d.json' % N), 'w'), indent=1)
