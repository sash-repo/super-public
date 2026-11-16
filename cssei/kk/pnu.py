#!/usr/bin/env python3
"""P(n;u): divided powers O(n;u) mod constants, {f,g}=sum d_i f d_i g over F_2; its derived
series, simple core, rank ad_x, restrictedness.  Also checks P1(6) thinness."""
import itertools, random, sys, json
import kk
def build(n, u):
    kk.N = n
    mons = [m for m in itertools.product(*[range(2**ui) for ui in u]) if any(m)]
    return [frozenset([m]) for m in mons]
def derived_basis(B):
    S = kk.Span()
    for a, b in itertools.combinations(B, 2):
        c = kk.br(a, b)
        if c: S.add(c)
    return S.basis
def stats(name, B):
    n = len(B); tab = kk.table(B)
    simple = all(kk.ideal_dim(tab, n, 1 << i) == n for i in range(n))
    random.seed(2)
    rk = max(kk.rank(kk.admat(tab, n, random.getrandbits(n))) for _ in range(300))
    print(name, 'dim', n, 'simple', simple, 'rank', rk, 'restricted', kk.restricted(tab, n)[0], flush=True)
    return tab
if __name__ == '__main__':
    for n, u in [(2, (3, 3)), (2, (4, 2)), (2, (5, 1)), (3, (2, 2, 2)), (3, (3, 2, 1)), (3, (4, 1, 1)),
                 (4, (2, 2, 1, 1)), (4, (3, 1, 1, 1)), (5, (2, 1, 1, 1, 1)), (6, (1,)*6)]:
        B = build(n, u)
        D = derived_basis(B)
        while True:
            D2 = derived_basis(D)
            if len(D2) == len(D): break
            D = D2
        stats('P(%d;%s) dim %d core' % (n, u, len(B)), D)
