"""brute check of N(K): random x; x in N(K) iff [x,k] in K for all basis k"""
import sys, random
from f2 import *
from inv1 import normalizer
for f in sys.argv[1:]:
    L = Alg.load(f); n = L.n
    ads = [L.ad(1 << i) for i in range(n)]
    V = span(flat(A, n) for A in ads)
    res = [V.reduce(flat(mul(A, A), n))[0] for A in ads]
    Kb = kernel_mod(res, V); K = span(Kb); N = span(normalizer(L, Kb)); random.seed(3); bad = 0
    for t in range(300):
        x = random.getrandbits(n)
        if t % 2:
            x = 0
            for b in N.basis():
                if random.random() < .5: x ^= b
        direct = all(K.reduce(L.br(x, k))[0] == 0 for k in Kb)
        bad += direct != (N.reduce(x)[0] == 0)
    print(f, 'dim N', len(N), 'mismatches', bad)
