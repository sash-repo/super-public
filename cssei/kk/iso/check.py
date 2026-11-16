"""Independent check of K = {x | (ad x)^2 in ad L}: for random x, test membership
directly (solve (ad x)^2 = ad y) and compare with x in span(K)."""
import sys, random
from f2 import *
from inv1 import *
for f in sys.argv[1:]:
    L = Alg.load(f); n = L.n
    ads = [L.ad(1 << i) for i in range(n)]
    V = span(flat(A, n) for A in ads)
    res = [V.reduce(flat(mul(A, A), n))[0] for A in ads]
    K = span(kernel_mod(res, V)); random.seed(7); bad = 0; inK = 0
    for t in range(400):
        x = random.getrandbits(n)
        if t % 2: x = K.basis()[0]; [x := x ^ b for b in K.basis() if random.random() < .5]
        A = L.ad(x); direct = V.reduce(flat(mul(A, A), n))[0] == 0
        mem = K.reduce(x)[0] == 0; inK += mem; bad += direct != mem
    print(f, 'mismatches', bad, 'of 400 (in K:', inK, ')')
