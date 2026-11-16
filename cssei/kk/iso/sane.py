import sys, random
from f2 import Alg
for f in sys.argv[1:]:
    L = Alg.load(f); n = L.n; random.seed(1)
    ok = all(L.tab[i][i] == 0 and L.tab[i][j] == L.tab[j][i] for i in range(n) for j in range(n))
    for _ in range(300):
        a, b, c = (random.getrandbits(n) for _ in range(3))
        ok &= L.br(a, L.br(b, c)) ^ L.br(b, L.br(c, a)) ^ L.br(c, L.br(a, b)) == 0
    print(f, n, 'alt+Jacobi', ok)
