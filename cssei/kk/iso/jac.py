import sys
from f2 import Alg
for f in sys.argv[1:]:
    L = Alg.load(f); n = L.n; bad = 0
    for i in range(n):
        for j in range(i+1, n):
            bij = L.tab[i][j]
            for k in range(j+1, n):
                if L.br(1<<i, L.tab[j][k]) ^ L.br(1<<j, L.tab[k][i]) ^ L.br(1<<k, bij): bad += 1
    print(f, 'Jacobi failures on basis triples', bad)
