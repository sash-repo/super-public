"""sh_Pi(2n;1) = {H_f | Delta f = 0}, Delta = sum_i d_i d_{i+n}, H_f = sum_i (d_i f d_{i+n} + d_{i+n} f d_i),
as matrices on O(2n;1) (dim 2^{2n}); its simple factor's fingerprint.  usage: shpi.py n"""
import sys, numpy as np
from lie import MatAmb, closure, table_of, factors
from known import fpr
from gf2 import nullspace
n = int(sys.argv[1]); m = 2 * n; N = 1 << m
amb = MatAmb(N, 'W(%d;1)' % m)
def fieldmat(g, j):   # g: set of monomials, field g d_j
    M = np.zeros((N, N), np.uint8)
    for b in range(N):
        if b >> j & 1:
            bp = b ^ (1 << j)
            for a in g:
                if a & bp == 0: M[a | bp, b] ^= 1
    return M
def d(f, i): return {a ^ (1 << i) for a in f if a >> i & 1}
def H(f):
    M = np.zeros((N, N), np.uint8)
    for i in range(n):
        M ^= fieldmat(d(f, i), i + n) ^ fieldmat(d(f, i + n), i)
    return M.ravel()
# kernel of Delta on O
Dl = np.zeros((N, N), np.uint8)
for a in range(N):
    for i in range(n):
        if a >> i & 1 and a >> (i + n) & 1: Dl[a ^ (1 << i) ^ (1 << (i + n)), a] ^= 1
K = nullspace(Dl)
gens = [H({a for a in range(N) if v[a]}) for v in K]
gens = [g for g in gens if g.any()]
E = closure(amb, gens, 10 ** 6)
rng = np.random.default_rng(2); out = []
ab = factors(table_of(amb, E), rng, out)
print('sh_Pi(%d;1): span dim %d, abelian %d' % (m, E.dim(), ab), [fpr(o[0], rng) for o in out])
