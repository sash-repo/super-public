import sys
from galg import *
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
F = GFField(K)
import random
random.seed(1)
alphas = [F.gen, F.exp[5], F.exp[17]]
if K % 2 == 0:
    om = F.exp[(F.q - 1) // 3]; alphas.append(om)       # alpha in F_4 \ F_2
for a in alphas:
    A = [[0, F.el(a), 0], [F.el(a), 0, 1], [0, 1, 0]]
    for lam in ([0, 0, 1], [1, 0, 0]):
        try:
            N, dims, el, central = generate(A, [1, 0, 0], lam, F)
        except Exception as ex:
            print('alpha', a, lam, 'fail', ex); continue
        print('alpha=%d lam=%s dimL=%d  dim=%d sdim=%s  1_L in deg0: %s  deg0 dim %d' % (
            a, lam, N, sum(dims.values()), sdim(dims, [1, 0, 0]), central, dims.get((0, 0, 0), 0)))
        print('   roots:', sorted(dims.items()))
