"""In the text: Theorem 4.6, 4) (the derived algebra of (V_8, gs(2))_* is 30-dimensional and absolutely simple)."""
import numpy as np, pickle, json, sys
from lalg import *
from buildalg import Graded, check_jacobi, assoc_dim
from analyze import series, derived
from f2lin import rank
mods8 = pickle.load(open('mods8.pkl', 'rb'))
for key, (Q, chi) in mods8.items():
    n = 8; Id = np.eye(n, dtype=np.int64)
    for nm, g0 in [('gs2', [Q[k] for k in range(NG)]), ('c gs2', [Q[k] for k in range(NG)] + [Id])]:
        G = Graded(g0, n, ())
        ser, core = series(G.C)
        D = derived(G.C); off = np.cumsum([0] + G.dims)
        k = core.shape[0]
        print(key, nm, 'dims', G.dims, 'Jacobi', check_jacobi(G.C), 'derived series', ser,
              'first derived by degree', [rank(D[:, off[i]:off[i + 1]]) for i in range(len(G.dims))],
              'core', k, 'abs simple', assoc_dim(core) == k * k, flush=True)
        if nm == 'gs2':
            tab = [[int(''.join(map(str, core[i, j][::-1])), 2) for j in range(k)] for i in range(k)]
            json.dump({'name': 'V8 %s' % (key,), 'dim': k, 'tab': tab}, open('alg_V8_%d_%d_core.json' % (key[0], key[1]), 'w'))
