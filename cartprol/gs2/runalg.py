#!/usr/bin/env python3
"""In the text: Theorem 4.6, 1)-3) (structure constants of the finite prolongs of V_7)."""
import numpy as np, pickle, sys
from lalg import *
from buildalg import *
from runprolong import parity, conj, Din, Dout, mods
M = mods[(1, 0)]; n = 7; Id = np.eye(n, dtype=np.int64)
gs = [M[k] for k in range(NG)]; Lm = [M[k] for k in range(N)]
cases = [('V7 + gs(2)', gs, ()), ('V7 + gs(2)+K1', gs + [Id], ()), ('Pi(V7) + L', Lm, tuple(range(n))),
         ('Pi(V7) + L+K1', Lm + [Id], tuple(range(n)))]
P, odd = parity(M, Din); Mc = conj(M, P)
sa = [Mc[k] for k in range(N) if NAMES[k] != '(u2)^2']
cases.append(('V7(3|4) + s_a', sa, tuple(odd)))
P, odd = parity(M, Dout); Mc = conj(M, P)
cases.append(('V7(3|4) + s_b', [Mc[k] for k in range(N)], tuple(odd)))
for nm, g0, od in cases:
    G = Graded(g0, n, od)
    ok = check_jacobi(G.C)
    D1 = span_derived(G.C); Z = center(G.C)
    print('%-16s dims %s total %d  Jacobi %s  center %d  dim [g,g] %d  assoc(ad) %d (d^2=%d)' % (
        nm, G.dims, G.d, ok, len(Z), len(D1), assoc_dim(G.C), G.d ** 2), flush=True)
    pickle.dump((G.C, G.dims), open('alg_%s.pkl' % nm.replace(' ', '').replace('+', '_').replace('(', '').replace(')', '').replace('|', ''), 'wb'))
