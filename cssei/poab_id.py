#!/usr/bin/env python3
"""Which of the three hyperplanes of m_0 is lambda = 0, 1, oo: the degrees of G/[G,G]
(BGLLS (2.34): b_1/b_1^(1) is spanned by M_{xi_1...xi_n}, of degree n-2; b_oo/b_oo^(1)
by M_{tau xi_1...xi_n}, of degree n; b_0 = b has the center K M_1 in degree -2)."""
import sys
import series as S
from f2vf import Space, parse_field, find_weights
from prolong import prolong, Graded

N = int(sys.argv[1])
X = ['tau'] + ['q%d' % i for i in range(1, N + 1)] + ['xi%d' % i for i in range(1, N + 1)]
neg = ['dtau'] + ['dq%d + xi%d dtau' % (i, i) for i in range(1, N + 1)] + ['dxi%d' % i for i in range(1, N + 1)]
nm = S.names(X)
sp0 = Space([1] * len(X), [2] + [1] * (2 * N))
W = find_weights(sp0, [parse_field(sp0, s, nm) for s in neg])
sp = Space([1] * len(X), [2] + [1] * (2 * N), None, W)
negf = [parse_field(sp, s, nm) for s in neg]
Gm = prolong(sp, negf, None, verbose=False, maxdeg=0)
for lab, g0b in S.poab_g0_choices(sp, Gm).items():
    G = prolong(sp, negf, [set(v) for v in g0b], verbose=False)
    D = Graded(sp)
    B = G.basis()
    for a in B:
        for b in B:
            D.add(sp.bracket(a, b))
    quot = {d: G.dim(d) - D.dim(d) for d in G.degrees() if G.dim(d) != D.dim(d)}
    # center
    cen = [v for v in G.basis(-2) if all(not sp.bracket(v, b) for b in B)]
    print(lab, 'dim G', G.dim(), 'dim [G,G]', D.dim(), 'G/[G,G] by degree', quot, 'center in deg -2:', len(cen))
