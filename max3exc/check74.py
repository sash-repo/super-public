#!/usr/bin/env python3
"""ag(2): every subalgebra h + S of sdim (7|4) (S symmetric closed): its ideals'
shape, and the proper closed sets strictly containing it."""
from regmax import RS, symmetric_maximal
from classify import GRAM, even_type, weyl_orbit
from ident import components
rs = RS("ag"); G = GRAM["ag"]
pairs = sorted({tuple(sorted((a, rs.neg[a]))) for a in range(rs.n)})
maxs = symmetric_maximal(rs)
hits = []
for bits in range(1, 1 << len(pairs)):
    S = 0
    for t, (a, b) in enumerate(pairs):
        if bits >> t & 1: S |= (1 << a) | (1 << b)
    if rs.closed(S) and rs.sdim(S) == (7, 4):
        hits.append(S)
print(len(hits), "symmetric closed sets of sdim (7|4)")
seen = set()
for S in hits:
    if S in seen: continue
    seen |= weyl_orbit(rs, S, G)
    print(" type", even_type(rs, S, G), "components", components(rs, S, G))
    over = [M for M in maxs if M & S == S]
    print("   contained in maximal ones of sdim", [rs.sdim(M) for M in over],
          [even_type(rs, M, G) for M in over])
    # is it maximal? (it cannot be if 'over' is nonempty)
    print("   is_max:", rs.is_max(S))
