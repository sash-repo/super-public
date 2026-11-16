#!/usr/bin/env python3
"""The parabolic answers of tables (22), (23) (osp_a(4|2)): g_{>=0}(h') for the
h' printed there; each contains the maximal torus, so Lemma 2.7 1) applies:
maximal iff closure(S + beta) = everything for every root beta outside S."""
from fractions import Fraction as F
from regmax import RS
rs = RS("osp")
for h in [(-F(1,2),-F(1,2),0),(F(1,2),F(1,2),0),(0,1,0),(-1,0,0),(0,0,1),(0,1,1),(-1,0,1),(-1,1,0),(1,0,0),(-1,0,0)]:
    S = 0
    for a, r in enumerate(rs.roots):
        if sum(x*y for x, y in zip(r, h)) >= 0: S |= 1 << a
    print(h, "sdim", rs.sdim(S), "closed", rs.closed(S), "MAX" if rs.is_max(S) else "not max")
