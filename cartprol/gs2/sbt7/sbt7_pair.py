#!/usr/bin/env python3
"""The pair (J_{-1}, J_0) of BGLLS's sb~(7;1) core J: J_0 exported as a
table (sbt7_J0.json, read by isoJ0w.py), and the 2-closure of ad(J_0)|J_{-1} in gl(7)."""
import sys, os, json, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from der import core
    from absutil import Coords
    sp, G, J = core('sb~7')
B = J.basis(); n = len(B); co = Coords(B)
deg = [sp.kdeg(next(iter(v))) for v in B]
print('J degrees:', {d: deg.count(d) for d in sorted(set(deg))})
tab = [[co(sp.bracket(B[i], B[j])) for j in range(n)] for i in range(n)]
tab = [[c if isinstance(c, int) else sum(1 << k for k in c) for c in row] for row in tab]
I0 = [i for i in range(n) if deg[i] == 0]; Im = [i for i in range(n) if deg[i] == min(deg)]
# J_0 table in its own coordinates
def sub(v, idx):  # bitmask over J -> bitmask over idx
    out = 0
    for t, i in enumerate(idx):
        if (v >> i) & 1: out |= 1 << t
    assert v & ~sum(1 << i for i in idx) == 0
    return out
t0 = [[sub(tab[i][j], I0) for j in I0] for i in I0]
json.dump({'name': 'sb~7 J_0', 'dim': len(I0), 'tab': t0}, open('../sbt7_J0.json', 'w'))
# matrices of ad(J_0) on J_{-1}
import numpy as np
mats = []
for i in I0:
    M = np.zeros((len(Im), len(Im)), dtype=np.int64)
    for c, j in enumerate(Im):
        v = sub(tab[i][j], Im)
        for r in range(len(Im)):
            M[r, c] = (v >> r) & 1
    mats.append(M)
sys.path.insert(0, '..')
from f2lin import rank
E = [m.flatten() for m in mats]
def inspan(M, E):
    return rank(np.array(E + [M.flatten()], dtype=np.uint8)) == rank(np.array(E, dtype=np.uint8))
basis = list(mats); changed = True
while changed:
    changed = False
    for m in list(basis):
        s = (m @ m) & 1
        if not inspan(s, [b.flatten() for b in basis]): basis.append(s); changed = True
print('dim J_0 =', len(I0), ' dim J_{-1} =', len(Im), ' 2-closure of ad(J_0)|J_{-1} in gl(7):', len(basis),
      ' contains 1:', inspan(np.eye(len(Im), dtype=np.int64), [b.flatten() for b in basis]))
