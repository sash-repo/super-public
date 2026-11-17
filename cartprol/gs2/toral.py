#!/usr/bin/env python3
"""All toral elements x = x^[2] of L over F_2; inner (in gs(2)) vs outer; eigen-gradings of L."""
import numpy as np, itertools
from gfla import GF
from lalg import *
F = GF(1)
tor = []
for bits in range(1, 1 << N):
    x = np.array([(bits >> k) & 1 for k in range(N)], dtype=np.int64)
    if (psq(F, x) == x).all(): tor.append(x)
inner = [x for x in tor if not x[NG:].any()]
print('toral elements over F_2:', len(tor), ' inner:', len(inner), ' outer:', len(tor) - len(inner))
def grading(x):
    A = admat(F, x)
    ev = F.nullspace(A); od = F.nullspace(A ^ np.eye(N, dtype=np.int64))
    return len(ev), len(od)
from collections import Counter
print('inner gradings (dim L_0, dim L_1):', Counter(grading(x) for x in inner))
print('outer gradings:', Counter(grading(x) for x in tor if x[NG:].any()))
# outer toral elements commuting with h?
H = [NAMES.index('u1d1'), NAMES.index('u2d2')]
e = np.eye(N, dtype=np.int64)
for x in tor[:0]: pass
out = [x for x in tor if x[NG:].any()]
lowest = min(out, key=lambda x: x.sum())
print('an outer toral element of least support:', ' + '.join(NAMES[k] for k in np.nonzero(lowest)[0]))
np.save('outerD.npy', lowest)
np.save('toral.npy', np.array(tor))
