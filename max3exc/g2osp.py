#!/usr/bin/env python3
"""ab(3) = F(4): u = g(2) + osp(1|2), where g(2) = stabilizer in o(7) of a
non-isotropic spinor w0 (spin_7 = 7 + C w0) and osp(1|2) = sl(2) + C^2 (x) w0.
(1) u is a subalgebra of sdim (17|2); (2) g/u = 7 + 2(x)7 as a g(2)+sl(2)-module,
both irreducible and non-isomorphic, so u is maximal iff neither u+7 nor
u + 2(x)7 is closed -- tested by brackets in the exact model."""
import sys, os
from fractions import Fraction as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exc3
from lin import span, inspan, kernel

m = exc3.build_F4()
N, dV, ds = m.N, m.dV, m.ds
G = m.G
# a non-isotropic spinor: first pair of basis vectors with G[a][b] != 0
a0, b0 = next((a, b) for a in range(dV) for b in range(dV) if G[a][b] != 0 and a != b)
w0 = [F(0)] * dV; w0[a0] = F(1); w0[b0] = F(1)
norm = sum(w0[i] * G[i][j] * w0[j] for i in range(dV) for j in range(dV))
assert norm != 0
# stabilizer of w0 in o(7)
rows = []
for k in range(dV):
    rows.append([sum(m.rho[i][k][j] * w0[j] for j in range(dV)) for i in range(ds)])
stab = kernel(rows, ds)
print("dim stab_o(7)(w0) =", len(stab))
assert len(stab) == 14

def vec_s(c):
    v = [F(0)] * N
    for i in range(ds): v[i] = c[i]
    return v

def odd(eps, vecV):
    v = [F(0)] * N
    for a in range(dV):
        if vecV[a]: v[m.iodd(a, eps)] = vecV[a]
    return v

U = [vec_s(c) for c in stab] + [m.e(m.iE), m.e(m.iH), m.e(m.iF)] + [odd(0, w0), odd(1, w0)]
U = span(U)
def closed(B):
    for x in B:
        for y in B:
            if not inspan(m.br(x, y), B):
                return False
    return True
print("u closed:", closed(U), " sdim (17|2) expected, dim", len(U))
# complements: 7 in o(7) and 2(x)7 in odd part
o7 = span([m.e(i) for i in range(ds)])
odd_all = [m.e(m.iodd(a, e)) for a in range(dV) for e in (0, 1)]
U7 = span(U + o7)                      # u + 7  (= g_ev + C^2 w0)
U27 = span(U + odd_all)                # u + 2(x)7 (= g2+sl2 + all odd)
print("u + 7 closed:", closed(U7), " dim", len(U7))
print("u + 2(x)7 closed:", closed(U27), " dim", len(U27))
