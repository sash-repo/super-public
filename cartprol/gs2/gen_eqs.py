#!/usr/bin/env python3
r"""Equations for a representation of gs(2) (Brown's monomial basis, shenL.npz) on
V = sum_{eps in F_2^2} V_eps, dim V_eps = shape[eps], with u1d1, u2d2 acting by eps_1, eps_2
(the weights mod 2 of the torus h).  Unknowns: the blocks of the generators
u1, u2, u1vol, u2vol, u1u2; the other root vectors are their brackets; every bracket
relation of gs(2) is imposed.  Output: a file eqs_*.m read by gb.py (sympy) or gb.wl (Mathematica).
Usage: gen_eqs.py d00 d01 d10 d11 [out.m]
In the text: Statement 4.4 (no non-trivial irreducible module of dim 2..6)."""
import sys, itertools, numpy as np, sympy as sp
z = np.load('shenL.npz'); CL = z['CL'][:14, :14, :14]; names = [str(s) for s in z['names'][:14]]
I = {n: k for k, n in enumerate(names)}
d00, d01, d10, d11 = map(int, sys.argv[1:5])
GEN = '--generic' in sys.argv; sys.argv = [a for a in sys.argv if a != '--generic']
shape = {(0, 0): d00, (0, 1): d01, (1, 0): d10, (1, 1): d11}
EPS = [(0, 0), (0, 1), (1, 0), (1, 1)]
basis = [e for e in EPS for _ in range(shape[e])]      # weight of each basis vector
n = len(basis)
def modwt(k):
    return (int(CL[I['u1d1'], k, k]), int(CL[I['u2d2'], k, k]))
xs = []
def unknown(name):
    w = modwt(I[name]); M = sp.zeros(n, n)
    for r in range(n):
        for c in range(n):
            if tuple((a + b) % 2 for a, b in zip(basis[c], w)) == basis[r]:
                v = sp.Symbol('x%d' % len(xs)); xs.append(v); M[r, c] = v
    return M
def br(A, B): return (A * B + B * A).applyfunc(lambda e: sp.expand(e))
rho = {}
for g in ['u1', 'u2', 'u1vol', 'u2vol', 'u1u2']: rho[g] = unknown(g)
if GEN:   # torus eigenvalues a+eps_1, b+eps_2 with a, b unknown (not necessarily in F_2)
    a, b = sp.Symbol('x%d' % len(xs)), sp.Symbol('x%d' % (len(xs) + 1)); xs += [a, b]
else:
    a = b = 0
rho['u1d1'] = sp.diag(*[a + e[0] for e in basis]); rho['u2d2'] = sp.diag(*[b + e[1] for e in basis])
rho['vol'] = br(rho['u2'], rho['u1']); rho['d1'] = br(rho['vol'], rho['u2'])
rho['d2'] = br(rho['vol'], rho['u1']); rho['u1d2'] = br(rho['u1'], rho['u1vol'])
rho['u2d1'] = br(rho['u2'], rho['u2vol']); rho['u1u2d1'] = br(rho['u2vol'], rho['u1u2'])
rho['u1u2d2'] = br(rho['u1vol'], rho['u1u2'])
# sanity: the definitions agree with the structure constants
for a, (b, c) in {'vol': ('u2', 'u1'), 'd1': ('vol', 'u2'), 'd2': ('vol', 'u1'), 'u1d2': ('u1', 'u1vol'),
                  'u2d1': ('u2', 'u2vol'), 'u1u2d1': ('u2vol', 'u1u2'), 'u1u2d2': ('u1vol', 'u1u2')}.items():
    assert [names[k] for k in range(14) if CL[I[b], I[c], k]] == [a], (a, b, c)
eqs = set()
for i in range(14):
    for j in range(i + 1, 14):
        R = br(rho[names[i]], rho[names[j]])
        for k in range(14):
            if CL[i, j, k]: R = R - rho[names[k]]
        for e in R:
            p = sp.Poly(sp.expand(e), *xs, modulus=2) if xs else None
            if p is not None and not p.is_zero:
                eqs.add(str(p.as_expr()).replace('**', '^'))
            elif p is None and e % 2: eqs.add('1')
out = sys.argv[5] if len(sys.argv) > 5 else 'eqs_%d%d%d%d%s.m' % (d00, d01, d10, d11, 'g' if GEN else '')
with open(out, 'w') as f:
    f.write('vars = {%s};\n' % ', '.join(str(v) for v in xs))
    f.write('eqs = {%s};\n' % ',\n '.join(sorted(eqs)))
print('shape', shape, 'unknowns', len(xs), 'equations', len(eqs), '->', out)
