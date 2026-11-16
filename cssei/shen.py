#!/usr/bin/env python3
r"""Shen's analog gs(2) of g(2) = me^(1)(5; N) (BGLLS Sect. 23, after Brown):
L(N) = vect(2;N) + Vol(2;N) + O(2;N) with [f vol, g vol] = 0, [f vol, g] = f H_g,
[f, g] = H_f(g) vol, H_f = f_1 d_2 + f_2 d_1, vect acting naturally.
N = (1,1): the restricted case."""
import itertools, sys, json
from abslie import Lie, span, derived, center, two_envelope, is_irreducible_adjoint
from f2vf import BitEchelon

N = (1, 1) if len(sys.argv) < 2 else tuple(int(a) for a in sys.argv[1:3])
MON = list(itertools.product(range(1 << N[0]), range(1 << N[1])))


def mul(a, b):
    if a[0] & b[0] or a[1] & b[1]:
        return None
    return (a[0] | b[0], a[1] | b[1])


def d(i, a):
    if a[i] == 0:
        return None
    b = list(a); b[i] -= 1
    return tuple(b)


# basis: ('v', mon, i) vector field mon d_i ; ('w', mon) mon vol ; ('f', mon) function
B = [('v', a, i) for a in MON for i in (0, 1)] + [('w', a) for a in MON] + [('f', a) for a in MON]
IDX = {b: k for k, b in enumerate(B)}
n = len(B)


def vfield_apply(a, i, f):   # (x^a d_i)(x^f)
    df = d(i, f)
    return None if df is None else mul(a, df)


def br(x, y):
    X, Y = B[x], B[y]
    out = 0
    def tog(key):
        nonlocal out
        out ^= 1 << IDX[key]
    if X[0] == 'v' and Y[0] == 'v':
        (_, a, i), (_, b, j) = X, Y
        m = vfield_apply(a, i, b)
        if m is not None: tog(('v', m, j))
        m = vfield_apply(b, j, a)
        if m is not None: tog(('v', m, i))
        return out
    if X[0] != 'v' and Y[0] == 'v':
        return br(y, x)
    if X[0] == 'v':
        (_, a, i) = X
        if Y[0] == 'f':
            m = vfield_apply(a, i, Y[1])
            if m is not None: tog(('f', m))
        else:   # volume form: D(f) + div(D) f,  div(x^a d_i) = d_i x^a
            m = vfield_apply(a, i, Y[1])
            if m is not None: tog(('w', m))
            da = d(i, a)
            if da is not None:
                m = mul(da, Y[1])
                if m is not None: tog(('w', m))
        return out
    if X[0] == 'w' and Y[0] == 'w':
        return 0
    if X[0] == 'f' and Y[0] == 'w':
        return br(y, x)
    if X[0] == 'w' and Y[0] == 'f':
        f, g = X[1], Y[1]          # f H_g = f g_1 d_2 + f g_2 d_1
        for i, j in ((0, 1), (1, 0)):
            dg = d(i, g)
            if dg is not None:
                m = mul(f, dg)
                if m is not None: tog(('v', m, j))
        return out
    # f, f : H_f(g) vol = (f_1 g_2 + f_2 g_1) vol
    f, g = X[1], Y[1]
    for i, j in ((0, 1), (1, 0)):
        df, dg = d(i, f), d(j, g)
        if df is not None and dg is not None:
            m = mul(df, dg)
            if m is not None: tog(('w', m))
    return out


L = Lie(n, br)
# Jacobi
bad = 0
for a in range(n):
    for b in range(n):
        for c in range(n):
            s = L.brv(1 << a, L.br(b, c)) ^ L.brv(1 << b, L.br(c, a)) ^ L.brv(1 << c, L.br(a, b))
            if s: bad += 1
print('N =', N, 'dim L =', n, 'Jacobi failures:', bad)
Z = center(L)
print('center dim', len(Z))
D1 = derived(L)
print('dim [L,L] =', len(D1))
# me = L / center ; me^(1) = [me, me] = ([L,L] + Z)/Z
DZ = span(list(D1.piv.values()) + [z for z in Z])
print('dim me = L/Z =', n - len(Z), ';  dim me^(1) = ([L,L]+Z)/Z =', len(DZ) - len(Z))
# work in the quotient: basis of L/Z via complement; simpler: iterate derived of L until stable
S = D1
while True:
    S2 = derived(L, S)
    if len(S2) == len(S):
        break
    S = S2
print('derived series stabilizes at dim', len(S), '(contains the center?)',
      all(S.reduce(z) == 0 for z in Z))

# ---- the quotient gs(2) = ([L,L] + Z)/Z as an abstract algebra
from absutil import Coords
Zv = list(Z)
Dv = list(D1.piv.values())
# complement of Z inside [L,L]
E = BitEchelon()
for z in Zv:
    E.add(z)
comp = []
for v in Dv:
    if E.add(v):
        comp.append(v)
co = Coords(Zv + comp, as_int=True)
k = len(comp)
nz = len(Zv)
def qbr(i, j):
    c = co(L.brv(comp[i], comp[j]))
    return c >> nz          # drop the Z coordinates
Q = Lie(k, qbr)
from abslie import two_envelope, is_irreducible_adjoint, center as ctr
print('gs(2): dim', k, 'center', len(ctr(Q)))
allb = [1 << i for i in range(k)]
print('absolutely simple (adjoint module absolutely irreducible):', is_irreducible_adjoint(Q, allb))
env, d0 = two_envelope(Q, allb)
print('2-closure of ad gs(2) in End(gs(2)): dim %d (ad image %d)' % (env, d0))
# which squares are missing
import os; os.makedirs('results', exist_ok=True)
json.dump({'N': N, 'dim L': n, 'center': len(Z), 'dim me': n - len(Z), 'dim gs2': k,
           'simple': True, '2-envelope': env}, open('results/gs2_brown.json', 'w'))
