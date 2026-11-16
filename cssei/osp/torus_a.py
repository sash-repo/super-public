"""s_a: the centralizer C of its split torus T (rank 2 over F_2), and the toral elements of C over
GF(4), GF(8), GF(16): is the maximal torus of s_a of rank 2 over the algebraic closure too?
Also: is C nilpotent (a Cartan subalgebra)?"""
import json, itertools
from galg import mm, br, flat, Ech
from gf2k import GFField
S = json.load(open('s_gs2.json')); ev = S['inner']['even']; N = len(ev[0])
F2 = GFField(2)
def comb(F, c, B):
    X = [[0] * N for _ in range(N)]
    for ci, M in zip(c, B):
        if ci:
            for a in range(N):
                for b in range(N):
                    if M[a][b]: X[a][b] ^= ci
    return X
tor = [X for c in itertools.product([0, 1], repeat=len(ev)) if any(c) for X in [comb(F2, c, ev)] if mm(F2, X, X) == X]
# split torus: span of commuting toral elements (from torus.py: rank 2)
E = Ech(F2); T = []
for X in tor:
    if all(not any(any(r) for r in br(F2, X, Y)) for Y in T) and E.add(flat(X)): T.append(X)
print('split torus rank over F_2:', len(T))
# centralizer of T in s_a_ev (and of s_a_odd)
def centralizer(B):
    out = []
    for c in itertools.product([0, 1], repeat=len(B)):
        if not any(c): continue
        X = comb(F2, c, B)
        if all(not any(any(r) for r in br(F2, X, Y)) for Y in T): out.append(X)
    e = Ech(F2); bas = []
    for X in out:
        if e.add(flat(X)): bas.append(X)
    return bas
C = centralizer(ev); Co = centralizer(S['inner']['odd'])
print('centralizer of T: even dim', len(C), ', odd dim', len(Co))
# lower central series of C
L = C; series = [len(C)]
while True:
    e = Ech(F2); nb = []
    for X in C:
        for Y in L:
            Z = br(F2, X, Y)
            if e.add(flat(Z)): nb.append(Z)
    series.append(len(nb))
    if len(nb) == len(L) or not nb: break
    L = nb
print('lower central series of C:', series)
for K in (2, 3, 4):
    F = GFField(K); best = 0; cnt = 0
    tt = []
    for c in itertools.product(range(F.q), repeat=len(C)):
        if not any(c): continue
        X = comb(F, c, C)
        if mm(F, X, X) == X: cnt += 1; tt.append(X)
    e = Ech(F)
    for X in tt: e.add(flat(X))
    print('GF(%d): toral elements in C: %d, their span has dim %d' % (F.q, cnt, len(e)))
# nilpotent elements of C (ad on gs(2)-realization = the p-map of the envelope): X^(2^k) = 0
nil = 0
for c in itertools.product([0, 1], repeat=len(C)):
    if not any(c): continue
    X = comb(F2, c, C); Y = X
    for _ in range(5): Y = mm(F2, Y, Y)
    if not any(any(r) for r in Y): nil += 1
print('nonzero x in C with x^[2^5] = 0 (nilpotent, over F_2):', nil)
