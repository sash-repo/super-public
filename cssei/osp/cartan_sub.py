"""For s_a and s_b, is the centralizer C of a maximal (split) torus T a Cartan
subalgebra, i.e., nilpotent and self-normalizing in s (normalizer taken in s_ev and in s_od)?"""
import json, itertools
from galg import mm, br, flat, Ech
from gf2k import GFField
F2 = GFField(2)
S = json.load(open('s_gs2.json'))
def comb(c, B):
    N = len(B[0]); X = [[0]*N for _ in range(N)]
    for ci, M in zip(c, B):
        if ci:
            for a in range(N):
                for b in range(N):
                    if M[a][b]: X[a][b] ^= 1
    return X
def zero(X): return not any(any(r) for r in X)
def span_dim(L):
    e = Ech(F2); return sum(1 for X in L if e.add(flat(X)))
def basis(L):
    e = Ech(F2); return [X for X in L if e.add(flat(X))]
for name in ('inner', 'outer'):
    ev, od = S[name]['even'], S[name]['odd']
    tor = [X for c in itertools.product([0,1], repeat=len(ev)) if any(c) for X in [comb(c, ev)] if mm(F2, X, X) == X]
    best = []
    for X0 in tor:
        fam = [X0]
        for X in tor:
            if all(zero(br(F2, X, Y)) for Y in fam): fam.append(X)
        b = basis(fam)
        if len(b) > len(best): best = b
    T = best
    def cent(B):
        return basis([X for c in itertools.product([0,1], repeat=len(B)) if any(c) for X in [comb(c, B)] if all(zero(br(F2, X, Y)) for Y in T)])
    C, Co = cent(ev), cent(od)
    Cdim = len(C)
    def normal(B):
        out = []
        for c in itertools.product([0,1], repeat=len(B)):
            if not any(c): continue
            X = comb(c, B)
            if all(span_dim(C + [br(F2, X, Y)]) == Cdim for Y in C): out.append(X)
        return basis(out)
    Ne, No = normal(ev), normal(od)
    # nilpotency of C: lower central series
    L = C; ser = [len(C)]
    while L:
        nb = basis([br(F2, X, Y) for X in C for Y in L])
        ser.append(len(nb))
        if len(nb) == len(L): break
        L = nb
    s = 's_a' if name == 'inner' else 's_b'
    print(f'{s}: torus rank {len(T)}; C sdim ({len(C)}|{len(Co)}); normalizer sdim ({len(Ne)}|{len(No)}); '
          f'lower central series of C {ser}; toral elements in C: {sum(1 for c in itertools.product([0,1],repeat=len(C)) if any(c) and mm(F2,comb(c,C),comb(c,C))==comb(c,C))}')
