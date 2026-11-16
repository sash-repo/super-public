"""Search for Chevalley generators in the outer (9|8) superization of gs(2) over F_2: a basis of
simple weights w_1,w_2,w_3 of the split 3-torus T, e_i,f_i in the w_i-weight space with
[e_i,f_j] = delta_ij h_i, h_i in T, [h_i,e_j] = a_ij e_j, [h_i,f_j] = a_ij f_j, generating."""
import json, itertools, sys
from collections import Counter
from galg import GFField, mm, br, flat, Ech
from msa import MSA, nullspace
from torus import analyse, add, comb
F = GFField(2)
S = json.load(open('s_gs2.json'))
which = sys.argv[1] if len(sys.argv) > 1 else 'outer'
ev, od = S[which]['even'], S[which]['odd']
A = MSA(F, ev, od); d = A.d; allb = ev + od
N = len(ev[0])
def vec2mat(c): return comb(allb, c)
def iszero(X): return not any(any(r) for r in X)
# torus: redo the greedy search to get the basis
tor = []
for c in itertools.product([0, 1], repeat=len(ev)):
    if not any(c): continue
    X = comb(ev, c)
    if mm(F, X, X) == X: tor.append(X)
def comm(X, Y): return iszero(br(F, X, Y))
best = []
for X0 in tor:
    fam = [X0]
    for X in tor:
        if all(comm(X, Y) for Y in fam): fam.append(X)
    E = Ech(F); basis = []
    for X in fam:
        if E.add(flat(X)): basis.append(X)
    if len(basis) > len(best): best = basis
T = best; r = len(T)
Tech = Ech(F)
for X in T: Tech.add(flat(X))
# weight spaces
def wspace(w):
    out = []
    for c in itertools.product([0, 1], repeat=d):
        if not any(c): continue
        X = vec2mat(c)
        ok = True
        for t, wt in zip(T, w):
            Y = br(F, t, X)
            if Y != (X if wt else [[0] * N for _ in range(N)]): ok = False; break
        if ok: out.append((c, X))
    return out
W = {}
for w in itertools.product([0, 1], repeat=r):
    if any(w): W[w] = wspace(w)
print('torus rank', r, 'weight space sizes (nonzero vectors):', {w: len(v) for w, v in W.items()})
def par_of(c): return sum(ci for ci, p in zip(c, A.par) if p) and 1 or 0
def homog(c):
    e = any(ci for ci, p in zip(c, A.par) if not p); o = any(ci for ci, p in zip(c, A.par) if p)
    return None if (e and o) else (1 if o else 0)
def gen_dim(gens, pars):
    """closure under brackets and squares of ODD homogeneous elements (parities tracked)"""
    E = Ech(F); q = []
    for X, p in zip(gens, pars):
        if E.add(flat(X)): q.append((X, p))
    while q:
        X, p = q.pop()
        for Y, pY in zip(gens, pars):
            Z = br(F, X, Y)
            if E.add(flat(Z)): q.append((Z, (p + pY) % 2))
        if p:
            Z = mm(F, X, X)
            if E.add(flat(Z)): q.append((Z, 0))
    return len(E)
def gen_dim_nosq(gens):
    E = Ech(F); q = []
    for X in gens:
        if E.add(flat(X)): q.append(X)
    while q:
        X = q.pop()
        for Y in gens:
            Z = br(F, X, Y)
            if E.add(flat(Z)): q.append(Z)
    return len(E)
weights = list(W)
found = Counter()
examples = {}
for B in itertools.combinations(weights, r):
    # linearly independent over F_2
    M = [list(b) for b in B]
    if len(nullspace(F, [list(col) for col in zip(*M)], r)) != 0: continue
    cand = [[(c, X) for c, X in W[w] if homog(c) is not None] for w in B]
    for choice in itertools.product(*[list(itertools.permutations(cd, 2)) for cd in cand]):
        es = [ch[0] for ch in choice]; fs = [ch[1] for ch in choice]
        if any(homog(e[0]) != homog(f[0]) for e, f in zip(es, fs)): continue
        ok = True; hs = []
        for i in range(r):
            for j in range(r):
                Z = br(F, es[i][1], fs[j][1])
                if i != j and not iszero(Z): ok = False; break
                if i == j:
                    if iszero(Z) or any(Tech.reduce(flat(Z))): ok = False; break
                    hs.append(Z)
            if not ok: break
        if not ok: continue
        Am = []
        for i in range(r):
            row = []
            for j in range(r):
                Ze = br(F, hs[i], es[j][1]); Zf = br(F, hs[i], fs[j][1])
                if iszero(Ze) and iszero(Zf): row.append(0)
                elif Ze == es[j][1] and Zf == fs[j][1]: row.append(1)
                else: ok = False; break
            if not ok: break
            Am.append(tuple(row))
        if not ok: continue
        gd = gen_dim([e[1] for e in es] + [f[1] for f in fs], [homog(e[0]) for e in es] * 2)
        gdn = gen_dim_nosq([e[1] for e in es] + [f[1] for f in fs])
        parv = tuple(homog(e[0]) for e in es)
        key = (tuple(Am), parv, gd, gdn)
        found[key] += 1
        examples.setdefault(key, (B, es, fs))
for k, v in sorted(found.items()):
    print('A =', k[0], ' parity', k[1], ' generated dim (with squares) %d, (brackets only) %d' % (k[2], k[3]), ':', v)
