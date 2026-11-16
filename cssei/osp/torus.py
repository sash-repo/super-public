"""Split maximal tori (over F_2) of the even part of a restricted matrix superalgebra, and the
weight decomposition of the whole algebra with respect to them."""
import json, itertools, sys
from collections import Counter
from galg import GFField, mm, br, flat
from msa import MSA
F = GFField(2)
def add(X, Y): return [[a ^ b for a, b in zip(r, s)] for r, s in zip(X, Y)]
def comb(B, c):
    N = len(B[0]); X = [[0] * N for _ in range(N)]
    for ci, M in zip(c, B):
        if ci: X = add(X, M)
    return X
def analyse(name, ev, od):
    N = len(ev[0])
    tor = []
    for c in itertools.product([0, 1], repeat=len(ev)):
        if not any(c): continue
        X = comb(ev, c)
        if mm(F, X, X) == X: tor.append((c, X))
    # commuting maximal families (exhaustive over subsets would be big; greedy from each start)
    def comm(X, Y): return not any(any(r) for r in br(F, X, Y))
    best = None
    for c0, X0 in tor:
        fam = [X0]
        for c, X in tor:
            if all(comm(X, Y) for Y in fam): fam.append(X)
        # rank of span
        from galg import Ech
        E = Ech(F); basis = []
        for X in fam:
            if E.add(flat(X)): basis.append(X)
        if best is None or len(basis) > len(best): best = basis
    r = len(best)
    # weight decomposition: each torus basis element T acts on the algebra by ad; eigenvalues 0,1
    A = MSA(F, ev, od)
    allb = ev + od
    # simultaneous eigenspaces: brute force over the F_2-span of algebra with weights
    tcoords = [A.coords(T) for T in best]
    admats = []
    for tc in tcoords:
        M = [[0] * A.d for _ in range(A.d)]
        for i, ci in enumerate(tc):
            if ci:
                Ai = A.admat(i)
                M = [[a ^ b for a, b in zip(r1, r2)] for r1, r2 in zip(M, Ai)]
        admats.append(M)
    from msa import nullspace
    wts = Counter()
    for w in itertools.product([0, 1], repeat=r):
        for par, idxs in ((0, [i for i in range(A.d) if A.par[i] == 0]), (1, [i for i in range(A.d) if A.par[i] == 1])):
            rows = []
            for M, wt in zip(admats, w):
                for k in range(A.d):
                    row = [0] * len(idxs)
                    for jj, j in enumerate(idxs):
                        row[jj] = M[k][j] ^ (wt if k == j else 0)
                    rows.append(row)
            dim = len(nullspace(F, rows, len(idxs)))
            if dim: wts[(w, par)] = dim
    print('%s: %d toral elements over F_2; split torus rank %d' % (name, len(tor), r))
    for k in sorted(wts): print('   weight', k[0], 'even' if k[1] == 0 else 'odd', wts[k])
    return len(tor), r, wts
if __name__ == '__main__':
    S = json.load(open('s_gs2.json'))
    for t in ('inner', 'outer'):
        analyse('gs2 ' + t, S[t]['even'], S[t]['odd'])
    from ospc import candidate, I, Pi
    ev, od = candidate(I(4), Pi(2), True, False)
    analyse('osp(1_4|Pi_2) alt', ev, od)
