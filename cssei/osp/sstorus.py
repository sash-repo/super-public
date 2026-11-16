"""Semisimple elements (as matrices in End gs(2)) of [s,s]_ev = ad g_0 over GF(4) and GF(8),
for both kinds of gradings: is there a torus of rank >= 2 inside the bracket-derived algebra?"""
import json, itertools, sys
S = json.load(open('s_gs2.json'))
from galg import GFField, br, mm, Ech, flat
for K in (2, 3):
    F = GFField(K)
    for t in ('inner', 'outer'):
        ev = S[t]['even']; od = S[t]['odd']
        # [s,s]_ev basis over F_2
        E = Ech(GFField(2)); basis = []
        allb = ev + od; ne = len(ev)
        for i, X in enumerate(allb):
            for j, Y in enumerate(allb):
                if (i >= ne) == (j >= ne):
                    Z = br(F, X, Y)
                    if E.add(flat(Z)): basis.append(Z)
        n = len(basis); N = len(basis[0])
        ss = []
        for c in itertools.product(range(F.q), repeat=n):
            if not any(c): continue
            if next(x for x in c if x) != 1: continue      # projective
            X = [[0] * N for _ in range(N)]
            for ci, B in zip(c, basis):
                if ci:
                    for a in range(N):
                        for b in range(N):
                            if B[a][b]: X[a][b] ^= ci
            Y = X; semis = False
            for m in range(1, 13):
                Y = mm(F, Y, Y)
                if Y == X: semis = True; break
                if not any(any(r) for r in Y): break
            if semis: ss.append(X)
        # rank of tori: max number of pairwise commuting linearly independent semisimple elements (greedy)
        best = 0
        for X in ss[:200]:
            fam = [X]
            for Y in ss:
                if all(not any(any(r) for r in br(F, Y, Z)) for Z in fam):
                    fam.append(Y)
            e = Ech(F)
            for Z in fam: e.add(flat(Z))
            best = max(best, len(e))
        print('GF(%d) %s: dim [s,s]_ev = %d; semisimple projective points %d; torus rank inside [s,s]_ev >= %d (greedy)' % (F.q, t, n, len(ss), best))
