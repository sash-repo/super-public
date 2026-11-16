"""Is Q = sl(W)/u0 an irreducible u0-module?  Every nonzero submodule meets
K = common kernel (in Q) of the u0 elements of positive weight for a generic functional;
so it suffices to test the weight vectors of K (generically, if a weight space of K is >1-dim)."""
import sys, random
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
exec(open(os.path.join(_HERE, "lattice2.py")).read().split("C = {q")[0])
from exactlin import nullspace
random.seed(1)
nq = len(reps)
# generic functional on torus weights
lam = [random.randint(1, 10**6) for _ in tor]
uw = []
for a, (A, pa) in enumerate(U):
    # weight of u0 element: it is a monomial field, so a weight vector; read via [t, A] = w(t) A
    w = None
    for i in range(N):
        for j in range(N):
            if A[i][j] != 0: w = wt(i, j); break
        if w: break
    uw.append(w if w is not None else tuple([0]*len(tor)))
pos = [a for a in range(len(U)) if sum(l*x for l, x in zip(lam, uw[a])) > 0]
# matrix of action of u0 element a on Q: column q = act[(a,q)]
rowsK = []
for a in pos:
    for t in range(nq):
        rowsK.append([act[(a, q)][t] for q in range(nq)])
K = nullspace(rowsK) if rowsK else [[int(i == j) for j in range(nq)] for i in range(nq)]
print("dim K", len(K))
def gen(v):
    S = [list(v)]; r = rank(S); frontier = [list(v)]
    while frontier:
        new = []
        for x in frontier:
            for a in range(len(U)):
                y = [sum(act[(a, q)][t]*x[q] for q in range(nq)) for t in range(nq)]
                if any(y) and rank(S+[y]) > r:
                    S.append(y); r += 1; new.append(y)
        frontier = new
    return r
# split K by (weight, parity) of reps
from collections import defaultdict
blocks = defaultdict(list)
for v in K:
    keys = {(reps[q][2], reps[q][1]) for q in range(nq) if v[q] != 0}
    blocks[frozenset(keys)].append(v)
for key, vs in blocks.items():
    for trial in range(3):
        c = [random.randint(-50, 50) for _ in vs]
        v = [sum(ci*vi[q] for ci, vi in zip(c, vs)) for q in range(nq)]
        print(sorted(key)[:2], "block dim", len(vs), "generates", gen(v), "of", nq)

def genbasis(v):
    S = [list(v)]; frontier = [list(v)]
    while frontier:
        new = []
        for x in frontier:
            for a in range(len(U)):
                y = [sum(act[(a, q)][t]*x[q] for q in range(nq)) for t in range(nq)]
                if any(y) and rank(S+[y]) > rank(S):
                    S.append(y); new.append(y)
        frontier = new
    return S
def tomat(v):
    M = [[0]*N for _ in range(N)]
    for q in range(nq):
        if v[q]:
            for i in range(N):
                for j in range(N): M[i][j] += v[q]*reps[q][0][i][j]
    return M
def vpar(v):
    ps = {reps[q][1] for q in range(nq) if v[q] != 0}; assert len(ps) == 1; return ps.pop()
print("--- closure of u0 + <generated submodule>")
for key, vs in blocks.items():
    S = genbasis(vs[0])
    # homogeneous basis: split by parity
    Sh = []
    for v in S:
        for p in (0, 1):
            w = [v[q] if reps[q][1] == p else 0 for q in range(nq)]
            if any(w) and rank(Sh+[w]) > rank(Sh): Sh.append(w)
    r = rank(Sh); ok = True
    for x in range(len(Sh)):
        for y in range(x, len(Sh)):
            c = proj(sbr(tomat(Sh[x]), vpar(Sh[x]), tomat(Sh[y]), vpar(Sh[y])))
            if rank(Sh+[c]) > r: ok = False; break
        if not ok: break
    ev = sum(1 for v in Sh if vpar(v) == 0)
    print(sorted(key), "submodule sdim (%d|%d)" % (ev, len(Sh)-ev), "u0+M closed:", ok)
