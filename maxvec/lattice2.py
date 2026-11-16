"""Intermediate subalgebras u0 < h < sl(W): submodule lattice of Q = sl/u0 via cyclic submodules
(complete when all weight-parity spaces of Q are 1-dimensional)."""
import sys, itertools
from fractions import Fraction
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, solve
from sub import main
from collections import Counter

m, n, k = map(int, sys.argv[1:4])
V, W, U0, mats, pW = main(m, n, k)
N = len(W)
def tolist(M): return [[int(M[i, j]) for j in range(N)] for i in range(N)]
U = [(tolist(M), p) for M, p in mats]
def mul(A, B): return [[sum(A[i][l]*B[l][j] for l in range(N)) for j in range(N)] for i in range(N)]
def sbr(A, pa, B, pb):
    X, Y = mul(A, B), mul(B, A); s = (-1)**(pa*pb)
    return [[X[i][j]-s*Y[i][j] for j in range(N)] for i in range(N)]
flat = lambda M: [x for r in M for x in r]
def E(i, j):
    M = [[0]*N for _ in range(N)]; M[i][j] = 1; return M
tor = [A for A, p in U if p == 0 and all(A[i][j] == 0 for i in range(N) for j in range(N) if i != j)]
wt = lambda i, j: tuple(A[i][i]-A[j][j] for A in tor)
U0rows = [flat(A) for A, p in U]
ru = rank(U0rows)
reps = []
rows = list(U0rows)
for i in range(N):
    for j in range(N):
        if i == j: continue
        if rank(rows+[flat(E(i, j))]) > len(rows) if False else rank(rows+[flat(E(i, j))]) > rank(rows):
            rows.append(flat(E(i, j))); reps.append((E(i, j), (pW[i]+pW[j]) % 2, wt(i, j), (i, j)))
for i in range(N-1):
    M = [[0]*N for _ in range(N)]; M[i][i] = (-1)**pW[i]; M[i+1][i+1] = -(-1)**pW[i+1]
    if rank(rows+[flat(M)]) > rank(rows):
        rows.append(flat(M)); reps.append((M, 0, wt(0, 0), "diag%d" % i))
print("sdim Q", sum(1 for r in reps if r[1] == 0), sum(1 for r in reps if r[1] == 1))
mult = Counter((r[2], r[1]) for r in reps)
print("max weight multiplicity", max(mult.values()))
# basis of sl = u0 rows + reps ; projection coefficients
B = U0rows + [flat(r[0]) for r in reps]
nU = len(U0rows); nb = len(B)
def proj(M):
    # solve sum c_b B_b = flat(M)
    # columns = basis vectors
    A = [[B[b][t] for b in range(nb)] for t in range(N*N)]
    x = solve(A, flat(M))
    return x[nU:]
# action of u0 on Q
act = {}
for a, (A, pa) in enumerate(U):
    for q, (R, pr, w, lab) in enumerate(reps):
        act[(a, q)] = proj(sbr(A, pa, R, pr))
def cyc(S):
    S = set(S); ch = True
    while ch:
        ch = False
        for a in range(len(U)):
            for q in list(S):
                for t, c in enumerate(act[(a, q)]):
                    if c != 0 and t not in S: S.add(t); ch = True
    return frozenset(S)
C = {q: cyc([q]) for q in range(len(reps))}
# all submodules = unions of cyclic ones
subs = set([frozenset()])
for q in range(len(reps)):
    new = set()
    for s in subs: new.add(s | C[q])
    subs |= new
subs.discard(frozenset()); subs.discard(frozenset(range(len(reps))))
print("proper nonzero submodules:", len(subs))
def closed(S):
    elems = [(A, p) for A, p in U] + [(reps[q][0], reps[q][1]) for q in S]
    for x in range(len(elems)):
        for y in range(x, len(elems)):
            if x < len(U) and y < len(U): continue
            c = proj(sbr(elems[x][0], elems[x][1], elems[y][0], elems[y][1]))
            if any(c[t] != 0 and t not in S for t in range(len(reps))): return False
    return True
good = [S for S in subs if closed(S)]
for S in sorted(good, key=len):
    ev = sum(1 for q in S if reps[q][1] == 0); od = len(S)-ev
    print("subalgebra: u0 + (%d|%d):" % (ev, od), sorted(reps[q][3] for q in S))
