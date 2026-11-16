#!/usr/bin/env python3
"""P1(6) is thin: label the basis x_i, x_ix_j, X_J+X_J' by the weight of the torus
T1=<x_i^(2)+x_j^(2)> (F_2^6 mod (1,...,1) = F_2^5); build its thin table and test
GL_5(F_2)-equivalence with the CSS thin table of LS (file PO.txt)."""
import json, itertools, sys, ast
import kk
B = kk.gen(kk.Gm1 + kk.G0p + kk.M)
B.sort(key=kk.deg)
ex = kk.coords(B)
def w5(f):
    ws = set()
    for m in f:
        v = sum(1 << i for i in range(6) if m[i] % 2)   # weights of x^(a): a mod 2 (a_i<=1 here)
        if v >> 5 & 1: v ^= 63
        ws.add(v)
    assert len(ws) == 1, f; return ws.pop()
lab = [w5(f) for f in B]
assert sorted(lab) == list(range(1, 32)), lab
n = 31
T = [[0]*32 for _ in range(32)]
for i in range(n):
    for j in range(n):
        c = kk.br(B[i], B[j])
        if c:
            k = ex(c); assert k == 1 << [t for t in range(n) if lab[t] == lab[i] ^ lab[j]][0]
            T[lab[i]][lab[j]] = 1
# Jacobi-thin sanity: T symmetric, zero diagonal
assert all(T[a][a] == 0 for a in range(32)) and all(T[a][b] == T[b][a] for a in range(32) for b in range(32))
def load(p):
    M = ast.literal_eval(open(p).read())
    S = [[0]*32 for _ in range(32)]
    for a in range(31):
        for b in range(31): S[a+1][b+1] = M[a][b]
    return S
def equiv(T, S):
    """g in GL5 with S[g a][g b] = T[a][b]; g given by images of 1,2,4,8,16."""
    def img(g, a):
        r = 0
        for k in range(5):
            if a >> k & 1: r ^= g[k]
        return r
    sols = []
    def rec(g):
        k = len(g)
        span = {img(g, a) for a in range(1 << k)}
        # check consistency on the span of the assigned basis
        for a in range(1, 1 << k):
            for b in range(1, 1 << k):
                if T[a][b] != S[img(g, a)][img(g, b)]: return
        if k == 5: sols.append(tuple(g)); return
        for c in range(1, 32):
            if c not in span: rec(g + [c])
    rec([]); return sols
if __name__ == '__main__':
    lib = sys.argv[1]
    import os
    for f in sorted(os.listdir(lib)):
        S = load(os.path.join(lib, f))
        s = equiv(T, S)
        print('%-22s %s' % (f, ('EQUIVALENT, %d maps, e.g. %s' % (len(s), s[0])) if s else 'no'), flush=True)
    json.dump(T, open('P1_thin.json', 'w'))
