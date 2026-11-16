"""
Statement 10.1.7 of the paper (and Conjecture 10.1.4a, Theorem 10.1.6):
the family of (67) -- x_2 odd, A_22 = A_13 = A_31 = 0 -- with the parameter

        t = A_23 / A_21 .

(67): t = -a, a in Z_{>0}  ==>  (ad_[x1,x2])^a [x2,x3] = 0.
Conjecture: t negative rational, neither integer nor 1/integer ==> one more relation.

Method: dim g(A)_alpha computed directly as the rank of the contravariant form,
recursively: Q^alpha = L^alpha / r^alpha, r^alpha = {x : D_i x in r^(alpha - e_i) for all i}.
Exact over Q (python-flint).  For each t the dimensions are compared with those at
a generic t; a drop in multidegree alpha is a relation of g(A) not present generically.

usage: ratio.py H d1 d3 A12 A32 t1 t2 ...   (t given as p/q)
"""
import sys
from fractions import Fraction
from itertools import product
import flint

N = 3
import os
PAR = [int(c) for c in os.environ.get("PAR", "010")]


def perms(alpha):
    out = []
    def rec(pre, rem):
        if not any(rem):
            out.append(tuple(pre)); return
        for i in range(N):
            if rem[i]:
                r = list(rem); r[i] -= 1
                rec(pre + [i], r)
    rec([], list(alpha))
    return out


def pw(w):
    return sum(PAR[j] for j in w) % 2


def run(A, H):
    words, idx, Lb, P, dim = {}, {}, {}, {}, {}
    alphas = [a for h in range(1, H + 1) for a in product(range(h + 1), repeat=3) if sum(a) == h]
    for al in alphas:
        W = perms(al); words[al] = W; idx[al] = {w: k for k, w in enumerate(W)}
    for al in alphas:
        h = sum(al); W = words[al]; I = idx[al]
        if h == 1:
            Lb[al] = flint.fmpq_mat(1, 1, [1]); P[al] = flint.fmpq_mat(1, 1, [1]); dim[al] = 1
            continue
        # L^alpha = sum_i [x_i, L^{alpha-e_i}]
        rows = []
        for i in range(N):
            if al[i] == 0: continue
            be = tuple(al[j] - (j == i) for j in range(N))
            B = Lb[be]; Wb = words[be]
            for r in range(B.nrows()):
                v = [0] * len(W)
                for c in range(B.ncols()):
                    x = B[r, c]
                    if x == 0: continue
                    w = Wb[c]
                    s = -1 if (PAR[i] * pw(w)) % 2 else 1
                    v[I[(i,) + w]] += x
                    v[I[w + (i,)]] -= s * x
                rows.append(v)
        M = flint.fmpq_mat(len(rows), len(W), [x for r in rows for x in r])
        R, rk = M.rref()
        Lb[al] = flint.fmpq_mat(rk, len(W), [R[r, c] for r in range(rk) for c in range(len(W))])
        # T: words of T^alpha -> (+)_i Q^{alpha - e_i}
        blocks = []
        for i in range(N):
            if al[i] == 0: continue
            be = tuple(al[j] - (j == i) for j in range(N))
            if dim.get(be, 0) == 0: continue
            Wb = idx[be]
            D = [[0] * len(words[be]) for _ in W]
            for k, w in enumerate(W):
                for t, j in enumerate(w):
                    if j != i: continue
                    sgn = -1 if (PAR[i] * sum(PAR[w[s]] for s in range(t))) % 2 else 1
                    tail = sum(A[i][w[s]] for s in range(t + 1, len(w)))
                    if tail:
                        D[k][Wb[w[:t] + w[t + 1:]]] += -sgn * tail
            Dm = flint.fmpq_mat(len(W), len(words[be]), [x for r in D for x in r])
            blocks.append(Dm * P[be])
        if not blocks:
            dim[al] = 0; continue
        K = sum(b.ncols() for b in blocks)
        T = flint.fmpq_mat(len(W), K, [b[r, c] for r in range(len(W)) for b in blocks for c in range(b.ncols())])
        Mm = Lb[al] * T
        R2, rk2 = Mm.rref()
        dim[al] = rk2
        if rk2 == 0: continue
        piv = []
        r = 0
        for c in range(K):
            if r < rk2 and R2[r, c] != 0:
                piv.append(c); r += 1
        P[al] = flint.fmpq_mat(len(W), rk2, [T[x, c] for x in range(len(W)) for c in piv])
    return dim


def mat(d1, d3, a12, a32, t):
    # row 2 normalized by A_21 = 1, so t = A_23/A_21 = A_23
    q = lambda x: flint.fmpq(Fraction(x).numerator, Fraction(x).denominator)
    return [[q(d1), q(a12), q(0)], [q(1), q(0), q(t)], [q(0), q(a32), q(d3)]]


if __name__ == '__main__':
    H = int(sys.argv[1]); d1, d3, a12, a32 = (Fraction(x) for x in sys.argv[2:6])
    ts = [Fraction(x) for x in sys.argv[6:]]
    gen = Fraction(-7919, 104729)
    g = run(mat(d1, d3, a12, a32, gen), H)
    print('H=%d diag=(%s,0,%s) A12=%s A32=%s; generic t=%s' % (H, d1, d3, a12, a32, gen), flush=True)
    for t in ts:
        d = run(mat(d1, d3, a12, a32, t), H)
        diff = sorted((al, g[al], d[al]) for al in g if d[al] != g[al])
        first = [x for x in diff if not any(all(y[0][j] <= x[0][j] for j in range(3)) and y[0] != x[0] for y in diff)]
        print('t=%-6s drops (minimal multidegrees): %s' % (t, [(al, '%d->%d' % (a, b)) for al, a, b in first]), flush=True)
