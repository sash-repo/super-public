#!/usr/bin/env python3
"""H^1(osp(1|2); gl(V)) over F_p, V = L^m, by parity; the sign convention is
tested on coboundaries first.  dc(x,y) = (-1)^{p(x)p(c)} x.c(y)
 - (-1)^{p(y)(p(x)+p(c))} y.c(x) - c([x,y]),  x.A = [rho(x), A]."""
import sys, itertools
from hwmod import Fp, rref
from prol import Rep, Image, op_bracket, Span, nullspace_sparse

def run(p, m):
    r = Rep([[1]], [1], p, [m])      # osp(1|2), A = (1): the adjoint is L(2)
    K = r.K
    N = r.N
    vpar = r.parity
    img = Image(r)
    g = [(X, pp) for d, pp, X in img.basis]
    # structure constants of the image (a rep of osp(1|2); faithful for m >= 1)
    def solve_in(Z):
        keys = sorted(set(k for X, _ in g for k in X) | set(Z))
        rows = [[X.get(k, 0) for X, _ in g] + [Z.get(k, 0)] for k in keys]
        R, piv = rref(K, rows)
        n = len(g)
        assert n not in piv
        sol = [0] * n
        for i, c in enumerate(piv): sol[c] = R[i][n]
        return sol
    SC = {}
    for i, j in itertools.product(range(len(g)), repeat=2):
        SC[(i, j)] = solve_in(op_bracket(K, g[i][0], g[i][1], g[j][0], g[j][1]))
    # gl(V) basis E_ab with parity vpar[a]+vpar[b]
    E = [((a, b), (vpar[a] + vpar[b]) % 2) for a in range(N) for b in range(N)]
    idx = {e: t for t, (e, _) in enumerate(E)}
    def act(i, A, pA):        # x_i . A = [rho(x_i), A], A as dict
        return op_bracket(K, g[i][0], g[i][1], A, pA)
    out = {}
    for pc in (0, 1):
        # unknowns c(x_i) = sum over E with parity p(x_i)+pc
        unk = {}
        for i, (X, px) in enumerate(g):
            for t, (e, pe) in enumerate(E):
                if pe == (px + pc) % 2:
                    unk[(i, t)] = len(unk)
        rows = []
        sgn = lambda e: -1 if e % 2 else 1
        for i, j in itertools.combinations_with_replacement(range(len(g)), 2):
            px, py = g[i][1], g[j][1]
            row = {}
            def addop(coef, i_unk, Aop):
                for key, v in Aop.items():
                    row.setdefault(key, {})
                    u = i_unk
                    row[key][u] = K.norm(row[key].get(u, 0) + coef * v)
            for (ii, t), u in unk.items():
                e, pe = E[t]
                unit = {e: 1}
                if ii == j:
                    addop(sgn(px * pc), u, act(i, unit, pe))
                if ii == i:
                    addop(-sgn(py * (px + pc)), u, act(j, unit, pe))
                for k, ck in enumerate(SC[(i, j)]):
                    if ck and ii == k:
                        addop(-ck, u, unit)
            for key, rr in row.items():
                rr = {u: v for u, v in rr.items() if v % p}
                if rr: rows.append(rr)
        Z = nullspace_sparse(K, rows, len(unk))
        # coboundaries of A in gl(V) of parity pc: c(x) = (-1)^{p(x)pc} x.A
        B = Span(K)
        for t, (e, pe) in enumerate(E):
            if pe != pc: continue
            vec = {}
            for i, (X, px) in enumerate(g):
                for key, v in act(i, {e: 1}, pe).items():
                    u = unk[(i, idx[key])]
                    vec[u] = K.norm(vec.get(u, 0) + sgn(px * pc) * v)
            vec = {u: v for u, v in vec.items() if v}
            if vec: B.add(vec)
        # sanity: each coboundary is a cocycle
        for b in B.basis:
            for rr in rows:
                assert sum(rr.get(u, 0) * v for u, v in b.items()) % p == 0, "sign convention broken"
        out[pc] = (len(Z), len(B.basis))
    h = (out[0][0] - out[0][1], out[1][0] - out[1][1])
    print("p=%d  V=L^%d sdim %s: Z^1 %s|%s, B^1 %s|%s, H^1 = %d|%d"
          % (p, m, r.sdim(), out[0][0], out[1][0], out[0][1], out[1][1], h[0], h[1]), flush=True)
    return h

if __name__ == "__main__":
    for p in (3, 5, 7):
        for m in range(1, p + 1):
            try:
                run(p, m)
            except Exception as e:
                print("p=%d m=%d: %s" % (p, m, e))
