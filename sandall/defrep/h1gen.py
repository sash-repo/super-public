#!/usr/bin/env python3
"""H^1(g; V) by parity and by Z^n-degree, for g = image of g(A) in gl(L) and
V = gl(L) (so H^1 = infinitesimal deformations of L) or V = g itself (adjoint:
H^1 = outer derivations).  Sign convention checked on coboundaries."""
import sys, itertools
from hwmod import Fp, rref
from prol import Rep, Image, op_bracket, Span, nullspace_sparse, vadd

REPS = {}
CTX = {}

def h1(A, par, p, lam, target="end", verbose=True):
    r = Rep(A, par, p, lam)
    K = r.K
    img = Image(r)
    g = [(X, pp, d) for d, pp, X in img.basis]
    n = len(g)
    sgn = lambda e: -1 if e % 2 else 1
    def coords(Z, basis):
        keys = sorted(set(k for X in basis for k in X) | set(Z))
        rows = [[X.get(k, 0) for X in basis] + [Z.get(k, 0)] for k in keys]
        R, piv = rref(K, rows)
        m = len(basis)
        if m in piv: return None
        sol = [0] * m
        for i, c in enumerate(piv): sol[c] = R[i][m]
        return sol
    gb = [X for X, _, _ in g]
    SC = {(i, j): coords(op_bracket(K, g[i][0], g[i][1], g[j][0], g[j][1]), gb)
          for i in range(n) for j in range(n)}
    if target == "end":
        N = r.N
        V = [({(a, b): 1}, (r.parity[a] + r.parity[b]) % 2,
              tuple(x - y for x, y in zip(r.deg[a], r.deg[b]))) for a in range(N) for b in range(N)]
        def act(i, t):   # x_i . v_t as coordinates over V
            Z = op_bracket(K, g[i][0], g[i][1], V[t][0], V[t][1])
            out = {}
            for (a, b), v in Z.items():
                out[a * N + b] = v
            return out
    elif target in ("tensor", "dualtensor"):
        N = r.N
        # V = L (x) L  (or L (x) L^*); basis t = a*N + b
        V = []
        for a in range(N):
            for b in range(N):
                db = r.deg[b] if target == "tensor" else tuple(-x for x in r.deg[b])
                V.append((None, (r.parity[a] + r.parity[b]) % 2, vadd(r.deg[a], db)))
        cols = []
        for X, _, _ in g:
            c = {}
            for (row, col), v in X.items(): c.setdefault(col, {})[row] = v
            rws = {}
            for (row, col), v in X.items(): rws.setdefault(row, {})[col] = v
            cols.append((c, rws))
        def act(i, t):
            a, b = divmod(t, N)
            c, rws = cols[i]
            out = {}
            for a2, v in c.get(a, {}).items():
                out[a2 * N + b] = K.norm(out.get(a2 * N + b, 0) + v)
            s = -1 if (g[i][1] and r.parity[a]) else 1
            if target == "tensor":
                for b2, v in c.get(b, {}).items():
                    out[a * N + b2] = K.norm(out.get(a * N + b2, 0) + s * v)
            else:
                # x . xi = -(-1)^{p(x)p(xi)} xi o x ; on the dual basis: (x.b*)(e_c) = -(-1)^.. X[b][c]
                for c2, v in rws.get(b, {}).items():
                    sb = -1 if (g[i][1] and r.parity[b]) else 1
                    out[a * N + c2] = K.norm(out.get(a * N + c2, 0) - s * sb * v)
            return {k: v for k, v in out.items() if not K.is_zero(v)}
    else:
        V = g
        def act(i, t):
            return {k: v for k, v in enumerate(SC[(i, t)]) if v}
    res = {}
    for pc in (0, 1):
        # group unknowns by total degree (deg c = deg v - deg x)
        blocks = {}
        for i in range(n):
            for t in range(len(V)):
                if V[t][1] == (g[i][1] + pc) % 2:
                    dc = tuple(x - y for x, y in zip(V[t][2], g[i][2]))
                    blocks.setdefault(dc, []).append((i, t))
        for dc, lst in blocks.items():
            unk = {it: u for u, it in enumerate(lst)}
            rows = []
            for i, j in itertools.combinations_with_replacement(range(n), 2):
                px, py = g[i][1], g[j][1]
                row = {}
                for (ii, t), u in unk.items():
                    if ii == j:
                        for s, v in act(i, t).items():
                            row.setdefault(s, {}); row[s][u] = K.norm(row[s].get(u, 0) + sgn(px * pc) * v)
                    if ii == i:
                        for s, v in act(j, t).items():
                            row.setdefault(s, {}); row[s][u] = K.norm(row[s].get(u, 0) - sgn(py * (px + pc)) * v)
                    ck = SC[(i, j)][ii]
                    if ck:
                        row.setdefault(t, {}); row[t][u] = K.norm(row[t].get(u, 0) - ck)
                for s, rr in row.items():
                    rr = {u: v for u, v in rr.items() if not K.is_zero(v)}
                    if rr: rows.append(rr)
            Z = nullspace_sparse(K, rows, len(unk))
            if not Z: continue
            B = Span(K)
            for t in range(len(V)):
                if V[t][1] != pc or V[t][2] != dc: continue
                vec = {}
                for i in range(n):
                    for s, v in act(i, t).items():
                        u = unk.get((i, s))
                        if u is None: continue
                        vec[u] = K.norm(vec.get(u, 0) + sgn(g[i][1] * pc) * v)
                vec = {u: v for u, v in vec.items() if not K.is_zero(v)}
                if vec: B.add(vec)
            for b in B.basis:
                for rr in rows:
                    assert K.is_zero(K.norm(sum(rr.get(u, 0) * v for u, v in b.items()))), "sign"
            dimh = len(Z) - len(B.basis)
            if dimh:
                res[(pc, dc)] = dimh
                # a representative: the first cocycle not in the span of the coboundaries
                inv = {u: it for it, u in unk.items()}
                for z in Z:
                    if B.reduce(z):
                        REPS[(pc, dc)] = {inv[u]: v for u, v in z.items()}
                        break
    CTX.update(dict(r=r, g=g, V=V, K=K))
    tot = [0, 0]
    for (pc, dc), d in res.items():
        tot[pc] += d
    if verbose:
        print("A=%s par=%s p=%d lam=%s target=%s: H^1 = %d|%d; by (parity, degree): %s"
              % (A, par, p, lam, target, tot[0], tot[1], sorted(res.items())), flush=True)
    return tot, res

if __name__ == "__main__":
    for p in (3, 5, 7):
        h1([[2]], [0], p, [p - 1], "end")
        h1([[2]], [0], p, [p - 2], "end")
    sl3 = [[2, -1], [-1, 2]]
    h1(sl3, [0, 0], 3, [0, 1], "end")
    h1(sl3, [0, 0], 3, [0, 1], "ad")
    h1(sl3, [0, 0], 3, [1, 1], "ad")

def h1_tensor(A, par, p, lam1, lam2, dual2=False):
    """H^1(g; L1 (x) L2) (or L1 (x) L2^*), g = image of g(A) in gl(L1 + L2) -- here both
    modules are built separately and g acts diagonally through the Chevalley generators."""
    from prol import Rep, op_from_cols
    r1 = Rep(A, par, p, lam1); r2 = Rep(A, par, p, lam2)
    K = r1.K
    n = len(A)
    N1, N2 = r1.N, r2.N
    # block-diagonal generators on L1 + L2 so that the image is faithful on g
    def blk(X1, X2):
        out = dict(X1)
        for (a, b), v in X2.items(): out[(a + N1, b + N1)] = v
        return out
    class R: pass
    rr = R(); rr.K = K; rr.n = n; rr.par = par; rr.N = N1 + N2
    rr.deg = r1.deg + r2.deg; rr.parity = r1.parity + r2.parity
    rr.E = []; rr.F = []
    for i in range(n):
        e = blk(op_from_cols(r1.E[i]), op_from_cols(r2.E[i]))
        f = blk(op_from_cols(r1.F[i]), op_from_cols(r2.F[i]))
        ce, cf = {}, {}
        for (a, b), v in e.items(): ce.setdefault(b, {})[a] = v
        for (a, b), v in f.items(): cf.setdefault(b, {})[a] = v
        rr.E.append(ce); rr.F.append(cf)
    img = Image(rr)
    g = [(X, pp, d) for d, pp, X in img.basis]
    return img, g, N1, N2
