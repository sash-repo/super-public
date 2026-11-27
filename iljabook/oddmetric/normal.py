"""Statement 5.A.8.4: does the odd conformal structure have a NORMAL tractor connection?
g = pe(n+1) graded by ad_h, h = diag(E11,-E11): g_{-1}+g_0+g_1+g_2, g_0 = cpe(n).
Spencer map d: C1 = Hom(g_{-1},g_1) -> C2 = Hom(g_{-1} x g_{-1}, g_0).
A normalization (a g_0-invariant complement of Im d, i.e. the analog of 'Ricci contraction
determines the Schouten tensor') exists iff there is an even g_0-equivariant L: C2 -> C1 with
d L d = d.  Also: the same test for the even conformal algebra o(n+2) (control: must exist),
and ker d (must be the prolong g_2).  Exact over Q (exactlin)."""
import sys, itertools, random
from fractions import Fraction
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from exactlin import rank, nullspace, solve

def build_pe(n):
    m = n + 1; D = 2*m
    par = [0]*m + [1]*m
    h = [0]*D; h[0] = 1; h[m] = -1
    basis = []   # (matrix as dict {(i,j):c}, key entry)
    for i in range(m):
        for j in range(m):
            basis.append(({(i, j): 1, (m+j, m+i): -1}, (i, j)))
    for i in range(m):
        for j in range(i, m):
            M = {(i, m+j): 1}
            M[(j, m+i)] = M.get((j, m+i), 0) + 1
            basis.append((M, (i, m+j)))
    for i in range(m):
        for j in range(i+1, m):
            basis.append(({(m+i, j): 1, (m+j, i): -1}, (m+i, j)))
    return D, par, h, basis

def build_o(n):
    """o(n+2) with form <p,q>=1, Euclidean on V: matrices X with X^t G + G X = 0; basis via pairs."""
    D = n + 2; par = [0]*D
    # coordinates: 0 = p, 1..n = V, n+1 = q ; G[0][n+1]=G[n+1][0]=1, G[i][i]=1
    G = [[0]*D for _ in range(D)]
    G[0][D-1] = G[D-1][0] = 1
    for i in range(1, D-1): G[i][i] = 1
    h = [0]*D; h[0] = 1; h[D-1] = -1
    basis = []
    for i in range(D):
        for j in range(i+1, D):
            # X = E_ij G^{-1}-ish: use X = G^{-1}(E_ij - E_ji): skew wrt G
            Gi = G  # G is its own inverse
            M = {}
            for a in range(D):
                c = Gi[a][i]
                if c: M[(a, j)] = M.get((a, j), 0) + c
                c = Gi[a][j]
                if c: M[(a, i)] = M.get((a, i), 0) - c
            key = max(M, key=lambda k: (abs(M[k]), k))
            basis.append((M, None))
    return D, par, h, basis

def setup(D, par, h, basis):
    # coordinates of a matrix in the basis: solve linear system once
    ents = sorted({k for M, _ in basis for k in M})
    col = {k: t for t, k in enumerate(ents)}
    B = [[Fraction(M.get(k, 0)) for M, _ in basis] for k in ents]  # ents x nb
    nb = len(basis)
    def coords(M):
        b = [Fraction(M.get(k, 0)) for k in ents]
        for k in M:
            if k not in col and M[k] != 0: raise ValueError("not in algebra")
        x = solve(B, b)
        return x
    def mpar(M):
        ps = {(par[i] + par[j]) % 2 for (i, j), c in M.items() if c}
        assert len(ps) == 1; return ps.pop()
    def mw(M):
        ws = {h[i] - h[j] for (i, j), c in M.items() if c}
        assert len(ws) == 1; return ws.pop()
    P = [mpar(M) for M, _ in basis]; W = [mw(M) for M, _ in basis]
    def mm(X, Y):
        R = {}
        for (i, j), a in X.items():
            for (k, l), b in Y.items():
                if j == k: R[(i, l)] = R.get((i, l), 0) + a*b
        return R
    br = {}
    for s in range(nb):
        for t in range(nb):
            X, Y = basis[s][0], basis[t][0]
            sg = (-1)**(P[s]*P[t])
            Z = mm(X, Y)
            for k, v in mm(Y, X).items(): Z[k] = Z.get(k, 0) - sg*v
            Z = {k: v for k, v in Z.items() if v}
            br[s, t] = coords(Z) if Z else [Fraction(0)]*nb
    return P, W, br

def analyse(name, D, par, h, basis, k=1):
    P, W, br = setup(D, par, h, basis)
    nb = len(basis)
    gm = [i for i in range(nb) if W[i] == -1]; g0 = [i for i in range(nb) if W[i] == 0]
    g1 = [i for i in range(nb) if W[i] == 1]; g2 = [i for i in range(nb) if W[i] == 2]
    print(f"{name}: dim g_-1={len(gm)} g_0={len(g0)} g_1={len(g1)} g_2={len(g2)}", flush=True)
    # C1 basis (v,w): beta(v)=w ; C2 basis (u,v,z)
    gk = [i for i in range(nb) if W[i] == k]; gk1 = [i for i in range(nb) if W[i] == k-1]
    print(f"  degree k={k}: C1=Hom(g_-1,g_{k}), C2=Hom(g_-1 x g_-1, g_{k-1})", flush=True)
    g1, g0z = gk, gk1
    C1 = [(v, w) for v in gm for w in g1]; i1 = {b: k for k, b in enumerate(C1)}
    C2 = [(u, v, z) for u in gm for v in gm for z in g0z]; i2 = {b: k for k, b in enumerate(C2)}
    p1 = [(P[v]+P[w]) % 2 for v, w in C1]; p2 = [(P[u]+P[v]+P[z]) % 2 for u, v, z in C2]
    def coef(x, y, target):  # coefficient of basis 'target' in [x,y]
        return br[x, y][target]
    def act1(x):  # matrix of x on C1: (x.beta)(v) = [x,beta(v)] - (-1)^{p(x)p(beta)} beta([x,v])
        M = [[Fraction(0)]*len(C1) for _ in C1]
        for k, (v, w) in enumerate(C1):
            for w2 in g1:
                c = coef(x, w, w2)
                if c: M[i1[v, w2]][k] += c
            s = (-1)**(P[x]*p1[k])
            for v2 in gm:   # beta([x,v2]) has component when [x,v2] contains v
                c = coef(x, v2, v)
                if c: M[i1[v2, w]][k] -= s*c
        return M
    def act2(x):
        M = [[Fraction(0)]*len(C2) for _ in C2]
        for k, (u, v, z) in enumerate(C2):
            for z2 in g0z:
                c = coef(x, z, z2)
                if c: M[i2[u, v, z2]][k] += c
            s = (-1)**(P[x]*p2[k])
            for u2 in gm:
                c = coef(x, u2, u)
                if c: M[i2[u2, v, z]][k] -= s*c
            for v2 in gm:
                c = coef(x, v2, v)
                if c:
                    # c(u', [x,v']) with u'=u: sign (-1)^{p(x)p(u)}
                    M[i2[u, v2, z]][k] -= s*(-1)**(P[x]*P[u])*c
        return M
    # Spencer d: (d beta)(u,v) = (-1)^{p(u)p(beta)}[u,beta(v)] - (-1)^{p(u)p(v)} (-1)^{p(v)p(beta)}[v,beta(u)]
    dM = [[Fraction(0)]*len(C1) for _ in C2]
    for k, (v0, w) in enumerate(C1):
        pb = p1[k]
        for u in gm:
            # term [u, beta(v0)] at (u, v0)
            for z in g0z:
                c = coef(u, w, z)
                if c: dM[i2[u, v0, z]][k] += (-1)**(P[u]*pb)*c
            # term -(-1)^{p(u)p(v0)} (-1)^{p(v0)... } : (d beta)(v0,u) contains -[u,beta(v0)] with sign
            for z in g0z:
                c = coef(u, w, z)
                if c: dM[i2[v0, u, z]][k] -= (-1)**(P[v0]*P[u])*(-1)**(P[u]*pb)*c
    # equivariance check of d
    def matmul(A, B):
        return [[sum((A[i][l]*B[l][j] for l in range(len(B)) if A[i][l] and B[l][j]), Fraction(0)) for j in range(len(B[0]))] for i in range(len(A))]
    bad = 0
    for x in g0:
        L = matmul(act2(x), dM); R = matmul(dM, act1(x))
        bad += sum(1 for i in range(len(C2)) for j in range(len(C1)) if L[i][j] != R[i][j])
    print(f"  d equivariant: {bad == 0} ({bad} bad)", flush=True)
    rd = rank(dM); print(f"  dim C1={len(C1)}, rank d={rd}, dim ker d={len(C1)-rd}", flush=True)
    # equivariant even L: C2 -> C1 ; unknowns L[i][j] with p1[i]==p2[j]
    unk = [(i, j) for i in range(len(C1)) for j in range(len(C2)) if p1[i] == p2[j]]
    ui = {u: k for k, u in enumerate(unk)}
    rows = []
    for x in g0:
        A1 = act1(x); A2 = act2(x)
        # (A1 L - L A2)[i][j] = 0
        for i in range(len(C1)):
            for j in range(len(C2)):
                row = {}
                for l in range(len(C1)):
                    if A1[i][l] and (l, j) in ui: row[ui[l, j]] = row.get(ui[l, j], 0) + A1[i][l]
                for l in range(len(C2)):
                    if A2[l][j] and (i, l) in ui: row[ui[i, l]] = row.get(ui[i, l], 0) - A2[l][j]
                row = {k: v for k, v in row.items() if v}
                if row: rows.append(row)
    print(f"  equations {len(rows)}, unknowns {len(unk)}", flush=True)
    # dedupe and solve
    seen = set(); dense = []
    for r in rows:
        key = tuple(sorted(r.items()))
        if key in seen: continue
        seen.add(key); dense.append([r.get(k, 0) for k in range(len(unk))])
    ns = nullspace(dense) if dense else [[Fraction(int(k == t)) for k in range(len(unk))] for t in range(len(unk))]
    print(f"  dim Hom_g0(C2,C1) (even) = {len(ns)}", flush=True)
    # does some L give d L d = d ?
    Ls = []
    for vec in ns:
        L = [[Fraction(0)]*len(C2) for _ in C1]
        for k, (i, j) in enumerate(unk): L[i][j] = vec[k]
        Ls.append(L)
    target = [dM[i][j] for i in range(len(C2)) for j in range(len(C1))]
    cols = []
    for L in Ls:
        T = matmul(dM, matmul(L, dM))
        cols.append([T[i][j] for i in range(len(C2)) for j in range(len(C1))])
    if not cols:
        print("  no equivariant L at all -> no normalization", flush=True); return
    Amat = [[cols[k][r] for k in range(len(cols))] for r in range(len(target))]
    try:
        t = solve(Amat, target); print(f"  d L d = d solvable: True  (normalization EXISTS)", flush=True)
    except Exception as e:
        rk = rank(Amat)
        # best achievable: rank of d L d over generic combination
        best = max((rank(matmul(dM, matmul(L, dM))) for L in Ls), default=0)
        gen = [[sum((random.randint(1, 97)*L[i][j] for L in Ls), Fraction(0)) for j in range(len(C2))] for i in range(len(C1))]
        print(f"  d L d = d solvable: False  (normalization does NOT exist); rank of d L d for generic L = {rank(matmul(dM, matmul(gen, dM)))} < rank d = {rd}", flush=True)

if __name__ == "__main__":
    random.seed(1)
    n = int(sys.argv[1]); k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if k == 1: analyse(f"o({n+2}) [control]", *build_o(n))
    analyse(f"pe({n+1}) [odd conformal]", *build_pe(n), k=k)
