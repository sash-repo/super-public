"""Even 2-cocycles on the simple vectorial Lie superalgebras vect(0|n), svect(0|n) and the
deformed svect(0|n) (Remark 2.4).
Question: does an even super-anti-symmetric 2-cocycle B on g exist that is non-degenerate?
(The constant solutions of the CYBE with values in g satisfying condition (A) are the inverses
of such B, see the proof of Theorem 2.1.)
The whole space Z^2_even of even 2-cocycles is computed (the cocycle identity is imposed on
every triple of basis elements), and the rank of a random element of it is printed.
Arithmetic is mod the prime P = 2147483629; the random element is drawn with a fixed seed.

Run:  python3 cocyc.py vect:3 vect:4 svect:4 svectdef:4
  (kind:n with kind = vect, svect, or svectdef -- the deformed svect(0|n), the kernel of
   X |-> X(xi_1...xi_n) + (1 + xi_1...xi_n) div X)
"""
import itertools, random, sys

P = 2147483629
rng = random.Random(1)

# Grassmann polynomials on n generators: dict {tuple(sorted idx): coeff}
def mono_mul(a, b):
    if set(a) & set(b): return None, 0
    lst = list(a) + list(b); sign = 1
    # bubble sort parity
    lst2 = lst[:]
    for i in range(len(lst2)):
        for j in range(len(lst2) - 1 - i):
            if lst2[j] > lst2[j + 1]:
                lst2[j], lst2[j + 1] = lst2[j + 1], lst2[j]; sign = -sign
    return tuple(lst2), sign
def pmul(f, g):
    r = {}
    for a, x in f.items():
        for b, y in g.items():
            m, s = mono_mul(a, b)
            if s: r[m] = (r.get(m, 0) + s * x * y) % P
    return {k: v for k, v in r.items() if v}
def padd(f, g, c=1):
    r = dict(f)
    for k, v in g.items(): r[k] = (r.get(k, 0) + c * v) % P
    return {k: v for k, v in r.items() if v}
def dleft(i, f):  # left derivative d/dxi_i
    r = {}
    for a, x in f.items():
        if i in a:
            pos = a.index(i); s = -1 if pos % 2 else 1
            m = tuple(t for t in a if t != i); r[m] = (r.get(m, 0) + s * x) % P
    return {k: v for k, v in r.items() if v}
def par_poly(f):
    ps = {len(a) % 2 for a in f}; assert len(ps) <= 1, f
    return ps.pop() if ps else 0

# vector field: tuple of n polys (X_i = coefficient of d_i), X = sum X_i d_i
def vf_apply(X, f):
    r = {}
    for i, Xi in enumerate(X):
        if Xi: r = padd(r, pmul(Xi, dleft(i, f)))
    return r
def vf_par(X):
    for Xi in X:
        if Xi: return (par_poly(Xi) + 1) % 2
    return 0
def vf_br(X, Y):
    px, py = vf_par(X), vf_par(Y); s = -1 if (px and py) else 1
    return tuple(padd(vf_apply(X, Y[j]), vf_apply(Y, X[j]), -s) for j in range(len(X)))

def monos(n):
    for k in range(n + 1):
        for a in itertools.combinations(range(n), k): yield a

def vect_basis(n):
    B = []
    for a in monos(n):
        for i in range(n):
            X = [dict() for _ in range(n)]; X[i] = {a: 1}; B.append(tuple(X))
    return B

def coords(X, index):  # vector field -> coordinate vector in vect basis
    v = {}
    for i, Xi in enumerate(X):
        for a, c in Xi.items(): v[index[(a, i)]] = c % P
    return v

def nullspace_modp(rows, ncols):
    """rows: list of dict col->val. returns basis of nullspace (list of lists) mod P."""
    M = [dict(r) for r in rows if r]
    piv = {}  # col -> row dict (normalized)
    for r in M:
        r = dict(r)
        for c, pr in piv.items():
            if c in r and r[c]:
                f = r[c]
                for k, v in pr.items(): r[k] = (r.get(k, 0) - f * v) % P
                r = {k: v for k, v in r.items() if v}
        if not r: continue
        c0 = min(r); inv = pow(r[c0], P - 2, P)
        r = {k: v * inv % P for k, v in r.items()}
        for c, pr in list(piv.items()):
            if c0 in pr and pr[c0]:
                f = pr[c0]
                for k, v in r.items(): pr[k] = (pr.get(k, 0) - f * v) % P
                piv[c] = {k: v for k, v in pr.items() if v}
        piv[c0] = r
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        v = [0] * ncols; v[f] = 1
        for c, pr in piv.items(): v[c] = (-pr.get(f, 0)) % P
        basis.append(v)
    return basis

def rank_modp(A):
    A = [row[:] for row in A]; m = len(A); nc = len(A[0]) if m else 0; r = 0
    for c in range(nc):
        p = next((i for i in range(r, m) if A[i][c] % P), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; inv = pow(A[r][c], P - 2, P)
        A[r] = [x * inv % P for x in A[r]]
        for i in range(m):
            if i != r and A[i][c]:
                f = A[i][c]; A[i] = [(x - f * y) % P for x, y in zip(A[i], A[r])]
        r += 1
    return r

def analyse(name, basis_fields, n, vbasis_index):
    """basis_fields: list of vector fields spanning g (a basis)."""
    N = len(basis_fields); par = [vf_par(X) for X in basis_fields]
    # express brackets in the g-basis: solve via vect coordinates
    vecs = [coords(X, vbasis_index) for X in basis_fields]
    # build a reduction: echelon of g-basis in vect coordinates
    nv = len(vbasis_index)
    # matrix G (N x nv); solve lambda G = w via dense mod p
    G = [[0]*nv for _ in range(N)]
    for i, v in enumerate(vecs):
        for k, c in v.items(): G[i][k] = c
    # pick pivot columns
    Gl = G
    # Gaussian elimination to find pivots and inverse on them
    cols = []; A = [row[:] + [1 if j == i else 0 for j in range(N)] for i, row in enumerate(Gl)]
    r = 0
    for c in range(nv):
        p = next((i for i in range(r, N) if A[i][c] % P), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; inv = pow(A[r][c], P - 2, P); A[r] = [x * inv % P for x in A[r]]
        for i in range(N):
            if i != r and A[i][c]:
                f = A[i][c]; A[i] = [(x - f * y) % P for x, y in zip(A[i], A[r])]
        cols.append(c); r += 1
    assert r == N, (name, r, N)
    # For w in span: lambda_i = sum over pivot rows: w[cols[k]] * A[k][nv+i]
    def express(w):
        lam = [0] * N
        for k, c in enumerate(cols):
            if c in w and w[c]:
                for i in range(N):
                    if A[k][nv + i]: lam[i] = (lam[i] + w[c] * A[k][nv + i]) % P
        return lam
    C = {}
    for i in range(N):
        for j in range(N):
            w = coords(vf_br(basis_fields[i], basis_fields[j]), vbasis_index)
            lam = express(w)
            # check membership
            back = {}
            for t, l in enumerate(lam):
                if l:
                    for k, c in vecs[t].items(): back[k] = (back.get(k, 0) + l * c) % P
            back = {k: v for k, v in back.items() if v}
            assert back == {k: v for k, v in w.items() if v % P}, (name, 'not closed', i, j)
            C[(i, j)] = {t: l for t, l in enumerate(lam) if l}
    # unknowns: B(i,j) for i<=j with par equal (even form), super-anti-symmetric
    unk = {}
    for i in range(N):
        for j in range(i, N):
            if par[i] != par[j]: continue
            if i == j and par[i] == 0: continue  # anti-symmetric even-even diag = 0
            unk[(i, j)] = len(unk)
    def Bvar(i, j):  # returns (index, sign) or None
        if par[i] != par[j]: return None
        if i <= j:
            return (unk[(i, j)], 1) if (i, j) in unk else None
        s = 1 if (par[i] and par[j]) else -1   # B(y,x) = -(-1)^{pp} B(x,y)
        return (unk[(j, i)], s) if (j, i) in unk else None
    def addB(row, coefdict, k, sign):
        for t, l in coefdict.items():
            bv = Bvar(t, k)
            if bv: row[bv[0]] = (row.get(bv[0], 0) + sign * l * bv[1]) % P
    rows = []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if (par[x] + par[y] + par[z]) % 2: continue
                px, py, pz = par[x], par[y], par[z]
                row = {}
                addB(row, C[(x, y)], z, 1)
                addB(row, C[(x, z)], y, -(-1) ** (py * pz))
                addB(row, C[(y, z)], x, (-1) ** (px * (py + pz)))
                row = {k: v for k, v in row.items() if v}
                if row: rows.append(row)
    Z = nullspace_modp(rows, len(unk))
    # random element
    coeff = [rng.randrange(P) for _ in Z]
    Bm = [[0] * N for _ in range(N)]
    for (i, j), u in unk.items():
        val = sum(c * z[u] for c, z in zip(coeff, Z)) % P
        Bm[i][j] = val; Bm[j][i] = (val * (1 if (par[i] and par[j]) else -1)) % P
    rk = rank_modp(Bm) if Z else 0
    ne = par.count(0); no = N - ne
    print(f"{name}: sdim=({ne}|{no}) unknowns={len(unk)} dim Z^2_even={len(Z)} rank(random cocycle)={rk} of {N} ->",
          "NON-DEGENERATE EXISTS" if rk == N else "all degenerate", flush=True)

def build(n, kind):
    vb = vect_basis(n)
    idx = {}
    for X in vb:
        for i, Xi in enumerate(X):
            for a in Xi: idx[(a, i)] = len(idx)
    if kind == 'vect':
        return vb, idx
    # div X = sum (-1)^{p(X_i)} d_i X_i  -- sign chosen by closure test below
    def div(X, sg):
        r = {}
        for i, Xi in enumerate(X):
            if Xi:
                s = (-1) ** par_poly(Xi) if sg else 1
                r = padd(r, dleft(i, Xi), s)
        return r
    return vb, idx, div

def subalg(n, cond_rows, vb, idx):
    """kernel of a linear condition on vect, given as function X-> poly; returns basis fields"""
    N = len(vb)
    # build matrix: for each basis field compute condition polynomial coordinates
    allm = list(monos(n)); mi = {a: k for k, a in enumerate(allm)}
    rows = {}
    for col, X in enumerate(vb):
        for a, c in cond_rows(X).items():
            rows.setdefault(mi[a], {})[col] = c
    ker = nullspace_modp(list(rows.values()), N)
    fields = []
    for v in ker:
        X = [dict() for _ in range(n)]
        for col, c in enumerate(v):
            if c:
                for i, Xi in enumerate(vb[col]):
                    for a, cc in Xi.items():
                        X[i][a] = (X[i].get(a, 0) + c * cc) % P
        fields.append(tuple({k: v for k, v in Xi.items() if v} for Xi in X))
    # split into homogeneous parity components (kernel vectors may mix parities)
    out = []
    for X in fields:
        for want in (0, 1):
            Y = tuple({a: c for a, c in Xi.items() if (len(a) + 1) % 2 == want} for Xi in X)
            if any(Y): out.append(Y)
    # reduce to a basis
    vecs = [coords(Y, idx) for Y in out]
    basis = []; ech = []
    for Y, v in zip(out, vecs):
        tmp = nullspace_modp([], 0)
        basis.append((Y, v))
    # select independent ones
    sel = []; rowsel = []
    for Y, v in basis:
        test = rowsel + [v]
        M = [[r.get(k, 0) for k in range(len(idx))] for r in test]
        if rank_modp(M) == len(test): rowsel.append(v); sel.append(Y)
    return sel

if __name__ == '__main__':
    which = sys.argv[1:]
    for w in which:
        kind, n = w.split(':'); n = int(n)
        if kind == 'vect':
            vb, idx = build(n, 'vect'); analyse(f"vect(0|{n})", vb, n, idx)
        elif kind in ('svect', 'svectdef'):
            vb, idx, div = build(n, 'svect')
            for sg in (0, 1):
                if kind == 'svect':
                    cond = lambda X, sg=sg: div(X, sg)
                else:
                    top = {tuple(range(n)): 1}
                    cond = lambda X, sg=sg: padd(vf_apply(X, top), pmul(padd({(): 1}, top), div(X, sg)))
                try:
                    g = subalg(n, cond, vb, idx)
                    analyse(f"{kind}(0|{n}) sign{sg}", g, n, idx); break
                except AssertionError as e:
                    print(f"  sign {sg}: not closed ({str(e)[:60]})", flush=True)
