"""Statement SderH 2): the derived algebra of h_{omega_{2,j}}(2k;N). Bracket (1.15) for the 2nd type form, Omega^{-1} by Neumann series.
Check Jacobi, injectivity of f -> D_f, and dim of the derived algebra vs B(F)."""
import os, itertools, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import rank
from math import comb
from derdim import *  # noqa

def setup(p, N):
    n = len(N); tops = [p**Ni-1 for Ni in N]
    def mul(A, B):
        C = {}
        for a, x in A.items():
            for b, y in B.items():
                c = tuple(ai+bi for ai, bi in zip(a, b))
                if any(ci > t for ci, t in zip(c, tops)): continue
                co = x*y
                for ai, bi in zip(a, b): co *= comb(ai+bi, ai)
                co %= p
                if co: C[c] = (C.get(c, 0)+co) % p
        return {m: v for m, v in C.items() if v}
    def add(A, B, s=1):
        C = dict(A)
        for m, v in B.items(): C[m] = (C.get(m, 0)+s*v) % p
        return {m: v for m, v in C.items() if v}
    def d(A, i):
        C = {}
        for m, v in A.items():
            if m[i]:
                mm = list(m); mm[i] -= 1; C[tuple(mm)] = v
        return C
    def mono(i, e):
        m = [0]*n; m[i] = e; return {tuple(m): 1}
    return tops, mul, add, d, mono

def run2(p, N, t, beta):
    n = len(N); k = n//2
    tops, mul, add, d, mono = setup(p, N)
    zero = tuple([0]*n)
    tl = lambda i: (i+k) % n
    pi = lambda i: 1 if i < k else -1
    half = pow(2, -1, p)
    ub = lambda i: mono(i, tops[i])
    W = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            e = {}
            if j == tl(i): e = {zero: pi(i) % p}
            if (i == t) != (j == t):
                e = add(e, mul({zero: beta[i][j] % p}, mul(ub(i), ub(j))))
            s = {}
            if j == t: s = add(s, {m: c*pi(i) for m, c in mono(tl(i), 1).items()})
            if i == t: s = add(s, {m: c*pi(j) for m, c in mono(tl(j), 1).items()}, -1)
            e = add(e, mul({zero: half}, mul(ub(t), s)))
            W[i][j] = e
    # invert: W = A + B, A constant part
    A = [[W[i][j].get(zero, 0) for j in range(n)] for i in range(n)]
    import sympy
    Ai = sympy.Matrix(A).inv_mod(p)
    Ainv = [[{zero: int(Ai[i, j]) % p} if int(Ai[i, j]) % p else {} for j in range(n)] for i in range(n)]
    B = [[{m: c for m, c in W[i][j].items() if m != zero} for j in range(n)] for i in range(n)]
    def mm(X, Y):
        Z = [[{} for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                acc = {}
                for l in range(n): acc = add(acc, mul(X[i][l], Y[l][j]))
                Z[i][j] = acc
        return Z
    # W^{-1} = sum (-Ainv B)^m Ainv
    T = mm(Ainv, B); T = [[{m: (-c) % p for m, c in T[i][j].items()} for j in range(n)] for i in range(n)]
    term = Ainv; inv = Ainv
    for _ in range(4*n+4):
        term = mm(T, term)
        if all(not term[i][j] for i in range(n) for j in range(n)): break
        inv = [[add(inv[i][j], term[i][j]) for j in range(n)] for i in range(n)]
    dv = [mono(t, tops[t]) if i == t else {} for i in range(n)]  # d_i v, v = u_t^{(p^N_t)}
    def br(a, b):
        R = {}
        for i in range(n):
            if True:
                Ai_ = add(d(a, i), mul(a, dv[i]))
                if not Ai_: continue
                for j in range(n):
                    if not inv[i][j]: continue
                    R = add(R, mul(inv[i][j], mul(Ai_, add(d(b, j), mul(b, dv[j])))))
        return R
    mons = list(itertools.product(*[range(x+1) for x in tops]))
    idx = {m: i for i, m in enumerate(mons)}
    basis = [{m: 1} for m in mons]
    # Jacobi on a random sample
    random.seed(0)
    bad = 0
    for _ in range(30):
        x, y, z = (random.choice(basis) for _ in range(3))
        J = add(add(br(x, br(y, z)), br(y, br(z, x))), br(z, br(x, y)))
        if J: bad += 1
    rows = []
    for x, y in itertools.combinations(basis, 2):
        zz = br(x, y)
        if zz:
            v = [0]*len(mons)
            for m, c in zz.items(): v[idx[m]] = c
            rows.append(v)
    der = rank(rows, p)
    topm = tuple(tops)
    has_top = rank(rows + [[1 if m == topm else 0 for m in mons]], p) == der
    Op=[[1 if m==mm_ else 0 for m in mons] for mm_ in mons if mm_!=topm]
    same = rank(rows+Op,p)==der==len(Op)
    import sympy
    M=sympy.Matrix(rows)
    # annihilator: vectors y with M y = 0 mod p
    from exactlin import nullspace
    ann=nullspace(rows,p)
    a=ann[0] if ann else [0]*len(mons); a={mons[i]:int(c)%p for i,c in enumerate(a) if int(c)%p}
    # Pfaffian of W (4x4): W01W23 - W02W13 + W03W12
    Pf={} if n<4 else add(add(mul(W[0][1],W[2][3]),mul(W[0][2],W[1][3]),-1),mul(W[0][3],W[1][2]))
    return a, Pf, topm, der, len(mons), bad

def pfcheck(p, a, Pf, top):
    mons = list(itertools.product(*[range(x+1) for x in top]))
    def coef(m):
        tot = 0
        for mm_, c in Pf.items():
            s_ = tuple(x+y for x, y in zip(m, mm_))
            if s_ == top:
                co = c
                for x, y in zip(m, mm_): co *= comb(x+y, x)
                tot += co
        return tot % p
    v = {m: coef(m) for m in mons}
    return any(all((lam*v[m]-a.get(m, 0)) % p == 0 for m in mons) for lam in range(1, p))

def job(args):
    p, N, t, beta = args
    a, Pf, top, der, dimO, bad = run2(p, N, t, beta)
    k = len(N)//2
    if (k+1) % p:
        ok = (der == dimO)
    else:
        ok = (der == dimO-1) and pfcheck(p, a, Pf, top)
    return (p, N, t, beta, ok, bad)

if __name__ == "__main__":
    import itertools
    from math import comb
    from multiprocessing import Pool
    jobs = []
    for p, N in ((3, (1, 1)), (3, (2, 1)), (3, (1, 2)), (5, (1, 1)), (5, (2, 1)), (3, (1, 1, 1, 1))):
        n = len(N); k = n//2
        for t in range(n):
            free = [j for j in range(n) if j != t and j != (t+k) % n]
            for vals in itertools.product(range(p), repeat=len(free)):
                b = [[0]*n for _ in range(n)]
                for j, x in zip(free, vals):
                    b[t][j] = x; b[j][t] = (-x) % p
                jobs.append((p, N, t, b))
    with Pool(9) as P:
        res = P.map(job, jobs)
    print("cases", len(res), "mismatches", sum(1 for r in res if not r[4]), "jacobi failures", sum(r[5] for r in res))
