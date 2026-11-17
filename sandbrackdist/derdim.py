"""Statement SderH 1): dim of h^O_{omega_1}(2k;N) and of its derived algebra, from the bracket (1.20).
Functions on O(2k;N) over F_p, divided powers; alpha a k x k matrix; f_ij from (f_ij)."""
import os, itertools, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank
from math import comb

def run(p, N, alpha):
    k = len(alpha); n = 2*k
    tops = [p**Ni - 1 for Ni in N]
    mons = list(itertools.product(*[range(t+1) for t in tops]))
    idx = {m: i for i, m in enumerate(mons)}
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
    def ubar(i):
        m = [0]*n; m[i] = tops[i]; return {tuple(m): 1}
    one = {tuple([0]*n): 1}
    # f_ij = sum_{1<=m<=k} (-1)^m sum_{t1..t_{m-1}} alpha_{i t1}...alpha_{t_{m-1} j} ubar_t1..ubar_t{m-1} ubar_~t1..
    f = [[{} for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            acc = {}
            for m in range(1, k+1):
                for ts in itertools.product(range(k), repeat=m-1):
                    path = [i]+list(ts)+[j]
                    c = 1
                    for a, b in zip(path, path[1:]): c *= alpha[a][b]
                    c = ((-1)**m * c) % p
                    if not c: continue
                    term = {tuple([0]*n): c}
                    for t in ts: term = mul(mul(term, ubar(t)), ubar(t+k))
                    acc = add(acc, term)
            f[i][j] = acc
    def br(a, b):
        R = {}
        for i in range(k):
            R = add(R, add(mul(d(a, i+k), d(b, i)), mul(d(a, i), d(b, i+k)), -1))
        for i in range(k):
            for j in range(k):
                if not f[i][j]: continue
                coef = mul(mul(f[i][j], ubar(i)), ubar(j+k))
                inner = add(mul(d(a, i+k), d(b, j)), mul(d(a, j), d(b, i+k)), -1)
                R = add(R, mul(coef, inner))
        return R
    basis = [{m: 1} for m in mons if any(m)]
    rows = []
    for x, y in itertools.combinations(basis, 2):
        z = br(x, y)
        z.pop(tuple([0]*n), None)   # D_const = 0
        if z:
            v = [0]*len(mons)
            for m, c in z.items(): v[idx[m]] = c
            rows.append(v)
    r = rank(rows, p) if rows else 0
    return len(mons), len(basis), r

def det(M, p):
    import sympy
    return int(sympy.Matrix(M).det()) % p

if __name__ == "__main__":
    random.seed(1)
    cases = 0
    for p in (3, 5):
        for k in (1, 2):
            Ns = [(1,)*(2*k)] + ([(2, 1)] if k == 1 else [])
            for N in Ns:
                mats = []
                for _ in range(6):
                    mats.append([[random.randrange(p) for _ in range(k)] for _ in range(k)])
                mats.append([[0]*k for _ in range(k)])
                if k == 2: mats += [[[0, 1], [0, 0]], [[1, 1], [1, 1]], [[1, 0], [0, 0]]]
                for A in mats:
                    if p == 5 and k == 2: continue  # too big for a quick run
                    dimO, dimh, dder = run(p, N, A)
                    dt = det(A, p)
                    pred = dimO-1 if dt else dimO-2
                    cases += 1
                    print(p, k, N, A, "det", dt, "dim h^O", dimh, "dim derived", dder, "claim", pred, "OK" if dder == pred else "MISMATCH", flush=True)
    print("cases", cases)
