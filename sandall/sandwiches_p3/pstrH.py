"""Statement pstrH (The p-structure of h_omega(2k;N), 1st type) 1)-3) checked on every normal shape of the 1st type
(J_k(0), C_k, J_{k,r}(lambda), lambda in F_p^*) for the listed (p, k, N)."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skrpb import Ham, J_k_0, C_k, J_kr
from multiprocessing import Pool

def shapes(k, p):
    yield "J_%d(0)" % k, J_k_0(k)
    if k >= 2: yield "C_%d" % k, C_k(k)
    for r in range(1, k+1):
        if k % r: continue
        for lam in range(1, p):
            yield "J_{%d,%d}(%d)" % (k, r, lam), J_kr(k, r, lam)

def paths(B, i, j, k):
    """admissible paths i -> j: interior vertices distinct, not i, not j"""
    out = []
    def rec(cur, interior, w):
        for nx in range(k):
            if not B[cur][nx]: continue
            ww = w * B[cur][nx]
            if nx == j: out.append((list(interior), ww))
            if nx != i and nx != j and nx not in interior:
                rec(nx, interior + [nx], ww)
    rec(i, [], 1)
    return out

def job(args):
    p, N, name, B = args
    H = Ham(p, N, B)
    A, k = H.A, H.k
    top = A.top
    nz = lambda f: {e: c for e, c in f.items() if any(e)}
    bad = []
    count = 0
    for i in range(k):
        h, obs, ok = H.p_power(A.mul(A.var(i), A.var(i+k)))
        count += 1
        if not ok or nz(A.sub(h, A.mul(A.var(i), A.var(i+k)))): bad.append(("torus", i))
    for i, j in itertools.combinations(range(k), 2):
        for (a, b) in ((i, j), (i+k, j+k)):
            h, obs, ok = H.p_power(A.mul(A.var(a), A.var(b)))
            count += 1
            if not ok or nz(h): bad.append(("half", a, b))
    for i in range(k):
        for j in range(k):
            if i == j: continue
            Phi = A.zero()
            for interior, w in paths(B, i, j, k):
                term = A.mono([0]*(2*k), (-1)**(len(interior)+2) * w)   # (-1)^{l(P)+1}, l = |interior|+1
                for t in interior:
                    term = A.mul(term, A.mul(A.var(t, top[t]-1), A.var(t+k, top[t+k]-1)))
                Phi = A.add(Phi, term)
            S = A.zero()
            for a in range(1, p):
                m = [0]*(2*k)
                m[i] = top[i]-a; m[j+k] = top[j+k]-(p-a); m[j] = a; m[i+k] = p-a
                S = A.add(S, A.mono(m, (-1)**(a-1) * a))
            claim = A.mul(Phi, S)
            h, obs, ok = H.p_power(A.mul(A.var(j), A.var(i+k)))
            count += 1
            if not ok or nz(A.sub(h, claim)): bad.append(("mixed", i, j))
    return (p, N, name, count, bad)

if __name__ == "__main__":
    jobs = []
    for p, k in ((3, 1), (5, 1), (7, 1), (3, 2), (5, 2), (7, 2), (3, 3)):
        Ns = [(1,)*(2*k)]
        if k == 2 and p == 3: Ns += [(2,1,1,2), (1,2,2,1), (2,2,2,2), (1,2,1,2)]
        if k == 1: Ns += [(2,1), (1,2)] if p <= 5 else []
        for N in Ns:
            for name, B in shapes(k, p):
                jobs.append((p, N, name, B))
    with Pool(9) as P:
        res = P.map(job, jobs, chunksize=1)
    tot = sum(r[3] for r in res)
    print("shape-cases", len(res), "elements", tot, "mismatches", sum(len(r[4]) for r in res))
    for r in res:
        if r[4]: print(r)
