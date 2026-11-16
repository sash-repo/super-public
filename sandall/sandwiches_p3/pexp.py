"""Statement pstrexp on h_{exp_j}(2k;1): which (ad_f)^p, f a monomial, are inner;
and is (ad_{u_{~j}})^p diagonal in the monomial basis."""
import os, sys, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from skrpb import HamExp
from exactlin import solve
from multiprocessing import Pool

def job(args):
    p, k, jj = args
    H = HamExp(p, (1,)*(2*k), jj)
    A = H.A
    mons = list(itertools.product(*[range(t) for t in A.top]))
    idx = {m: i for i, m in enumerate(mons)}
    basis = [A.mono(m) for m in mons]
    rnd = random.Random(1)
    X = basis if len(basis) <= 100 else rnd.sample(basis, 60)
    BR = [[H.br(b, x) for x in X] for b in basis]          # BR[m][x]
    noninner = []
    diag_ok = None
    tl = (jj + k) % (2*k)
    for mi, f in enumerate(basis):
        if not any(mons[mi]): continue
        D = [H.ad_pow(f, x, p) for x in X]
        # rows: for each x and each monomial e, sum_m c_m BR[m][x][e] = D[x][e]
        rows, rhs = [], []
        for xi in range(len(X)):
            keys = set(D[xi])
            for m in range(len(basis)): keys |= set(BR[m][xi])
            for e in keys:
                rows.append([BR[m][xi].get(e, 0) for m in range(len(basis))]); rhs.append(D[xi].get(e, 0))
        sol = solve(rows, rhs, p)
        inner = sol is not None
        if inner and X is not basis:   # verify on every basis element
            g = {}
            for m, c in enumerate(sol):
                if int(c) % p: g = A.add(g, A.mono(mons[m], int(c)))
            inner = all(not A.sub(H.ad_pow(f, x, p), H.br(g, x)) for x in basis)
        if not inner: noninner.append(mons[mi])
        if mons[mi] == tuple(1 if i == tl else 0 for i in range(2*k)):
            diag_ok = all(set(H.ad_pow(f, b, p)) <= {mons[bi]} for bi, b in enumerate(basis))
    expect = [tuple(1 if i == tl else 0 for i in range(2*k))]
    return (p, k, jj, noninner == expect, noninner, diag_ok)

if __name__ == "__main__":
    jobs = [(p, k, jj) for p, k in ((3, 1), (5, 1), (7, 1), (3, 2)) for jj in range(2*k)]
    with Pool(9) as P:
        res = P.map(job, jobs, chunksize=1)
    for r in res: print(r)
    print("cases", len(res), "all as claimed", all(r[3] and r[5] for r in res))
