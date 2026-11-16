"""Statement SderH 1) over EVERY alpha in Mat_k(F_p), for the listed (p,k,N)."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derdim import run, det
from multiprocessing import Pool
def job(a):
    p, N, A = a
    dimO, dimh, dder = run(p, N, A)
    pred = dimO-1 if det(A, p) else dimO-2
    return (p, N, A, dder == pred)
if __name__ == "__main__":
    jobs = []
    for p, k, N in ((3,1,(1,1)), (3,1,(2,1)), (3,1,(1,2)), (5,1,(1,1)), (5,1,(2,1)), (3,2,(1,1,1,1))):
        for ent in itertools.product(range(p), repeat=k*k):
            jobs.append((p, N, [list(ent[i*k:(i+1)*k]) for i in range(k)]))
    with Pool(9) as P:
        res = P.map(job, jobs)
    bad = [r for r in res if not r[3]]
    print("cases", len(res), "mismatches", len(bad))
    for r in bad: print(r)
