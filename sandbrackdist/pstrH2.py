import sys
from pstrH import job, shapes
from multiprocessing import Pool
if __name__ == "__main__":
    jobs = [(p, (1,)*(2*k), n, B) for p, k in ((5, 3), (7, 3), (3, 4), (11, 2), (11, 1), (13, 2)) for n, B in shapes(k, p)]
    with Pool(9) as P:
        res = P.map(job, jobs, chunksize=1)
    print("shape-cases", len(res), "elements", sum(r[3] for r in res), "mismatches", sum(len(r[4]) for r in res))
    for r in res:
        if r[4]: print(r)
