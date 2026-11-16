"""Parallel H_q (= dim H^q) of positive parts, per weight.  usage: gonch_par.py kind W qmin qmax [T]
Each (q, w) job computes dim C_q,w - rk d_q - rk d_{q+1}.  Output: one line per nonzero class."""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from multiprocessing import Pool
from nsgonch import algebra, chains, dmat, rank, F

def job(args):
    kind, W, q, n = args
    B, br = algebra(kind, W); w = F(n, 2)
    def rk(qq):
        if qq <= 0: return 0
        M, a, b = dmat(B, br, qq, w)
        return rank(M) if a and b else 0
    dim = len(chains(B, q, w))
    if dim == 0: return (q, n, 0, 0)
    return (q, n, dim, dim - rk(q) - rk(q + 1))

if __name__ == "__main__":
    kind, W, q0, q1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    T = F(sys.argv[5]) if len(sys.argv) > 5 else None   # top generator weight -> per-q window
    def top(q):
        return 2 * W if T is None else min(2 * W, int(2 * (q * T + F(3, 2) * q * (q - 1) + 3)))
    jobs = [(kind, W, q, n) for q in range(q0, q1 + 1) for n in range(1, top(q) + 1)]
    jobs.sort(key=lambda j: -j[3])
    res = {}
    with Pool(9) as p:
        for (q, n, dim, h) in p.imap_unordered(job, jobs):
            res.setdefault(q, []).append((n, h))
    for q in sorted(res):
        L = sorted((n, h) for n, h in res[q] if h)
        print(kind, q, sum(h for _, h in L), [(str(F(n, 2)), h) for n, h in L], flush=True)
