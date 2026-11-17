"""Statement 3.2.6: structure of s = g_x/r(g_x): s', centroid of s', Killing form of s_0.
Usage: python3 probe.py m [m ...]   (m = 7 for the rows of Statement 3.2.6 with m = 7)
"""
import sys, numpy as np
from radical2 import *

def sub_alg(s, B):
    """subalgebra spanned by rows B (homogeneous rref rows) as an Alg."""
    B = rref(B)[0]; k = len(B)
    piv = rref(B)[1]
    T2 = np.zeros((k, k, k), dtype=np.int64)
    for a in range(k):
        for b in range(k):
            w = s.brk(B[a], B[b])   # reduced mod P at every step (no int64 overflow)
            T2[a, b] = w[piv]   # coordinates in the rref basis = values at pivots
    par = []
    for r in B:
        ps = {s.par[i] for i in np.nonzero(r)[0]}
        par.append(ps.pop() if len(ps) == 1 else None)
    return Alg(T2, par)

def killing_rank(a, idx):
    A = [a.ad_of(np.eye(a.d, dtype=np.int64)[i]) for i in idx]
    K = np.array([[int(np.trace(matmul(X, Y)) % P) for Y in A] for X in A], dtype=np.int64)
    return rank(K)

def describe(s):
    I = np.eye(s.d, dtype=np.int64)
    D = s.span_br(I, I)
    out = f"s'={len(D)}"
    if len(D) < s.d:
        sp = sub_alg(s, D)
        out += f" sdim s'={sp.sdim()} dim s''={sp.derived_dim()} centroid(s')<={sp.centroid_bound()}"
    ev = [i for i in range(s.d) if s.par[i] == 0]
    if ev:
        e = sub_alg(s, I[ev])
        out += f" | s_0: dim {e.d}, [s_0,s_0]={e.derived_dim()}, Killing rank {killing_rank(e, range(e.d))}, toral rank {e.toral_rank()}, centroid(s_0)<={e.centroid_bound()}"
    return out

if __name__ == "__main__":
    for m in map(int, sys.argv[1:]):
        po = Po(m)
        reps, _ = type_reps(po)
        for ty, msk in sorted(reps.items()):
            x = {msk: 1}
            if not homologic(po, x): continue
            q = Quot(po, x)
            if q.dim == 0: continue
            T = np.array([[[modp(c) for c in v] for v in row] for row in q.table()], dtype=np.int64)
            s, steps = radical_quotient(Alg(T, q.par))
            if s.d == 0: continue
            print(f"m={m} {po.name(msk):>22} g_x/r sdim={s.sdim()} {describe(s)}", flush=True)
