"""The 8P^2 bilinear maps B_ij and theta B_ij (B_ij = m o D1^i D2^j, i,j < 2P) on O(1;N|1) over F_p are linearly
independent (Lemma 7.2): prints their ranks.  Also the library (basics, bilin, flat) used by the other bin*.py.
Usage: python3 binnum.py p N"""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace
from osp1 import setup, generators, rep
from math import comb

def basics(p, N):
    S = setup(p, N); n = S['n']; P = S['P']
    D = S['add'](S['mul'](S['TH'], S['Dt']), S['Dth'])
    pows = [S['I']]
    for _ in range(2*P): pows.append(S['mul'](D, pows[-1]))
    def prod(u, v):   # basis u=2r+e, v=2s+f -> dict
        r, e = divmod(u, 2); s, f = divmod(v, 2)
        if e and f: return {}
        if r+s >= P: return {}
        return {2*(r+s)+(e or f): comb(r+s, r) % p}
    return S, D, pows, prod

def bilin(S, pows, prod, eps, i, j):
    p, n = S['p'], S['n']
    T = {}
    Di, Dj = pows[i], pows[j]
    for u in range(n):
        su = -1 if (j % 2 and u % 2) else 1
        for v in range(n):
            out = {}
            for x in range(n):
                if not Di[x][u]: continue
                for y in range(n):
                    if not Dj[y][v]: continue
                    for z, w in prod(x, y).items():
                        if eps:
                            r, e = divmod(z, 2)
                            if e: continue
                            z = z + 1
                        out[z] = (out.get(z, 0) + su*Di[x][u]*Dj[y][v]*w) % p
            for z, w in out.items():
                if w: T[(z, u, v)] = w
    return T

def flat(T, n): 
    v = [0]*(n**3)
    for (z, u, v_), w in T.items(): v[(z*n+u)*n+v_] = w
    return v

if __name__ == "__main__":
    p, N = int(sys.argv[1]), int(sys.argv[2])
    S, D, pows, prod = basics(p, N); n, P = S['n'], S['P']
    fam0 = [flat(bilin(S, pows, prod, 0, i, j), n) for i in range(2*P) for j in range(2*P)]
    fam1 = [flat(bilin(S, pows, prod, 1, i, j), n) for i in range(2*P) for j in range(2*P)]
    print("rank m(D1^i D2^j):", rank(fam0, p), "of", len(fam0))
    print("rank theta m(..):", rank(fam1, p), "of", len(fam1))
    print("rank all:", rank(fam0+fam1, p), "of", len(fam0)+len(fam1))
