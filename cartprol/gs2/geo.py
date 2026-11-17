#!/usr/bin/env python3
"""Invariants of a Lie algebra over F_2 that do not change under extension of the
ground field: dim Der, dim of the 2-envelope of ad L in gl(L), dim of invariant
bilinear forms (all / symmetric), generic rank of ad x and degree of the minimal
polynomial of a generic ad x (computed at random points over F_{2^31}).
usage: geo.py TABLE.json ...   (tab[i][j] = bitmask of [e_i,e_j])
In the text: Theorem 4.6, 5) and the evidence for Conjecture 4.7."""
import sys, json, random
random.seed(1599)

def load(p): return json.load(open(p))['tab']

def rank2(rows):
    """rank over F_2 of a list of python-int bit rows"""
    piv = {}
    r = 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in piv: v ^= piv[h]
            else: piv[h] = v; r += 1; break
    return r

def adm(tab, i, n):  # matrix of ad e_i: column j = [e_i,e_j]; M[k][j]
    return [[(tab[i][j] >> k) & 1 for j in range(n)] for k in range(n)]

def der_dim(tab):
    n = len(tab)
    # unknown D[k][l] (D e_l = sum_k D[k][l] e_k), index k*n+l
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            for t in range(n):  # coefficient of e_t
                v = 0
                # D[e_i,e_j]: sum_m c_ij^m D[t][m]
                b = tab[i][j]; m = 0
                while b:
                    if b & 1: v ^= 1 << (t * n + m)
                    b >>= 1; m += 1
                # [D e_i, e_j] = sum_k D[k][i] [e_k,e_j]; coefficient of e_t: c_kj^t
                for k in range(n):
                    if (tab[k][j] >> t) & 1: v ^= 1 << (k * n + i)
                    if (tab[i][k] >> t) & 1: v ^= 1 << (k * n + j)
                rows.append(v)
    return n * n - rank2(rows)

def inv_forms(tab):
    n = len(tab)
    # B[a][b], index a*n+b ; B([e_i,e_j],e_k) = B(e_i,[e_j,e_k])
    rows = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = 0
                b = tab[i][j]; m = 0
                while b:
                    if b & 1: v ^= 1 << (m * n + k)
                    b >>= 1; m += 1
                b = tab[j][k]; m = 0
                while b:
                    if b & 1: v ^= 1 << (i * n + m)
                    b >>= 1; m += 1
                if v: rows.append(v)
    allf = n * n - rank2(rows)
    sym = [((1 << (a * n + b)) ^ (1 << (b * n + a))) for a in range(n) for b in range(a + 1, n)]
    symf = n * n - rank2(rows + sym)
    return allf, symf

def matmul2(A, B, n):
    return [[sum(A[i][k] & B[k][j] for k in range(n)) & 1 for j in range(n)] for i in range(n)]

def envelope(tab):
    n = len(tab)
    flat = lambda M: sum(M[i][j] << (i * n + j) for i in range(n) for j in range(n))
    vecs = []
    for i in range(n):
        M = adm(tab, i, n)
        for _ in range(12):
            vecs.append(flat(M)); M = matmul2(M, M, n)
    restricted = rank2([flat(adm(tab, i, n)) for i in range(n)])
    return rank2(vecs), restricted

# ---- GF(2^31), modulus x^31 + x^3 + 1
K = 31; MOD = (1 << 31) | (1 << 3) | 1
def gmul(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        b >>= 1; a <<= 1
        if a >> K: a ^= MOD
    return r
def ginv(a):
    r = 1; e = (1 << K) - 2
    while e:
        if e & 1: r = gmul(r, a)
        a = gmul(a, a); e >>= 1
    return r
def grank(M):
    M = [row[:] for row in M]; r = 0; m = len(M); c = len(M[0])
    for col in range(c):
        p = next((i for i in range(r, m) if M[i][col]), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        iv = ginv(M[r][col]); M[r] = [gmul(iv, x) for x in M[r]]
        for i in range(m):
            if i != r and M[i][col]:
                f = M[i][col]; M[i] = [x ^ gmul(f, y) for x, y in zip(M[i], M[r])]
        r += 1
    return r

def generic(tab, samples=6):
    n = len(tab)
    best_rank = 0; best_deg = 0
    for _ in range(samples):
        x = [random.getrandbits(K) for _ in range(n)]
        # ad x as matrix: column j = sum_i x_i [e_i,e_j]
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            if not x[i]: continue
            for j in range(n):
                b = tab[i][j]; k = 0
                while b:
                    if b & 1: A[k][j] ^= x[i]
                    b >>= 1; k += 1
        best_rank = max(best_rank, grank(A))
        # Krylov: degree of minimal polynomial = rank of {v, Av, ..., A^n v} for generic v
        v = [random.getrandbits(K) for _ in range(n)]
        kr = []
        for _ in range(n + 1):
            kr.append(v)
            v = [0 ^ 0 for _ in range(n)] if False else [
                __import__('functools').reduce(lambda s, t: s ^ t, (gmul(A[i][j], v_j) for j, v_j in enumerate(v)), 0)
                for i in range(n)]
        best_deg = max(best_deg, grank(kr))
    return best_rank, best_deg

if __name__ == '__main__':
    for p in sys.argv[1:]:
        tab = load(p)
        env, rk = envelope(tab)
        a, s = inv_forms(tab)
        gr, gd = generic(tab)
        print(f"{p.split('/')[-1]:40s} dim {len(tab)} Der {der_dim(tab)} 2-env {env} "
              f"invforms {a} sym {s} generic rk ad {gr} (centralizer {len(tab)-gr}) minpoly deg {gd}",
              flush=True)
