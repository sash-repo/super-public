#!/usr/bin/env python3
"""#{x in L (x) F_4 : (ad x)^2 = 0} and #{x : (ad x)^2 = ad x}, incl. 0.  x = a + b w,
w^2 = w + 1, ad x = A + B w: (ad x)^2 = (A^2 + B^2) + (AB + BA + B^2) w.
Matrices by columns as bitmasks (M[j] = image of e_j).  usage: countF4.py (all dim-15 + L15)"""
import sys, json, glob, numpy as np, numba as nb
from loadcss import load, library

@nb.njit(cache=True)
def app(M, v):
    r = np.uint32(0); k = 0
    while v:
        if v & 1: r ^= M[k]
        v >>= 1; k += 1
    return r

@nb.njit(parallel=True, cache=True)
def countF4(ad, n):
    NB = 1 << n
    nil = np.zeros(NB, np.int64); tor = np.zeros(NB, np.int64)
    for bb in nb.prange(NB):
        B = np.zeros(n, np.uint32)
        for k in range(n):
            if (bb >> k) & 1:
                for j in range(n): B[j] ^= ad[k, j]
        BB = np.zeros(n, np.uint32)
        for j in range(n): BB[j] = app(B, B[j])
        A = np.zeros(n, np.uint32); a_n = 0; a_t = 0
        for g in range(NB):
            if g:
                k = 0
                while not (g >> k) & 1: k += 1
                for j in range(n): A[j] ^= ad[k, j]
            isn = True; ist = True
            for j in range(n):
                p0 = app(A, A[j]) ^ BB[j]
                p1 = app(A, B[j]) ^ app(B, A[j]) ^ BB[j]
                if p0 != 0 or p1 != 0: isn = False
                if p0 != A[j] or p1 != B[j]: ist = False
                if not isn and not ist: break
            if isn: a_n += 1
            if ist: a_t += 1
        nil[bb] = a_n; tor[bb] = a_t
    return nil.sum(), tor.sum()

def cols(ad):
    d = ad.shape[0]
    w = (1 << np.arange(d)).astype(np.uint64)
    return np.array([[int((ad[i, :, j].astype(np.uint64) * w).sum()) for j in range(d)] for i in range(d)], np.uint32)

if __name__ == '__main__':
    todo = []
    z = [np.load(p) for p in glob.glob('found/*.npz') if str(np.load(p)['fp']).startswith('15:1:False:simple:14')][0]
    todo.append(('L15', z['ad']))
    for nm, p in library():
        ad = load(p)
        if ad.shape[0] == 15: todo.append((nm, ad))
    res = {}
    for nm, ad in todo:
        a, t = countF4(cols(ad), 15)
        res[nm] = [int(a), int(t)]
        print('%-14s over F_4: 2-nilp %d  toral %d' % (nm, a, t), flush=True)
    json.dump(res, open('countF4.json', 'w'), indent=0)
