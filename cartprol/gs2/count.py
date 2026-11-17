#!/usr/bin/env python3
"""#{x : (ad x)^2 = 0} and #{x : (ad x)^2 = ad x} over F_2, INCLUDING x = 0
(the convention of Table C.3 of Cushing--Stagg--Stewart).  Gray-code enumeration of
L, compiled by numba, parallel over chunks.   usage: count.py TABLE [TABLE ...]
In the text: Theorem 4.6, 4) and Conjecture 4.7 (42752 and 12545)."""
import sys, numpy as np, numba as nb
from tab2txt import load, jacobi_ok

@nb.njit(cache=True)
def app(M, v):
    r = np.uint32(0)
    k = 0
    while v:
        if v & 1:
            r ^= M[k]
        v >>= 1
        k += 1
    return r

@nb.njit(parallel=True, cache=True)
def count(ad, n, low):
    nch = 1 << (n - low)
    nil = np.zeros(nch, np.int64); tor = np.zeros(nch, np.int64)
    for c in nb.prange(nch):
        M = np.zeros(n, np.uint32)
        for b in range(n - low):
            if (c >> b) & 1:
                for j in range(n):
                    M[j] ^= ad[low + b, j]
        a = 0; t = 0
        for g in range(1 << low):
            if g:
                b = 0
                while not (g >> b) & 1:
                    b += 1
                for j in range(n):
                    M[j] ^= ad[b, j]
            isn = True; ist = True
            for j in range(n):
                w = app(M, M[j])
                if w != 0:
                    isn = False
                if w != M[j]:
                    ist = False
                if not isn and not ist:
                    break
            if isn: a += 1
            if ist: t += 1
        nil[c] = a; tor[c] = t
    return nil.sum(), tor.sum()

def invariants(tab):
    n = len(tab)
    ad = np.array(tab, dtype=np.uint32)
    return count(ad, n, min(n, 16))

if __name__ == '__main__':
    for p in sys.argv[1:]:
        tab = load(p)
        ok = jacobi_ok(tab) if len(tab) <= 31 else None
        print('%-50s dim %2d  Jacobi %s  2-nilp %d  toral %d' % ((p.split('/')[-1], len(tab), ok) + tuple(invariants(tab))), flush=True)
