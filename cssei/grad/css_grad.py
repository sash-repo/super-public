#!/usr/bin/env python3
r"""Z/2-gradings of the simple Lie algebras over F_2 found by Cushing--Stagg--Stewart
(arXiv:2207.01094) and the superdimensions of their "method 2" superizations.

As in shen_grad.py: a Z/2-grading is a derivation D with D^2 = D (g_0 = Ker D,
g_1 = Im D).  Der g is computed exactly over F_2; its F_2-points D with D^2 = D are
enumerated by a Gray-code walk compiled with numba (parallel over chunks); for each
grading we record
   (dim g_0, dim g_1, D inner?, dim L/ad g, dim [g_0,g_0], rank of torus-type data)
where L = ad g + Span{(ad x)^2 | x in g_1} (inside End g), so that the superization
s(g,gr) has sdim (dim g_0 + dim L/ad g | dim g_1).
usage: python3 grad/css_grad.py TABLE [TABLE ...]   (CSS .txt or our .json)
"""
import sys, os, json, time
from collections import Counter
import numpy as np
import numba as nb
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'cmp'))
from f2vf import kernel_bits, BitEchelon
from tab2txt import load, jacobi_ok


def der_basis(tab):
    """Der g over F_2 as a list of column-bitmask matrices (M[j] = D e_j as a bitmask)."""
    n = len(tab)
    C = np.zeros((n, n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j, k] = (tab[i][j] >> k) & 1
    iu, ju = np.triu_indices(n, 1)                 # equations only for i<j (antisymmetry)
    npairs = len(iu)
    pairidx = -np.ones((n, n), dtype=np.int64)
    pairidx[iu, ju] = np.arange(npairs); pairidx[ju, iu] = np.arange(npairs)
    imgs = []
    for l in range(n):
        for k in range(n):
            E = np.zeros((npairs, n), dtype=np.uint8)   # (pair, m)
            # D[e_i,e_j] contains C[i,j,k] * D e_k, D e_k = e_l
            E[:, l] ^= C[iu, ju, k]
            # the pair {k, o}: the Leibniz term with D e_k = e_l is [e_l, e_o], once
            # (char 2: [e_l,e_o] = [e_o,e_l]), coordinate m: C[l, o, m]
            for o in range(n):
                if o != k:
                    E[pairidx[k, o]] ^= C[l, o]
            bits = np.packbits(E.flatten(), bitorder='little')
            imgs.append(int.from_bytes(bits.tobytes(), 'little'))
    ker = kernel_bits(imgs)
    out = []
    for kv in ker:
        M = [0] * n                                   # column j: bits l with D[l,j]
        for idx in range(n * n):
            if (kv >> idx) & 1:
                l, k = divmod(idx, n)
                M[k] |= 1 << l
        out.append(M)
    return out, C


@nb.njit(cache=True)
def app(M, v):
    r = np.uint64(0); k = 0
    while v:
        if v & np.uint64(1):
            r ^= M[k]
        v >>= np.uint64(1); k += 1
    return r


@nb.njit(parallel=True, cache=True)
def walk(B, n, low, store, cap):
    """B: d x n uint64 basis of Der.  Count / store coefficient words of the D with D^2 = D."""
    d = B.shape[0]
    nch = 1 << (d - low)
    cnt = np.zeros(nch, np.int64)
    for c in nb.prange(nch):
        M = np.zeros(n, np.uint64)
        for b in range(d - low):
            if (c >> b) & 1:
                for j in range(n):
                    M[j] ^= B[low + b, j]
        t = 0
        for g in range(1 << low):
            if g:
                b = 0
                while not (g >> b) & 1:
                    b += 1
                for j in range(n):
                    M[j] ^= B[b, j]
            ok = True
            for j in range(n):
                if app(M, M[j]) != M[j]:
                    ok = False
                    break
            if ok:
                if store.shape[1] > 0 and t < cap:
                    store[c, t] = (np.int64(c) << low) | (g ^ (g >> 1))
                t += 1
        cnt[c] = t
    return cnt


def rank_bits(vs):
    E = BitEchelon()
    for v in vs:
        E.add(v)
    return len(E)


def analyse(path):
    t0 = time.time()
    tab = load(path)
    n = len(tab)
    assert n <= 64
    name = os.path.basename(path).rsplit('.', 1)[0]
    DER, C = der_basis(tab)
    d = len(DER)
    ADS = []
    for i in range(n):                       # ad e_i, columns
        ADS.append([tab[i][j] for j in range(n)])
    def flat(M):
        v = 0
        for j, col in enumerate(M):
            v |= col << (j * n)
        return v
    inner = BitEchelon()
    for A in ADS:
        inner.add(flat(A))
    outer = d - len(inner)
    print('%s: dim %d, Jacobi %s, dim Der %d, outer %d' % (name, n, jacobi_ok(tab) if n <= 31 else None, d, outer), flush=True)
    B = np.array(DER, dtype=np.uint64)
    low = min(d, 18)
    cnt = walk(B, n, low, np.zeros((1, 0), np.int64), 0)
    total = int(cnt.sum()) - 1               # minus D = 0
    cap = int(cnt.max())
    store = np.zeros((len(cnt), cap), np.int64)
    walk(B, n, low, store, cap)
    words = [int(store[c, t]) for c in range(len(cnt)) for t in range(int(cnt[c])) if store[c, t] != 0]
    print('  idempotent D != 0 over F_2: %d  (%.1fs)' % (total, time.time() - t0), flush=True)
    assert len(words) == total

    def mat(w):
        M = [0] * n
        for k in range(d):
            if (w >> k) & 1:
                for j in range(n):
                    M[j] ^= DER[k][j]
        return M
    def ad(x):                               # columns of ad x = XOR of the ad e_i
        M = [0] * n; i = 0
        while x:
            if x & 1:
                A = ADS[i]
                M = [M[j] ^ A[j] for j in range(n)]
            x >>= 1; i += 1
        return M
    def brv(a, b):
        return app_py(ad(a), b)
    def mul(A, Bm):                          # (A B) e_j = A(B e_j)
        return [app_py(A, Bm[j]) for j in range(n)]
    def app_py(A, v):
        r = 0; k = 0
        while v:
            if v & 1:
                r ^= A[k]
            v >>= 1; k += 1
        return r
    def basis_of(vs):
        E = BitEchelon(); out = []
        for v in vs:
            if E.add(v):
                out.append(v)
        return out
    json.dump({'name': name, 'dimDer': d, 'words': words, 'der': DER},
              open(os.path.join(HERE, 'results', 'cssgrad', name + '_words.json'), 'w'))
    rows = Counter(); examples = {}
    for w in words:
        D = mat(w)
        g1 = basis_of(D)                                    # image
        g0 = [kv for kv in kernel_bits(D)]                  # kernel (bitmask over basis)
        isin = inner.reduce(flat(D)) == 0 if hasattr(inner, 'reduce') else None
        Lsp = BitEchelon()
        for A in ADS:
            Lsp.add(flat(A))
        for x in g1:
            A = ad(x)
            Lsp.add(flat(mul(A, A)))
        extra = len(Lsp) - len(inner)
        der0 = rank_bits([brv(a, b) for a in g0 for b in g0])
        key = (len(g0), len(g1), bool(isin), extra, der0)
        rows[key] += 1
        examples.setdefault(key, w)
    print('  (dim g0, dim g1, inner, dim L/ad g, dim [g0,g0]) : number;  sdim s(g,gr)')
    for k in sorted(rows):
        print('    %s : %d;  (%d|%d)' % (k, rows[k], k[0] + k[3], k[1]), flush=True)
    return {'name': name, 'dim': n, 'dimDer': d, 'outer': outer, 'idempotents': total,
            'classes': [[list(k), v] for k, v in sorted(rows.items())],
            'examples': {str(list(k)): v for k, v in examples.items()},
            'der': DER}


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'results', 'cssgrad'), exist_ok=True)
    for p in sys.argv[1:]:
        r = analyse(p)
        json.dump(r, open(os.path.join(HERE, 'results', 'cssgrad', r['name'] + '.json'), 'w'))
