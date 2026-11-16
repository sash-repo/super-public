#!/usr/bin/env python3
"""Toral switching of thin tables (CSS 2207.01094 Sect. 3.7) and GL_5 equivalence:
is the thin table of P1(6) toral-switching-connected to the CSS table PO.txt (= LS)?"""
import json, sys, itertools
from thin import load, equiv
n = 31
def mats(T):
    """ad e_a as column bitmasks over basis index a-1"""
    ad = {}
    for a in range(1, 32):
        cols = []
        for b in range(1, 32):
            cols.append((1 << ((a ^ b) - 1)) if T[a][b] else 0)
        ad[a] = cols
    return ad
def app(cols, v):
    r = 0; i = 0
    while v:
        if v & 1: r ^= cols[i]
        v >>= 1; i += 1
    return r
def mul(A, B): return [app(A, b) for b in B]
def add(A, B): return [a ^ b for a, b in zip(A, B)]
def switched(T, beta):
    ad = mats(T)
    if mul(ad[beta], ad[beta]) != [0]*n: return None
    tor = []
    for i in range(5):
        D = [(1 << (a - 1)) if (a >> i & 1) else 0 for a in range(1, 32)]   # t_i e_a = (a_i) e_a
        c = bin(beta & (1 << i)).count('1') % 2        # (alpha_i, beta) = beta_i
        if c: D = add(D, ad[beta])
        tor.append(D)
    # toral and commuting?
    for D in tor:
        if mul(D, D) != D: return None
    for D, E in itertools.combinations(tor, 2):
        if mul(D, E) != mul(E, D): return None
    # simultaneous eigenvectors: for each weight w in F_2^5, eigenspace = cap ker(D_i - w_i)
    def kernel(cols):
        # vectors v with app(cols,v)=0 ; Gaussian on the transpose
        piv = {}; basis = []
        for j in range(n):
            v, t = cols[j], 1 << j
            while v:
                p = v.bit_length() - 1
                if p in piv: v ^= piv[p][0]; t ^= piv[p][1]
                else: piv[p] = (v, t); break
            if not v: basis.append(t)
        return basis
    vec = {}
    for w in range(1, 32):
        cols = []
        for j in range(n):
            col = 0
            for i in range(5):
                img = app(tor[i], 1 << j) ^ ((1 << j) if (w >> i & 1) else 0)
                col |= 0  # placeholder
            cols.append(None)
        # stack the 5 operators D_i - w_i into one map V -> V^5
        stacked = []
        for j in range(n):
            s = 0
            for i in range(5):
                img = app(tor[i], 1 << j) ^ ((1 << j) if (w >> i & 1) else 0)
                s |= img << (n * i)
            stacked.append(s)
        K = kernel(stacked)
        if len(K) != 1: return None
        vec[w] = K[0]
    # coordinates in the new basis
    piv = {}
    for w, v in vec.items():
        t = 1 << w
        while v:
            p = v.bit_length() - 1
            if p in piv: v ^= piv[p][0]; t ^= piv[p][1]
            else: piv[p] = (v, t); break
        assert v
    def coord(v):
        t = 0
        while v:
            p = v.bit_length() - 1
            v ^= piv[p][0]; t ^= piv[p][1]
        return t
    def br(u, v):
        r = 0; i = 0
        while u:
            if u & 1: r ^= app(ad[i + 1], v)
            u >>= 1; i += 1
        return r
    S = [[0]*32 for _ in range(32)]
    for a in range(1, 32):
        for b in range(1, 32):
            c = coord(br(vec[a], vec[b]))
            if c == 0: continue
            assert c == 1 << (a ^ b), (a, b, c)
            S[a][b] = 1
    return S
def closure(T0, depth):
    seen = [T0]; frontier = [T0]
    for _ in range(depth):
        nf = []
        for T in frontier:
            for beta in range(1, 32):
                S = switched(T, beta)
                if S is None: continue
                if any(equiv(S, U) for U in seen): continue
                seen.append(S); nf.append(S)
        frontier = nf
    return seen
if __name__ == '__main__':
    P1 = [[0]*32 for _ in range(32)]
    J = json.load(open('P1_thin.json'))
    LS = load('../css-lib/repo/ThinTables/dim31/PO.txt')
    C1 = closure(J, 2)
    print('P1 switching classes found:', len(C1), flush=True)
    hit = [k for k, S in enumerate(C1) if equiv(S, LS)]
    print('equivalent to PO.txt (LS):', hit, flush=True)
    if not hit:
        C2 = closure(LS, 2); print('LS classes', len(C2))
        print('cross:', [(i, j) for i, S in enumerate(C1) for j, U in enumerate(C2) if equiv(S, U)])
