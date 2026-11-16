#!/usr/bin/env python3
r"""
d4.py --- the simple Lie algebra of type D_4 over F_2, of dim 26, built from
the root system, and the isomorphism

        ir(9; \One)  ~  fo^{(2)}_\Pi(8)/fc  =  the D_4 algebra over F_2,

which is what \cite[\S1.1]{BGLLS} claims without proof ("for N = (1, ..., 1),
they turn into o'_Pi(8)/c ..."); item 5) of the Statement on ir(9; \One)
records only the equality of dimensions.

THE MODEL.  Over F_2 the Chevalley algebra of a simply-laced type is easy: all
its structure constants N_{alpha,beta} = +-1 reduce to 1, so

    [e_a, e_b] = e_{a+b} if a + b is a root,   [e_a, e_{-a}] = h_a,
    [h, e_a]   = <a, h> e_a,                   [h, h'] = 0,

with h_a the coroot of a written in the basis of the simple coroots, all mod 2.
For D_4 the Cartan matrix mod 2 is the adjacency matrix of the diagram, of rank
2, so the center of the 28-dimensional algebra is 2-dimensional and the
quotient is of dim 24 + 2 = 26 -- the algebra called D_4 in the table of
Cushing--Stagg--Stewart, whose census of 2-nilpotent and toral elements
(69616 and 69889) is the one we found for ir(9; \One).

The p-structure is e_a^{[2]} = 0 and h^{[2]} = h; the Cartan subalgebra is a
torus, and its image in the quotient is a 2-dimensional maximal torus with
three weight spaces of dim 8 -- the profile `torus2.py` finds for ir(9; \One)
and for Out ir(9; \One).

Usage:  python3 d4.py
"""

import itertools
import pickle
import sys
import time

import iso2
import lin2
import torus2
from lin2 import bits, echelon, in_span
from torus2 import Alg


def d4_roots():
    """The 24 roots of D_4 as vectors in Z^4, and the four simple ones."""
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 4
                    v[i], v[j] = si, sj
                    roots.append(tuple(v))
    simple = [(1, -1, 0, 0), (0, 1, -1, 0), (0, 0, 1, -1), (0, 0, 1, 1)]
    return roots, simple


def coords_in_simple(a, simple):
    """a = sum c_i alpha_i; returns the c_i (integers)."""
    import fractions
    n = 4
    # solve the 4x4 system over Q
    A = [[simple[j][i] for j in range(n)] for i in range(n)]
    b = list(a)
    M = [[fractions.Fraction(A[i][j]) for j in range(n)] + [fractions.Fraction(b[i])]
         for i in range(n)]
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = 1 / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(n):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        r += 1
    return [int(M[i][n]) for i in range(n)]


def build_d4():
    """The 26-dimensional D_4 algebra over F_2: (BR, SQ) and the basis names."""
    roots, simple = d4_roots()
    ip = lambda a, b: sum(x * y for x, y in zip(a, b))
    # a basis of the Cartan: h_1, ..., h_4; the center is its radical mod 2
    # first build the 28-dimensional algebra
    idx = {a: k for k, a in enumerate(roots)}
    N = len(roots)                                   # 24
    dim = N + 4
    BR = [[0] * dim for _ in range(dim)]
    for a in roots:
        ca = coords_in_simple(a, simple)
        for b in roots:
            if a == b:
                continue
            s = tuple(x + y for x, y in zip(a, b))
            if all(x == 0 for x in s):               # b = -a
                v = 0
                for i in range(4):
                    if ca[i] % 2:
                        v |= 1 << (N + i)
                BR[idx[a]][idx[b]] = v
            elif s in idx:
                BR[idx[a]][idx[b]] = 1 << idx[s]
        for i in range(4):
            if ip(a, simple[i]) % 2:                 # <a, alpha_i^vee> mod 2
                BR[N + i][idx[a]] = 1 << idx[a]
                BR[idx[a]][N + i] = 1 << idx[a]
    SQ = [0] * N + [1 << (N + i) for i in range(4)]
    A28 = Alg(BR, SQ, 'D4(28)')
    # the center
    cen = A28.centralizer([1 << k for k in range(dim)])
    assert len(cen) == 2, 'the center of the D_4 algebra mod 2 is not 2-dim'
    # quotient
    Q, BRq, SQq = quotient(A28, cen)
    return Alg(BRq, SQq, 'D4/c (26)'), A28, cen


def quotient(A, ideal):
    """L/I for a central ideal I: pick a complement and reduce."""
    ide = echelon(ideal)
    comp = []
    for k in range(A.n):
        v = 1 << k
        if not in_span(v, echelon(ide + comp)):
            comp.append(v)
    assert len(comp) + len(ide) == A.n
    m = len(comp)
    # coordinates of a vector modulo the ideal, in the basis `comp`
    rec = []
    for j, u in enumerate(comp):
        v, cm = u, 1 << j
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            rec.append((v.bit_length() - 1, v, cm))
            rec.sort(key=lambda t: -t[0])

    def cls(v):
        v = lin2.reduce_vec(v, ide)
        cm = 0
        for p, b, c in rec:
            if v >> p & 1:
                v ^= b
                cm ^= c
        assert v == 0, 'the complement is wrong'
        return cm

    BR = [[cls(A.br(comp[i], comp[j])) for j in range(m)] for i in range(m)]
    SQ = [cls(A.sq(comp[i])) for i in range(m)]
    return comp, BR, SQ


def census(A, limit=None):
    """(number of x with x^{[2]} = 0, number of toral x), zero included."""
    n = A.n
    nsq, ntor = 1, 1
    cur, prev, subset = 0, 0, []
    for g in range(1, 1 << n):
        gray = g ^ (g >> 1)
        j = (gray ^ prev).bit_length() - 1
        upd = A.SQ[j]
        for i in subset:
            upd ^= A.BR[i][j]
        cur ^= upd
        if j in subset:
            subset.remove(j)
        else:
            subset.append(j)
        prev = gray
        if cur == 0:
            nsq += 1
        elif cur == gray:
            ntor += 1
    return nsq, ntor


def main():
    t0 = time.time()
    D, A28, cen = build_d4()
    print('=== the D_4 algebra over F_2 ===')
    print('  dim of the Chevalley algebra: %d, center of dim %d, quotient of '
          'dim %d' % (A28.n, len(cen), D.n))
    print('  the quotient is perfect: %s'
          % (lin2.dim([D.br(1 << i, 1 << j) for i in range(D.n)
                       for j in range(D.n)]) == D.n))
    T = torus2.maximal_torus(D)
    W = torus2.weights(D, T)
    print('  maximal torus of dim %d, weight spaces %s'
          % (len(T), sorted((len(v) for v in W.values()), reverse=True)))

    print('\n  the census over all 2^26 elements (Gray code) ...')
    nsq, ntor = census(D)
    print('     x^{[2]} = 0: %d      toral: %d      (%.0f s)'
          % (nsq, ntor, time.time() - t0))
    print('     Cushing-Stagg-Stewart, row "26 1  D4": 69616 and 69889')

    with open('algs.pkl', 'rb') as f:
        d = pickle.load(f)
    L1 = Alg(d['BR1'], d['SQ1'], 'ir(9;One)')
    L2 = Alg(d['BR2'], d['SQ2'], 'Out ir(9;One)')
    for L in (L1, L2):
        print('\n  --- is %s isomorphic to the D_4 algebra? ---' % L.name)
        TL = torus2.maximal_torus(L)
        done = False
        for perm in iso2.linear_perms():
            phi, nodes = iso2.graded_search(L, D, TL, T, perm, verbose=False)
            if phi is not None:
                mat = phi.matrix()
                ok = iso2.verify(L, D, mat)
                print('     ISOMORPHISM FOUND (%d nodes), verified on all '
                      'brackets and squares: %s' % (nodes, ok))
                with open('iso-d4-%s.pkl' % L.name.split('(')[0].strip(),
                          'wb') as f:
                    pickle.dump(mat, f)
                done = True
                break
        if not done:
            print('     none found with the tori tried')
    print('\ndone in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
