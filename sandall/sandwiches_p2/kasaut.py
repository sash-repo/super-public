#!/usr/bin/env python3
r"""
kasaut.py --- an explicit automorphism of \fkas(7;\widetilde\un) carrying the W-grading
``1\xi" to the W-grading ``1\eta" (see kasiso.py for the question, kas2.py for the setting).

A contact transformation of \fk(7;\widetilde\un) that preserves the standard grading is
given by

    \Psi(\xi_i), \Psi(\eta_i) = A applied to the linear span V = <\xi_i, \eta_i>,
    \Psi(t) = t + Q,   Q a quadratic form in the \xi's and \eta's,

extended to \cO(7;\widetilde\un) as a homomorphism of divided power algebras:
\Psi(t^{(k)}) = \sum_i t^{(k-i)}\gamma_i(Q), where \gamma_i(Q) = e_i(m_1,\dots,m_r) is the
i-th elementary symmetric function of the monomials of Q (each monomial is a product of
height-one indeterminates, so \gamma_j of it vanishes for j\geq2).  Multiplicativity of the
resulting map is verified, not assumed.

To carry 1\xi = (2;0,1,1;2,1,1) to 1\eta = (2;2,1,1;0,1,1) such a \Psi must have

    A\xi_1 = \eta_1,  A\eta_1 = \xi_1,  A(W) = W  for W = <\xi_2,\xi_3,\eta_2,\eta_3>,
    Q  of 1\eta-degree 2, i.e. Q in <\xi_1\eta_1> + \Lambda^2 W  (7 dimensions).

We enumerate all such \Psi -- A|_W runs over Sp(W) = Sp_4(2), of order 720, and Q over the
128 elements of that 7-dimensional space -- and keep those that preserve \fkas.

Run:  python3 kasaut.py           (KAS_NT=3 by default; KAS_NT=2 is faster)
"""

import itertools
import os
import sys

import kas2 as K

NVAR, T, XI, ETA = K.NVAR, K.T, K.XI, K.ETA
MONS, IDX, DIMO = K.MONS, K.IDX, K.DIMO
NT = K.NT

D_XI = (2, 0, 1, 1, 2, 1, 1)
D_ETA = (2, 2, 1, 1, 0, 1, 1)

# ----------------------------------------------------------------- the generators

def var(i):
    e = [0] * NVAR
    e[i] = 1
    return 1 << IDX[tuple(e)]

TT = var(T)
X = [var(i) for i in XI]
E = [var(i) for i in ETA]
ONE = 1 << IDX[tuple([0] * NVAR)]

# V = <xi_1,xi_2,xi_3,eta_1,eta_2,eta_3>, in this order
VBASIS = [X[0], X[1], X[2], E[0], E[1], E[2]]
VIDX = {v: i for i, v in enumerate(VBASIS)}
# W = <xi_2,xi_3,eta_2,eta_3>: positions 1,2,4,5 of VBASIS
WPOS = [1, 2, 4, 5]

def bilin(u, v):
    """B(u,v) = {u,v} for u,v linear; a scalar in F_2"""
    b = K.kb(u, v)
    return 1 if b == ONE else (0 if b == 0 else None)

# ----------------------------------------------------------------- Sp(W) = Sp_4(2)

def sp4():
    """all A: W -> W preserving B|_W, as tuples (image of xi_2, xi_3, eta_2, eta_3)"""
    wb = [VBASIS[p] for p in WPOS]
    allw = []
    for mask in range(1, 16):
        v = 0
        for i in range(4):
            if (mask >> i) & 1:
                v ^= wb[i]
        allw.append(v)
    target = [[bilin(a, b) for b in wb] for a in wb]
    out = []
    def rec(img):
        k = len(img)
        if k == 4:
            out.append(tuple(img))
            return
        for w in allw:
            if w in img:
                continue
            ok = all(bilin(w, img[j]) == target[k][j] for j in range(k))
            if ok:
                rec(img + [w])
    rec([])
    return out

# ----------------------------------------------------------------- building Psi

def gamma(Q, i):
    """i-th divided power of Q = m_1 + ... + m_r: the elementary symmetric e_i"""
    mons = [1 << b for b in K.bits(Q)]
    if i == 0:
        return ONE
    r = 0
    for comb in itertools.combinations(mons, i):
        p = comb[0]
        for c in comb[1:]:
            p = K.mul(p, c)
            if p == 0:
                break
        r ^= p
    return r

def build_psi(Aimg, Q):
    """Aimg = images of xi_1,xi_2,xi_3,eta_1,eta_2,eta_3; returns the table Psi(monomial)
    or None if the map is not multiplicative"""
    # Psi of the t-divided-powers
    tpow = {0: ONE}
    for k in range(1, 2 ** NT):
        r = 0
        for i in range(0, k + 1):
            g = gamma(Q, i)
            if g == 0:
                continue
            e = [0] * NVAR
            e[T] = k - i
            r ^= K.mul(1 << IDX[tuple(e)], g)
        tpow[k] = r
    tab = []
    for m in MONS:
        v = tpow[m[T]]
        for j, i in enumerate(XI + ETA):
            if m[i]:
                v = K.mul(v, Aimg[j if j < 3 else j])
        tab.append(v)
    return tab

def apply_tab(tab, f):
    r = 0
    for k in K.bits(f):
        r ^= tab[k]
    return r

def is_multiplicative(tab):
    for i in range(DIMO):
        for j in range(DIMO):
            if apply_tab(tab, K.mul(1 << i, 1 << j)) != K.mul(tab[i], tab[j]):
                return False
    return True

def preserves_bracket(tab):
    for i in range(DIMO):
        for j in range(DIMO):
            if apply_tab(tab, K.kb(1 << i, 1 << j)) != K.kb(tab[i], tab[j]):
                return False
    return True

# ----------------------------------------------------------------- main

def main():
    print('kasaut.py --- an automorphism of kas(7;N) taking 1xi to 1eta,  N_t = %d' % NT)
    comp = K.build_kas()
    kas = K.echelon([v for j in sorted(comp) for v in comp[j]])
    U = comp[1]
    pvU = K.pivots(U)
    pvkas = K.pivots(kas)
    print('dim kas = %d, dim U = %d' % (len(kas), len(U)))

    # the 7-dimensional space of admissible Q
    qmons = [K.mul(X[0], E[0])]
    wb = [VBASIS[p] for p in WPOS]
    for a, b in itertools.combinations(range(4), 2):
        qmons.append(K.mul(wb[a], wb[b]))
    print('admissible Q: 2^%d = %d' % (len(qmons), 2 ** len(qmons)))

    sp = sp4()
    print('|Sp(W)| = %d' % len(sp))

    found = []
    tested = 0
    for Aw in sp:
        # A on V: xi_1 -> eta_1, eta_1 -> xi_1, W -> W by Aw
        Aimg = [0] * 6
        Aimg[0] = E[0]            # xi_1 -> eta_1
        Aimg[3] = X[0]            # eta_1 -> xi_1
        for k, p in enumerate(WPOS):
            Aimg[p] = Aw[k]
        for qmask in range(2 ** len(qmons)):
            Q = 0
            for i in range(len(qmons)):
                if (qmask >> i) & 1:
                    Q ^= qmons[i]
            tested += 1
            tab = build_psi(Aimg, Q)
            # cheap necessary conditions on the generators
            PT = tab[IDX[tuple([1] + [0] * 6)]]
            bad = False
            for i in range(3):
                if K.kb(PT, Aimg[i]) != 0:
                    bad = True
                    break
                if K.kb(PT, Aimg[3 + i]) != Aimg[3 + i]:
                    bad = True
                    break
            if bad:
                continue
            # does it preserve U (hence kas)?
            if not all(K.in_span(apply_tab(tab, u), pvU) for u in U):
                continue
            found.append((Aimg, Q, tab))

    print('candidates tested: %d;  preserving U: %d' % (tested, len(found)))
    if not found:
        print('NONE -- no standard-graded contact automorphism takes 1xi to 1eta')
        return

    # full verification of the first few
    good = []
    for Aimg, Q, tab in found:
        if not is_multiplicative(tab):
            continue
        if not preserves_bracket(tab):
            continue
        if not all(K.in_span(apply_tab(tab, v), pvkas) for v in kas):
            continue
        good.append((Aimg, Q, tab))
        if len(good) >= 3:
            break
    print('fully verified automorphisms of kas among them: %d (of the first checked)'
          % len(good))
    if not good:
        print('none of the U-preserving candidates is a bracket automorphism')
        return

    names = ['xi_1', 'xi_2', 'xi_3', 'eta_1', 'eta_2', 'eta_3']
    def show(v):
        if v == 0:
            return '0'
        out = []
        for b in K.bits(v):
            m = MONS[b]
            s = ''
            if m[T]:
                s += 't' + ('' if m[T] == 1 else '^(%d)' % m[T])
            for j, i in enumerate(XI + ETA):
                if m[i]:
                    s += names[j]
            out.append(s if s else '1')
        return ' + '.join(out)

    for Aimg, Q, tab in good:
        print('\n  ---- automorphism ----')
        for j in range(6):
            print('    %-6s -> %s' % (names[j], show(Aimg[j])))
        print('    %-6s -> %s' % ('t', show(TT ^ Q)))
        # does it really take the 1xi grading to the 1eta grading?
        gX = K.graded_pieces(kas, D_XI)
        gY = K.graded_pieces(kas, D_ETA)
        ok = True
        for c in sorted(gX):
            pv = K.pivots(gY.get(c, []))
            for v in gX[c]:
                if not K.in_span(apply_tab(tab, v), pv):
                    ok = False
                    break
            if not ok:
                print('    FAILS to map degree %d of 1xi into degree %d of 1eta' % (c, c))
                break
        if ok:
            print('    maps kas^{1xi}_c onto kas^{1eta}_c for every c:  YES')
            print('    ==> the two W-gradings are ISOMORPHIC as graded Lie algebras')


if __name__ == '__main__':
    main()
