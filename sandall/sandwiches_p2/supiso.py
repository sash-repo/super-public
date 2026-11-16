#!/usr/bin/env python3
r"""
supiso.py --- are the two superizations of ir(9; \One) of one and the same
superdimension isomorphic as ABSTRACT Lie superalgebras?

The Theorem on the admissible parity functions
gives 16 admissible parity functions, falling into five families,

  1) p = 0                          ir(9; \un)               sdim (26|0)
  2) 6 functions, x_1 even          ir(5; \un|4)             sdim (14|12)
  3) 1 function,  x_1 even          ir(1; \un|8)             sdim (10|16)
  4) 6 functions, x_1 odd           ir(4; \un|5)             sdim (14|12)
  5) 2 functions, x_1 odd           widetilde{ir}(4; \un|5)  sdim (10|16)

and `iriso.py` showed that the members of each family are pairwise isomorphic
(48 contact automorphisms, orbit sizes 1, 6, 1, 6, 2).  What was open is
whether 2) is isomorphic to 4) and 3) to 5) -- they are told apart by their
gradings, not by any invariant tried so far.

THE REDUCTION.  A Z/2-grading of a Lie algebra L is the same thing as a
derivation D with D^2 = D (D = 0 on the even part, D = 1 on the odd part).
Here every one of the four gradings is INNER: D = ad t with t toral (t^{[2]} =
t), because ad_t^2 = ad_{t^{[2]}} = ad_t.  So the superization is
        g_ev = Ker ad t = the centralizer of t,   g_od = Im ad t,
and two superizations are isomorphic as superalgebras exactly when some
automorphism of L carries t to t' (the center of L being 0, an automorphism
carrying one grading to the other carries t to t').

So the question becomes the conjugacy of two toral elements under Aut(L), and
that is decided by the torus-seeded search of `iso2.py`: extend t to a maximal
torus (2-dimensional here), and look for an isomorphism carrying the one torus
to the other by a map that sends t to t'.

Usage:  python3 supiso.py
"""

import itertools
import pickle
import random
import sys
import time

import ir as M
import iriso as I
import iso2
import lin2
import torus2
from lin2 import bits, echelon, in_span
from torus2 import Alg

DIM = I.DIM


def hat_algebra(A, D, name):
    """L^ = L + F_2 D, the semidirect sum by the idempotent derivation D.

    D is toral in it (D^{[2]} = D o D = D), L = [L^, L^] is characteristic, so
    an isomorphism L^_1 -> L^_2 carrying D_1 to D_2 restricts to an
    automorphism of L intertwining the two Z/2-gradings -- which is exactly
    an isomorphism of the two superizations.
    """
    n = A.n
    BR = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            BR[i][j] = A.BR[i][j]
        BR[i][n] = D[i]
        BR[n][i] = D[i]
    SQ = [A.SQ[i] for i in range(n)] + [1 << n]
    return Alg(BR, SQ, name)


def maximal_tori_through(A, t, tries=24, cap=200000):
    """Distinct maximal tori containing the toral element t."""
    out, seen = [], set()
    for _ in range(tries * 8):
        T = [t]
        while True:
            Cz = A.centralizer(T)
            if len(Cz) > 20:
                Cz = Cz[:20]
            pool = []
            span = echelon(T)
            for k in range(1, 1 << len(Cz)):
                v = 0
                for i in bits(k):
                    v ^= Cz[i]
                if v and not in_span(v, span) and A.sq(v) == v:
                    pool.append(v)
                if len(pool) > 500:
                    break
            if not pool:
                break
            T.append(random.choice(pool))
        key = tuple(echelon(T))
        if key not in seen:
            seen.add(key)
            out.append(T)
            if len(out) >= tries:
                break
    return out


def invertible_first_row_e0(r):
    """All invertible r x r matrices over F_2 whose first row is (1, 0, ..., 0)."""
    out = []
    rows = list(range(1 << r))
    for rest in itertools.product(rows, repeat=r - 1):
        M = [[1 if k == 0 else 0 for k in range(r)]]
        for v in rest:
            M.append([(v >> k) & 1 for k in range(r)])
        if iso2.inv_mat(M) is not None:
            out.append(M)
    return out


def outer_conj(A, D1, D2, verbose=True, budget=400000):
    """Is there phi in Aut(A) with phi D_1 phi^{-1} = D_2, for outer idempotent
    derivations D_i?  Searched inside the two 27-dimensional extensions."""
    H1 = hat_algebra(A, D1, 'ir + F_2 D1')
    H2 = hat_algebra(A, D2, 'ir + F_2 D2')
    d1 = 1 << A.n
    T1s = maximal_tori_through(H1, d1)
    T2s = maximal_tori_through(H2, d1)
    if verbose:
        print('     maximal tori through D: ranks %s / %s'
              % (sorted({len(T) for T in T1s}), sorted({len(T) for T in T2s})))
    for T1 in T1s:
        for T2 in T2s:
            if len(T1) != len(T2):
                continue
            for Mm in invertible_first_row_e0(len(T1)):
                phi, nodes = iso2.graded_search_M(H1, H2, T1, T2, Mm,
                                                  budget=budget, verbose=False)
                if phi is not None:
                    mat = phi.matrix()
                    assert iso2.verify(H1, H2, mat)
                    assert mat[A.n] == d1, 'D_1 does not go to D_2'
                    return mat
    return None


def op_of_grading(C, ev, od):
    """The grading operator D (list of 26 images of the basis vectors)."""
    evb = [C.co(v) for v in ev]
    odb = [C.co(v) for v in od]
    # write e_j = u + w with u even, w odd; D(e_j) = w
    rec = []                            # (pivot, vec, tag) tag = odd part
    for v in evb:
        s, tag = v, 0
        for p, b, t in rec:
            if s >> p & 1:
                s ^= b
                tag ^= t
        if s:
            rec.append((s.bit_length() - 1, s, tag))
            rec.sort(key=lambda x: -x[0])
    for v in odb:
        s, tag = v, v
        for p, b, t in rec:
            if s >> p & 1:
                s ^= b
                tag ^= t
        if s:
            rec.append((s.bit_length() - 1, s, tag))
            rec.sort(key=lambda x: -x[0])
    D = []
    for j in range(DIM):
        s, tag = 1 << j, 0
        for p, b, t in rec:
            if s >> p & 1:
                s ^= b
                tag ^= t
        assert s == 0, 'the even and odd parts do not span'
        D.append(tag)
    return D


def inner(A, D):
    """t with ad_t = D, or None."""
    rows = []
    for j in range(A.n):
        for t in range(A.n):
            e = 0
            for i in range(A.n):
                if A.BR[i][j] >> t & 1:
                    e |= 1 << i
            rhs = (D[j] >> t) & 1
            rows.append((e, rhs))
    # solve the linear system sum_i c_i BR[i][j] = D[j]
    eqs = [(e << 1) | r for e, r in rows]
    piv = lin2.rref(eqs)
    sol = 0
    for p, r in piv.items():
        pass
    # back-substitution: variables are bits 1..n, the constant is bit 0
    # rebuild by Gaussian elimination on (coeff | rhs)
    ech = []
    for e, r in rows:
        v = (e << 1) | r
        for b in ech:
            if v >> (b.bit_length() - 1) & 1:
                v ^= b
        if v:
            if v.bit_length() == 1:          # 0 = 1
                return None
            ech.append(v)
            ech.sort(key=lambda u: -u.bit_length())
    # solve
    sol = 0
    for b in sorted(ech, key=lambda u: u.bit_length()):
        p = b.bit_length() - 1
        s = b & ((1 << p) - 1)               # the rest
        val = 0
        for q in bits(s):
            if q == 0:
                val ^= 1
            else:
                val ^= (sol >> (q - 1)) & 1
        if val:
            sol |= 1 << (p - 1)
    t = sol
    # check
    for j in range(A.n):
        img = 0
        for i in bits(t):
            img ^= A.BR[i][j]
        if img != D[j]:
            return None
    return t


def tori_through(A, t, tries=60):
    """Maximal tori containing the toral element t."""
    out = []
    C = A.centralizer([t])
    pool = []
    for k in range(1, 1 << len(C)):
        v = 0
        for i in bits(k):
            v ^= C[i]
        if v and A.sq(v) == v and not in_span(v, [t]):
            pool.append(v)
    seen = set()
    for u in pool:
        T = echelon([t, u])
        key = tuple(T)
        if key in seen:
            continue
        seen.add(key)
        out.append([t, u])
        if len(out) >= tries:
            break
    return out, len(pool)


def conj_search(A, t1, t2, verbose=True, budget=200000):
    """Is there phi in Aut(A) with phi(t1) = t2?  Tries every maximal torus
    through t1 against one through t2, with both admissible torus maps."""
    T1s, n1 = tori_through(A, t1)
    T2s, n2 = tori_through(A, t2)
    if verbose:
        print('     toral elements in C(t) outside K.t: %d and %d '
              '(so %d resp. %d maximal tori through t)'
              % (n1, n2, n1 // 2, n2 // 2))
    for T1 in T1s:
        for T2 in T2s[:1]:
            for m10, m11 in ((0, 1), (1, 1)):
                Mmat = [[1, 0], [m10, m11]]
                phi, nodes = iso2.graded_search_M(A, A, T1, T2, Mmat,
                                                  budget=budget, verbose=False)
                if phi is not None:
                    mat = phi.matrix()
                    assert iso2.verify(A, A, mat)
                    return mat
    return None


def main():
    t0 = time.time()
    print('=== the superizations of ir(9; \\One): are 2) and 4), 3) and 5) '
          'isomorphic? ===')
    basis, _ = I.build_ir()
    C = I.Coords(basis)
    A = Alg(C.BR, C.SQ, 'ir(9;One)')
    adm = I.admissible([v for v in basis])
    print('  %d admissible parity functions' % len(adm))

    data = []
    for p, ev, od in adm:
        if not any(p):
            continue
        D = op_of_grading(C, ev, od)
        t = inner(A, D)
        sdim = (len(M.echelon(ev)), len(M.echelon(od)))
        fam = I.family(p)
        data.append((p, fam, sdim, t, D))

    print('\n  grading, family, sdim, is the grading inner, rank ad_t')
    for p, fam, sdim, t, D in data:
        r = A.rank_ad(t) if t is not None else -1
        print('    p=%s  family %d  sdim %s  inner: %-5s  t toral: %-5s  '
              'rank ad_t = %d'
              % (''.join(map(str, p)), fam, sdim, t is not None,
                 (A.sq(t) == t) if t is not None else '-', r))

    # representatives: one per orbit, by (family, sdim)
    reps = {}
    for p, fam, sdim, t, D in data:
        key = (fam, sdim)
        if key not in reps:
            reps[key] = (p, t, sdim, D)
    print('\n  representatives: %s' % sorted(reps))

    pairs = []
    for key1, key2 in itertools.combinations(sorted(reps), 2):
        if reps[key1][2] == reps[key2][2]:
            pairs.append((key1, key2))
    for key1, key2 in pairs:
        p1, t1, sdim = reps[key1][:3]
        p2, t2, _ = reps[key2][:3]
        print('\n  --- %s  versus  %s   (sdim %s)' % (key1, key2, sdim))
        if t1 is None or t2 is None:
            print('     the gradings are outer: searching inside ir + F_2 D')
            D1 = reps[key1][3]
            D2 = reps[key2][3]
            phi = outer_conj(A, D1, D2)
            if phi is None:
                print('     NO isomorphism found carrying D_1 to D_2 '
                      '(with the tori tried)')
            else:
                print('     ISOMORPHISM FOUND with phi(D_1) = D_2, verified: '
                      'the two superizations ARE isomorphic')
                with open('supiso-%s-%s.pkl' % (key1[0], key2[0]), 'wb') as f:
                    pickle.dump({'phi': phi, 'p1': p1, 'p2': p2}, f)
            continue
        phi = conj_search(A, t1, t2)
        if phi is None:
            print('     NO automorphism found carrying t1 to t2 '
                  '(with the tori tried)')
        else:
            print('     AUTOMORPHISM FOUND with phi(t1) = t2, verified: '
                  'the two superizations ARE isomorphic')
            with open('supiso-%s-%s.pkl' % (key1[0], key2[0]), 'wb') as f:
                pickle.dump({'phi': phi, 'p1': p1, 'p2': p2}, f)
    print('\ndone in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
