#!/usr/bin/env python3
r"""
irh2.py --- the psl(2|2) pattern for ir(9; \One): are the outer derivations and
the central extensions the same module?

For psl(2|2) in characteristic 0 one has Out psl(2|2) = sl(2) and
H^2(psl(2|2); K) = sl(2) again, as modules over that sl(2).  For ir(9; \One) over F_2,
`irder.py` finds  dim ir = dim Out ir = dim H^2(ir; K) = 26.  Both Out ir and
H^2(ir; K) are modules over the Lie algebra Out ir = Der ir / ad ir (the inner
derivations act trivially on cohomology, which is verified), so the question
has a~precise form:

    is  H^2(ir; K) = Out ir  as Out ir-modules?  and its dual?

The script computes Hom_{Out}(Out_adjoint, H^2), Hom_{Out}(H^2, Out_adjoint)
and the same for the dual module H^2 (the action transposed), exactly over F_2,
and tests every element of each Hom-space for invertibility.

Usage:  python3 irh2.py
"""

import sys
import time

import ir as M
import iriso as I
import irder as D

DIM = I.DIM
bits = M.bits
cech = I.cech
cdim = I.cdim


def hom_between(A, B, n, m, k):
    """Hom_g(V, W) for a k-dimensional Lie algebra g given by the action
    matrices A (on V, dim n) and B (on W, dim m): phi with
    phi(g . v) = g . phi(v).  Unknowns a_{j,l}, index j * m + l."""
    eqs = []
    for i in range(k):
        for j in range(n):
            cols = [0] * (n * m)
            for r in bits(A[i][j]):                 # phi(x_i . v_j)
                for l in range(m):
                    cols[r * m + l] ^= 1 << l
            for l in range(m):                      # x_i . phi(v_j)
                cols[j * m + l] ^= B[i][l]
            for t in range(m):
                e = 0
                for u in range(n * m):
                    if cols[u] >> t & 1:
                        e |= 1 << u
                if e:
                    eqs.append(e)
    ech = cech(eqs)
    piv = {p for p, _ in ech}
    sols = []
    for u in (u for u in range(n * m) if u not in piv):
        sol = 1 << u
        for p, row in sorted(ech, key=lambda pb: pb[0]):
            if bin(row & sol).count('1') % 2:
                sol ^= 1 << p
        sols.append(sol)
    return sols


def has_invertible(sols, n, m):
    if n != m or not sols or len(sols) > 22:
        return None
    for g in range(1, 1 << len(sols)):
        s = 0
        for k in bits(g):
            s ^= sols[k]
        cols = [(s >> (j * m)) & ((1 << m) - 1) for j in range(n)]
        if cdim(cols) == n:
            return True
    return False


def transpose_action(mats, n):
    """the dual module: (x . f)(v) = f(x . v), no signs in characteristic 2."""
    out = []
    for A in mats:
        T = []
        for l in range(n):                       # image of the dual basis f_l
            col = 0
            for j in range(n):
                if A[j] >> l & 1:
                    col ^= 1 << j
            T.append(col)
        out.append(T)
    return out


def main():
    t0 = time.time()
    print('=== ir(9; \\One): Out ir versus H^2(ir; K) as Out ir-modules ===')
    basis, comps = I.build_ir()
    C = I.Coords(basis)
    der = D.der_basis(C)
    adim = [D.flat(C.ad(1 << i)) for i in range(DIM)]
    Q = D.Quotient(der, adim)
    n = Q.dim
    ops = [D.unflat(v) for v in Q.out]
    BRout = [[Q.cls(D.flat(D.comm(ops[a], ops[b]))) for b in range(n)]
             for a in range(n)]
    print('  dim ir = %d, dim Out ir = %d' % (DIM, n))

    dimZ, dimB, dimH, idx, NP, echZ, cob = D.h2_trivial(C)
    m, inner_triv, hmats, hbasis = D.h2_out_module(C, Q, None, idx, NP, echZ,
                                                   cob)
    print('  dim H^2(ir; K) = %d; the inner derivations act trivially on it: %s'
          % (m, inner_triv))
    assert m == dimH

    # the adjoint module of Out
    adjoint = [[BRout[i][j] for j in range(n)] for i in range(n)]
    # hmats[i] is the action of the i-th basis vector of Out on H^2
    h2mod = [list(hmats[i]) for i in range(n)]

    # both actions must be representations of Out
    def is_rep(mats, d):
        for i in range(n):
            for j in range(n):
                for v in range(d):
                    lhs = 0
                    for r in bits(BRout[i][j]):
                        lhs ^= mats[r][v]
                    x = 0
                    for r in bits(mats[j][v]):
                        x ^= mats[i][r]
                    for r in bits(mats[i][v]):
                        x ^= mats[j][r]
                    if lhs != x:
                        return False
        return True

    print('  the adjoint action of Out is a representation:  %s'
          % is_rep(adjoint, n))
    print('  the action of Out on H^2 is a representation:    %s'
          % is_rep(h2mod, m))

    irr_a, dim_a = D.burnside_ok(adjoint, n)
    irr_h, dim_h = D.burnside_ok(h2mod, m)
    print('  Out is absolutely irreducible as its own adjoint module: %s '
          '(algebra of dim %d of %d) --- i.e. Out is simple'
          % (irr_a, dim_a, n * n))
    print('  H^2 is an absolutely irreducible Out-module: %s (dim %d of %d)'
          % (irr_h, dim_h, m * m))

    hom1 = hom_between(adjoint, h2mod, n, m, n)
    print('  dim Hom_Out(Out, H^2) = %d' % len(hom1))
    if hom1:
        print('    some element of it is invertible: %s'
              % has_invertible(hom1, n, m))

    dual = transpose_action(h2mod, m)
    print('  the dual action is a representation: %s' % is_rep(dual, m))
    hom2 = hom_between(adjoint, dual, n, m, n)
    print('  dim Hom_Out(Out, (H^2)^*) = %d' % len(hom2))
    if hom2:
        print('    some element of it is invertible: %s'
              % has_invertible(hom2, n, m))

    print()
    print('done in %.1f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
