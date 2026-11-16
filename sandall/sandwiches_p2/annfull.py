#!/usr/bin/env python3
r"""
annfull.py --- the exact number ann = |Ann(ir(9;\One))| of Eick's table.

`eick.py` shows that the homogeneous annihilator derivations are 441 in number
(63, 315, 63 in the degrees -2, 0, 2) and that every non-zero d in Ann has its
lowest homogeneous component in Ann again.  So the whole variety is swept by
taking each homogeneous annihilator as a seed and adding the components of
higher degree one at a time, and each stage is LINEAR in the new component:
writing d = d_{i0} + ... + u with u of degree i0 + s, the only terms of the
target degree in which u occurs are

    [d_{i0}(x), u(y)] + [u(x), d_{i0}(y)]      and     d_{i0} u + u d_{i0},

u paired with itself landing one degree higher.  Hence

  * the coefficient matrix of the stage depends on the SEED alone, not on the
    partial map built so far -- it is assembled once, and only its right-hand
    side is recomputed;
  * the right-hand side is needed only at the equations of one row basis,
    which is at most 22 of the 18252 of them, i.e. at most 22 brackets per
    partial map instead of 700;
  * anything that slips through (a partial map failing one of the redundant
    equations) is killed by the full verification of the result: every lift
    that is counted is checked to satisfy d^2 = 0 and [d(x), d(y)] = 0 on all
    the 26 x 26 basis pairs.

Usage:  python3 annfull.py
"""

import pickle
import sys
import time

import eick
import lin2
from lin2 import bits
from torus2 import Alg


def build_stage(A, degs, seed, i0, B, sdeg):
    """The coefficient rows of the stage, and how to evaluate a right-hand side.

    Returns (rows, spec) where rows[k] is a mask over the |B| unknowns and
    spec[k] is ('br', a, b, t) or ('sq', a, t): the equation says that bit t
    of the degree-(deg a + deg b + i0 + s) part of [d(e_a), d(e_b)] vanishes.
    """
    n = A.n
    rows, spec = [], []
    for a in range(n):
        for b in range(n):
            g = degs[a] + degs[b] + i0 + sdeg
            coef = [eick.project(A.br(seed[a], u[b]) ^ A.br(u[a], seed[b]),
                                 degs, g) for u in B]
            for t in range(n):
                e = 0
                for c in range(len(B)):
                    if coef[c] >> t & 1:
                        e |= 1 << c
                rows.append(e)
                spec.append(('br', a, b, t, g))
        g2 = degs[a] + i0 + sdeg
        coef = [eick.project(eick.apply_op(seed, u[a]) ^ eick.apply_op(u, seed[a]),
                             degs, g2) for u in B]
        for t in range(n):
            e = 0
            for c in range(len(B)):
                if coef[c] >> t & 1:
                    e |= 1 << c
            rows.append(e)
            spec.append(('sq', a, a, t, g2))
    return rows, spec


def row_basis(rows, spec, m):
    """Indices of a maximal independent set of rows, and the null space."""
    ech, keep = [], []
    for k, r in enumerate(rows):
        v = r
        for b in ech:
            if v and v >> (b.bit_length() - 1) & 1:
                v ^= b
        if v:
            ech.append(v)
            ech.sort(key=lambda u: -u.bit_length())
            keep.append(k)
    null = lin2.solve_null(rows, m)
    return keep, null


def solve_stage(rows, keep, spec, m, A, degs, P, null):
    """All u (as masks over the basis of the stage) with the equations of
    `keep` satisfied; [] if inconsistent."""
    lines = []
    for k in keep:
        kind, a, b, t, g = spec[k]
        if kind == 'br':
            w = A.br(P[a], P[b])
        else:
            w = eick.apply_op(P, P[a])
        rhs = (eick.project(w, degs, g) >> t) & 1
        lines.append((rows[k] << 1) | rhs)
    part = eick.affine_solutions(lines, m)
    return part


def main():
    t0 = time.time()
    with open('algs.pkl', 'rb') as f:
        d = pickle.load(f)
    A = Alg(d['BR1'], d['SQ1'], 'ir(9;One)')
    degs = d['degs']
    DER = eick.der_basis(A)
    pieces = eick.graded_pieces(A, DER, degs)
    found = eick.homogeneous_annihilators(A, pieces, verbose=False)
    print('homogeneous annihilator derivations: %d' % len(found))
    degmax = max(pieces)
    n = A.n

    def genuine(P):
        for a in range(n):
            if eick.apply_op(P, P[a]):
                return False
            for b in range(n):
                if A.br(P[a], P[b]):
                    return False
        return True

    total = 0
    for i0 in sorted(pieces):
        seeds = [D for (i, D) in found if i == i0]
        if not seeds:
            continue
        cnt = 0
        for si, seed in enumerate(seeds):
            cur = [seed]
            for sdeg in range(i0 + 1, degmax + 1):
                B = pieces.get(sdeg, [])
                if not B:
                    continue
                rows, spec = build_stage(A, degs, seed, i0, B, sdeg)
                keep, null = row_basis(rows, spec, len(B))
                nxt = []
                for P in cur:
                    for cm in solve_stage(rows, keep, spec, len(B), A, degs,
                                          P, null):
                        Q = list(P)
                        for c in bits(cm):
                            for j in range(n):
                                Q[j] ^= B[c][j]
                        nxt.append(Q)
                cur = nxt
            good = sum(1 for P in cur if genuine(P))
            cnt += good
            if si % 20 == 0:
                print('   degree %+d, seed %d/%d: %d lifts so far, %.0f s'
                      % (i0, si, len(seeds), cnt, time.time() - t0))
        print('  degree %+d: %d annihilator derivations with this lowest '
              'component' % (i0, cnt))
        total += cnt
    print('\n  ann = %d   (the zero derivation included)' % (total + 1))
    print('done in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
