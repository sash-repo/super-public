#!/usr/bin/env python3
r"""
irfam.py --- the superdimension of EVERY admissible superization of ir(9; \One),
one parity function at a time, computed twice by two different algorithms.

Why: ir.py grouped the 16 admissible parity functions by the NUMBER of odd
indeterminates and printed the superdimension of the first member of each group
(`rows[0]`), so a group whose members have different superdimensions would look
uniform.  iriso.py suggests family 4) of Theorem 2.3.4 is exactly such a group.
This script prints, for each of the 16 parity functions,

  * the odd indeterminates,
  * sdim ir = (dim ev | dim od) and the graded sdim, degree by degree,
  * the growth vector of the distribution preserved,

with the even/odd split computed in two independent ways:

  (i)  ir.py's split_by_parity, which intersects ir with the even and the odd
       part of vect(9; \One) by elimination;
  (ii) the projections: p splits every vector field v of vect(9; \One) into
       v = v_ev + v_od by sorting its terms by parity, and ir is p-homogeneous
       iff v_ev lies in ir for every v in ir; then the even part of ir is
       spanned by the v_ev and the odd part by the v_od.

The two must agree, and dim ev + dim od must be 26.

Usage:  python3 irfam.py
"""

import itertools
import sys

import ir as M

NVAR = M.NVAR
PAIRS = M.PAIRS
DIM = 26


def parts(v, p):
    """(even part, odd part) of a vector field, term by term."""
    ev = od = 0
    for i in M.bits(v):
        mask, j = divmod(i, NVAR)
        q = p[j]
        for b in M.bits(mask):
            q ^= p[b]
        if q:
            od ^= 1 << i
        else:
            ev ^= 1 << i
    return ev, od


def split_by_projection(elems, p):
    """(even basis, odd basis) via (ii), or None if not homogeneous."""
    ech = M.echelon(elems)
    ev, od = [], []
    for _, v in ech:
        a, b = parts(v, p)
        if not M.in_span(a, ech) or not M.in_span(b, ech):
            return None
        ev.append(a)
        od.append(b)
    return [v for _, v in M.echelon(ev)], [v for _, v in M.echelon(od)]


def main():
    part = M.prolong(M.G0T)
    idl, uniq, _ = M.minimal_ideal(M.flatten(part))
    assert uniq and len(idl) == DIM
    comps = {d: M.echelon([v for _, v in part[d] if M.in_span(v, idl)])
             for d in sorted(part)}
    ir = [v for _, v in idl]
    print('ir(9; \\One): dim = %d, by degree %s' % (len(ir), M.gdims(comps)))
    print()

    rows, bad = [], 0
    for p in itertools.product((0, 1), repeat=NVAR):
        sp = M.split_by_parity(ir, p)
        if sp is None:
            continue
        ev, od = sp
        evb = M.echelon(ev)
        if not all(M.in_span(M.compose_square(x), evb) for x in od):
            continue
        # cross-check by the projections
        sp2 = split_by_projection(ir, p)
        agree = (sp2 is not None and len(sp2[0]) == len(ev)
                 and len(sp2[1]) == len(od)
                 and all(M.in_span(v, M.echelon(ev)) for v in sp2[0])
                 and all(M.in_span(v, M.echelon(od)) for v in sp2[1]))
        if not agree or len(ev) + len(od) != DIM:
            bad += 1
        gr = {}
        for d in sorted(comps):
            if comps[d]:
                s = M.split_by_parity([v for _, v in comps[d]], p)
                gr[d] = (len(s[0]), len(s[1]))
        rows.append((p, len(ev), len(od), gr, agree))

    print('%-30s %-9s %-40s %s'
          % ('odd indeterminates', 'sdim', 'graded sdim', 'growth vector'))
    for p, ne, no, gr, agree in sorted(rows, key=lambda r: (r[0][0], -r[1])):
        odd = ', '.join('x_%d' % (i + 1) for i in range(NVAR) if p[i]) or '---'
        g = '  '.join('%+d:(%d|%d)' % (d, a, b) for d, (a, b) in sorted(gr.items()))
        a, b = gr[-1]
        c, e = gr[-2]
        print('%-30s (%2d|%2d)  %-40s (%d|%d, %d|%d)C%s'
              % (odd, ne, no, g, a, b, a + c, b + e,
                 '' if agree else '   <-- ALGORITHMS DISAGREE'))

    print()
    print('the two algorithms agree on all %d, and dim ev + dim od = 26: %s'
          % (len(rows), bad == 0))

    # the census by superdimension
    cen = {}
    for p, ne, no, gr, _ in rows:
        cen.setdefault((p[0], ne, no), []).append(p)
    print()
    print('census by (parity of x_1, sdim):')
    for k in sorted(cen):
        x1, ne, no = k
        print('  x_1 %-4s sdim (%2d|%2d): %d parity function(s)'
              % ('odd' if x1 else 'even', ne, no, len(cen[k])))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
