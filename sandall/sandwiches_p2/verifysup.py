#!/usr/bin/env python3
r"""
verifysup.py --- an independent check of the two isomorphisms of superalgebras
found by `supiso.py`:

        ir(5; \un|4)  ~  ir(4; \un|5)              sdim (14|12)
        ir(1; \un|8)  ~  widetilde{ir}(4; \un|5)   sdim (10|16)

Nothing is taken on trust from the search.  The algebra ir(9; \One) is rebuilt
from its realization by vector fields, the two parity functions of each pair
are re-derived, and the stored map phi is checked to be

  * a bijection of the 26-dimensional space,
  * an automorphism of the Lie algebra: phi[x, y] = [phi x, phi y] on all
    26 x 26 basis pairs and on 20000 random pairs,
  * compatible with the squaring: phi(x^{[2]}) = (phi x)^{[2]} -- so it is an
    automorphism of the RESTRICTED Lie algebra, and in particular carries the
    odd squaring of the one superalgebra to that of the other,
  * and carrying the Z/2-grading to the Z/2-grading: phi(g_ev) = g'_ev and
    phi(g_od) = g'_od.

Together this says exactly that the two superizations are isomorphic as Lie
superalgebras -- although not as GRADED ones: their Weisfeiler gradings have
different growth vectors, (4|4, 5|4) against (4|4, 4|5).

Usage:  python3 verifysup.py
"""

import pickle
import random
import sys
import time

import ir as M
import iriso as I
import supiso as S
from lin2 import bits, echelon, in_span

random.seed(1708202602)


def main():
    t0 = time.time()
    basis, _ = I.build_ir()
    C = I.Coords(basis)
    print('ir(9; \\One) rebuilt: dim %d  (%.0f s)' % (len(basis), time.time() - t0))
    adm = {p: (ev, od) for p, ev, od in I.admissible([v for v in basis])}

    def ap(phi, v):
        out = 0
        for i in bits(v):
            out ^= phi[i]
        return out

    allok = True
    for fname in ('supiso-2-4.pkl', 'supiso-3-4.pkl'):
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        phi, p1, p2 = d['phi'][:26], d['p1'], d['p2']
        ev1, od1 = adm[p1]
        ev2, od2 = adm[p2]
        ev1 = echelon([C.co(v) for v in ev1]); od1 = echelon([C.co(v) for v in od1])
        ev2 = echelon([C.co(v) for v in ev2]); od2 = echelon([C.co(v) for v in od2])
        print('\n%s:  p1 = %s  (sdim (%d|%d)),  p2 = %s  (sdim (%d|%d))'
              % (fname, ''.join(map(str, p1)), len(ev1), len(od1),
                 ''.join(map(str, p2)), len(ev2), len(od2)))
        ok = len(echelon(phi)) == 26
        print('  bijective: %s' % ok)
        bad = 0
        for i in range(26):
            if ap(phi, C.SQ[i]) != C.sq(phi[i]):
                bad += 1
            for j in range(26):
                if ap(phi, C.BR[i][j]) != C.br(phi[i], phi[j]):
                    bad += 1
        print('  failures on the 26 squares and the 676 brackets: %d' % bad)
        ok = ok and bad == 0
        rbad = 0
        for _ in range(20000):
            x, y = random.getrandbits(26), random.getrandbits(26)
            if ap(phi, C.br(x, y)) != C.br(ap(phi, x), ap(phi, y)):
                rbad += 1
            if ap(phi, C.sq(x)) != C.sq(ap(phi, x)):
                rbad += 1
        print('  failures on 20000 random pairs: %d' % rbad)
        ok = ok and rbad == 0
        gev = all(in_span(ap(phi, v), ev2) for v in ev1)
        god = all(in_span(ap(phi, v), od2) for v in od1)
        print('  phi(g_ev) = g\'_ev: %s      phi(g_od) = g\'_od: %s' % (gev, god))
        ok = ok and gev and god
        print('  ==> an isomorphism of Lie superalgebras: %s' % ok)
        allok = allok and ok
    print('\nboth: %s   (%.0f s)' % (allok, time.time() - t0))


if __name__ == '__main__':
    main()
