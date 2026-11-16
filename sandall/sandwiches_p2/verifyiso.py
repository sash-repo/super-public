#!/usr/bin/env python3
r"""
verifyiso.py --- an independent check of the isomorphism

        phi:  ir(9; \One)  -->  Out ir(9; \One) = Der ir / ad ir

found by `iso2.py`.  Nothing is taken from the cache `algs.pkl` here: both
algebras are rebuilt from scratch (ir from its realization by vector fields in
`ir.py`, Out ir from the derivation solver of `irder.py`), and the map stored
in `iso-found.pkl` is checked

  * to be bijective (rank 26 over F_2),
  * to satisfy phi([e_i, e_j]) = [phi e_i, phi e_j] for all 26 x 26 pairs,
  * to satisfy phi(e_i^{[2]}) = (phi e_i)^{[2]} for all 26 basis vectors, and
    hence, by polarization, on every element of the algebra,
  * and, as a redundant control, on 20000 random pairs (x, y) of elements.

Usage:  python3 verifyiso.py
"""

import pickle
import random
import sys
import time

import iriso as I
import irout as O
from lin2 import bits, echelon

random.seed(17082026)


def main():
    t0 = time.time()
    basis, _ = I.build_ir()
    C = I.Coords(basis)
    BR1, SQ1 = C.BR, C.SQ
    _, _, BR2, SQ2, n, _ = O.out_algebra()
    assert n == 26 and len(BR1) == 26
    print('both algebras rebuilt from scratch in %.0f s' % (time.time() - t0))

    with open('iso-found.pkl', 'rb') as f:
        phi = pickle.load(f)
    assert len(phi) == 26

    def ap(v):
        out = 0
        for i in bits(v):
            out ^= phi[i]
        return out

    def br(BR, x, y):
        out = 0
        for i in bits(x):
            row = BR[i]
            for j in bits(y):
                out ^= row[j]
        return out

    def sq(BR, SQ, x):
        out, bs = 0, list(bits(x))
        for k, i in enumerate(bs):
            out ^= SQ[i]
            row = BR[i]
            for j in bs[k + 1:]:
                out ^= row[j]
        return out

    ok = True
    if len(echelon(phi)) != 26:
        print('  NOT bijective'); ok = False
    bad = 0
    for i in range(26):
        if ap(SQ1[i]) != sq(BR2, SQ2, phi[i]):
            bad += 1
        for j in range(26):
            if ap(BR1[i][j]) != br(BR2, phi[i], phi[j]):
                bad += 1
    print('  bijective: %s' % (len(echelon(phi)) == 26))
    print('  failures on the 26 squares and the 676 brackets: %d' % bad)
    ok = ok and bad == 0

    bad = 0
    for _ in range(20000):
        x = random.getrandbits(26)
        y = random.getrandbits(26)
        if ap(br(BR1, x, y)) != br(BR2, ap(x), ap(y)):
            bad += 1
        if ap(sq(BR1, SQ1, x)) != sq(BR2, SQ2, ap(x)):
            bad += 1
    print('  failures on 20000 random pairs (bracket and squaring): %d' % bad)
    ok = ok and bad == 0
    print('  ==> phi is an isomorphism of restricted Lie algebras: %s' % ok)
    print('done in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
