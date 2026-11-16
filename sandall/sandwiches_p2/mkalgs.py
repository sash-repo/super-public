#!/usr/bin/env python3
r"""
mkalgs.py --- build the cache `algs.pkl` used by torus2.py, iso2.py, d4.py,
eick.py, annfull.py and f4.py.

It holds the structure constants over F_2 of the two 26-dimensional algebras

    BR1, SQ1 :  ir(9; \One)        (from its realization by vector fields, iriso.py)
    BR2, SQ2 :  Out ir(9; \One)    (Der ir / ad ir, from the derivation solver, irout.py)
    degs     :  the contact degree of each basis vector of ir(9; \One)

BR[i][j] is the bit vector of [e_i, e_j], SQ[i] that of e_i^{[2]}.

Usage:  python3 mkalgs.py        (takes about a minute)
"""
import pickle
import time

import iriso as I
import irout as O


def main():
    t0 = time.time()
    basis, comps = I.build_ir()
    C = I.Coords(basis)
    _, _, BR2, SQ2, n, _ = O.out_algebra()
    degs = [dg for dg in sorted(comps) for _ in comps[dg]]
    d = {'BR1': C.BR, 'SQ1': C.SQ, 'BR2': BR2, 'SQ2': SQ2, 'degs': degs}
    with open('algs.pkl', 'wb') as f:
        pickle.dump(d, f)
    print('algs.pkl written: dim %d and %d, in %.0f s' % (len(C.BR), n, time.time() - t0))


if __name__ == '__main__':
    main()
