#!/usr/bin/env python3
r"""fvle2.py --- fvle.py over the grading LATTICE (rank 4) rather than a box of
Z^7, deduplicated by the decomposition before the Weisfeiler test, in
parallel.  Usage: python3 fvle2.py B   (lattice coordinates in [-B, B]^4)"""
import itertools
import sys
from math import gcd
from multiprocessing import Pool

import fvle
from fvle import graded_pieces, normalize, wtest, growth, is_contact, signature, irw

BASIS = None
TMS = None


def init():
    global BASIS, TMS
    BASIS = fvle.build()
    TMS = [irw.terms(v) for v in BASIS]


def decomp(d):
    p = graded_pieces(BASIS, TMS, d)
    assert p is not None, d
    return d, signature(normalize(p))


def test(d):
    p = normalize(graded_pieces(BASIS, TMS, d))
    ok, msg = wtest(p)
    dims = tuple((c, len(p[c])) for c in sorted(p))
    return d, ok, msg, dims, growth(p), (is_contact(p) if ok else False)


def main():
    B = int(sys.argv[1])
    init()
    good = [d for d in itertools.product(range(-1, 2), repeat=7)
            if graded_pieces(BASIS, TMS, d) is not None]
    from sympy import Matrix
    H = Matrix(good).T.columnspace()
    # an integral basis of the lattice: Hermite form of the good vectors
    M = Matrix(good)
    from sympy.matrices.normalforms import hermite_normal_form
    Hn = hermite_normal_form(M.T).T          # rows span the same lattice
    L = [tuple(int(x) for x in Hn.row(i)) for i in range(Hn.rows) if any(Hn.row(i))]
    fvle.say('lattice basis:', L)
    assert len(L) == 4
    cand = set()
    for lam in itertools.product(range(-B, B + 1), repeat=4):
        d = tuple(sum(l * v[i] for l, v in zip(lam, L)) for i in range(7))
        if not any(d):
            continue
        g = 0
        for x in d:
            g = gcd(g, abs(x))
        cand.add(tuple(x // g for x in d))
    cand = sorted(cand)
    fvle.say('B =', B, ': primitive degree vectors', len(cand))
    with Pool(8, initializer=init) as P:
        sigs = P.map(decomp, cand, chunksize=50)
    rep = {}
    for d, s in sigs:
        rep.setdefault(s, d)
    fvle.say('distinct decompositions:', len(rep))
    with Pool(8, initializer=init) as P:
        res = P.map(test, list(rep.values()), chunksize=4)
    W = [r for r in res if r[1]]
    reasons = {}
    for r in res:
        if not r[1]:
            reasons[r[2]] = reasons.get(r[2], 0) + 1
    fvle.say('non-W:', reasons)
    fvle.say('W-gradings:', len(W))
    byclass = {}
    for d, ok, msg, dims, gv, ct in W:
        byclass.setdefault((sum(n for c, n in dims if c < 0), dims, gv, ct), []).append(d)
    for k in sorted(byclass):
        cod, dims, gv, ct = k
        fvle.say('  codim %d  dims %s  growth %s%s  x%d  e.g. d=%s' % (
            cod, dims, gv, ' C' if ct else '', len(byclass[k]), byclass[k][0]))


if __name__ == '__main__':
    main()
