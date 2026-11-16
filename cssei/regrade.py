#!/usr/bin/env python3
"""Regrading a vector-field algebra G (in vect(m;H)) by new coordinate degrees
r_i: split every basis vector into r-homogeneous parts (checking they lie in G),
and return the abstract algebra of the parts of r-degree <= 0."""
from f2vf import Echelon
from absutil import abstract


def rdeg_key(sp, k, r):
    A, i = k >> 5, k & 31
    e = sp.exps(A)
    return sum(a * rr for a, rr in zip(e, r)) - r[i]


def regrade(sp, G, r):
    parts = {}
    for v in G.basis():
        split = {}
        for k in v:
            split.setdefault((rdeg_key(sp, k, r), sp.kpar(k)), set()).add(k)
        for key, p in split.items():
            assert G.contains(p), 'G is not graded by %s' % (r,)
            parts.setdefault(key, Echelon()).add(p)
    dims = {}
    for (d, p), E in parts.items():
        dims.setdefault(d, [0, 0])[p] += len(E)
    return parts, dict(sorted(dims.items()))


def nonpos_abstract(sp, parts):
    basis, degs, pars = [], [], []
    for (d, p), E in sorted(parts.items()):
        if d <= 0:
            for v in E.basis():
                basis.append(v)
                degs.append(d)
                pars.append(p)
    return abstract(basis, degs, pars, sp.bracket)
