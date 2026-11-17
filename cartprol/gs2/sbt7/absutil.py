#!/usr/bin/env python3
"""Helpers: an Abstract algebra from a list of basis vectors of any kind with a
bracket, coordinates by tracked echelon over F_2."""
from blattner import Abstract


class Coords:
    """coordinates w.r.t. a basis of F_2-vectors given as python sets or ints"""

    def __init__(self, basis, as_int=False):
        self.piv = {}
        self.as_int = as_int
        for idx, v in enumerate(basis):
            c = 1 << idx
            v = self._norm(v)
            while v:
                p = self._top(v)
                b = self.piv.get(p)
                if b is None:
                    self.piv[p] = (v, c)
                    break
                v = self._xor(v, b[0])
                c ^= b[1]
            else:
                raise ValueError('dependent basis at %d' % idx)

    def _norm(self, v):
        return v if self.as_int else frozenset(v)

    def _top(self, v):
        return v.bit_length() - 1 if self.as_int else max(v)

    def _xor(self, a, b):
        return a ^ b if self.as_int else a.symmetric_difference(b)

    def __call__(self, w):
        w = self._norm(w)
        c = 0
        while w:
            p = self._top(w)
            b = self.piv.get(p)
            if b is None:
                raise ValueError('vector not in span')
            w = self._xor(w, b[0])
            c ^= b[1]
        return c


def abstract(basis, degs, pars, bracket, as_int=False, names=None):
    co = Coords(basis, as_int)
    return Abstract(degs, pars, lambda i, j: co(bracket(basis[i], basis[j])), names)
