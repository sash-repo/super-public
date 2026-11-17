"""Pseudodifferential symbols of complex order, exact (sympy).

A symbol is Sym(a, b, {k: c_k}) = sum_k c_k D^{a + b*lam - k}, c_k a differential
polynomial in jet variables  u{k}_{i} = d^i u_k/dx^i  (or q{k}_{i}), coefficients in Q(lam).
Composition is the general Leibniz rule
    (c D^al) o (e D^be) = sum_j binom(al, j) c e^{(j)} D^{al+be-j},
which is associative for complex orders; everything is truncated at depth N.
"""
import sympy as sp
from functools import lru_cache

lam = sp.Symbol('lam')
_jets = {}


def jet(name, k, i):
    key = (name, k, i)
    if key not in _jets:
        _jets[key] = sp.Symbol(f'{name}{k}_{i}')
    return _jets[key]


def _parse(s):
    n = s.name
    for name in ('u', 'q', 'v'):
        if n.startswith(name) and '_' in n:
            k, i = n[len(name):].split('_')
            return name, int(k), int(i)
    return None


def Dx(e):
    """total x-derivative of a differential polynomial"""
    e = sp.sympify(e)
    r = 0
    for s in e.free_symbols:
        p = _parse(s)
        if p:
            name, k, i = p
            r += sp.diff(e, s) * jet(name, k, i + 1)
    return sp.expand(r)


def Dxn(e, n):
    for _ in range(n):
        e = Dx(e)
    return e


def binom(a, b, j):
    """binom(a + b*lam, j) as a polynomial in lam"""
    al = a + b * lam
    r = sp.Integer(1)
    for i in range(j):
        r *= (al - i)
    return sp.expand(r / sp.factorial(j))


class Sym:
    def __init__(self, a, b, coeffs, N):
        self.a, self.b, self.N = a, b, N
        self.c = {k: sp.expand(v) for k, v in coeffs.items() if k <= N and sp.expand(v) != 0}

    def __add__(self, o):
        assert (self.a, self.b) == (o.a, o.b) or not o.c or not self.c, 'orders'
        a, b = (self.a, self.b) if self.c else (o.a, o.b)
        if self.c and o.c:
            a, b = self.a, self.b
        c = dict(self.c)
        for k, v in o.c.items():
            c[k] = c.get(k, 0) + v
        return Sym(a, b, c, min(self.N, o.N))

    def __neg__(self):
        return Sym(self.a, self.b, {k: -v for k, v in self.c.items()}, self.N)

    def __sub__(self, o):
        return self + (-o)

    def scale(self, s):
        return Sym(self.a, self.b, {k: s * v for k, v in self.c.items()}, self.N)

    def __mul__(self, o):
        N = min(self.N, o.N)
        a, b = self.a + o.a, self.b + o.b
        c = {}
        for k1, c1 in self.c.items():
            for k2, c2 in o.c.items():
                d = c2
                for j in range(0, N - k1 - k2 + 1):
                    if j > 0:
                        d = Dx(d)
                        if d == 0:
                            break
                    bn = binom(self.a - k1, self.b, j)
                    if bn == 0:
                        break
                    c[k1 + k2 + j] = c.get(k1 + k2 + j, 0) + bn * c1 * d
        return Sym(a, b, c, N)

    def plus(self):
        """differential part; integer order only"""
        assert self.b == 0
        return Sym(self.a, 0, {k: v for k, v in self.c.items() if self.a - k >= 0}, self.N)

    def coef_of_order(self, a, b):
        """coefficient of D^{a+b lam}"""
        if b != self.b:
            return 0
        k = self.a - a
        return self.c.get(k, 0)

    def subs(self, d):
        return Sym(self.a, self.b, {k: sp.expand(v.subs(d)) for k, v in self.c.items()}, self.N)


def comm(A, B):
    """[A,B]; orders add, so both products have the same order"""
    P, R = A * B, B * A
    c = dict(P.c)
    for k, v in R.c.items():
        c[k] = c.get(k, 0) - v
    return Sym(P.a, P.b, c, P.N)
