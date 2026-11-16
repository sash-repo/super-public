#!/usr/bin/env python3
"""
ratf.py --- exact arithmetic in the rational function field F_2(alpha).

This is the field the p = 2 tables of the paper need: the Cartan matrices of
wk(n; alpha) and bgl(n; alpha) have the entry alpha, which is a genuine
parameter (alpha != 0, 1), so no prime field will do.

Polynomials over F_2 are ints used as bit vectors: the bit 2^k is the
coefficient of alpha^k.  Addition is XOR, multiplication is carry-less.
A rational function is a reduced pair (num, den); over F_2 every non-zero
polynomial is monic, so the reduced pair is canonical and equality is
componentwise.

The class F2alpha is plugged into hwmod.HWModule through its `field=`
argument; it offers exactly the interface hwmod uses: norm, is_zero, inv,
zero, one.
"""


# ------------------------------------------------------- polynomials over F_2

def deg(a):
    return a.bit_length() - 1          # deg(0) = -1


def pmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def pdivmod(a, b):
    if b == 0:
        raise ZeroDivisionError
    q = 0
    db = deg(b)
    while a and deg(a) >= db:
        s = deg(a) - db
        q ^= 1 << s
        a ^= b << s
    return q, a


def pgcd(a, b):
    while b:
        a, b = b, pdivmod(a, b)[1]
    return a


# --------------------------------------------------------- rational functions

class RF:
    """An element of F_2(alpha)."""

    __slots__ = ("n", "d")

    def __init__(self, n=0, d=1, reduce=True):
        if d == 0:
            raise ZeroDivisionError
        if reduce and n:
            g = pgcd(n, d)
            if g != 1:
                n = pdivmod(n, g)[0]
                d = pdivmod(d, g)[0]
        if n == 0:
            d = 1
        self.n = n
        self.d = d

    # ---- arithmetic (characteristic 2: -x = x, a - b = a + b)

    def __add__(self, other):
        o = _rf(other)
        return RF(pmul(self.n, o.d) ^ pmul(o.n, self.d), pmul(self.d, o.d))

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__

    def __neg__(self):
        return self

    def __pos__(self):
        return self

    def __mul__(self, other):
        o = _rf(other)
        return RF(pmul(self.n, o.n), pmul(self.d, o.d))

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = _rf(other)
        return RF(pmul(self.n, o.d), pmul(self.d, o.n))

    def inv(self):
        return RF(self.d, self.n)

    def __pow__(self, k):
        r = RF(1)
        for _ in range(k):
            r = r * self
        return r

    # ---- comparison / display

    def __eq__(self, other):
        try:
            o = _rf(other)
        except TypeError:
            return NotImplemented
        return self.n == o.n and self.d == o.d

    def __hash__(self):
        return hash((self.n, self.d))

    def __bool__(self):
        return self.n != 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.d == 1:
            return _pstr(self.n)
        return "(%s)/(%s)" % (_pstr(self.n), _pstr(self.d))


def _rf(x):
    if isinstance(x, RF):
        return x
    if isinstance(x, int):
        return RF(x & 1)                # an integer scalar, taken mod 2
    raise TypeError("cannot coerce %r into F_2(alpha)" % (x,))


def _pstr(a):
    if a == 0:
        return "0"
    out = []
    k = 0
    while a:
        if a & 1:
            out.append("1" if k == 0 else ("al" if k == 1 else "al^%d" % k))
        a >>= 1
        k += 1
    return "+".join(reversed(out))


ZERO = RF(0)
ONE = RF(1)
AL = RF(0b10)                            # alpha itself


# --------------------------------------------------------------- the field

class F2alpha:
    """The field F_2(alpha), in the interface hwmod expects of a field."""

    p = 2

    def norm(self, x):
        return _rf(x)

    def is_zero(self, x):
        return not _rf(x)

    def inv(self, x):
        return _rf(x).inv()

    zero = property(lambda self: ZERO)
    one = property(lambda self: ONE)


def selftest():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("%-34s %-22s %s" % (name, str(got), "OK" if good else
                                  "MISMATCH, expected %s" % (want,)))

    a = AL
    check("alpha + alpha", a + a, ZERO)
    check("alpha * alpha^{-1}", a * a.inv(), ONE)
    check("(alpha+1)^2 = alpha^2+1", (a + ONE) * (a + ONE), a * a + ONE)
    check("1/(alpha+1) + 1/alpha", ONE / (a + ONE) + ONE / a,
          ONE / (a * (a + ONE)))
    check("alpha^3/alpha = alpha^2", (a ** 3) / a, a * a)
    check("2*alpha = 0 (char 2)", 2 * a, ZERO)
    check("3*alpha = alpha", 3 * a, a)
    check("-alpha = alpha", -a, a)
    check("1+alpha^2 != 0", bool(ONE + a * a), True)
    K = F2alpha()
    check("K.inv(alpha)*alpha", K.norm(K.inv(a) * a), ONE)
    print("ratf selftest:", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
