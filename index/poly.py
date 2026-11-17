"""Exact multivariate polynomials over Q -- a fast stand-in for sympy.

Variables are indexed 0..NV-1; the last one, index T_IDX = NV-1, is reserved for
the transgression parameter T (it is never differentiated by `diff`, and it is
the variable `integrate01` integrates out).

A polynomial is a dict {exponent tuple: Fraction}, with zero coefficients pruned.
"""

from fractions import Fraction

NV = 4              # x0, x1, x2, T
T_IDX = 3
NX = 3              # the even coordinates of the base are x0, x1, x2

ZERO_EXP = (0,) * NV


class P:
    __slots__ = ("c",)

    def __init__(self, c=None):
        if c is None:
            self.c = {}
        else:
            self.c = {k: v for k, v in c.items() if v}

    # ---- constructors
    @staticmethod
    def const(a):
        a = Fraction(a)
        return P({ZERO_EXP: a}) if a else P()

    @staticmethod
    def var(i):
        e = [0] * NV
        e[i] = 1
        return P({tuple(e): Fraction(1)})

    # ---- arithmetic
    def __bool__(self):
        return bool(self.c)

    def __eq__(self, o):
        return self.c == o.c

    def __hash__(self):
        return hash(frozenset(self.c.items()))

    def __add__(self, o):
        r = dict(self.c)
        for k, v in o.c.items():
            n = r.get(k, 0) + v
            if n:
                r[k] = n
            else:
                r.pop(k, None)
        return P(r)

    def __sub__(self, o):
        r = dict(self.c)
        for k, v in o.c.items():
            n = r.get(k, 0) - v
            if n:
                r[k] = n
            else:
                r.pop(k, None)
        return P(r)

    def __neg__(self):
        return P({k: -v for k, v in self.c.items()})

    def __mul__(self, o):
        if not self.c or not o.c:
            return P()
        r = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                k = tuple(a + b for a, b in zip(k1, k2))
                n = r.get(k, 0) + v1 * v2
                if n:
                    r[k] = n
                else:
                    r.pop(k, None)
        return P(r)

    def scal(self, a):
        a = Fraction(a)
        if not a:
            return P()
        return P({k: v * a for k, v in self.c.items()})

    # ---- calculus
    def diff(self, i):
        r = {}
        for k, v in self.c.items():
            if k[i] == 0:
                continue
            e = list(k)
            e[i] -= 1
            r[tuple(e)] = v * k[i]
        return P(r)

    def integrate01(self):
        """Integrate over the reserved variable T from 0 to 1."""
        r = {}
        for k, v in self.c.items():
            e = list(k)
            n = e[T_IDX]
            e[T_IDX] = 0
            key = tuple(e)
            val = r.get(key, 0) + Fraction(v, n + 1)
            if val:
                r[key] = val
            else:
                r.pop(key, None)
        return P(r)

    def subsT(self, value):
        """Substitute a rational value for T."""
        value = Fraction(value)
        r = {}
        for k, v in self.c.items():
            e = list(k)
            n = e[T_IDX]
            e[T_IDX] = 0
            key = tuple(e)
            val = r.get(key, 0) + v * value ** n
            if val:
                r[key] = val
            else:
                r.pop(key, None)
        return P(r)

    def __repr__(self):
        if not self.c:
            return "0"
        names = ["x0", "x1", "x2", "T"]
        out = []
        for k in sorted(self.c, key=lambda t: (sum(t), t)):
            v = self.c[k]
            mon = "*".join(f"{names[i]}" + (f"^{e}" if e > 1 else "")
                           for i, e in enumerate(k) if e)
            out.append((f"{v}" if not mon else
                        (mon if v == 1 else ("-" + mon if v == -1 else f"{v}*{mon}"))))
        return " + ".join(out).replace("+ -", "- ")


ONE = P.const(1)
NIL = P()
