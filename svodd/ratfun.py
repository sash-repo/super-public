#!/usr/bin/env python3
"""
ratfun.py -- the ground field K(mu) of rational functions in the parameters of
a family, and the arithmetic that records what it inverts.

WHY.  The enumerator meets nodes at which the classes of a step form a FAMILY

        r(mu) = r + mu_1 d_1 + ... + mu_t d_t ,

and the fibre over each member has to be classified by the stabiliser

        S(mu) = Ker ( M |--> Abar_M r(mu) ) ,

which in general MOVES with mu (affine.uniform_family names the two cases in
which it provably does not).  Classifying the fibre over one member and using
the answer for all of them is then a hypothesis, not a step.

The certificate is a change of the ground ring and not of the logic: carry out
the whole computation at the GENERIC point of the family, i.e. over the field

        K(mu) = Q(mu_1, ..., mu_t)

with mu_1, ..., mu_t indeterminates.  Every branch the machinery takes there --
a rank, a pivot, an invariant subspace, a transitivity certificate -- is a
statement about the generic member, and it specialises to the member at mu = a
as soon as none of the finitely many polynomials the elimination INVERTED
vanishes at a.  That is the whole content of the certificate, and it is why
this module records the inversions: `Rat.__truediv__` appends the numerator of
every divisor to the trace, so at the end of a run one holds the exact
exceptional locus

        V(p_1) u ... u V(p_N),      p_i the inverted polynomials,

outside of which the generic answer is proved for every member.  For a
one-parameter family that locus is the root set of a single univariate
polynomial -- finitely many members, handled one by one (see param.py).

Nothing here is heuristic: an element is zero iff its numerator is the zero
polynomial, which is decided exactly, so no branch is ever taken by sampling.

IMPLEMENTATION.  Two backends, chosen automatically:

  * FLINT (python-flint's fmpq_mpoly) -- exact multivariate polynomials with
    gcd and factorisation.  Elements are kept cancelled, hence canonical.
  * a pure-Python fallback -- dictionaries exponent-tuple |-> Fraction, with
    Euclid's gcd in one variable and content cancellation in more.  Correctness
    does not depend on cancelling (the zero test is the emptiness of the
    dictionary); only the size of the expressions does.

Both hash by the value at a fixed rational point, which is equal on equal
elements whatever the state of cancellation, so `Rat` is safe in a set or a
dict key even under the fallback.
"""

from fractions import Fraction as F

try:
    from flint import fmpq_mpoly, fmpq_mpoly_ctx, fmpq
    HAVE_FLINT = True
except Exception:                                            # pragma: no cover
    HAVE_FLINT = False


# --------------------------------------------------------------- the trace

INVERTED = []          # every polynomial that has been inverted, as a Poly
_TRACING = True


def trace_reset():
    global INVERTED
    INVERTED = []


def trace_polys():
    """the inverted polynomials, deduplicated, as Poly objects"""
    out, seen = [], set()
    for p in INVERTED:
        k = p.key()
        if k in seen or p.is_constant():
            continue
        seen.add(k)
        out.append(p)
    return out


# ------------------------------------------------------- polynomials, FLINT

class PolyFlint:
    """a polynomial in the parameters, over Q"""
    __slots__ = ("p",)
    ctx = None

    def __init__(self, p):
        self.p = p

    # -- construction
    @classmethod
    def setup(cls, t):
        cls.ctx = fmpq_mpoly_ctx.get([f"m{i}" for i in range(t)], "lex")
        cls.nvars = t

    @classmethod
    def const(cls, c):
        c = F(c)
        return cls(cls.ctx.from_dict({(0,) * cls.nvars:
                                      fmpq(c.numerator, c.denominator)}))

    @classmethod
    def gen(cls, i):
        return cls(cls.ctx.gens()[i])

    # -- ring operations
    def __add__(self, o):
        return PolyFlint(self.p + o.p)

    def __sub__(self, o):
        return PolyFlint(self.p - o.p)

    def __mul__(self, o):
        return PolyFlint(self.p * o.p)

    def __neg__(self):
        return PolyFlint(-self.p)

    def is_zero(self):
        return self.p.is_zero()

    def is_constant(self):
        return self.p.is_constant()

    def gcd(self, o):
        if self.is_zero():
            return o
        if o.is_zero():
            return self
        return PolyFlint(self.p.gcd(o.p))

    def divexact(self, o):
        q = self.p / o.p
        return PolyFlint(q)

    def scale(self, c):
        c = F(c)
        return PolyFlint(self.p * fmpq(c.numerator, c.denominator))

    def lead(self):
        """the leading coefficient, as a Fraction"""
        c = self.p.leading_coefficient()
        return F(int(c.numer()), int(c.denom()))

    def eval(self, vals):
        """the value at the rational point `vals`, as a Fraction"""
        out = F(0)
        for mon, c in self.p.to_dict().items():
            term = F(int(c.numer()), int(c.denom()))
            for i, e in enumerate(mon):
                term *= vals[i] ** int(e)
            out += term
        return out

    def factor(self):
        """[(irreducible Poly, multiplicity)], the constant dropped"""
        c, fs = self.p.factor()
        return [(PolyFlint(g), int(e)) for g, e in fs]

    def degree_in(self, i):
        return int(self.p.degrees()[i])

    def key(self):
        return str(self.p)

    def __str__(self):
        return str(self.p)


# ------------------------------------------------ polynomials, pure Python

class PolyPy:
    """dictionary exponent-tuple |-> Fraction; the fallback backend"""
    __slots__ = ("d",)
    nvars = 1

    def __init__(self, d):
        self.d = {k: v for k, v in d.items() if v}

    @classmethod
    def setup(cls, t):
        cls.nvars = t

    @classmethod
    def const(cls, c):
        c = F(c)
        return cls({(0,) * cls.nvars: c} if c else {})

    @classmethod
    def gen(cls, i):
        e = [0] * cls.nvars
        e[i] = 1
        return cls({tuple(e): F(1)})

    def __add__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d.get(k, F(0)) + v
        return PolyPy(d)

    def __sub__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d.get(k, F(0)) - v
        return PolyPy(d)

    def __mul__(self, o):
        d = {}
        for k1, v1 in self.d.items():
            for k2, v2 in o.d.items():
                k = tuple(a + b for a, b in zip(k1, k2))
                d[k] = d.get(k, F(0)) + v1 * v2
        return PolyPy(d)

    def __neg__(self):
        return PolyPy({k: -v for k, v in self.d.items()})

    def is_zero(self):
        return not self.d

    def is_constant(self):
        return all(not any(k) for k in self.d)

    # -- gcd: exact in one variable, content only in more
    def _uni(self):
        c = [F(0)] * (max((k[0] for k in self.d), default=0) + 1)
        for k, v in self.d.items():
            c[k[0]] = v
        return c

    @staticmethod
    def _from_uni(c):
        return PolyPy({(i,): v for i, v in enumerate(c) if v})

    def gcd(self, o):
        if self.is_zero():
            return o
        if o.is_zero():
            return self
        if PolyPy.nvars == 1:
            a, b = self._uni(), o._uni()
            while any(b):
                while b and not b[-1]:
                    b.pop()
                if not b:
                    break
                if len(a) < len(b):
                    a, b = b, a
                sh, f = len(a) - len(b), a[-1] / b[-1]
                a = [a[i] - (f * b[i - sh] if 0 <= i - sh < len(b) else F(0))
                     for i in range(len(a))]
                while a and not a[-1]:
                    a.pop()
                if not a:
                    a, b = b, []
                else:
                    a, b = b, a
            g = PolyPy._from_uni(a)
            lc = a[-1]
            return PolyPy({k: v / lc for k, v in g.d.items()})
        # several variables: the numerical content and the common monomial
        num = 0
        den = 1
        for v in list(self.d.values()) + list(o.d.values()):
            num = _igcd(num, abs(v.numerator))
            den = den * v.denominator // _igcd(den, v.denominator)
        mon = [min(min(k[i] for k in self.d), min(k[i] for k in o.d))
               for i in range(PolyPy.nvars)]
        return PolyPy({tuple(mon): F(num, den)})

    def divexact(self, o):
        """exact division; o must divide self"""
        if o.is_constant():
            c = next(iter(o.d.values()))
            return PolyPy({k: v / c for k, v in self.d.items()})
        # monomial divisor
        if len(o.d) == 1:
            mk, mv = next(iter(o.d.items()))
            return PolyPy({tuple(a - b for a, b in zip(k, mk)): v / mv
                           for k, v in self.d.items()})
        if PolyPy.nvars == 1:
            a, b = self._uni(), o._uni()
            q = [F(0)] * (len(a) - len(b) + 1)
            while len(a) >= len(b) and any(a):
                while a and not a[-1]:
                    a.pop()
                if len(a) < len(b):
                    break
                sh, f = len(a) - len(b), a[-1] / b[-1]
                q[sh] = f
                a = [a[i] - (f * b[i - sh] if 0 <= i - sh < len(b) else F(0))
                     for i in range(len(a))]
            return PolyPy._from_uni(q)
        raise ValueError("no exact division in several variables")

    def scale(self, c):
        c = F(c)
        return PolyPy({k: v * c for k, v in self.d.items()})

    def lead(self):
        k = max(self.d)
        return self.d[k]

    def eval(self, vals):
        out = F(0)
        for k, v in self.d.items():
            t = v
            for i, e in enumerate(k):
                t *= vals[i] ** e
            out += t
        return out

    def factor(self):
        """only the univariate case is factored; else the polynomial itself"""
        if PolyPy.nvars == 1:
            try:
                import sympy
                x = sympy.Symbol("m0")
                expr = sum(sympy.Rational(v.numerator, v.denominator) * x ** k[0]
                           for k, v in self.d.items())
                fs = sympy.factor_list(sympy.Poly(expr, x))
                out = []
                for g, e in fs[1]:
                    cs = g.all_coeffs()[::-1]
                    out.append((PolyPy({(i,): F(str(c))
                                        for i, c in enumerate(cs) if c}), int(e)))
                return out
            except Exception:
                pass
        return [(self, 1)]

    def degree_in(self, i):
        return max((k[i] for k in self.d), default=0)

    def key(self):
        return tuple(sorted((k, str(v)) for k, v in self.d.items()))

    def __str__(self):
        if not self.d:
            return "0"
        out = []
        for k in sorted(self.d, reverse=True):
            v = self.d[k]
            mon = "*".join(f"m{i}^{e}" if e > 1 else f"m{i}"
                           for i, e in enumerate(k) if e)
            out.append(f"{v}*{mon}" if mon and v != 1 else (mon or str(v)))
        return " + ".join(out)


def _igcd(a, b):
    while b:
        a, b = b, a % b
    return a


Poly = PolyFlint if HAVE_FLINT else PolyPy


# ------------------------------------------------------ the field K(mu)

_HASHPT = None          # the rational point at which elements are hashed


def setup(t):
    """make K(mu_1..mu_t) the ground field; returns the list of generators"""
    global _HASHPT
    Poly.setup(t)
    _HASHPT = [F(p, q) for p, q in
               [(7919, 3), (104729, 5), (1299709, 7), (15485863, 11),
                (32452843, 13), (49979687, 17)][:t]]
    trace_reset()
    return [Rat(Poly.gen(i), Poly.const(1)) for i in range(t)]


def backend():
    return "flint" if HAVE_FLINT else "python"


class Rat:
    """an element of K(mu) = Q(mu_1, ..., mu_t)"""
    __slots__ = ("n", "d")

    def __init__(self, n, d=None):
        self.n = n
        self.d = d if d is not None else Poly.const(1)

    # ---- coercion
    @staticmethod
    def of(x):
        if isinstance(x, Rat):
            return x
        return Rat(Poly.const(F(x)), Poly.const(1))

    @staticmethod
    def _cancel(n, d):
        if n.is_zero():
            return Rat(Poly.const(0), Poly.const(1))
        g = n.gcd(d)
        if not g.is_constant():
            n, d = n.divexact(g), d.divexact(g)
        c = d.lead()
        if c != 1:
            n, d = n.scale(F(1) / c), d.scale(F(1) / c)
        return Rat(n, d)

    # ---- ring operations
    def __add__(self, o):
        o = Rat.of(o)
        return Rat._cancel(self.n * o.d + o.n * self.d, self.d * o.d)

    __radd__ = __add__

    def __sub__(self, o):
        o = Rat.of(o)
        return Rat._cancel(self.n * o.d - o.n * self.d, self.d * o.d)

    def __rsub__(self, o):
        return Rat.of(o) - self

    def __mul__(self, o):
        o = Rat.of(o)
        return Rat._cancel(self.n * o.n, self.d * o.d)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Rat.of(o)
        if o.n.is_zero():
            raise ZeroDivisionError("division by zero in K(mu)")
        if _TRACING:
            INVERTED.append(o.n)
        return Rat._cancel(self.n * o.d, self.d * o.n)

    def __rtruediv__(self, o):
        return Rat.of(o) / self

    def __neg__(self):
        return Rat(-self.n, self.d)

    def __pow__(self, e):
        out = Rat.of(1)
        for _ in range(int(e)):
            out = out * self
        return out

    # ---- comparison, truth, hashing
    def __bool__(self):
        return not self.n.is_zero()

    def __eq__(self, o):
        if not isinstance(o, Rat):
            if isinstance(o, (int, F)):
                o = Rat.of(o)
            else:
                return NotImplemented
        return (self.n * o.d - o.n * self.d).is_zero()

    def __ne__(self, o):
        r = self.__eq__(o)
        return r if r is NotImplemented else not r

    def __hash__(self):
        return hash(self._val())

    def _val(self):
        """the value at the fixed hashing point; equal on equal elements"""
        try:
            dv = self.d.eval(_HASHPT)
            if dv == 0:
                return F(0)
            return self.n.eval(_HASHPT) / dv
        except Exception:
            return F(0)

    # a total order, so that the enumerator may sort its representatives
    def __lt__(self, o):
        return self._val() < (o._val() if isinstance(o, Rat) else F(o))

    def __gt__(self, o):
        return self._val() > (o._val() if isinstance(o, Rat) else F(o))

    def __le__(self, o):
        return not self.__gt__(o)

    def __ge__(self, o):
        return not self.__lt__(o)

    # ---- specialisation
    def at(self, vals):
        """the value at mu = vals, as a Fraction; None if the denominator dies"""
        dv = self.d.eval(vals)
        if dv == 0:
            return None
        return self.n.eval(vals) / dv

    def is_constant(self):
        return self.n.is_constant() and self.d.is_constant()

    def __str__(self):
        if self.d.is_constant():
            c = self.d.lead()
            s = str(self.n if c == 1 else self.n.scale(F(1) / c))
        else:
            return f"({self.n})/({self.d})"
        return s if (" " not in s) else f"({s})"

    __repr__ = __str__


def const(c):
    return Rat.of(c)


def to_field(v):
    """a vector (or matrix) of Fractions, over K(mu)"""
    if isinstance(v, (list, tuple)):
        return [to_field(x) for x in v]
    return Rat.of(v)


def specialize(v, vals):
    """a vector (or matrix) over K(mu), at mu = vals; None if any denominator dies"""
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            y = specialize(x, vals)
            if y is None:
                return None
            out.append(y)
        return out
    if isinstance(v, Rat):
        return v.at(vals)
    return F(v)
