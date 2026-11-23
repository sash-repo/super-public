"""
Calculus of functions and vector fields on the superstring (exact, over Q).

Realisation: functions on the superstring
    R(n) = C[t^{-1}][[t]] (x) Lambda(zeta, theta_1, ..., theta_{n-1})     (the "M" case)
    R(n) = C[t^{-1}][[t]] (x) Lambda(theta_1, ..., theta_n)               (the "L" case)

A function is a dict {(a, S): coeff}, where a in Z is the power of t, S is a
strictly increasing tuple of odd-generator indices (canonical order), coeff a
sympy expression.  Odd generators are labelled by ints; the caller fixes the
labelling (we use 0 for zeta and 1..n-1 for theta_i in the M case, 1..n in the L case).

Vector fields are dicts {coord: function}, coord in {'t'} u {odd labels},
meaning X = sum_a X_a d_a with the coefficient written on the LEFT of d_a.
"""

from fractions import Fraction

def sympify_S(c):
    return c if isinstance(c, Fraction) else Fraction(c)

def expand(v):
    return v

# ---------------------------------------------------------------- functions

def const(c, ):
    return {(0, ()): sympify_S(c)} if c != 0 else {}

def mono(a, S=(), c=1):
    """c * t^a * (product of odd gens in S, in the given order)."""
    S = tuple(S)
    srt, sgn = _sort(S)
    if sgn == 0:
        return {}
    return {(a, srt): sympify_S(c) * sgn}

def _sort(S):
    """Sort a tuple of odd labels, returning (sorted, sign); 0 if repeated."""
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sgn

def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, 0) + v
    return {k: expand(v) for k, v in out.items() if expand(v) != 0}

def smul(c, f):
    c = sympify_S(c)
    return {k: expand(c * v) for k, v in f.items() if expand(c * v) != 0}

def mul(f, g):
    out = {}
    for (a1, S1), c1 in f.items():
        for (a2, S2), c2 in g.items():
            S, sgn = _sort(S1 + S2)
            if sgn == 0:
                continue
            key = (a1 + a2, S)
            out[key] = out.get(key, 0) + sgn * c1 * c2
    return {k: expand(v) for k, v in out.items() if expand(v) != 0}

def parity(f):
    ps = {len(S) % 2 for (a, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity: %s" % f)
    return ps.pop() if ps else 0

def d_t(f):
    out = {}
    for (a, S), c in f.items():
        if a != 0:
            out[(a - 1, S)] = out.get((a - 1, S), 0) + a * c
    return {k: expand(v) for k, v in out.items() if expand(v) != 0}

def d_odd(j, f):
    """Left derivative with respect to the odd generator labelled j."""
    out = {}
    for (a, S), c in f.items():
        if j in S:
            p = S.index(j)
            sgn = (-1) ** p
            T = S[:p] + S[p + 1:]
            out[(a, T)] = out.get((a, T), 0) + sgn * c
    return {k: expand(v) for k, v in out.items() if expand(v) != 0}

def eq(f, g):
    return add(f, smul(-1, g)) == {}

def show(f):
    if not f:
        return "0"
    names = {0: "z"}
    parts = []
    for (a, S), c in sorted(f.items()):
        s = "".join(names.get(j, "th%d" % j) for j in S)
        parts.append("(%s)*t^%s%s" % (c, a, ("*" + s) if s else ""))
    return " + ".join(parts)

# ------------------------------------------------------------ vector fields

class VF:
    """X = sum_a X_a d_a ; coeff on the left of d_a."""

    def __init__(self, comps, odd_labels):
        self.c = {k: v for k, v in comps.items() if v}
        self.odd = tuple(odd_labels)   # labels of odd coordinates

    def p_coord(self, a):
        return 0 if a == 't' else 1

    def parity(self):
        ps = set()
        for a, f in self.c.items():
            ps.add((parity(f) + self.p_coord(a)) % 2)
        if len(ps) > 1:
            raise ValueError("inhomogeneous vector field")
        return ps.pop() if ps else 0

    def d(self, a, f):
        return d_t(f) if a == 't' else d_odd(a, f)

    def apply(self, f):
        out = {}
        for a, Xa in self.c.items():
            out = add(out, mul(Xa, self.d(a, f)))
        return out

    def bracket(self, Y):
        pX, pY = self.parity(), Y.parity()
        sgn = (-1) ** (pX * pY)
        comps = {}
        for c in set(list(self.c) + list(Y.c) + ['t'] + list(self.odd)):
            term = {}
            for a, Xa in self.c.items():
                if c in Y.c:
                    term = add(term, mul(Xa, self.d(a, Y.c[c])))
            for a, Ya in Y.c.items():
                if c in self.c:
                    term = add(term, smul(-sgn, mul(Ya, self.d(a, self.c[c]))))
            if term:
                comps[c] = term
        return VF(comps, self.odd)

    def __eq__(self, Y):
        keys = set(self.c) | set(Y.c)
        return all(eq(self.c.get(k, {}), Y.c.get(k, {})) for k in keys)

    def __hash__(self):
        items = []
        for k in sorted(self.c, key=str):
            items.append((k, tuple(sorted(self.c[k].items(), key=str))))
        return hash(tuple(items))

    def scal(self, c):
        return VF({k: smul(c, v) for k, v in self.c.items()}, self.odd)

    def plus(self, Y):
        keys = set(self.c) | set(Y.c)
        return VF({k: add(self.c.get(k, {}), Y.c.get(k, {})) for k in keys}, self.odd)

    def is_zero(self):
        return all(not v for v in self.c.values())

    def __repr__(self):
        names = {'t': 'd_t', 0: 'd_z'}
        return " + ".join("(%s)%s" % (show(v), names.get(k, "d_th%s" % k))
                          for k, v in self.c.items() if v) or "0"
