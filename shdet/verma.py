"""Verma modules and Shapovalov forms over the distinguished stringy (super)algebras.

Exact symbolic arithmetic (sympy over Q[h, c] resp. Q[h, q, t]); no floats.

The algebra is given by a spec: a set of generators indexed by (name, mode),
each with a parity, a level (= -mode; a generator of negative level lowers the
weight), a charge, together with the bracket and the anti-involution sigma.

The Verma module M^lambda is U(n^-) v, with the PBW basis of monomials in the
generators of negative level written in a fixed weakly-decreasing order; an odd
generator occurs at most once in such a monomial.  The Shapovalov form is
   (u v | w v) = the coefficient of v in sigma(u) w v.

Used to check
  1) the Kac determinant of vir and of ns(1) as printed in Subsec. 3.2--3.3
     of the paper -- in particular whether the inner index of the product is
     the running i or the fixed n;
  2) the Boucher--Friedan--Kent determinant of ns(2).
"""

import sys
from fractions import Fraction
from itertools import combinations

import sympy


# ---------------------------------------------------------------------------
#  generators
# ---------------------------------------------------------------------------

class Gen:
    """A generator (name, mode).  level = -mode: negative level lowers weight."""
    __slots__ = ("name", "mode", "parity", "charge")

    def __init__(self, name, mode, parity, charge=0):
        self.name = name
        self.mode = sympy.Rational(mode)
        self.parity = parity
        self.charge = charge

    @property
    def key(self):
        return (self.name, self.mode)

    def __repr__(self):
        return "%s(%s)" % (self.name, self.mode)


# ---------------------------------------------------------------------------
#  an algebra spec
# ---------------------------------------------------------------------------

class Algebra:
    """Subclasses provide gens(), bracket(), sigma(), weight()."""

    #: order in which the generators of a PBW monomial are written; a monomial
    #: is a tuple of keys, weakly decreasing with respect to ``ordkey``.
    def ordkey(self, g):
        return (g.mode, self.namerank(g.name))

    def namerank(self, name):
        return self.NAMES.index(name)

    def gen(self, key):
        return self._gens[key]

    def register(self, gens):
        self._gens = {g.key: g for g in gens}

    # -- to be provided --------------------------------------------------
    def bracket(self, g1, g2):
        """[g1, g2] as (list of (coeff, key), central coeff)."""
        raise NotImplementedError

    def sigma(self, key):
        raise NotImplementedError

    def weight(self, g):
        """the scalar by which a zero-level generator acts on the vacuum."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  straightening
# ---------------------------------------------------------------------------

class Module:
    """The Verma module of the algebra ``alg``; elements are dict{monomial: coeff}."""

    def __init__(self, alg):
        self.alg = alg
        self._cache = {}

    # ---- the left action of one generator on a straightened monomial ----
    def act(self, key, mono):
        ck = (key, mono)
        got = self._cache.get(ck)
        if got is not None:
            return got
        out = self._act(key, mono)
        out = {m: sympy.expand(c) for m, c in out.items() if sympy.expand(c) != 0}
        self._cache[ck] = out
        return out

    def _act(self, key, mono):
        alg = self.alg
        g = alg.gen(key)
        if not mono:
            if g.mode < 0:
                return {(key,): sympy.Integer(1)}
            if g.mode > 0:
                return {}
            return {(): alg.weight(g)}

        head, rest = mono[0], mono[1:]
        h = alg.gen(head)

        # already in order -> just prepend
        if g.mode < 0 and alg.ordkey(g) >= alg.ordkey(h):
            if not (key == head and g.parity == 1):
                return {(key,) + mono: sympy.Integer(1)}
            # odd generator squared: g g = 1/2 {g, g}
            terms, cen = alg.bracket(g, g)
            out = {}
            for coeff, k2 in terms:
                for m2, c2 in self.act(k2, rest).items():
                    out[m2] = out.get(m2, 0) + sympy.Rational(1, 2) * coeff * c2
            if cen != 0:
                for m2, c2 in self.mono_value(rest).items():
                    out[m2] = out.get(m2, 0) + sympy.Rational(1, 2) * cen * c2
            return out

        # g head = (-1)^{p p} head g + [g, head]
        sign = -1 if (g.parity == 1 and h.parity == 1) else 1
        out = {}
        for m1, c1 in self.act(key, rest).items():
            for m2, c2 in self.act(head, m1).items():
                out[m2] = out.get(m2, 0) + sign * c1 * c2
        terms, cen = alg.bracket(g, h)
        for coeff, k2 in terms:
            for m2, c2 in self.act(k2, rest).items():
                out[m2] = out.get(m2, 0) + coeff * c2
        if cen != 0:
            for m2, c2 in self.mono_value(rest).items():
                out[m2] = out.get(m2, 0) + cen * c2
        return out

    def mono_value(self, mono):
        """the element of the module the (already straightened) monomial is."""
        return {mono: sympy.Integer(1)}

    def apply_word(self, word, elt):
        """apply the generators of ``word`` right to left to a module element."""
        cur = elt
        for key in reversed(word):
            nxt = {}
            for m, c in cur.items():
                for m2, c2 in self.act(key, m).items():
                    nxt[m2] = nxt.get(m2, 0) + c * c2
            cur = {m: sympy.expand(c) for m, c in nxt.items() if sympy.expand(c) != 0}
        return cur

    # ---- the Shapovalov form -------------------------------------------
    def pairing(self, u, w):
        """(u v | w v): the coefficient of v in sigma(u) w v."""
        sig = [self.alg.sigma(k) for k in reversed(u)]
        res = self.apply_word(sig, {w: sympy.Integer(1)})
        return sympy.expand(res.get((), sympy.Integer(0)))

    # ---- the PBW basis -------------------------------------------------
    def basis(self, level, charge=None):
        """monomials of the given level (and charge, if the algebra is charged)."""
        alg = self.alg
        neg = [g for g in alg._gens.values() if g.mode < 0]
        neg.sort(key=lambda g: alg.ordkey(g), reverse=True)
        out = []

        def rec(idx, lev, chg, acc):
            if lev == 0 and (charge is None or chg == charge):
                out.append(tuple(acc))
            if idx >= len(neg) or lev <= 0:
                return
            g = neg[idx]
            step = -g.mode
            if g.parity == 1:
                rec(idx + 1, lev, chg, acc)
                if step <= lev:
                    rec(idx + 1, lev - step, chg + g.charge, acc + [g.key])
            else:
                k = 0
                while k * step <= lev:
                    rec(idx + 1, lev - k * step, chg + k * g.charge, acc + [g.key] * k)
                    k += 1
        rec(0, level, 0, [])
        # dedupe (the recursion above enumerates each monomial once, but be safe)
        seen, uniq = set(), []
        for m in out:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq

    def gram(self, level, charge=None):
        b = self.basis(level, charge)
        M = sympy.zeros(len(b), len(b))
        for i in range(len(b)):
            for j in range(len(b)):
                M[i, j] = self.pairing(b[i], b[j])
        return b, M

    def par(self, mono):
        p = 0
        for k in mono:
            p += self.alg.gen(k).parity
        return p % 2


# ---------------------------------------------------------------------------
#  the Virasoro algebra
# ---------------------------------------------------------------------------

h, c = sympy.symbols("h c")


class Vir(Algebra):
    NAMES = ["L"]

    def __init__(self, nmax):
        gens = [Gen("L", n, 0) for n in range(-nmax, nmax + 1)]
        self.register(gens)

    def bracket(self, g1, g2):
        m, n = g1.mode, g2.mode
        terms = []
        if m + n in [g.mode for g in self._gens.values() if g.name == "L"]:
            terms.append(((m - n), ("L", m + n)))
        cen = sympy.Rational(1, 12) * c * (m ** 3 - m) if m + n == 0 else 0
        return terms, cen

    def sigma(self, key):
        return ("L", -key[1])

    def weight(self, g):
        return h


# ---------------------------------------------------------------------------
#  the Neveu-Schwarz algebra ns(1)
# ---------------------------------------------------------------------------

class NS1(Algebra):
    """[L_m,L_n]=(m-n)L + c/12 (m^3-m); [L_m,G_r]=(m/2-r)G_{m+r};
       {G_r,G_s}=2L_{r+s} + c/3 (r^2-1/4) delta."""
    NAMES = ["L", "G"]

    def __init__(self, nmax):
        gens = [Gen("L", n, 0) for n in range(-nmax, nmax + 1)]
        gens += [Gen("G", sympy.Rational(2 * r + 1, 2), 1)
                 for r in range(-nmax - 1, nmax + 1)]
        self.register(gens)

    def has(self, key):
        return key in self._gens

    def bracket(self, g1, g2):
        m, n = g1.mode, g2.mode
        if g1.name == "L" and g2.name == "L":
            terms = [((m - n), ("L", m + n))] if self.has(("L", m + n)) else []
            cen = sympy.Rational(1, 12) * c * (m ** 3 - m) if m + n == 0 else 0
            return terms, cen
        if g1.name == "L" and g2.name == "G":
            k = ("G", m + n)
            terms = [((m / 2 - n), k)] if self.has(k) else []
            return terms, 0
        if g1.name == "G" and g2.name == "L":
            k = ("G", m + n)
            terms = [(-(n / 2 - m), k)] if self.has(k) else []
            return terms, 0
        # {G_r, G_s}
        k = ("L", m + n)
        terms = [(sympy.Integer(2), k)] if self.has(k) else []
        cen = sympy.Rational(1, 3) * c * (m ** 2 - sympy.Rational(1, 4)) if m + n == 0 else 0
        return terms, cen

    def sigma(self, key):
        return (key[0], -key[1])

    def weight(self, g):
        return h


# ---------------------------------------------------------------------------
#  the N=2 Neveu-Schwarz algebra ns(2)
# ---------------------------------------------------------------------------

q, t = sympy.symbols("q t")


class NS2(Algebra):
    """Doerrzapf's conventions (hep-th/9403124, eq. (2.7)):
       [L_m,L_n]=(m-n)L+c/12(m^3-m); [L_m,G^pm_r]=(m/2-r)G^pm;
       [L_m,T_n]=-n T_{m+n}; [T_m,T_n]=c/3 m delta; [T_m,G^pm_r]=pm G^pm;
       {G^+_r,G^-_s}=2L+(r-s)T+c/3(r^2-1/4)delta; {G^+,G^+}={G^-,G^-}=0."""
    NAMES = ["L", "T", "Gm", "Gp"]

    def __init__(self, nmax, cc):
        self.c = cc
        gens = [Gen("L", n, 0) for n in range(-nmax, nmax + 1)]
        gens += [Gen("T", n, 0) for n in range(-nmax, nmax + 1)]
        gens += [Gen("Gp", sympy.Rational(2 * r + 1, 2), 1, +1)
                 for r in range(-nmax - 1, nmax + 1)]
        gens += [Gen("Gm", sympy.Rational(2 * r + 1, 2), 1, -1)
                 for r in range(-nmax - 1, nmax + 1)]
        self.register(gens)

    def has(self, key):
        return key in self._gens

    def bracket(self, g1, g2):
        cc = self.c
        m, n = g1.mode, g2.mode
        a, b = g1.name, g2.name
        if a == "L" and b == "L":
            terms = [((m - n), ("L", m + n))] if self.has(("L", m + n)) else []
            return terms, (sympy.Rational(1, 12) * cc * (m ** 3 - m) if m + n == 0 else 0)
        if a == "L" and b in ("Gp", "Gm"):
            k = (b, m + n)
            return ([((m / 2 - n), k)] if self.has(k) else []), 0
        if a in ("Gp", "Gm") and b == "L":
            k = (a, m + n)
            return ([(-(n / 2 - m), k)] if self.has(k) else []), 0
        if a == "L" and b == "T":
            k = ("T", m + n)
            return ([(-n, k)] if self.has(k) else []), 0
        if a == "T" and b == "L":
            k = ("T", m + n)
            return ([(m, k)] if self.has(k) else []), 0
        if a == "T" and b == "T":
            return [], (sympy.Rational(1, 3) * cc * m if m + n == 0 else 0)
        if a == "T" and b in ("Gp", "Gm"):
            k = (b, m + n)
            s = 1 if b == "Gp" else -1
            return ([(s, k)] if self.has(k) else []), 0
        if a in ("Gp", "Gm") and b == "T":
            k = (a, m + n)
            s = 1 if a == "Gp" else -1
            return ([(-s, k)] if self.has(k) else []), 0
        # odd x odd
        if a == b:
            return [], 0
        s = 1 if a == "Gp" else -1     # {G^+_r, G^-_s} with (r,s) = (m,n)
        r_, s_ = (m, n) if a == "Gp" else (n, m)
        terms = []
        if self.has(("L", m + n)):
            terms.append((sympy.Integer(2), ("L", m + n)))
        if self.has(("T", m + n)):
            terms.append(((r_ - s_), ("T", m + n)))
        cen = (sympy.Rational(1, 3) * cc * (r_ ** 2 - sympy.Rational(1, 4))
               if m + n == 0 else 0)
        return terms, cen

    def sigma(self, key):
        nm = {"L": "L", "T": "T", "Gp": "Gm", "Gm": "Gp"}[key[0]]
        return (nm, -key[1])

    def weight(self, g):
        if g.name == "L":
            return h
        if g.name == "T":
            return q
        return sympy.Integer(0)
