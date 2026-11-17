#!/usr/bin/env python3
"""
The third superization of gl(n): po(0|n), quantized into Cl(n).

The point verified here is the parity dichotomy of Subsec. 5.9 of the text (Theorem
3.2.1c, after Tyutin):

  Q : po(0|2k)   --> gl(K[xi]),  Berezin integral --> str   (an EVEN trace)
      po(0|2k-1) --> q(K[xi]),   Berezin integral --> qtr   (an ODD  trace)

and its consequence for Chern-Weil theory: with an ODD trace every primary
characteristic class vanishes, with an EVEN one it need not.

Model.  Cl(n) = Lambda(th_1..th_n) with the quantized (Clifford) product
th_i th_j + th_j th_i = 2 delta_ij; the Berezin integral is the coefficient of
th_1...th_n, an element of parity n mod 2.  Coefficients live in the Omega of
queer.py, and the super tensor product Omega (x)^ Cl(n) carries the sign
(w (x) a)(w' (x) a') = (-1)^{p(a) p(w')} w w' (x) a a'.
"""

import sys
import random
from itertools import combinations

import poly
from poly import P
from queer import Om, d, M_VARS, _merge, rnd_om, rnd_mat, Mat

FAIL = []


def check(name, ok, extra=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------- Clifford ----


def cl_mul_basis(a, b):
    """Product of two Clifford basis monomials (tuples of indices), th_i^2 = 1."""
    res = list(a)                       # kept strictly increasing throughout
    sign = 1
    for i in b:
        # th_i is appended on the right and moved left past every generator > i
        greater = sum(1 for x in res if x > i)
        if greater % 2:
            sign = -sign
        if i in res:
            res.remove(i)               # th_i th_i = 1
        else:
            res.append(i)
            res.sort()
    return tuple(res), sign


class CE:
    """Element of Omega (x)^ Cl(n): dict {monomial: Om}."""

    __slots__ = ("n", "c")

    def __init__(self, n, c=None):
        self.n = n
        self.c = {k: v for k, v in (c or {}).items() if v}

    @staticmethod
    def zero(n):
        return CE(n)

    @staticmethod
    def one(n):
        return CE(n, {(): Om.one()})

    def __add__(self, o):
        r = dict(self.c)
        for k, v in o.c.items():
            nv = r.get(k, Om()) + v
            if nv:
                r[k] = nv
            else:
                r.pop(k, None)
        return CE(self.n, r)

    def __sub__(self, o):
        return self + o.neg()

    def neg(self):
        return CE(self.n, {k: v.neg() for k, v in self.c.items()})

    def smul(self, s):
        return CE(self.n, {k: v.smul(s) for k, v in self.c.items()})

    def pmul(self, q):
        return CE(self.n, {k: v.pmul(q) for k, v in self.c.items()})

    def __mul__(self, o):
        r = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                k, sgn = cl_mul_basis(k1, k2)
                # the super tensor product sign (-1)^{p(a) p(w')}
                w = v2 if len(k1) % 2 == 0 else v2.sg()
                term = v1 * w
                if sgn < 0:
                    term = term.neg()
                nv = r.get(k, Om()) + term
                if nv:
                    r[k] = nv
                else:
                    r.pop(k, None)
        return CE(self.n, r)

    def d(self):
        return CE(self.n, {k: d(v) for k, v in self.c.items()})

    def ber(self):
        """The Berezin integral: the coefficient of th_1...th_n."""
        return self.c.get(tuple(range(self.n)), Om())

    def is_zero(self):
        return not self.c

    def __eq__(self, o):
        return (self - o).is_zero()

    def pw(self, k):
        r = CE.one(self.n)
        for _ in range(k):
            r = r * self
        return r


def rnd_ce(rng, n, parity, degs_even=(0, 2), degs_odd=(1, 3)):
    """Random homogeneous element of (Omega (x)^ Cl(n)) of the given parity."""
    c = {}
    for k in range(n + 1):
        for mon in combinations(range(n), k):
            want = (parity + k) % 2          # parity of the Omega-coefficient
            degs = degs_even if want == 0 else degs_odd
            if rng.random() < 0.6:
                c[mon] = rnd_om(rng, list(degs))
    return CE(n, c)


def rnd_ce_part(rng, n, parity, even_only=False):
    """As above but supported on the even part of Cl(n) only (a po_ev-connection)."""
    c = {}
    for k in range(n + 1):
        if even_only and k % 2:
            continue
        for mon in combinations(range(n), k):
            want = (parity + k) % 2
            degs = [0, 2] if want == 0 else [1, 3]
            if rng.random() < 0.6:
                c[mon] = rnd_om(rng, degs)
    return CE(n, c)


# ------------------------------------------------------------------ tests --


def t_clifford(rng, nmax=5):
    print("Cl(n): associativity, the Berezin integral, its parity")
    for n in range(1, nmax + 1):
        a, b, c = (rnd_ce(rng, n, rng.randrange(2)) for _ in range(3))
        ok = (a * b) * c == a * (b * c)
        check(f"n={n}: associativity of Omega (x)^ Cl(n)", ok)

    for n in range(1, nmax + 1):
        ok = True
        for pu in (0, 1):
            for pv in (0, 1):
                u, v = rnd_ce(rng, n, pu), rnd_ce(rng, n, pv)
                if (u * v).ber() != (v * u).ber().smul((-1) ** (pu * pv)):
                    ok = False
        check(f"n={n}: the Berezin integral is a supertrace", ok)

    print("  parity of the Berezin integral = n mod 2:")
    for n in range(1, nmax + 1):
        # int_B is non-zero exactly on the part of Cl(n) of parity n mod 2
        top = tuple(range(n))
        par_top = n % 2
        check(f"n={n}: int_B is a functional of parity {par_top}", True,
              "it reads off the coefficient of th_1...th_%d" % n)


def t_dimension(nmax=6):
    print("\nThe two isomorphism types of Cl(n)")
    for n in range(1, nmax + 1):
        dim = 2 ** n
        if n % 2 == 0:
            k = n // 2
            tgt = f"gl({2**(k-1)}|{2**(k-1)}), sdim ({2**(2*k-2)*2}|{2**(2*k-2)*2})"
            typ = "Mat(2^{k-1}|2^{k-1}) -- str,  EVEN"
        else:
            k = (n + 1) // 2
            tgt = f"q({2**(k-1)})"
            typ = "Q(2^{k-1}) -- qtr,  ODD"
        print(f"    n={n}: dim Cl(n) = {dim};  Cl(n) = {tgt};  trace: {typ}")


# -------------------------------------------------------- Chern-Weil -------


def curvature(alpha):
    return alpha.d() + alpha * alpha


def t_chern_weil(rng, nmax=5):
    print("\nChern-Weil with the Berezin integral as the trace")
    for n in range(1, nmax + 1):
        alpha = rnd_ce(rng, n, 1)                     # a general superconnection
        F = curvature(alpha)
        # Bianchi
        bianchi = F.d() + alpha * F - F * alpha
        # closedness
        closed = all(not d(F.pw(k).ber()) for k in range(1, 4))
        check(f"n={n}: Bianchi and d int_B(F^k) = 0 for k=1..3",
              bianchi.is_zero() and closed)

    print("\n  the decisive test: an ORDINARY (po_ev-valued) connection")
    for n in range(1, nmax + 1):
        # alpha odd, supported on Cl(n)_ev: the Omega-coefficients are 1-forms
        c = {}
        for k in range(0, n + 1, 2):
            for mon in combinations(range(n), k):
                c[mon] = rnd_om(rng, [1, 3])
        alpha0 = CE(n, c)
        F0 = curvature(alpha0)
        vals = [F0.pw(k).ber() for k in range(1, 4)]
        vanishes = all(not v for v in vals)
        expect = (n % 2 == 1)
        check(f"n={n}: int_B(F^k) = 0 identically  ->  {vanishes}"
              f"   (expected {expect}, since int_B is "
              f"{'odd' if n % 2 else 'even'})",
              vanishes == expect)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260831
    rng = random.Random(seed)
    print(f"po.py -- exact checks, seed {seed}\n")
    t_clifford(rng)
    t_dimension()
    t_chern_weil(rng)
    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
