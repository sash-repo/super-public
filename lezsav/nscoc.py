#!/usr/bin/env python3
"""
nscoc.py -- from the super-Liouville anomaly  Phi_a = -1/2 D^5 a  to the value
of the central charge of the Neveu-Schwarz superalgebra ns = the non-trivial
central extension of k^L(1|1).

Superfunctions on the superstring C^{1|1} are pairs (f0, f1), meaning
f = f0(x) + theta f1(x), carrying a parity p (so f0 has parity p and f1 has
parity p+1).  D = d/dtheta + theta d/dx,  D^2 = d/dx.

Contact fields:  K_a = a D^2 + 1/2 (Da) D   (checked in sliouv.py to be the
general even one, with [K_a, D] = -1/2 (D^2 a) D).

Everything is exact (sympy Rationals / symbolic powers of x).
"""

import sympy as sp

x = sp.Symbol("x", positive=True)


# --------------------------------------------------------------------------
# superfunctions as (f0, f1, parity)
# --------------------------------------------------------------------------

class SF:
    def __init__(self, f0, f1, p):
        self.f0, self.f1, self.p = sp.expand(f0), sp.expand(f1), p % 2

    def __repr__(self):
        return f"({self.f0}) + theta*({self.f1})   [p={self.p}]"

    def __add__(self, o):
        assert self.p == o.p
        return SF(self.f0 + o.f0, self.f1 + o.f1, self.p)

    def __sub__(self, o):
        return self + o.scal(-1)

    def scal(self, c):
        return SF(c * self.f0, c * self.f1, self.p)

    def __mul__(self, o):
        s = (-1) ** self.p
        return SF(self.f0 * o.f0, self.f1 * o.f0 + s * self.f0 * o.f1,
                  self.p + o.p)

    def dx(self):
        return SF(sp.diff(self.f0, x), sp.diff(self.f1, x), self.p)

    def D(self):
        return SF(self.f1, sp.diff(self.f0, x), self.p + 1)

    def Dn(self, n):
        r = self
        for _ in range(n):
            r = r.D()
        return r

    def is_zero(self):
        return sp.simplify(self.f0) == 0 and sp.simplify(self.f1) == 0

    def simp(self):
        return SF(sp.simplify(self.f0), sp.simplify(self.f1), self.p)


def K(a, g):
    """K_a applied to the superfunction g.

    K_a = a D^2 + (-1)^{p(a)}/2 (Da) D.  The parity sign is forced: it is the
    only one for which [K_a, D] = -1/2 (D^2 a) D for BOTH parities of a (see
    sliouv.py, where both cases are re-derived from scratch)."""
    s = sp.Rational((-1) ** a.p, 2)
    return a * g.dx() + (a.D() * g.D()).scal(s)


def bracket(a, b):
    """the contact bracket, [K_a, K_b] = K_{a,b}:

        {a,b} = a b' - a' b + (-1)^{p(a)}/2 (Da)(Db).

    The three coefficients are not quoted but were determined by a scan and
    verified on 25 (pair, test-function) combinations; check_bracket re-verifies
    any given pair."""
    return (a * b.dx()) - (a.dx() * b) + \
           (a.D() * b.D()).scal(sp.Rational((-1) ** a.p, 2))


def check_bracket(a, b):
    """verify [K_a,K_b] = K_{a,b} on a battery of test superfunctions."""
    c = bracket(a, b)
    s = (-1) ** (a.p * b.p)
    tests = [SF(x ** 3, 0, 0), SF(0, x ** 2, 1), SF(x ** -2, 0, 0),
             SF(0, x ** sp.Rational(1, 2), 1), SF(sp.log(x), x ** 4, 0)]
    for g in tests:
        d = (K(a, K(b, g)) - K(b, K(a, g)).scal(s) - K(c, g)).simp()
        if not d.is_zero():
            return False, c, g, d
    return True, c, None, None


# --------------------------------------------------------------------------
# the anomaly and the cocycle
# --------------------------------------------------------------------------

def Phi_geom(a):
    """the F-free part of delta_a T, as found in sliouv.py for both parities:
           -(-1)^{p(a)}/2 * D^5 a .
    This is the 1-cocycle of k^L(1|1) with values in F_{3/2}."""
    return a.Dn(5).scal(sp.Rational(-((-1) ** a.p), 2))


def Phi(a):
    """the same anomaly, with the sign (-1)^{p(a)} moved into the
    identification F_{3/2} = g^*; what remains is  -1/2 D^5 a, uniformly in the
    parity, and THIS is what the 2-cocycle is built from.  Which of the two
    carries the sign is a convention; the tests below (super-skew-symmetry and
    the cocycle identity) decide it for the 2-cocycle, and the 1-cocycle test
    in pairing.py decides it for Phi_geom."""
    return a.Dn(5).scal(sp.Rational(-1, 2))


def Res(h):
    """Berezin residue: the coefficient of x^{-1} theta in h."""
    return sp.expand(h.f1).coeff(x, -1)


def cocycle(a, b):
    """c(a,b) = Res( b . Phi_a )."""
    return sp.simplify(Res(b * Phi(a)))


# --------------------------------------------------------------------------
# the standard Neveu-Schwarz basis
# --------------------------------------------------------------------------

def L(m):
    """L_m = K_a with a = -x^{m+1}; its d/dx-part is -x^{m+1} d/dx."""
    return SF(-x ** (m + 1), 0, 0)


def G(r, sigma):
    """G_r = K_b with b = sigma * theta * x^{r+1/2}."""
    return SF(0, sigma * x ** (sp.Rational(1, 2) + r), 1)


if __name__ == "__main__":
    print("=" * 74)
    print("1) the contact bracket  [K_a,K_b] = K_{a,b}")
    print("=" * 74)
    A = SF(x ** 3, 0, 0)
    B = SF(x ** -1, 0, 0)
    ok, c, g, d = check_bracket(A, B)
    print("   {x^3, x^-1}  =", c, "   verified:", ok)
    A = SF(-x ** 2, 0, 0)                       # L_1
    B = SF(0, x ** sp.Rational(1, 2), 1)        # ~ G_0
    ok2, c2, _, _ = check_bracket(A, B)
    print("   even x odd   verified:", ok2)
    A = SF(0, x ** sp.Rational(1, 2), 1)
    B = SF(0, x ** sp.Rational(3, 2), 1)
    ok3, c3, _, _ = check_bracket(A, B)
    print("   {theta x^1/2, theta x^3/2} =", c3, "  verified:", ok3)
    print()

    print("=" * 74)
    print("2) normalizing G_r so that {G_r,G_s} = 2 L_{r+s}")
    print("=" * 74)
    sig = sp.Symbol("sigma")
    r, s = sp.Rational(1, 2), sp.Rational(-1, 2)
    br = bracket(G(r, sig), G(s, sig))
    tgt = L(r + s)
    # br should equal 2*L_0
    eq = sp.simplify(br.f0 - 2 * tgt.f0)
    sols = sp.solve(sp.Eq(eq, 0), sig)
    print("   {G_1/2, G_-1/2} =", br.f0, "+ theta*(", br.f1, ")")
    print("   2 L_0 =", 2 * tgt.f0)
    print("   sigma =", sols)
    sigma = [q for q in sols if q != 0][0]
    print("   chosen sigma =", sigma)
    # re-check for other r,s
    for (rr, ss) in [(sp.Rational(3, 2), sp.Rational(-1, 2)),
                     (sp.Rational(5, 2), sp.Rational(-3, 2)),
                     (sp.Rational(1, 2), sp.Rational(3, 2))]:
        br = bracket(G(rr, sigma), G(ss, sigma))
        tg = L(rr + ss).scal(2)
        assert sp.simplify(br.f0 - tg.f0) == 0 and sp.simplify(br.f1 - tg.f1) == 0, (rr, ss)
    print("   {G_r,G_s} = 2 L_{r+s} verified for four more pairs   OK")
    # and [L_m, G_r] = (m/2 - r) G_{m+r}
    for m in [-2, -1, 0, 1, 2, 3]:
        for rr in [sp.Rational(-3, 2), sp.Rational(-1, 2), sp.Rational(1, 2), sp.Rational(5, 2)]:
            br = bracket(L(m), G(rr, sigma))
            tg = G(m + rr, sigma).scal(sp.Rational(m, 2) - rr)
            assert sp.simplify(br.f0 - tg.f0) == 0 and sp.simplify(br.f1 - tg.f1) == 0, (m, rr)
    print("   [L_m,G_r] = (m/2 - r) G_{m+r} verified, 24 pairs         OK")
    # and [L_m,L_n] = (m-n) L_{m+n}
    for m in range(-3, 4):
        for n in range(-3, 4):
            br = bracket(L(m), L(n))
            tg = L(m + n).scal(m - n)
            assert sp.simplify(br.f0 - tg.f0) == 0, (m, n)
    print("   [L_m,L_n] = (m-n) L_{m+n} verified, 49 pairs             OK")
    print()

    print("=" * 74)
    print("3) the cocycle c(a,b) = Res( b . (-1/2 D^5 a) )")
    print("=" * 74)
    print("   c(L_m,L_n), should be (c/12)(m^3-m) delta_{m+n,0}:")
    bad = 0
    for m in range(-4, 5):
        for n in range(-4, 5):
            v = cocycle(L(m), L(n))
            if m + n != 0:
                if v != 0:
                    bad += 1
            else:
                pred = sp.Rational(-1, 2) * (m ** 3 - m)
                if sp.simplify(v - pred) != 0:
                    bad += 1
                    print("      MISMATCH", m, n, v, pred)
    print("      c(L_m,L_-m) = -(1/2)(m^3-m) for -4<=m<=4, and 0 off the"
          " antidiagonal;  failures:", bad)
    print("      => c/12 = -1/2, i.e.  c = -6  (|c| = 6)")
    print()
    print("   c(G_r,G_s), should be (c/3)(r^2-1/4) delta_{r+s,0}:")
    bad = 0
    halves = [sp.Rational(2 * k + 1, 2) for k in range(-5, 5)]
    for rr in halves:
        for ss in halves:
            v = cocycle(G(rr, sigma), G(ss, sigma))
            if rr + ss != 0:
                if v != 0:
                    bad += 1
                    print("      OFF-DIAG", rr, ss, v)
            else:
                pred = -2 * (rr ** 2 - sp.Rational(1, 4))
                if sp.simplify(v - pred) != 0:
                    bad += 1
                    print("      MISMATCH", rr, ss, v, pred)
    print("      c(G_r,G_-r) = -2(r^2-1/4) for |r|<=9/2;  failures:", bad)
    print("      => c/3 = -2, i.e. c = -6 -- the SAME c, so the cocycle is the")
    print("         Neveu-Schwarz one and the two sectors agree.")
    print()

    print("=" * 74)
    print("4) super-skew-symmetry and the cocycle identity")
    print("=" * 74)
    basis = [L(-1), L(0), L(1), L(2), G(sp.Rational(-1, 2), sigma),
             G(sp.Rational(1, 2), sigma), G(sp.Rational(3, 2), sigma),
             G(sp.Rational(-3, 2), sigma)]
    bad = 0
    for a in basis:
        for b in basis:
            lhs = cocycle(a, b)
            rhs = -(-1) ** (a.p * b.p) * cocycle(b, a)
            if sp.simplify(lhs - rhs) != 0:
                bad += 1
    print("   super-skew-symmetry: failures", bad, "of", len(basis) ** 2)
    bad = tot = 0
    for a in basis:
        for b in basis:
            for cc in basis:
                tot += 1
                t1 = (-1) ** (a.p * cc.p) * cocycle(bracket(a, b), cc)
                t2 = (-1) ** (b.p * a.p) * cocycle(bracket(b, cc), a)
                t3 = (-1) ** (cc.p * b.p) * cocycle(bracket(cc, a), b)
                if sp.simplify(t1 + t2 + t3) != 0:
                    bad += 1
    print("   cocycle identity   : failures", bad, "of", tot)
