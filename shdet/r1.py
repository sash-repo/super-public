"""The Shapovalov determinant of r(1) with the (1|1) vacuum.

Two things are done here.

A) The basis (21) and the table (22) of Subsec. 3.3 are checked against
   the Ramond bracket of the paper, eq. (kb), restricted to the one twisted odd
   indeterminate zeta, i.e., against k^M(1|1).  This settles

     -- whether the modes of G are integers or half-integers (the label
        G_{n+1/2} of (21) versus the element G_0 the text uses right after),
     -- the normalization of L_n and G_n as multiples of K_{t^{n+1}} and
        K_{t^{n+1}zeta},
     -- the value of the parameter alpha of (22) in terms of the central
        charge c of the standard normalization.

B) The Verma module over r(1) is built with the (1|1) vacuum

        V = Span(v, w = G_0 v),   (v|v) = (w|w) = 1,  (v|w) = 0,

   which is the normalization (23) of the text, and the determinant of
   the Shapovalov form is computed on every weight space of level <= NMAX and
   factored over Q(h, c).  Both the whole determinant and the two parity blocks
   (at each level only vectors of the same parity pair) are given.

Exact arithmetic throughout (sympy over Q[h, c]).
"""

import sys
from fractions import Fraction as Q

import sympy

from verma import Algebra, Gen, Module, h, c

LOG = open("run-r1.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ===========================================================================
#  A) the algebra r(1) read off the Ramond bracket (kb)
# ===========================================================================

def check_bracket():
    """k^M(1|1): functions t^a zeta^e, bracket (kb) of the paper."""
    from stringy import mono, add, mul, smul, d_t, d_odd, parity
    Z = 0

    def nabla(f):
        out = {}
        for (a, S), co in f.items():
            v = (2 - len(S)) * co
            if v:
                out[(a, S)] = v
        return out

    def D(f):
        return add(d_t(f), smul(Q(-1, 2), mul(mono(-1, (Z,)), d_odd(Z, f))))

    def br(f, g):
        """eq. (kb): the odd part enters with the sign -(-1)^{p(f)}."""
        res = add(mul(nabla(f), D(g)), smul(-1, mul(D(f), nabla(g))))
        s = smul(Q(1), mul(mono(-1), mul(d_odd(Z, f), d_odd(Z, g))))
        return add(res, smul(-((-1) ** parity(f)), s))

    say("=" * 74)
    say("A) the basis (21) and the table (22) against the bracket (kb)")
    say("=" * 74)

    # the generating functions
    def KL(n):
        return mono(n + 1)              # t^{n+1}

    def KG(n):
        return mono(n + 1, (Z,))        # t^{n+1} zeta

    say("")
    say("Raw brackets of the generating functions:")
    for a, b in [(1, 2), (2, -1), (0, 3)]:
        say("   {t^%d, t^%d}     = %s" % (a + 1, b + 1, fmt(br(KL(a), KL(b)))))
    for a, b in [(1, 2), (0, 0), (2, -1)]:
        say("   {t^%d, t^%d z}   = %s" % (a + 1, b + 1, fmt(br(KL(a), KG(b)))))
    for a, b in [(1, -1), (0, 0), (2, -2)]:
        say("   {t^%d z, t^%d z} = %s" % (a + 1, b + 1, fmt(br(KG(a), KG(b)))))

    # normalization: L_n := u * K_{t^{n+1}}, G_n := s * K_{t^{n+1} zeta}
    # {t^{a+1}, t^{b+1}} = 2(b-a) t^{a+b+1}  =>  u = 1/2 gives [L_a,L_b]=(b-a)L
    # {t^{a+1}z, t^{b+1}z} = k * t^{a+b+1}   =>  fix s by {G_a,G_b} = 2 L_{a+b}
    say("")
    ok = True
    for a in (-2, -1, 0, 1, 2, 3):
        for b in (-3, -1, 0, 2):
            got = br(KL(a), KL(b))
            want = smul(2 * (b - a), KL(a + b))
            if got != want:
                ok = False
                say("   MISMATCH {L,L} at (%d,%d): %s vs %s"
                    % (a, b, fmt(got), fmt(want)))
    say("   {t^{a+1}, t^{b+1}} = 2(b-a) t^{a+b+1} on all pairs tested: %s" % ok)

    ok = True
    for a in (-2, -1, 0, 1, 2):
        for b in (-3, -1, 0, 2):
            got = br(KL(a), KG(b))
            want = smul(2 * b - a, KG(a + b))
            if got != want:
                ok = False
                say("   MISMATCH {L,G} at (%d,%d): %s vs %s"
                    % (a, b, fmt(got), fmt(want)))
    say("   {t^{a+1}, t^{b+1}z} = (2b-a) t^{a+b+1}z on all pairs tested: %s" % ok)

    ok = True
    for a in (-2, -1, 0, 1, 2):
        for b in (-3, -1, 0, 2):
            got = br(KG(a), KG(b))
            want = KL(a + b)
            if got != want:
                ok = False
                say("   MISMATCH {G,G} at (%d,%d): %s vs %s"
                    % (a, b, fmt(got), fmt(want)))
    say("   {t^{a+1}z, t^{b+1}z} = t^{a+b+1} on all pairs tested: %s" % ok)

    say("")
    say("   Hence, with  L_n := (1/2) K_{t^{n+1}}  and  G_n := K_{t^{n+1} zeta},")
    say("      [L_a, L_b] = (b-a) L_{a+b},   [L_a, G_b] = (b - a/2) G_{a+b},")
    say("      [G_a, G_b] = 2 L_{a+b},")
    say("   which is the table (22) with the central terms switched off.")
    say("   The modes of G are INTEGERS: G_b = i K_{t^{b+1}zeta} has the same")
    say("   ad_{K_t}-degree 2b as L_b, and G_0 -- the element the text uses right")
    say("   after (21) -- is i K_{t zeta}.  So the label G_{n+1/2} of")
    say("   (21) is a misprint for G_n.")


def fmt(f):
    if not f:
        return "0"
    out = []
    for (a, S), co in sorted(f.items()):
        m = "t^%d" % a + ("z" if S else "")
        out.append("%s*%s" % (co, m))
    return " + ".join(out)


# ===========================================================================
#  B) the Verma module over r(1) with the (1|1) vacuum
# ===========================================================================

class R1(Algebra):
    """The Ramond algebra r(1) in the standard normalization

         [L_m,L_n] = (m-n)L_{m+n} + (c/12)(m^3-m) delta,
         [L_m,G_n] = (m/2-n) G_{m+n},
         {G_m,G_n} = 2 L_{m+n} + (c/3)(m^2 - 1/4) delta,      m, n in Z.

    The table (22) is this algebra written in the letters
    L' = -L, G' = sqrt(-1) G, alpha = c/12; see part A.
    """
    NAMES = ["L", "G"]

    def __init__(self, nmax):
        gens = [Gen("L", n, 0) for n in range(-nmax, nmax + 1)]
        gens += [Gen("G", n, 1) for n in range(-nmax, nmax + 1)]
        self.register(gens)

    def has(self, key):
        return key in self._gens

    def islow(self, g):
        """the generators a PBW monomial may contain: G_0 is one of them."""
        return g.mode < 0 or (g.name == "G" and g.mode == 0)

    def ordkey(self, g):
        if g.name == "G" and g.mode == 0:
            return (sympy.Integer(-10 ** 6), 0)     # G_0 sits at the right end
        return (g.mode, self.namerank(g.name))

    def bracket(self, g1, g2):
        m, n = g1.mode, g2.mode
        a, b = g1.name, g2.name
        if a == "L" and b == "L":
            terms = [((m - n), ("L", m + n))] if self.has(("L", m + n)) else []
            cen = sympy.Rational(1, 12) * c * (m ** 3 - m) if m + n == 0 else 0
            return terms, cen
        if a == "L" and b == "G":
            k = ("G", m + n)
            return ([((sympy.Rational(m, 2) - n), k)] if self.has(k) else []), 0
        if a == "G" and b == "L":
            k = ("G", m + n)
            return ([(-(sympy.Rational(n, 2) - m), k)] if self.has(k) else []), 0
        k = ("L", m + n)
        terms = [(sympy.Integer(2), k)] if self.has(k) else []
        cen = (sympy.Rational(1, 3) * c * (m ** 2 - sympy.Rational(1, 4))
               if m + n == 0 else 0)
        return terms, cen

    def sigma(self, key):
        return (key[0], -key[1])

    def weight(self, g):
        return h if g.name == "L" else sympy.Integer(0)


G0 = ("G", sympy.Integer(0))


class RModule(Module):
    """The Verma module with the (1|1) vacuum V = Span(v, w = G_0 v)."""

    def basis(self, level, charge=None):
        alg = self.alg
        neg = [g for g in alg._gens.values() if g.mode < 0]
        neg.sort(key=lambda g: alg.ordkey(g), reverse=True)
        out = []

        def rec(idx, lev, acc):
            if lev == 0:
                out.append(tuple(acc))              # ... v
                out.append(tuple(acc) + (G0,))      # ... w
            if idx >= len(neg) or lev <= 0:
                return
            g = neg[idx]
            step = -g.mode
            if g.parity == 1:
                rec(idx + 1, lev, acc)
                if step <= lev:
                    rec(idx + 1, lev - step, acc + [g.key])
            else:
                k = 0
                while k * step <= lev:
                    rec(idx + 1, lev - k * step, acc + [g.key] * k)
                    k += 1
        rec(0, level, [])
        seen, uniq = set(), []
        for m in out:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq

    def _act(self, key, mono):
        """as in Module, but G_0 may sit in a monomial and the vacuum is (1|1)."""
        alg = self.alg
        g = alg.gen(key)
        if not mono:
            if g.mode < 0:
                return {(key,): sympy.Integer(1)}
            if g.mode > 0:
                return {}
            if g.name == "G":                      # G_0 v = w
                return {(G0,): sympy.Integer(1)}
            return {(): alg.weight(g)}

        head, rest = mono[0], mono[1:]
        hh = alg.gen(head)

        if alg.islow(g) and alg.ordkey(g) >= alg.ordkey(hh):
            if not (key == head and g.parity == 1):
                return {(key,) + mono: sympy.Integer(1)}
            terms, cen = alg.bracket(g, g)         # odd square: g g = 1/2 {g, g}
            out = {}
            for coeff, k2 in terms:
                for m2, c2 in self.act(k2, rest).items():
                    out[m2] = out.get(m2, 0) + sympy.Rational(1, 2) * coeff * c2
            if cen != 0:
                for m2, c2 in self.mono_value(rest).items():
                    out[m2] = out.get(m2, 0) + sympy.Rational(1, 2) * cen * c2
            return out

        sign = -1 if (g.parity == 1 and hh.parity == 1) else 1
        out = {}
        for m1, c1 in self.act(key, rest).items():
            for m2, c2 in self.act(head, m1).items():
                out[m2] = out.get(m2, 0) + sign * c1 * c2
        terms, cen = alg.bracket(g, hh)
        for coeff, k2 in terms:
            for m2, c2 in self.act(k2, rest).items():
                out[m2] = out.get(m2, 0) + coeff * c2
        if cen != 0:
            for m2, c2 in self.mono_value(rest).items():
                out[m2] = out.get(m2, 0) + cen * c2
        return out

    def pairing(self, u, w):
        """(u | w) in the normalization (v|v) = (w|w) = 1, (v|w) = 0."""
        eps = 1 if u and u[-1] == G0 else 0
        m = u[:-1] if eps else u
        sig = [self.alg.sigma(k) for k in reversed(m)]
        res = self.apply_word(sig, {w: sympy.Integer(1)})
        want = (G0,) if eps else ()
        return sympy.expand(res.get(want, sympy.Integer(0)))

    def pairing_induced(self, u, w):
        """(u | w) in the induced normalization (w|w) = (v | G_0^2 v) = h - c/24."""
        eps = 1 if u and u[-1] == G0 else 0
        val = self.pairing(u, w)
        return sympy.expand(val * ((h - sympy.Rational(1, 24) * c) ** eps))


CNS = sympy.Rational(3, 2) * c      # the section's c: c_vir = (3/2) c_ns


def phiR(k, s):
    """The quadratic in h whose roots are

         (1/32)( (5-c)(k^2+s^2) +- sqrt(c^2-10c+9)(k^2-s^2) - 8ks + 2c ),

    i.e., the phi_{k,s} of Subsec. 3.3 (Neveu--Schwarz) with its constant term
    2c-2 replaced by 2c.  Here c is the section's central charge, so the engine's
    c has to be substituted for by CNS.
    """
    cc = sympy.Symbol("cns")
    base = sympy.Rational(1, 32) * ((5 - cc) * (k ** 2 + s ** 2) - 8 * k * s + 2 * cc)
    dev = sympy.Rational(1, 32) * sympy.sqrt(cc ** 2 - 10 * cc + 9) * (k ** 2 - s ** 2)
    out = sympy.expand((h - base - dev) * (h - base + dev))
    return sympy.expand(out.subs(cc, sympy.Rational(2, 3) * c))


def psiR(j):
    """psi^R_j = prod over the unordered pairs {k,s} with ks = 2j and k+s ODD."""
    out = sympy.Integer(1)
    for k in range(1, 2 * j + 1):
        if (2 * j) % k == 0 and k < 2 * j // k and (k + 2 * j // k) % 2 == 1:
            out *= phiR(k, 2 * j // k)
    return sympy.expand(out)


def PR(m):
    """the number of partitions of m into parts of two colors, one of the two
       colors allowed at most once: prod (1+x^k)/(1-x^k)."""
    tab = [1] + [0] * m
    for k in range(1, m + 1):
        for n in range(k, m + 1):
            tab[n] += tab[n - k]
    tab2 = list(tab)
    for k in range(1, m + 1):
        for n in range(m, k - 1, -1):
            tab2[n] += tab2[n - k]
    return tab2[m]


def run(NMAX=3):
    say("")
    say("=" * 74)
    say("B) the Shapovalov determinant of r(1), levels 1 .. %d" % NMAX)
    say("   vacuum V = Span(v, w = G_0 v), (v|v) = (w|w) = 1, (v|w) = 0")
    say("=" * 74)
    alg = R1(2 * NMAX + 6)
    M = RModule(alg)
    for n in range(0, NMAX + 1):
        b = M.basis(n)
        G = sympy.zeros(len(b), len(b))
        for i in range(len(b)):
            for j in range(len(b)):
                G[i, j] = M.pairing(b[i], b[j])
        det = sympy.factor(sympy.expand(G.det()))
        say("")
        say("level %d: dim = %d" % (n, len(b)))
        say("   basis = %s" % (b,))
        say("   det   = %s" % det)
        # the two parity blocks
        ev = [i for i in range(len(b)) if M.par(b[i]) == 0]
        od = [i for i in range(len(b)) if M.par(b[i]) == 1]
        say("   sdim of the level = (%d|%d)" % (len(ev), len(od)))
        for nm, idx in (("even", ev), ("odd ", od)):
            if not idx:
                continue
            B = sympy.zeros(len(idx), len(idx))
            for x, i in enumerate(idx):
                for y, j in enumerate(idx):
                    B[x, y] = G[i, j]
            say("   det on the %s block = %s"
                % (nm, sympy.factor(sympy.expand(B.det()))))
        # the off-diagonal blocks must vanish: the form is even
        bad = [(i, j) for i in ev for j in od if sympy.expand(G[i, j]) != 0]
        say("   even-odd pairings that do not vanish: %d" % len(bad))
        # the induced normalization
        Gi = sympy.zeros(len(b), len(b))
        for i in range(len(b)):
            for j in range(len(b)):
                Gi[i, j] = M.pairing_induced(b[i], b[j])
        deti = sympy.factor(sympy.expand(Gi.det()))
        say("   det in the induced normalization = %s" % deti)
        if det != 0:
            say("   ratio = %s" % sympy.factor(sympy.cancel(deti / det)))
        # the closed expression
        pr = sympy.Integer(1)
        for j in range(1, n + 1):
            pr *= psiR(j) ** PR(n - j)
        if pr != 1:
            B = sympy.zeros(len(ev), len(ev))
            for x, i in enumerate(ev):
                for y, j2 in enumerate(ev):
                    B[x, y] = G[i, j2]
            r = sympy.simplify(sympy.cancel(sympy.expand(B.det()) / pr))
            say("   (even block) / prod_j (psi^R_j)^{P_R(n-j)} = %s  -> %s"
                % (r, "AGREES (a scalar)" if r.free_symbols == set() else "DIFFERS"))


# ===========================================================================
#  C) the Bernstein--Shapovalov form: the Berezin integral of the U(h)-valued one
# ===========================================================================

def bernstein(NMAX=3):
    """On M = U(g_-) (x) V, the U(h)-valued form is HC(sigma(m) m') in
       U(h) = S(h_ev) (x) Lambda(G_0); write it as alpha + beta G_0.  Then

         alpha = the form one gets when h_od is made to act trivially,
                 i.e., under the restriction (38) of the text,
         beta  = the Berezin integral, i.e., Bernstein's form B of (40).
    """
    say("")
    say("=" * 74)
    say("C) the Bernstein--Shapovalov determinant of r(1) against the classical one")
    say("=" * 74)
    alg = R1(2 * NMAX + 6)
    M = RModule(alg)
    for n in range(0, NMAX + 1):
        b = [m for m in M.basis(n) if not (m and m[-1] == G0)]   # U(g_-) only
        A = sympy.zeros(len(b), len(b))
        B = sympy.zeros(len(b), len(b))
        for i in range(len(b)):
            sig = [alg.sigma(k) for k in reversed(b[i])]
            for j in range(len(b)):
                res = M.apply_word(sig, {b[j]: sympy.Integer(1)})
                A[i, j] = sympy.expand(res.get((), sympy.Integer(0)))
                B[i, j] = sympy.expand(res.get((G0,), sympy.Integer(0)))
        dA = sympy.factor(sympy.expand(A.det()))
        dB = sympy.factor(sympy.expand(B.det()))
        pr = sympy.Integer(1)
        for j in range(1, n + 1):
            pr *= psiR(j) ** PR(n - j)
        say("")
        say("level %d: dim U(g_-)(n) = %d, P_R(%d) = %d" % (n, len(b), n, PR(n)))
        say("   det of the even part alpha    = %s" % dA)
        say("   det of the Berezin integral B = %s" % dB)
        say("   prod_j (psi^R_j)^{P_R(n-j)}   = %s" % sympy.factor(pr))
        for nm, d in (("alpha", dA), ("B", dB)):
            if d == 0:
                say("   det %s vanishes identically" % nm)
                continue
            r = sympy.simplify(sympy.cancel(d / pr))
            say("   det %s / the product = %s  -> %s"
                % (nm, r, "AGREES (a scalar)" if r.free_symbols == set() else "DIFFERS"))


def bernstein2(NMAX=2):
    """The Bernstein form on the WHOLE weight space of M = U(g_-) (x) V.

    The Shapovalov form takes values in U(h) = Clifford algebra on G_0 over
    S(h_ev): for x, y in M of level n, the level-0 part of sigma(x) y is
    (A + B G_0) v, and B(x|y) := int (x|y) is the coefficient B of G_0.
    """
    say("")
    say("=" * 74)
    say("D) the Bernstein form on the whole (1|1)-vacuum module, levels <= %d" % NMAX)
    say("=" * 74)
    alg = R1(2 * NMAX + 6)
    M = RModule(alg)
    for n in range(0, NMAX + 1):
        b = M.basis(n)
        B = sympy.zeros(len(b), len(b))
        for i in range(len(b)):
            sig = [alg.sigma(k) for k in reversed(b[i])]
            for j in range(len(b)):
                res = M.apply_word(sig, {b[j]: sympy.Integer(1)})
                B[i, j] = sympy.expand(res.get((G0,), sympy.Integer(0)))
        det = sympy.factor(sympy.expand(B.det()))
        pr = sympy.Integer(1)
        for j in range(1, n + 1):
            pr *= psiR(j) ** PR(n - j)
        say("")
        say("level %d: dim = %d" % (n, len(b)))
        say("   det B                       = %s" % det)
        say("   prod_j (psi^R_j)^{P_R(n-j)} = %s" % sympy.factor(pr))
        if det != 0:
            for nm, tgt in (("the product", pr), ("its square", sympy.expand(pr ** 2))):
                r = sympy.simplify(sympy.cancel(det / tgt))
                say("   det B / %s = %s  -> %s"
                    % (nm, r, "AGREES (a scalar)" if r.free_symbols == set()
                       else "DIFFERS"))


if __name__ == "__main__":
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if what == "bern":
        bernstein2(n)
    else:
        check_bracket()
        run(n)
        bernstein(n)
        bernstein2(min(n, 2))
