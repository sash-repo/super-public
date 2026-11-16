"""The Shapovalov determinant of the centrally extended svect^L_lambda(1|2).

The algebra, its cocycle and its anti-involution come from ``svectlam.py`` and
``sigmalam.py``; nothing here is typed in by hand.

Roots.  With mu := (lambda+1)/2, the root of a basis element of mode m is the
pair (its LL_0-weight, its T0_0-weight):

    LL_m, T0_m : (m, 0)      Tp_m : (m, 2)       Tm_m : (m, -2)
    S1_m : (m+mu,  1)        S2_m : (m+mu, -1)
    P1_m : (m-mu, -1)        P2_m : (m-mu,  1)

so g_0 = t = Span(LL_0, T0_0, z) is purely even and 3-dimensional.  A root is
called positive if its first entry is > 0, or is 0 and its second entry is > 0;
for 0 < mu < 1 -- the chamber that contains the small N=4 algebra, mu = 1/2 --
the lowering generators are

    LL_{-k}, T0_{-k}, Tp_{-k}, Tm_{-k} (k >= 1),  Tm_0,
    S^j_{-k} (k >= 1),                            P^j_{-k} (k >= 0).

The Verma module M(h, l) is U(n^-)v with LL_0 v = h v, T0_0 v = l v, z v = kap v.
It is graded by the triple (N, B, Q): a lowering generator of mode -k and root
(-k + b mu, q) contributes (k, b, q).  Each graded piece is finite-dimensional,
and the Shapovalov form pairs the piece (N, B, Q) with itself.

Exact arithmetic: sympy over Q(h, l, kap, lambda).
"""

import itertools
import sys

import sympy

from svectlam import NAMES, PARITY, ROOT, LAM, KAP, br

H = sympy.Symbol("h")            # the LL_0-weight of the vacuum
ELL = sympy.Symbol("ell")        # the T0_0-weight of the vacuum

#: sigma, as computed by sigmalam.py: X_m |-> X'_{-m} with unit coefficients.
MIRROR = {"LL": "LL", "T0": "T0", "Tp": "Tm", "Tm": "Tp",
          "S1": "P1", "P1": "S1", "S2": "P2", "P2": "S2"}


def sigma(key):
    name, m = key
    return (MIRROR[name], -m)


def bq(name):
    """(b, q): the mu-part and the charge of the root."""
    _, b, q = ROOT[name]
    return b, q


def role(key):
    """'-' lowering, '0' toral, '+' raising."""
    name, m = key
    b, q = bq(name)
    #  weight = m + b*mu with 0 < mu < 1 generic
    if m + b * sympy.Rational(1, 2) != 0 or b != 0:
        w = sympy.Rational(2 * m + b, 2)
        if w < 0:
            return "-"
        if w > 0:
            return "+"
    if q < 0:
        return "-"
    if q > 0:
        return "+"
    return "0"


def level(key):
    """(N, B, Q) of a lowering generator."""
    name, m = key
    b, q = bq(name)
    return (-m, b, q)


ORDER = {n: i for i, n in enumerate(["LL", "T0", "Tp", "Tm", "S1", "S2", "P1", "P2"])}


def ordkey(key):
    name, m = key
    return (m, ORDER[name])


def weight_of(key):
    name, m = key
    b, q = bq(name)
    return sympy.Rational(2 * m + b, 2), q


# ---------------------------------------------------------------------------
#  the Verma module
# ---------------------------------------------------------------------------

class Verma:
    def __init__(self):
        self._cache = {}

    def toral_value(self, key):
        name, m = key
        if name == "LL":
            return H
        if name == "T0":
            return ELL
        raise KeyError(key)

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
        r = role(key)
        if not mono:
            if r == "-":
                return {(key,): sympy.Integer(1)}
            if r == "+":
                return {}
            return {(): self.toral_value(key)}

        head, rest = mono[0], mono[1:]
        podd = PARITY[key[0]] == 1

        if r == "-" and ordkey(key) >= ordkey(head):
            if not (key == head and podd):
                return {(key,) + mono: sympy.Integer(1)}
            # an odd generator squared: g g = 1/2 [g, g]
            terms, cen = br(key, key)
            out = {}
            for k2, co in terms.items():
                for m2, c2 in self.act(k2, rest).items():
                    out[m2] = out.get(m2, 0) + sympy.Rational(1, 2) * co * c2
            if cen != 0:
                out[rest] = out.get(rest, 0) + sympy.Rational(1, 2) * cen
            return out

        sign = -1 if (podd and PARITY[head[0]] == 1) else 1
        out = {}
        for m1, c1 in self.act(key, rest).items():
            for m2, c2 in self.act(head, m1).items():
                out[m2] = out.get(m2, 0) + sign * c1 * c2
        terms, cen = br(key, head)
        for k2, co in terms.items():
            for m2, c2 in self.act(k2, rest).items():
                out[m2] = out.get(m2, 0) + co * c2
        if cen != 0:
            out[rest] = out.get(rest, 0) + cen
        return out

    def apply_word(self, word, elt):
        cur = elt
        for key in reversed(word):
            nxt = {}
            for m, c in cur.items():
                for m2, c2 in self.act(key, m).items():
                    nxt[m2] = nxt.get(m2, 0) + c * c2
            cur = {m: sympy.expand(c) for m, c in nxt.items() if sympy.expand(c) != 0}
        return cur

    def pairing(self, u, w):
        sig = [sigma(k) for k in reversed(u)]
        res = self.apply_word(sig, {w: sympy.Integer(1)})
        return sympy.expand(res.get((), sympy.Integer(0)))


# ---------------------------------------------------------------------------
#  the PBW basis of a graded piece
# ---------------------------------------------------------------------------

def lowering(kmax):
    """the lowering generators with N <= kmax, in the PBW order."""
    out = []
    for name in NAMES:
        for m in range(-kmax, 1):
            key = (name, m)
            if role(key) == "-" and -m <= kmax:
                out.append(key)
    out.sort(key=ordkey, reverse=True)
    return out


def monomials(kmax, qmin, target):
    """all PBW monomials of the given (N, B, Q)."""
    gens = lowering(kmax)
    N0, B0, Q0 = target
    out = []

    def rec(i, N, B, Q, acc):
        if (N, B, Q) == (N0, B0, Q0):
            out.append(tuple(acc))
        if i >= len(gens):
            return
        if N > N0:
            return
        g = gens[i]
        n, b, q = level(g)
        if PARITY[g[0]] == 1:
            rec(i + 1, N, B, Q, acc)
            rec(i + 1, N + n, B + b, Q + q, acc + [g])
        else:
            r = 0
            while N + r * n <= N0 and r <= 8:
                rec(i + 1, N + r * n, B + r * b, Q + r * q, acc + [g] * r)
                if n == 0 and abs(Q + r * q) > abs(Q0) + 4:
                    break
                r += 1
    rec(0, 0, 0, 0, [])
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    uniq.sort()
    return uniq


def gram(V, mons):
    """the full Gram matrix -- every entry computed, symmetry never assumed."""
    n = len(mons)
    M = sympy.zeros(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = V.pairing(mons[i], mons[j])
    return M


def sym_defect(M):
    """how far the Gram matrix is from being symmetric."""
    n = M.rows
    bad = []
    for i in range(n):
        for j in range(n):
            if sympy.simplify(M[i, j] - M[j, i]) != 0:
                bad.append((i, j))
    return bad


def parity(mono):
    return sum(PARITY[k[0]] for k in mono) % 2


def show(x):
    return sympy.factor(sympy.simplify(x))


if __name__ == "__main__":
    log = open("run-shsvect.log", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")
        log.flush()

    V = Verma()

    say("=== the triangular decomposition (chamber 0 < mu < 1) ===")
    for name in NAMES:
        rs = {m: role((name, m)) for m in range(-2, 3)}
        say("   %-3s  %s" % (name, "  ".join("%d:%s" % (m, r) for m, r in rs.items())))

    say("\n=== the graded pieces (N, B, Q) and their Shapovalov determinants ===")
    #  the pieces are listed by the LL_0-weight drop  N - B*mu, smallest first
    TARGETS = [
        (0, 0, -2), (0, 0, -4), (0, 0, -6),
        (0, -1, 1), (0, -1, -1),
        (0, -2, 0),
        (1, 1, 1), (1, 1, -1),
        (1, 0, 0), (1, 0, 2), (1, 0, -2),
        (1, -1, 1), (1, -1, -1),
    ]
    if len(sys.argv) > 1:
        TARGETS = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]]
    import time
    for tgt in TARGETS:
        t0 = time.time()
        mons = monomials(2, -8, tgt)
        if not mons:
            say("   (N,B,Q) = %-12s empty" % (str(tgt),))
            continue
        M = gram(V, mons)
        bad = sym_defect(M)
        d = sympy.factor(sympy.simplify(M.det()))
        say("   (N,B,Q) = %-12s dim %d  %s  det = %s   [%.1f s]"
            % (str(tgt), len(mons),
               "symmetric" if not bad else "NOT symmetric at %s" % bad[:3],
               d, time.time() - t0))
        for m in mons:
            say("        %s" % (" ".join("%s_%d" % (k[0], k[1]) for k in m) or "1",))
    log.close()
