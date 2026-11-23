"""
The central extensions of the distinguished stringy Lie superalgebras that are not spanned by
the monomials K_{t^a theta_S}, and the restriction of their cocycles to abelian subalgebras.

`coc.py` computes the degree-0 2-cocycles of an algebra given by monomial keys.  Three kinds of
algebra need more:

  * the DERIVED algebras.  The derived algebra of a monomial algebra is again spanned by
    monomials whenever all but finitely many monomials lie in it; `derived_missing` finds the
    monomials that do not, degree by degree, and `Derived` deletes those keys.  In k^L(1|4)
    exactly one monomial is missing, K_{t^{-1} theta_1 theta_2 theta_3 theta_4}; in k^M(1|4),
    in the realization of yb.py, none is.

  * vect^L(1|2) and m^L(1), whose bases are monomial (f d_mu, resp. M_f); for them coc.py
    applies verbatim, the Laurent exponent of a key being read off by the `expo` methods of
    yb.py.

  * svect^L_lambda(1|2) = Ker div_lambda, whose homogeneous components are kernels of a linear
    map and NOT spanned by monomials.  `Spanned` keys the basis by (degree, index) and
    re-expresses every bracket in that basis.  For `Spanned` the windows A, B of coc.py are
    windows of the DEGREE, not of the Laurent exponent (the two bound each other,
    deg = 2a + |S| - shift).

Over a distinguished g, a rank-1 solution r = x (x) x of the CYBE is a solution with values in
the central extension g^ if and only if c(x,x) = 0, c the cocycle; on an abelian subalgebra of
homological elements c is a symmetric form depending only on the class of c (Theorem 5.9).

Run:  python3 coc2.py      (Table (19) itself, all eight rows, is liftforms.py)
"""

from fractions import Fraction as Q
from itertools import combinations
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"),
                os.path.dirname(os.path.abspath(__file__))]

from yb import Contact, Ramond, Vect, SVect, MPer                  # noqa: E402
from coc import report, quadratic_form, backend                    # noqa: E402

try:
    from exactlin import rank, solve
except ImportError:                                                # pragma: no cover
    rank = solve = None


# --------------------------------------------------------------- the derived
# algebra: which monomials drop out?

def derived_missing(alg, degrees, D=10):
    """The monomials of `alg` that are NOT in [alg, alg], degree by degree.

    The derived algebra is graded, so it is enough to intersect each graded
    component with the span of the brackets landing in it.
    """
    missing = []
    for d in degrees:
        tgt = alg.homog_keys(d)
        if not tgt:
            continue
        idx = {k: i for i, k in enumerate(tgt)}
        rows = []
        for d1 in range(-D, D + 1):
            for k1 in alg.homog_keys(d1):
                for k2 in alg.homog_keys(d - d1):
                    br = alg.brk({k1: Q(1)}, {k2: Q(1)})
                    if not br:
                        continue
                    row = [Q(0)] * len(tgt)
                    for k, v in br.items():
                        row[idx[k]] = Q(v)
                    rows.append(row)
        r = rank(rows) if rows else 0
        if r == len(tgt):
            continue
        for k in tgt:
            e = [Q(0)] * len(tgt)
            e[idx[k]] = Q(1)
            if rank(rows + [e]) > r:
                missing.append(k)
    return missing


class Derived:
    """A subalgebra of a monomial algebra given by deleting some keys."""

    def __init__(self, alg, drop, name):
        self.alg = alg
        self.drop = set(drop)
        self.name = name

    def brk(self, f, g):
        return self.alg.brk(f, g)

    def par(self, key):
        return self.alg.par(key)

    def deg(self, key):
        return self.alg.deg(key)

    def expo(self, key):
        return self.alg.expo(key)

    def keys(self, arange, parities=(0, 1)):
        return [k for k in self.alg.keys(arange, parities) if k not in self.drop]

    def homog_keys(self, d, parities=(0, 1)):
        return [k for k in self.alg.homog_keys(d, parities) if k not in self.drop]


# ------------------------------------------------- a non-monomial basis (svect)

class Spanned:
    """An algebra whose graded components come with an explicit linear basis.

    Keys are (degree, index).  `A` and `B` in coc.report are then windows of
    the degree, not of the Laurent exponent.
    """

    def __init__(self, alg, name, degrees):
        self.alg = alg
        self.name = name
        self.basis = {d: alg.homog_basis(d) for d in degrees}
        self.degrees = sorted(d for d in degrees if self.basis[d])

    # -- coordinates of an ambient vector in the basis of its degree
    def _coords(self, d, vec):
        bas = self.basis.get(d) or []
        if not vec:
            return [Q(0)] * len(bas)
        mons = sorted({m for b in bas for m in b} | set(vec), key=repr)
        A = [[Q(b.get(m, 0)) for b in bas] for m in mons]
        rhs = [Q(vec.get(m, 0)) for m in mons]
        x = solve(A, rhs)
        if x is None:
            raise ValueError("bracket left the span in degree %d" % d)
        return list(x)

    def brk(self, f, g):
        out = {}
        for k1, c1 in f.items():
            for k2, c2 in g.items():
                v1, v2 = self.basis[k1[0]][k1[1]], self.basis[k2[0]][k2[1]]
                br = self.alg.brk(v1, v2)
                if not br:
                    continue
                d = k1[0] + k2[0]
                for i, x in enumerate(self._coords(d, br)):
                    if x:
                        out[(d, i)] = out.get((d, i), Q(0)) + c1 * c2 * x
        return {k: v for k, v in out.items() if v}

    def _parity(self, d, i):
        vec = self.basis[d][i]
        return self.alg.par(next(iter(vec)))

    def par(self, key):
        return self._parity(*key)

    def deg(self, key):
        return key[0]

    def expo(self, key):
        return key[0]

    def keys(self, arange, parities=(0, 1)):
        out = []
        for d in arange:
            for i in range(len(self.basis.get(d) or [])):
                if self._parity(d, i) in parities:
                    out.append((d, i))
        return out

    def homog_keys(self, d, parities=(0, 1)):
        return [(d, i) for i in range(len(self.basis.get(d) or []))
                if self._parity(d, i) in parities]

    def coords(self, vec, d):
        """the key-basis expression of an ambient vector of degree d."""
        return {(d, i): x for i, x in enumerate(self._coords(d, vec)) if x}


# ------------------------------------------- the abelian family of svect

def sv_family(g, a, i=1):
    """X^{(i)}_a = -/+ (a+lambda) t^{a-1} theta_1 theta_2 d_{theta_i}
                   + t^a theta_{3-i} d_t   in svect^L_lambda(1|2).

    Divergence-free for every lambda, odd, homological, and the two families
    (i = 1, 2) together span an abelian subalgebra of superdimension (0|infty).
    """
    j = 3 - i
    s = 1 if i == 1 else -1
    return {(i, a - 1, (1, 2)): -s * (a + g.lam), ('t', a, (j,)): Q(1)}


# -------------------------------------------------------------- the reporting

def pairing(P, cocycle, x, y):
    """c(x,y) for x, y written in the key basis."""
    tot = Q(0)
    for k1, c1 in x.items():
        for k2, c2 in y.items():
            hit = P.lookup(k1, k2)
            if isinstance(hit, tuple):
                tot += c1 * c2 * cocycle[hit[0]] * hit[1]
    return tot


def form_span_vec(P, vectors, elts, label, tag):
    """Same as `form_span`, but the family is given by arbitrary vectors."""
    pairs = list(combinations(range(len(elts)), 2)) + \
        [(i, i) for i in range(len(elts))]
    rows = [[pairing(P, v, elts[i], elts[j]) for i, j in pairs] for v in vectors]
    r = rank(rows) if rows else 0
    if r == 0:
        print("      %s | %s: c|_a = 0 for EVERY cocycle -- the family lifts "
              "into every central extension" % (label, tag))
        return True
    print("      %s | %s: the restriction map has rank %d -- the extension "
          "obstructs part of the family" % (label, tag, r))
    for row in rows:
        nz = [(pairs[i], x) for i, x in enumerate(row) if x]
        if nz:
            for (i, j), x in nz[:8]:
                print("         c(x_%d, x_%d) = %s" % (i, j, x))
            break
    return False


def form_span(alg, P, vectors, keys, label, tag):
    """The restriction map  c |--> c|_a  applied to every `vectors` element.

    The point of doing it for the WHOLE space of cocycles, and not for one
    representative, is that the answer then does not depend on which central
    extension is meant: if the image is zero, every rank-1 solution supported on
    a lifts into every central extension of the algebra.
    """
    pairs = list(combinations(keys, 2)) + [(k, k) for k in keys]
    rows = []
    for v in vectors:
        q = quadratic_form(alg, P, v, keys)
        rows.append([q.get(kk, Q(0)) for kk in pairs])
    r = rank(rows) if rows else 0
    if r == 0:
        print("      %s | %s: c|_a = 0 for EVERY cocycle -- the family lifts "
              "into every central extension" % (label, tag))
        return True
    print("      %s | %s: the restriction map has rank %d, so some extensions "
          "do obstruct; e.g." % (label, tag, r))
    for row, v in zip(rows, vectors):
        nz = [(kk, x) for kk, x in zip(pairs, row) if x]
        if nz:
            for kk, x in nz[:6]:
                print("         c%s = %s" % (kk, x))
            break
    return False


def run(alg, A, B, families, C=None, show=False):
    print("  %s:" % alg.name)
    P, mat, basis, cob, dh = report(alg, A, B, C)
    for label, keys in families:
        # a is abelian, so every coboundary restricts to 0 there: check it
        form_span(alg, P, cob, keys, label, "coboundaries (must be 0)")
        ok = form_span(alg, P, basis, keys, label, "all cocycles")
        if show and not ok:
            shape(alg, P, basis, keys, label)
    return P, basis, cob


def shape(alg, P, vectors, keys, label):
    """Print the whole symmetric form on `keys` for a representative that is
    non-zero there, so that its shape (which pairs, which constant) is seen."""
    for v in vectors:
        q = quadratic_form(alg, P, v, keys)
        nz = {kk: x for kk, x in q.items() if x}
        if nz:
            vals = sorted({x for x in nz.values()})
            print("      %s: %d non-zero pairs, values %s" % (label, len(nz), vals))
            for kk in sorted(nz)[:40]:
                print("         c%s = %s" % (kk, nz[kk]))
            return
    print("      %s: the form is identically zero" % label)


def main():
    print("backend: %s" % backend())

    print()
    print("=== which monomials are missing from the derived algebra? ===")
    for alg, rng in ((Contact(4), range(-12, 13)), (Ramond(4), range(-12, 13))):
        miss = derived_missing(alg, rng, D=14)
        print("  %s (degrees %d..%d): %s"
              % (alg.name, rng[0], rng[-1],
                 miss if miss else "none -- the algebra is perfect there"))

    print()
    print("=== {k^L}'(1|4): a = Lambda^3(theta) (x) C[t^-1,t] ===")
    kL4 = Derived(Contact(4), [(-1, (1, 2, 3, 4))], "{k^L}'(1|4)")
    fam = [("Lambda^3", [(a, S) for a in range(-2, 3)
                         for S in combinations((1, 2, 3, 4), 3)])]
    run(kL4, 6, 3, fam, show=True)

    print()
    print("=== {k^M}'(1|4) = k^M(1|4): a = zeta Lambda^2(theta) (x) C[t^-1,t] ===")
    kM4 = Ramond(4)
    fam = [("zeta Lambda^2", [(a, S) for a in range(-2, 3)
                              for S in [(0, 1, 2), (0, 1, 3), (0, 2, 3)]])]
    run(kM4, 6, 3, fam, show=True)

    print()
    print("=== m^L(1): a = {M_{f(q)}} and {M_{f(q) tau xi}} ===")
    m1 = MPer()
    fam = [("M_{q^a}", [(a, ()) for a in range(-3, 4)]),
           ("M_{q^a tau xi}", [(a, (1, 2)) for a in range(-3, 4)])]
    run(m1, 8, 4, fam)

    print()
    print("=== vect^L(1|2): a = {a(t) theta_1 theta_2 d_{theta_i}} ===")
    v2 = Vect(2)
    fam = [("theta_1 theta_2 d_{theta_1}",
            [(1, a, (1, 2)) for a in range(-3, 4)]),
           ("theta_1 theta_2 d_{theta_2}",
            [(2, a, (1, 2)) for a in range(-3, 4)])]
    run(v2, 8, 4, fam)

    print()
    print("=== svect^L_lambda(1|2) and {svect^L}'(1|2) ===")
    print("    a = span(X^{(i)}_a), X^{(i)}_a = -/+(a+lambda) t^{a-1} th1 th2 "
          "d_{th_i} + t^a th_{3-i} d_t")
    # For `Spanned` the windows are windows of the DEGREE.  Only the modes
    # whose degree (2a - 1) lies inside the core are used: a form that is
    # non-zero only at the edge of the window is an artifact of the truncation,
    # no equation reaching there.  (With A = 8, B = 4 and the modes -3..3 one
    # does see such an artifact, at c(X_{-2}, X_2); it disappears here.)
    A, B = 12, 6
    modes = [a for a in range(-6, 7) if abs(2 * a - 1) <= B - 1]
    print("    windows: |deg| <= %d for the unknowns, <= %d for the equations, "
          "core <= %d; modes %s" % (A, B, B - 1, modes))
    for lam in (Q(1, 3), Q(1), Q(0), Q(2)):
        g = SVect(lam)
        S = Spanned(g, "svect^L_{lambda=%s}(1|2)" % lam, range(-2 * A, 2 * A + 1))
        elts = []
        for i in (1, 2):
            for a in modes:
                v = sv_family(g, a, i)
                d = g.deg(next(iter(v)))
                assert all(g.deg(k) == d for k in v)
                elts.append(S.coords(v, d))
        print("  %s:" % S.name)
        P, mat, basis, cob, dh = report(S, A, B)
        form_span_vec(P, cob, elts, "the family", "coboundaries (must be 0)")
        form_span_vec(P, basis, elts, "the family", "all cocycles")
        if lam == 0:
            miss = derived_missing(S, range(-6, 7), D=8)
            print("    {svect^L}'(1|2): basis vectors outside the derived "
                  "algebra: %s" % (miss if miss else "none in this range"))


if __name__ == "__main__":
    main()
