"""
The central extensions of the contact stringy superalgebras, from scratch, and
the question they raise for the rank-1 solutions of the CYBE.

A central extension of g by Kee z (z even, central) is a bracket

    [x,y]^ = [x,y] + c(x,y) z

on g + Kee z; the super Jacobi identity in the form used by yb.py,

    [f,[g,h]] = [[f,g],h] + (-1)^{p(f)p(g)} [g,[f,h]],

has as its z-component exactly the 2-cocycle identity

    c(f,[g,h]) = c([f,g],h) + (-1)^{p(f)p(g)} c(g,[f,h]),                (CC)

and superskewness of the bracket forces c(f,g) = -(-1)^{p(f)p(g)} c(g,f).
So no cohomological convention has to be fixed by hand: (CC) *is* the Jacobi
identity of the extended algebra, in the engine's own convention.

We solve (CC) for the cocycles of degree 0 (the grading being the contact one,
deg t = 2, deg theta = 1, deg K_f = deg f - 2) inside a Laurent window, and
quotient by the coboundaries c(x,y) = phi([x,y]), phi a degree-0 functional.

Why this is here: "distinguished" means "possesses a non-trivial central
extension".  For a distinguished g there is therefore a question that a
non-distinguished stringy superalgebra does not have -- whether a rank-1
solution r = x (x) x of the CYBE over g survives in the extension, i.e.,
whether the homological x stays homological in g^: since

    [x,x]^ = [x,x] + c(x,x) z = c(x,x) z,

x is homological in g^ if and only if c(x,x) = 0, and on odd elements c is a
symmetric bilinear form, so this is a genuine quadratic condition.

This file is a library for liftforms.py (Table (19)); its main() is the calibration on witt
followed by H^2_0 of k^L(1|n), k^M(1|n) for n = 1, ..., 5.  Run:  python3 coc.py
"""

from fractions import Fraction as Q
from itertools import combinations, product
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"),
                os.path.dirname(os.path.abspath(__file__))]

from yb import Contact, Ramond, Witt                       # noqa: E402
from stringy import mono, add, smul                        # noqa: E402

try:
    from exactlin import nullspace as _null, rank as _rank, backend
except ImportError:                                        # stand-alone fallback
    from yb import _nullspace as _ns

    def _null(mat):
        return _ns([list(r) for r in mat], len(mat[0]) if mat else 0)

    def _rank(mat):
        if not mat:
            return 0
        return len(mat[0]) - len(_null(mat))

    def backend():
        return "pure-python"


def nullspace(mat, ncols):
    """kernel basis as rows; `ncols` is used when there is no equation at all."""
    if not mat:
        return [[Q(1) if i == j else Q(0) for i in range(ncols)]
                for j in range(ncols)]
    return [list(v) for v in _null(mat)]


def rank(mat):
    return _rank(mat) if mat else 0


# ------------------------------------------------------------------ unknowns

class Pairs:
    """the unknowns c(k1,k2), one per unordered pair of total degree 0."""

    def __init__(self, alg, keys):
        self.alg = alg
        self.index = {}
        self.pairs = []
        for k1, k2 in product(keys, repeat=2):
            self.get(k1, k2)

    def _norm(self, k1, k2):
        """(canonical pair, sign) with c(k1,k2) = sign * u[pair]."""
        alg = self.alg
        if alg.deg(k1) + alg.deg(k2) != 0:
            return None
        p1, p2 = alg.par(k1), alg.par(k2)
        if (p1 + p2) % 2:                       # c is even
            return None
        if k1 == k2 and p1 == 0:                # c(x,x) = -c(x,x) for x even
            return None
        try:
            first = k1 <= k2
        except TypeError:              # heterogeneous key slots (vect: 't' vs 1)
            first = repr(k1) <= repr(k2)
        if first:
            return (k1, k2), 1
        return (k2, k1), -((-1) ** (p1 * p2))

    def get(self, k1, k2):
        nr = self._norm(k1, k2)
        if nr is None:
            return None
        key, sgn = nr
        if key not in self.index:
            self.index[key] = len(self.pairs)
            self.pairs.append(key)
        return self.index[key], sgn

    def lookup(self, k1, k2):
        """(j, sign) | "zero" (the pair is forced to vanish) | "outside"."""
        nr = self._norm(k1, k2)
        if nr is None:
            return "zero"
        key, sgn = nr
        j = self.index.get(key)
        return "outside" if j is None else (j, sgn)

    def __len__(self):
        return len(self.pairs)


def _row_add(row, j, v):
    row[j] = row.get(j, Q(0)) + v


def cocycle_space(alg, A, B, verbose=False):
    """Basis of the space of degree-0 2-cocycles inside the window |a| <= A,
    the identity (CC) being imposed for all triples of keys with |a| <= B whose
    brackets stay inside the window.  Returns (unknowns, basis, skipped)."""
    ukeys = alg.keys(range(-A, A + 1))
    P = Pairs(alg, ukeys)
    ekeys = alg.keys(range(-B, B + 1))
    by_deg = {}
    for k in ekeys:
        by_deg.setdefault(alg.deg(k), []).append(k)

    rows, skipped = [], 0
    for kf, kg in product(ekeys, repeat=2):
        d = -(alg.deg(kf) + alg.deg(kg))        # only total degree 0 matters
        for kh in by_deg.get(d, ()):
            f, g, h = {kf: Q(1)}, {kg: Q(1)}, {kh: Q(1)}
            row, ok = {}, True
            # c(f,[g,h]) - c([f,g],h) - (-1)^{p(f)p(g)} c(g,[f,h]) = 0
            terms = [("L", kf, alg.brk(g, h), Q(1)),
                     ("R", kh, alg.brk(f, g), Q(-1)),
                     ("L", kg, alg.brk(f, h),
                      Q(-1) * (-1) ** (alg.par(kf) * alg.par(kg)))]
            for side, k0, br, sgn in terms:
                for k1, cf in br.items():
                    hit = P.lookup(k0, k1) if side == "L" else P.lookup(k1, k0)
                    if hit == "zero":
                        continue
                    if hit == "outside":
                        ok = False
                        break
                    j, s = hit
                    _row_add(row, j, sgn * s * Q(cf))
                if not ok:
                    break
            if not ok:
                skipped += 1
                continue
            if row:
                rows.append(row)

    # the same equation comes up many times (up to a scalar); keep one copy of
    # each -- the dense matrix is what costs, not the loop above
    seen, uniq = set(), []
    for r in rows:
        r = {j: v for j, v in r.items() if v}
        if not r:
            continue
        js = sorted(r)
        lead = r[js[0]]
        sig = tuple((j, r[j] / lead) for j in js)
        if sig not in seen:
            seen.add(sig)
            uniq.append(r)
    mat = [[r.get(j, Q(0)) for j in range(len(P))] for r in uniq]
    if verbose:
        print("    unknowns %d, equations %d (%d before dedup), skipped %d"
              % (len(P), len(mat), len(rows), skipped))
    basis = nullspace(mat, len(P))
    return P, mat, basis, skipped


def core_columns(alg, P, C):
    """the indices of the pairs both of whose keys sit in the window |a| <= C.
    A truncated computation says nothing about the unknowns near the edge of
    the window (no equation reaches them), so all dimensions are measured on
    the core."""
    expo = getattr(alg, "expo", lambda k: k[0])
    out = []
    for (k1, k2), j in P.index.items():
        if abs(expo(k1)) <= C and abs(expo(k2)) <= C:
            out.append(j)
    return sorted(out)


def restrict(vectors, cols):
    return [[v[j] for j in cols] for v in vectors]


def coboundaries(alg, P, A):
    """the degree-0 coboundaries c(x,y) = phi([x,y]), phi supported in degree 0."""
    zero_keys = [k for k in alg.keys(range(-A, A + 1)) if alg.deg(k) == 0
                 and alg.par(k) == 0]
    out = []
    for kz in zero_keys:
        row = [Q(0)] * len(P)
        for (k1, k2) in P.pairs:
            br = alg.brk({k1: Q(1)}, {k2: Q(1)})
            row[P.index[(k1, k2)]] = Q(br.get(kz, 0))
        if any(row):
            out.append(row)
    return out


def report(alg, A, B, C=None):
    """A = window of the unknowns, B = window of the equations (A >= 2B, so
    that no bracket of the equations leaves the unknown window), C = the core
    on which the dimensions are measured (C <= B)."""
    C = B - 1 if C is None else C
    P, mat, basis, skipped = cocycle_space(alg, A, B, verbose=True)
    cob = coboundaries(alg, P, A)
    cols = core_columns(alg, P, C)
    zc = restrict(basis, cols)
    bc = restrict(cob, cols)
    dz, dc = rank(zc), rank(bc)
    dh = rank(zc + bc) - dc if zc else 0
    print("    on the core |a| <= %d (%d pairs): Z^2_0 = %d, B^2_0 = %d, "
          "H^2_0 = %d" % (C, len(cols), dz, dc, dh))
    return P, mat, basis, cob, dh


# --------------------------------------------------------------- calibration

def virasoro_check(A=10, B=5):
    """Witt: the degree-0 cocycles must be 1-dimensional modulo coboundaries,
    and the class must be the Gelfand-Fuks one,
        c(t^{a+1} d_t, t^{b+1} d_t) = const * (a^3 - a) delta_{a+b,0}.
    Since the coboundaries add multiples of (a - b) delta_{a+b,0}, we test the
    class rather than the representative: the third difference must match."""
    alg = Witt()
    print("  witt:")
    P, mat, basis, cob, dh = report(alg, A, B)
    # the Gelfand-Fuks vector itself must satisfy every equation
    gf = [Q(0)] * len(P)
    for (k1, k2), j in P.index.items():
        m = k1[0] - 1                       # t^{a} d_t = L_{a-1}
        gf[j] = Q(m ** 3 - m)
    bad = 0
    for row in mat:
        if sum(r * v for r, v in zip(row, gf)):
            bad += 1
    print("    Gelfand-Fuks c(L_m,L_{-m}) = m^3-m: %d of %d equations violated"
          % (bad, len(mat)))
    print("    H^2_0 = %d (expected 1)" % dh)
    return bad == 0 and dh == 1


# ------------------------------------------------------- lifting to the ext.

def quadratic_form(alg, P, cocycle, keys):
    """the symmetric form c(x,x) on the odd span of `keys`, as a dict
    (k1,k2) -> value with c(x,y) for x,y odd."""
    out = {}
    for k1, k2 in list(combinations(keys, 2)) + [(k, k) for k in keys]:
        hit = P.lookup(k1, k2)
        out[(k1, k2)] = (cocycle[hit[0]] * hit[1]
                         if isinstance(hit, tuple) else Q(0))
    return out


def lifts(alg, P, cocycle, keys):
    """Does the abelian homological space spanned by `keys` stay homological in
    the central extension?  Yes iff c vanishes identically on it."""
    q = quadratic_form(alg, P, cocycle, keys)
    bad = {kk: v for kk, v in q.items() if v}
    return (not bad), bad


def main():
    print("backend: %s" % backend())
    print()
    print("=== calibration on witt (the Virasoro cocycle) ===")
    virasoro_check()
    print()
    print("=== k^L(1|n): the degree-0 central extensions ===")
    for n in range(1, 6):
        alg = Contact(n)
        B = 5 if n <= 2 else (4 if n == 3 else 3)
        print("  %s:" % alg.name)
        report(alg, 2 * B, B)
    print()
    print("=== k^M(1|n): the degree-0 central extensions ===")
    for n in range(1, 6):
        alg = Ramond(n)
        B = 5 if n <= 2 else (4 if n == 3 else 3)
        print("  %s:" % alg.name)
        report(alg, 2 * B, B)


if __name__ == "__main__":
    main()
