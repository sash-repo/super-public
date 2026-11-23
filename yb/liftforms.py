"""Table (19) and Theorem 5.9: the restriction c |-> c|_a of the 2-cocycles of the eight
distinguished stringy superalgebras to their abelian subalgebras a of Examples 5.20.

For each row of Table (19), the degree-0 2-cocycles are computed in a Laurent window
(coc.report: the cocycle identity is the Jacobi identity of the central extension, no sign
convention is fixed by hand), and the restriction map c |-> c|_a is applied to the whole space
of cocycles, not to one representative.  Every class has a representative of degree 0 (the
grading element is inner), and a coboundary restricts to 0 on an abelian a (checked: the line
"coboundaries (must be 0)"), so the rank printed on the line "all cocycles" is the rank of the
restriction map on H^2(g).  Only the modes in the core of the window are used.

Expected: rank 1 for k^L(1|3), {k^L}'(1|4), k^M(1|3), k^M(1|4), with the values
c(K_{t^a theta_S}, K_{t^b theta_T}) non-zero exactly for S = T and a + b = -1 (Neveu--Schwarz),
a + b = 0 (Ramond); rank 0 for k^L(1|2), vect^L(1|2), svect^L_lambda(1|2), {svect^L}'(1|2).
Keys: (a, S) is K_{t^a theta_S} (in the Ramond sector the index 0 is zeta); (mu, a, S) is
t^a theta_S d_mu; for svect, (d, i) is the i-th basis vector of the degree-d component.

Run:  python3 liftforms.py            (all eight rows)
      python3 liftforms.py kL2 kL3 ... (a selection: kL2 kL3 kL4 kM3 kM4 vect svect svectp)
"""
import os
import sys
import time
from fractions import Fraction as Q
from itertools import combinations

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"),
                os.path.dirname(os.path.abspath(__file__))]
from yb import Contact, Ramond, Vect, SVect                                    # noqa: E402
from coc import report, backend, rank                                          # noqa: E402
from coc2 import (Derived, Spanned, sv_family, form_span, form_span_vec,       # noqa: E402
                  shape, derived_missing)


def contact_row(alg, A, B, fam_keys, label):
    """rows k^L(1|3), {k^L}'(1|4), k^M(1|3), k^M(1|4): the family is a monomial span"""
    print("  %s:  family %s, %d basis elements, %d pairs"
          % (alg.name, label, len(fam_keys), len(fam_keys) * (len(fam_keys) + 1) // 2))
    P, mat, basis, cob, dh = report(alg, A, B)
    form_span(alg, P, cob, fam_keys, label, "coboundaries (must be 0)")
    form_span(alg, P, basis, fam_keys, label, "all cocycles")
    shape(alg, P, basis, fam_keys, label)


def kL2(A=10, B=5):
    """k^L(1|2): a = C[t^-1,t](theta_1 +- i theta_2).  For x = u(theta_1 +- i theta_2),
    y = v(theta_1 +- i theta_2) and c real, c(x, y) = (c11 - c22) +- i(c12 + c21), where
    cij = c(K_{u theta_i}, K_{v theta_j}); so c|_a = 0 iff c11 = c22 and c12 + c21 = 0."""
    alg = Contact(2)
    print("  %s:  family u(t)(theta_1 +- i theta_2)" % alg.name)
    P, mat, basis, cob, dh = report(alg, A, B)
    C = B - 1

    def val(vec, k1, k2):
        hit = P.lookup(k1, k2)
        if hit == "zero":
            return Q(0)
        assert hit != "outside", (k1, k2)
        return vec[hit[0]] * hit[1]

    pairs = [(a, b) for a in range(-C, C + 1) for b in range(-C, C + 1) if a <= b]
    for vecs, tag in ((cob, "coboundaries (must be 0)"), (basis, "all cocycles")):
        rows = []
        for v in vecs:
            row = []
            for a, b in pairs:
                f1, f2, g1, g2 = (a, (1,)), (a, (2,)), (b, (1,)), (b, (2,))
                row.append(val(v, f1, g1) - val(v, f2, g2))
                row.append(val(v, f1, g2) + val(v, f2, g1))
            rows.append(row)
        r = rank(rows) if rows else 0
        print("      %d pairs of modes |a|, |b| <= %d | %s: rank of the restriction map = %d"
              % (len(pairs), C, tag, r))


def kL3():
    contact_row(Contact(3), 8, 4, [(a, (1, 2, 3)) for a in range(-3, 4)], "theta_1 theta_2 theta_3")


def kM3():
    contact_row(Ramond(3), 8, 4, [(a, (0, 1, 2)) for a in range(-3, 4)], "zeta theta_1 theta_2")


def kL4():
    # the derived algebra: k^L(1|4) without the one monomial K_{t^-1 theta_1...theta_4}
    # (`missing` below shows that it is the only one, in the degrees -12..12)
    base = Contact(4)
    miss = derived_missing(base, range(-12, 13), D=14)
    print("  monomials of k^L(1|4) outside [k^L(1|4), k^L(1|4)], degrees -12..12: %s" % miss)
    g = Derived(base, miss, "{k^L}'(1|4)")
    fam = [(a, S) for a in range(-3, 4) for S in combinations((1, 2, 3, 4), 3)]
    contact_row(g, 8, 4, fam, "Lambda^3(theta)")


def kM4():
    base = Ramond(4)
    miss = derived_missing(base, range(-12, 13), D=14)
    print("  monomials of k^M(1|4) outside [k^M(1|4), k^M(1|4)], degrees -12..12: %s"
          % (miss if miss else "none (k^M(1|4) is perfect there)"))
    fam = [(a, S) for a in range(-3, 4) for S in [(0, 1, 2), (0, 1, 3), (0, 2, 3)]]
    contact_row(base, 8, 4, fam, "zeta Lambda^2(theta)")


def vect():
    # a = C[t^-1,t] theta_1 theta_2 d_theta_i, i = 1 and 2 together (also the pairs i =/= j)
    keys = [(i, a, (1, 2)) for i in (1, 2) for a in range(-3, 4)]
    contact_row(Vect(2), 8, 4, keys, "theta_1 theta_2 d_theta_i")


def _svect_rows(lam, derived):
    # windows of the DEGREE for svect (deg X^{(i)}_a = 2a - 1); only core modes are used
    A, B = 12, 6
    modes = [a for a in range(-6, 7) if abs(2 * a - 1) <= B - 1]
    g = SVect(lam)
    S = Spanned(g, "svect^L_{lambda=%s}(1|2)" % lam, range(-2 * A, 2 * A + 1))
    alg = S
    if derived:
        miss = derived_missing(S, range(-6, 7), D=8)
        print("  svect^L_{lambda=%s}(1|2): basis vectors outside the derived algebra, "
              "degrees -6..6: %s" % (lam, miss))
        alg = Derived(S, miss, "{svect^L}'(1|2) (lambda = %s)" % lam)
    elts = []
    for i in (1, 2):
        for a in modes:
            v = sv_family(g, a, i)
            d = g.deg(next(iter(v)))
            assert all(g.deg(k) == d for k in v)
            e = S.coords(v, d)
            if derived:
                assert not set(e) & alg.drop, "X^(%d)_%d is not in the derived algebra" % (i, a)
            elts.append(e)
    print("  %s:  family X^(i)_a, i = 1, 2, a in %s: %d elements, %d pairs"
          % (alg.name, modes, len(elts), len(elts) * (len(elts) + 1) // 2))
    P, mat, basis, cob, dh = report(alg, A, B)
    form_span_vec(P, cob, elts, "the family", "coboundaries (must be 0)")
    form_span_vec(P, basis, elts, "the family", "all cocycles")


def svect():
    for lam in (Q(1, 3), Q(1, 2), Q(-1, 3)):
        _svect_rows(lam, derived=False)


def svectp():
    # for lambda in Z the list contains the derived algebra; lambda = 0 and lambda = 1
    for lam in (Q(0), Q(1)):
        _svect_rows(lam, derived=True)


ROWS = [("kL2", "k^L(1|2)", kL2), ("kL3", "k^L(1|3)", kL3), ("kL4", "{k^L}'(1|4)", kL4),
        ("kM3", "k^M(1|3)", kM3), ("kM4", "k^M(1|4)", kM4), ("vect", "vect^L(1|2)", vect),
        ("svect", "svect^L_lambda(1|2)", svect), ("svectp", "{svect^L}'(1|2)", svectp)]

if __name__ == "__main__":
    want = sys.argv[1:] or [r[0] for r in ROWS]
    print("backend: %s" % backend())
    for tag, name, fn in ROWS:
        if tag in want:
            t0 = time.time()
            print("\n=== %s ===" % name)
            fn()
            print("  [%.1f s]" % (time.time() - t0), flush=True)
