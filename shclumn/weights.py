"""Which weights carry cohomology.

If g has an INNER grading element E (an E in g_0 with [E,x] = (deg x)x), then
Cartan's formula L_E = d iota_E + iota_E d makes the identity of the weight-w
part chain homotopic to multiplication by w, so H^*(g;g) sits in weight 0
alone.  Where E is only an outer derivation -- which, among our algebras, is
exactly svect^L_lambda(1|2) for generic lambda -- the other weights are free to
carry classes, and they are what separates the coefficient rings:

    Laurent POLYNOMIALS  C[t,t^{-1}]  ->  C^2 = sum over weights,
    formal Laurent SERIES C((t))      ->  C^2 = PRODUCT over weights.

A class in a single nonzero weight lives in both.  What only the completion can
hold is a class whose weight components are infinitely many; and that can only
happen if infinitely many weights carry cohomology at all.  So this table is
the test of whether the series answer can differ from the polynomial one.

Run:  ../venv/bin/python -u weights.py
"""

from fractions import Fraction as Q
import sys, time

from defco import Graded, h2_core, algebras
import yb

import os
LOG = open(os.devnull, "w")


def say(s):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def row(g, N, M, ws):
    hs = []
    for w in ws:
        t0 = time.time()
        r = h2_core(g, N, M, weight=w)
        hs.append(r["h"])
    say("  %-22s E %-5s  N=%d  h(w) for w=%s:  %s"
        % (g.name, "inner" if g.grading_element() is not None else "OUTER",
           N, list(ws), hs))
    return hs


def main():
    say("\n" + "=" * 78)
    say("H^2 by weight   %s" % time.strftime("%Y-%m-%d %H:%M"))
    say("=" * 78)
    ws = (-2, -1, 0, 1, 2)

    say("\n--- algebras with an inner grading element: only w = 0 may be nonzero")
    for g in [Graded(yb.Witt(), "witt"),
              Graded(yb.Contact(1), "k^L(1|1)"),
              Graded(yb.Contact(2), "k^L(1|2)"),
              Graded(yb.Ramond(1), "k^M(1|1)"),
              Graded(yb.Ramond(2), "k^M(1|2)"),
              Graded(yb.Vect(1), "vect^L(1|1)"),
              Graded(yb.MPer(), "m^L(1)")]:
        row(g, 5, 2, ws)

    say("\n--- svect^L_lambda(1|2): the grading element is outer for lambda != 0")
    for lam in (Q(0), Q(1), Q(1, 3), Q(2, 5), Q(-1)):
        g = Graded(yb.SVect(lam), "svect^L_{%s}(1|2)" % lam)
        row(g, 4, 1, ws)

    say("\ndone")


if __name__ == "__main__":
    main()
