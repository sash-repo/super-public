#!/usr/bin/env python3
"""The census of invariant bilinear forms on cF_{lambda;mu} over ns = k^L(1|1)
and r = k^M(1|1): the analog of Proposition 7.3 for one odd indeterminate.
Every (lambda, mu) on a grid of quarters is tried; the window is raised from 6
to 9 to make sure nothing depends on the truncation."""
from fractions import Fraction as Q
from ns1 import act_NS, act_R, invariant_forms, form_report

LAMS = [Q(k, 4) for k in range(-4, 9)] + [Q(1, 3), Q(2, 3), Q(1, 5), Q(3, 7)]
MUS = [Q(0), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4), Q(1, 5)]

for name, act in (("ns = k^L(1|1)", act_NS),
                  ("r  = k^M(1|1)", lambda l, m, f, u: act_R(l, m, f, u, 1, Q(1)))):
    print("=" * 70)
    print(name)
    print("=" * 70)
    for N in (6, 9):
        print("  window |a| <= %d" % N)
        for lam in LAMS:
            for mu in MUS:
                forms, _ = invariant_forms(act, lam, mu, lam, mu, N=N)
                for f in forms:
                    par, eps, nondeg, dead = form_report(f, N)
                    print("    lambda=%-5s mu=%-4s dim=%d  parity=%s  eps=%s  %s%s"
                          % (lam, mu, len(forms), par, sorted(map(str, eps)),
                             "nondegenerate" if nondeg else "degenerate",
                             "" if nondeg else ", kernel %s" % (dead[:4],)))
