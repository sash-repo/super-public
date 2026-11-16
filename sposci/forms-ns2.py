#!/usr/bin/env python3
r"""forms-ns2.py -- the analog of Proposition 7.3 for two odd
indeterminates: a wide sweep for invariant bilinear forms on cF_{lambda,nu;mu}
over k^L(1|2) and on cF^M_{lambda;mu} over k^M(1|2), at two windows."""

from fractions import Fraction as Q

from ns2 import invariant_forms, form_report, act_L, act_M

CZ, CT = Q(1), Q(1)
LAMS = [Q(l, 4) for l in range(-4, 9)] + [Q(1, 3), Q(2, 3), Q(1, 5), Q(3, 7)]
NUS = [Q(n, 4) for n in range(-4, 9)] + [Q(1, 3), Q(2, 3), Q(1, 5)]
MUS = [Q(0), Q(1, 5), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4)]


def actM(par, f, u):
    return act_M(par, f, u, CZ, CT)


def sweep(name, act, pars, N):
    hits = []
    for par in pars:
        forms, _ = invariant_forms(act, par, par, N=N)
        for form in forms:
            p, eps, nd, _ = form_report(form, N)
            hits.append((par, len(forms), p, tuple(sorted(map(str, eps))), nd))
    return hits


def main():
    parsL = [(l, n, m) for l in LAMS for n in NUS for m in MUS]
    parsM = [(l, m) for l in LAMS for m in MUS]
    for N in (5, 6):
        print("=" * 70)
        print("window |a| <= %d" % N)
        hs = sweep("L", act_L, parsL, N)
        print("k^L(1|2): %d (lambda, nu, mu) triples tried, %d carry a form"
              % (len(parsL), len(hs)))
        for h in hs:
            print("   lambda=%-6s nu=%-6s mu=%-5s dim=%d parity=%s eps=%s nondeg=%s"
                  % (h[0][0], h[0][1], h[0][2], h[1], h[2], h[3], h[4]))
        hs = sweep("M", actM, parsM, N)
        print("k^M(1|2): %d (lambda, mu) pairs tried, %d carry a form"
              % (len(parsM), len(hs)))
        for h in hs:
            print("   lambda=%-6s mu=%-5s dim=%d parity=%s eps=%s nondeg=%s"
                  % (h[0][0], h[0][1], h[1], h[2], h[3], h[4]))


if __name__ == "__main__":
    main()
