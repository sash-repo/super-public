"""
mabloci.py -- are Kochetkov's confined families really confined?

Items 2b), 2c), 3b), 4) of his Theorem 2 are stated on a hypersurface in (a:b): 2b) on
a + nb = 0 (= sm(n)), 2c) on 2a + nb = 0, 3b) and 4) on b = 0.  Items 2a), 3a), 3c) are
stated for the whole family (3a), 3c) with a != 0).  Here each item is tested at its own
weight for a whole range of t = a/b, the weight being read off his own (a,b)-linear
formula -- and, where the formula is written with a special (a,b) already substituted
(item 2b: "lambda + 2n"), at BOTH natural continuations, nu = lambda + 2a and
nu = lambda - 2nb, which agree on the locus.

    usage:  python3 mabloci.py <n> [item ...]
"""

import sys
from fractions import Fraction as F

from mab import scan_mab
from mabsweep2 import TS, ab_of, tname, lam_of


def hdr(s):
    print("\n" + "=" * 78 + "\n" + s)


def line(n, a, b, mu, nu, m, want, tag=""):
    a, b = F(a), F(b)
    print("  t = %-5s (a,b) = (%s,%s)  lambda = %-6s  %s"
          % (tname(F(a, b) if b else 'oo'), a, b, lam_of(n, a, b), tag))
    hits = scan_mab(n, mu, nu, m, a, b, want=want, quiet=True)
    return len(hits)


def run(n, items):
    Z = tuple([0] * (n - 1))
    E = tuple([0] * (n - 2) + [1])
    ts = [t for t in TS]

    if '2a' in items:
        hdr("2a)  mu = 0, nu = lam + 2a, m = 2, wt(y) = (0; lam);  lam = 1 fixed")
        for t in ts:
            a, b = ab_of(t)
            line(n, a, b, list(Z), 1 + 2 * a, 2, (Z, F(1)))

    if '2b' in items:
        # the (a,b)-homogeneous form of his weight: on the locus (a,b) = (n,-1) the module
        # is V_pe(1,...,1,0) (x) C_{c=2}, i.e. nu = a c + b|lambda| = 2a + (n-1)b, and
        # wt(y) has nu(y) = nu - 2a = (n-1)b.  On the locus this is his (lambda+2n; lambda),
        # lambda = 1-n.
        hdr("2b)  mu = E, nu = 2a + (n-1)b, m = 2, wt(y) = (E; (n-1)b)")
        for t in ts:
            a, b = ab_of(t)
            line(n, a, b, list(E), 2 * a + (n - 1) * b, 2, (E, F((n - 1) * b)))

    if '2c' in items and n >= 3:
        hdr("2c)  mu = (0^{n-3},1,0), nu = -2b, m = 2, wt(y) = ((0^{n-2},2); -2(a+b))")
        mu = tuple([0] * (n - 3) + [1, 0])
        wy = tuple([0] * (n - 2) + [2])
        for t in ts:
            a, b = ab_of(t)
            line(n, a, b, list(mu), -2 * b, 2, (wy, F(-2 * (a + b))))

    if '3a' in items:
        hdr("3a)  mu = E, nu = (2n-1)b + 3a, m = 3, wt(y) = (0; 2nb)")
        for t in ts:
            a, b = ab_of(t)
            line(n, a, b, list(E), (2 * n - 1) * b + 3 * a, 3, (Z, F(2 * n * b)))

    if '3c' in items:
        hdr("3c)  mu = 0, nu = 2a, m = 3, wt(y) = ((0^{n-2},1); -(a+b))")
        for t in ts:
            a, b = ab_of(t)
            line(n, a, b, list(Z), 2 * a, 3, (E, F(-(a + b))))

    if '4' in items:
        hdr("4)  mu = E, nu = 3a, m = 4, wt(y) = (E; -a);  (his b=0, a=1 case)")
        for t in ts:
            a, b = ab_of(t)
            line(n, a, b, list(E), 3 * a, 4, (E, F(-a)))


if __name__ == '__main__':
    n = int(sys.argv[1])
    items = sys.argv[2:] or ['2a', '2b', '2c', '3a', '3c', '4']
    print("=== are the confined families confined?  n = %d ===" % n)
    run(n, items)
