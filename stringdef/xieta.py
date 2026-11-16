"""k^L(1|2) in the HYPERBOLIC frame alpha_1 = dt + xi d eta + eta d xi.

yb.Contact(2) works in the orthonormal frame alpha_1 = dt + sum theta_i
d theta_i.  The two are equivalent over C, but the second divergence Div_2 of
k(1|2) -- the one that exists for this algebra alone, k(1|2)_0 being
commutative of dimension 2 -- is written in the xi-eta frame,
and its operators Kt_xi, Kt_eta are frame-specific.  So the frame is built
here from scratch, and the first thing asked of it is that it reproduce the
central charge computed in the other frame.

Odd labels: 1 = xi, 2 = eta.
"""

from fractions import Fraction as Q
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from itertools import combinations
from stringy import add, mul, smul, mono, d_t, d_odd, parity


class ContactXiEta:
    name = "k^L(1|2) [xi-eta]"

    def __init__(self, sigma=1):
        self.n = 2
        self.odd = (1, 2)
        self.sigma = sigma

    def nabla(self, h):
        out = {}
        for (a, S), c in h.items():
            v = (2 - len(S)) * c
            if v:
                out[(a, S)] = v
        return out

    def brk(self, f, g):
        res = add(mul(self.nabla(f), d_t(g)), smul(-1, mul(d_t(f), self.nabla(g))))
        s = add(mul(d_odd(1, f), d_odd(2, g)), mul(d_odd(2, f), d_odd(1, g)))
        return add(res, smul(self.sigma * (-1) ** parity(f), s))

    def par(self, key):
        return len(key[1]) % 2

    def expo(self, key):
        return key[0]

    def deg(self, key):
        a, S = key
        return 2 * a + len(S) - 2

    def keys(self, arange, parities=(0, 1)):
        out = []
        for a in arange:
            for r in range(3):
                for S in combinations(self.odd, r):
                    if r % 2 in parities:
                        out.append((a, S))
        return out

    def homog_keys(self, d, parities=(0, 1)):
        out = []
        for r in range(3):
            if (d + 2 - r) % 2:
                continue
            a = (d + 2 - r) // 2
            for S in combinations(self.odd, r):
                if r % 2 in parities:
                    out.append((a, S))
        return out

    def check_jacobi(self, span=range(-2, 3)):
        ks = self.keys(span)
        bad = tested = 0
        for kf in ks:
            f = {kf: Q(1)}
            for kg in ks:
                g = {kg: Q(1)}
                for kh in ks:
                    h = {kh: Q(1)}
                    tested += 1
                    acc = {}
                    for (x, y, z) in ((f, g, h), (g, h, f), (h, f, g)):
                        s = (-1) ** (parity(x) * parity(z))
                        acc = add(acc, smul(s, self.brk(self.brk(x, y), z)))
                    if any(v for v in acc.values()):
                        bad += 1
        return bad, tested
