"""
blam.py -- the dictionary between Kochetkov's b_{a,b}(n) = M_{a,b}(n), given as the line
C(az+bh) added to spe(n) in the Cartan prolong (ba(n), spe(n)+C(az+bh))_*, and the
one-parameter family b_lambda(n) of [LKae] / of the survey [LSh].

The survey singles b_{A,B}(n) out inside m(n) by

    A Div M_g = (-1)^{p(g)} 2(A - Bn) d g/d tau,   b_{A,B}(n) = b_lambda(n),
    lambda = 2A/(n(A-B)),                                          (survey, (2.7.2)-(2.7.3))

and Kochetkov's own Remark says the same thing about his (a,b):

    Div(h) = (-1)^{p(h)} ((nb+a)/a) [1,h],   and   [1,h]_M = 2 d h/d tau.

Both right-hand sides are the intrinsic multiplier of L_X alpha = g_X alpha (same alpha in
both papers), so matching the two conditions gives (A:B) = (a:-b) and

    lambda = 2a/(n(a+b)).

What is computed here, in Kochetkov's realisation and with exact arithmetic:

  1) his Remark itself: the ratio kappa with Div X = kappa (-1)^{p(f)} 2 d_tau f is
     constant on the whole of b_{a,b}(n) and equals (nb+a)/a  -- checked degree by degree;
  2) the two anchors of the dictionary: a = 0 gives lambda = 0 (= B(n), the central
     extension of le(n)) and a + nb = 0 gives lambda = 2/(n-1) (= sm(n), divergence-free);
  3) the third, independent, check: b_lambda(n) is non-simple exactly at lambda = 0, 1,
     infty, i.e. -- under the dictionary -- at a = 0, at 2a = n(a+b) and at a+b = 0; we
     test this by computing the codimension of the derived algebra degree by degree.

usage:  python3 blam.py [n] [dmax]
"""

import sys
from fractions import Fraction as F

from le import is_hom_par
from mn import Mvf, dtau, layer, mbr, mdeg, one
from mimod import Mn, Mab


def kappa(n, a, b, dmax=3):
    """the constant kappa with Div X = kappa (-1)^{p(f)} 2 d_tau f on b_{a,b}(n), or None"""
    alg = Mab(Mn(n), F(a), F(b))
    rows = []
    for j in range(-2, dmax + 1):
        for f in alg.layer(j):
            p = is_hom_par(f)
            d = Mvf(f, n).div()
            t = {m: 2 * ((-1) ** p) * c for m, c in dtau(f, n).items()}
            for m in set(d) | set(t):
                rows.append((d.get(m, F(0)), t.get(m, F(0))))
    k = None
    for u, v in rows:
        if v:
            k = F(u, 1) / v
            break
    if k is None:
        return "infty" if any(u for u, v in rows) else None   # Div == 0 identically
    return k if all(u == k * v for u, v in rows) else "not constant"


def lam(n, a, b):
    """lambda = 2a/(n(a+b))"""
    d = n * (F(a) + F(b))
    return "infty" if d == 0 else F(2 * a, 1) / d


def derived_codim(n, a, b, dmax=4):
    """codim of [g,g] in g, degree by degree (a codim-1 ideal shows up as a single 1)"""
    from mgencheck import span_dim
    alg = Mab(Mn(n), F(a), F(b))
    out = {}
    for j in range(-2, dmax + 1):
        tgt = alg.layer(j)
        if not tgt:
            continue
        amb = [next(iter(f)) for f in layer(n, j)]
        gen = []
        for p in range(-2, j + 3):
            q = j - p
            if q < -2 or q > dmax + 2:
                continue
            for f in alg.layer(p):
                for g in alg.layer(q):
                    x = mbr(f, g, n)
                    if x and mdeg(x, n) == j:
                        gen.append(x)
        out[j] = len(tgt) - span_dim(gen, amb)
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("n = %d, degrees -2..%d\n" % (n, dmax))
    print("1) Kochetkov's Remark:  Div X = kappa (-1)^p 2 d_tau f,  kappa = (nb+a)/a ?")
    tests = [(0, 1), (1, 0), (1, 1), (1, -1), (2, 1), (1, 2), (3, 1), (1, 3),
             (n, -1), (2, -1), (n, 1), (1, -2)]
    for a, b in tests:
        k = kappa(n, a, b, dmax)
        pred = "infty" if a == 0 else F(n * b + a, a)
        ok = "ok" if str(k) == str(pred) else "MISMATCH"
        print("   (a,b) = (%2s,%2s):  kappa = %-8s  (nb+a)/a = %-8s  %s  ->  lambda = %s"
              % (a, b, k, pred, ok, lam(n, a, b)))
    print("\n2) anchors")
    print("   a = 0      -> lambda = %s   (want 0, B(n) = b_0(n))" % lam(n, 0, 1))
    print("   a + nb = 0 -> lambda = %s   (want %s, sm(n) = b_{2/(n-1)}(n))"
          % (lam(n, n, -1), F(2, n - 1)))
    print("\n3) non-simplicity: codim of [g,g] in g, degree by degree")
    for a, b in [(0, 1), (n, -1), (1, -1), (1, 0), (2, 1), (1, 2)]:
        print("   (a,b) = (%2s,%2s), lambda = %-8s : %s"
              % (a, b, lam(n, a, b), derived_codim(n, a, b, dmax)))
