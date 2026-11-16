"""
mabsweep2.py -- the family b_{a,b}(n) as a family, in Kochetkov's own coordinates.

A b_{a,b}(n)_0-module is labelled by (mu; nu): mu the spe(n)-highest weight and nu the
eigenvalue of a z + b h.  Both are intrinsic; the pair (lambda, c) used inside the engine
is only a lift of it.  So the right sweep fixes (mu; nu) -- integers -- and varies
t = a/b over P^1(Q), (a,b) coprime.  (Fixing (lambda, c) instead would confound the
answer: a family that lives at nu = alpha a + beta b would show up at a different c for
every t, and look as if it existed only for special t.)

For every (mu, nu, t) we list all singular vectors of degree <= mmax, over every weight
that can occur, and then group the hits by (mu, m, wt(y) - the mu-part) -- i.e. by "the
family" -- and print, for each group, the map t |-> nu.  If nu is a linear function
alpha a + beta b of the coprime representative (a,b) of t, the group is a family present
for *every* member of b_{a,b}(n), and (alpha, beta) is printed; if the group occurs at a
few values of t only, those t are printed instead.

    usage:  python3 mabsweep2.py <n> <mmax> <mu-hi> <nulo> <nuhi> [t ...]
"""

import sys
from fractions import Fraction as F
from itertools import product

from mn import layer, mbr, mdeg
from mimod import Mn, Mab, MInduced
from mgencheck import span_dim
from mone import weights_of
from mreport import render
from mab import lift, mu_of, nu_of

NRANGE = [0]
NULO = [0]
NUHI = [0]

TS = ['oo', 5, 3, 2, 1, F(1, 2), F(1, 3), 0, F(-1, 3), F(-1, 2), -1, F(-3, 2), -2,
      F(-5, 2), -3, F(-7, 2), -4, -5, F(7, 3)]


def ab_of(t):
    if t == 'oo':
        return F(1), F(0)
    t = F(t)
    return F(t.numerator), F(t.denominator)


def tname(t):
    return 'oo' if t == 'oo' else str(t)


def lam_of(n, a, b):
    """the b_lambda(n)-parameter of [LKae]:  lambda = 2a/(n(a+b))"""
    d = n * (a + b)
    return 'oo' if d == 0 else str(F(2 * a, 1) / d)


def ops_for(mn, a, b):
    n = mn.n
    A = Mab(mn, a, b)
    ops = list(A.layer(1))
    amb = [next(iter(f)) for f in layer(n, 2)]
    gen = []
    for f in A.layer(1):
        for g in A.layer(1):
            x = mbr(f, g, n)
            if x and mdeg(x, n) == 2:
                gen.append(x)
    extra = span_dim(gen, amb) < len(A.layer(2))
    if extra:
        ops += A.layer(2)
    return ops, extra


def mus(n, hi):
    for mu in product(range(hi, -1, -1), repeat=n - 1):
        yield mu


def sweep(n, mmax, hi, nulo, nuhi, ts):
    mn = Mn(n)
    opsmap = {}
    for t in ts:
        a, b = ab_of(t)
        opsmap[tname(t)] = ops_for(mn, a, b)
        if opsmap[tname(t)][1]:
            print("# t = %-5s (lambda = %-5s): L_1 does not generate L_{>=1}; L_2 imposed"
                  % (tname(t), lam_of(n, a, b)))
    hits = {}          # (mu, m, shift) -> {t: [(nu, dim, text)]}
    for mu in mus(n, hi):
        for nu in range(nulo, nuhi + 1):
            for t in ts:
                a, b = ab_of(t)
                ops, _ = opsmap[tname(t)]
                lam, c = lift(n, mu, nu, a, b)
                I = MInduced(mn, lam, c, ops)
                ws = weights_of(I.V, mn.pe)
                for m in range(1, mmax + 1):
                    rhos = set()
                    for u in I.umonos(m):
                        w = I.uwt(u)
                        for q in ws:
                            rhos.add(tuple(x + y for x, y in zip(q, w)))
                    for rho in sorted(rhos):
                        sv = I.singular(m, rho)
                        if not sv:
                            continue
                        shift = tuple(x - y for x, y in zip(mu_of(rho), mu))
                        key = (mu, m, shift)
                        nuy = nu_of(rho, c - m, a, b)
                        hits.setdefault(key, {}).setdefault(tname(t), []).append(
                            (nu, nuy, len(sv), render(I, sv[0])))
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('|')
        sys.stdout.flush()
    print()
    return hits


def fit(pts):
    """is nu = alpha a + beta b on the given [(a, b, nu)]?  return (alpha, beta) or None"""
    if len(pts) < 2:
        return None
    # solve from two independent rows, then check
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a1, b1, n1 = pts[i]
            a2, b2, n2 = pts[j]
            det = a1 * b2 - a2 * b1
            if det == 0:
                continue
            al = (n1 * b2 - n2 * b1) / det
            be = (a1 * n2 - a2 * n1) / det
            if all(al * a + be * b == nu for a, b, nu in pts):
                return al, be
            return None
    return None


def report(n, hits, ts):
    names = [tname(t) for t in ts]
    print("\n=== b_{a,b}(%d): every singular vector of the sweep, as a function of "
          "t = a/b ===" % n)
    print("(mu; nu) is the b_{a,b}(n)_0-highest weight, (a,b) the coprime representative "
          "of t,\nlambda = 2a/(n(a+b)) the parameter of [LKae].\n")
    for key in sorted(hits, key=lambda k: (k[1], k[0], k[2])):
        mu, m, shift = key
        d = hits[key]
        got = [s for s in names if s in d]
        print("-- V(mu = %s), degree m = %d, wt(y) - wt(V) = %s  :  %d of %d values of t"
              % (list(mu), m, list(shift), len(got), len(names)))
        pts, pts_y = [], []
        for s in got:
            for (nu, nuy, dim, txt) in d[s]:
                t = 'oo' if s == 'oo' else F(s)
                a, b = ab_of(t)
                pts.append((a, b, F(nu)))
                pts_y.append((a, b, F(nuy)))
        f, fy = fit(pts), fit(pts_y)
        fd = fit([(a, b, nuy - nu) for (a, b, nu), (_, _, nuy) in zip(pts, pts_y)])
        full = all(len({x[0] for x in d[s]}) >= NRANGE[0] for s in got)
        if len(got) == len(names) and full:
            if fd:
                print("   every t, and every nu in the range scanned (nu is free);"
                      "   nu(y) = nu + (%s) a + (%s) b" % (fd[0], fd[1]))
            else:
                diffs = sorted({(str(a), str(b), str(nuy - nu))
                                for (a, b, nu), (_, _, nuy) in zip(pts, pts_y)})
                print("   every t, and every nu in the range scanned (nu is free);"
                      "   (a,b,nu(y)-nu) = %s" % diffs[:8])
        elif f and all(not (NULO[0] <= f[0] * ab_of('oo' if s == 'oo' else F(s))[0]
                            + f[1] * ab_of('oo' if s == 'oo' else F(s))[1] <= NUHI[0])
                       for s in names if s not in d):
            # present wherever the window allows: the t's not seen are exactly those whose
            # predicted nu falls outside the range scanned
            print("   nu = %s a + %s b   for EVERY t (%d seen; the other %d have nu outside "
                  "[%d,%d]);   nu(y) = %s a + %s b"
                  % (f[0], f[1], len(got), len(names) - len(got), NULO[0], NUHI[0],
                     fy[0] if fy else '?', fy[1] if fy else '?'))
        else:
            for s in got:
                a, b = ab_of('oo' if s == 'oo' else F(s))
                print("   t = %-5s (a,b) = (%s,%s), lambda = %-5s :  nu = %s"
                      % (s, a, b, lam_of(n, a, b),
                         ", ".join(str(x[0]) for x in d[s])))
        print("      %s" % d[got[0]][0][3])


if __name__ == '__main__':
    n = int(sys.argv[1])
    mmax = int(sys.argv[2])
    hi = int(sys.argv[3])
    nulo, nuhi = int(sys.argv[4]), int(sys.argv[5])
    ts = sys.argv[6:] or TS
    ts = [('oo' if str(x) == 'oo' else F(str(x))) for x in ts]
    print("=== sweep: n=%d, m<=%d, 0<=mu_i<=%d, %d<=nu<=%d, %d values of t ==="
          % (n, mmax, hi, nulo, nuhi, len(ts)))
    NRANGE[0] = nuhi - nulo + 1
    NULO[0], NUHI[0] = nulo, nuhi
    h = sweep(n, mmax, hi, nulo, nuhi, ts)
    report(n, h, ts)
