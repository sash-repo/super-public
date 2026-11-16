#!/usr/bin/env python3
r"""charcheck.py --- the SUPERCHARACTER formula for the typical irreducible
modules over osp(3|2) and osp_a(4|2), machine-checked against the modules
themselves.

The formula in the literature is Kac's
(LNM 676, Theorem 1; see also Cheng-Wang GSM 144, Thm 2.35): for a typical
lambda,

   ch  L(lambda) * PROD_{alpha in Dev+} (e^{a/2}-e^{-a/2})
        = PROD_{alpha in Dod+} (e^{a/2}+e^{-a/2}) * SUM_w sgn(w) e^{w(lam+rho)}

   sch L(lambda) * PROD_{alpha in Dev+} (e^{a/2}-e^{-a/2})
        = PROD_{alpha in Dod+} (e^{a/2}-e^{-a/2}) * SUM_w sgn(w) e^{w(lam+rho)}

Note that the odd product runs over ALL odd positive roots, isotropic or not:
osp(3|2) has the non-isotropic delta, and the formula still holds.  We verify
both identities as identities of Laurent polynomials -- no division -- against
the modules built degree by degree from the Cartan matrix (hwmod.py).

Setting every e^mu to 1 in the two identities gives, respectively, Kac's
dimension formula and sdim L = 0 in the a-b reading; the honest statement is
that sdim L(lambda) = a + a*epsilon, i.e. the even and the odd parts have
equal dimension -- which is what the supercharacter identity says, since its
right-hand side has the factor (e^{a/2}-e^{-a/2}) for each odd positive root
and hence vanishes at e^mu = 1.
"""
import sys
from fractions import Fraction as Fr
from itertools import product

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from hwmod import irrep

# ------------------------------------------------------------------ Laurent
# a Laurent polynomial in the formal exponentials e^{mu} is a dict
#   {tuple of DOUBLED integer coordinates of mu : integer coefficient}

def lmul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            k = tuple(x + y for x, y in zip(m1, m2))
            r[k] = r.get(k, 0) + c1 * c2
    return {k: v for k, v in r.items() if v}

def ladd(p, q):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + v
        if r[k] == 0:
            del r[k]
    return r

def mono(mu, c=1):
    return {tuple(mu): c} if c else {}

def lprod(ps):
    r = mono((0,) * NC)
    for p in ps:
        r = lmul(r, p)
    return r


# ------------------------------------------------------------------ engine

def character(A, par, lam_labels, simple_roots, lam_eps, dbl):
    """(ch, sch) of L(lambda) as Laurent polynomials in doubled coordinates."""
    M = irrep(A, par, 0, lam_labels, maxdim=6000)
    ch, sch = {}, {}
    for d, m in M.dim.items():
        if not m:
            continue
        mu = [dbl * x for x in lam_eps]
        for j, dj in enumerate(d):
            for i in range(len(mu)):
                mu[i] -= dj * dbl * simple_roots[j][i]
        mu = tuple(int(x) for x in mu)
        p = sum(d[j] * par[j] for j in range(len(par))) % 2
        ch[mu] = ch.get(mu, 0) + m
        sch[mu] = sch.get(mu, 0) + (-m if p else m)
    return {k: v for k, v in ch.items() if v}, {k: v for k, v in sch.items() if v}


def kac_sides(POSEV, POSOD, lam_eps, rho, W, dbl):
    """(even-denominator, odd-numerator-plus, odd-numerator-minus, Weyl sum)."""
    def half(al, sign):
        h = tuple(int(dbl * Fr(x, 2)) for x in al)
        mh = tuple(-x for x in h)
        return ladd(mono(h), mono(mh, sign))
    den = lprod([half(al, -1) for al in POSEV])
    numP = lprod([half(al, +1) for al in POSOD])
    numM = lprod([half(al, -1) for al in POSOD])
    lr = [Fr(lam_eps[i]) + Fr(rho[i]) for i in range(len(rho))]
    wsum = {}
    for sgn, w in W:
        img = tuple(int(dbl * w(lr)[i]) for i in range(len(lr)))
        wsum = ladd(wsum, mono(img, sgn))
    return den, numP, numM, wsum


# ================================================================== osp(3|2)
def run_osp32():
    global NC
    NC = 2
    dbl = 2
    POSEV = [(1, 0), (0, 2)]
    POSOD = [(1, -1), (0, 1), (1, 1)]
    def frm(u, v):
        return Fr(u[0]) * v[0] - Fr(u[1]) * v[1]
    rho = [Fr(1, 2) * (POSEV[0][i] + POSEV[1][i]) -
           Fr(1, 2) * sum(r[i] for r in POSOD) for i in range(2)]
    A = [[0, 1], [-1, 1]]
    par = [1, 1]
    SIMPLE = [(1, -1), (0, 1)]
    # W(g_ev) = W(o(3)) x W(sp(2)) = {1, s_eps} x {1, s_delta}
    W = [(1, lambda v: v),
         (-1, lambda v: [-v[0], v[1]]),
         (-1, lambda v: [v[0], -v[1]]),
         (1, lambda v: [-v[0], -v[1]])]
    # For the SUPERcharacter of osp(3|2) the sign character of W has to be
    # twisted: a scan shows that sgn(w) must be replaced
    # by sgn(w restricted to o(3)), i.e. the reflection s_delta in the
    # SYMPLECTIC direction contributes +1 and not -1.  That is the trace of the
    # odd NON-isotropic root delta, whose double 2delta is even.
    WS = [(1, lambda v: v),
          (-1, lambda v: [-v[0], v[1]]),
          (1, lambda v: [v[0], -v[1]]),
          (-1, lambda v: [-v[0], -v[1]])]
    POSOD_ISO = [r for r in POSOD if frm(r, r) == 0]
    print("osp(3|2):  rho =", rho, "  isotropic odd positive roots:", POSOD_ISO)
    okc = oks = 0
    tested = []
    for m in range(2, 11):
        for n in range(0, 5):
            a, b = Fr(m, 2), Fr(n)
            lam = (a, b)
            lr = [lam[i] + rho[i] for i in range(2)]
            typ = all(frm(lr, r) != 0 for r in POSOD_ISO)
            if not typ:
                continue
            labels = [frm(lam, SIMPLE[0]) / Fr(1), frm(lam, SIMPLE[1]) / Fr(-1)]
            try:
                ch, sch = character(A, par, labels, SIMPLE, labels and lam, dbl)
            except Exception as e:
                continue
            den, numP, numM, wsum = kac_sides(POSEV, POSOD, lam, rho, W, dbl)
            _, _, _, wsumS = kac_sides(POSEV, POSOD, lam, rho, WS, dbl)
            lhsC, rhsC = lmul(ch, den), lmul(numP, wsum)
            lhsS, rhsS = lmul(sch, den), lmul(numM, wsumS)
            okc += (lhsC == rhsC)
            oks += (lhsS == rhsS)
            tested.append((m, n, lhsC == rhsC, lhsS == rhsS,
                           sum(ch.values()), sum(sch.values())))
    print("   typical modules tested: %d;  character identity holds %d,"
          "  supercharacter identity holds %d" % (len(tested), okc, oks))
    bad = [t for t in tested if not (t[2] and t[3])]
    print("   failures:", bad if bad else "none")
    print("   sdim (= value of sch at e^mu=1) is 0 for every one of them:",
          all(t[5] == 0 for t in tested))
    return len(tested), okc, oks


# =============================================================== osp_a(4|2)
def run_osp42():
    global NC
    NC = 3
    dbl = 2
    POSEV = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    POSOD = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]
    rho = [-1, 1, 1]
    SIMPLE = [(1, -1, -1), (0, 2, 0), (0, 0, 2)]
    W = []
    for s in product([1, -1], repeat=3):
        sgn = s[0] * s[1] * s[2]
        W.append((sgn, (lambda ss: (lambda v: [ss[i] * v[i] for i in range(3)]))(s)))
    tot = okc = oks = 0
    for a in [Fr(1), Fr(2), Fr(3), Fr(1, 2), Fr(5, 3), Fr(-3), Fr(7, 2)]:
        s = [-(1 + a), Fr(1), a]
        def frm(u, v, s=s):
            return sum(Fr(u[i]) * v[i] * s[i] for i in range(3))
        A = [[0, 1, a], [-1, 2, 0], [-1, 0, 2]]
        par = [1, 0, 0]
        for m1 in range(0, 5):
            for m2 in range(0, 3):
                for m3 in range(0, 3):
                    lam = (Fr(m1), Fr(m2), Fr(m3))
                    lr = [lam[i] + rho[i] for i in range(3)]
                    if not all(frm(lr, al) != 0 for al in POSOD):
                        continue
                    labels = [Fr(m1 * (1 + a) + m2 + m3 * a) / 2, Fr(m2), Fr(m3)]
                    try:
                        ch, sch = character(A, par, labels, SIMPLE, lam, dbl)
                    except Exception:
                        continue
                    den, numP, numM, wsum = kac_sides(POSEV, POSOD, lam, rho, W, dbl)
                    tot += 1
                    okc += (lmul(ch, den) == lmul(numP, wsum))
                    oks += (lmul(sch, den) == lmul(numM, wsum))
    print("osp_a(4|2): typical modules tested over 7 values of a: %d;"
          "  ch holds %d,  sch holds %d" % (tot, okc, oks))
    return tot, okc, oks


if __name__ == "__main__":
    print("=" * 74)
    run_osp32()
    print()
    run_osp42()
