"""The exponents in the Boucher--Friedan--Kent determinant of ns(2).

The determinant of the Shapovalov form on the weight space of level n and
relative charge m is computed from scratch and factored; the multiplicity of
each vanishing surface is read off and compared with

   (h - h_{r,s})   ->   P(n - rs/2, m),
   (h - h_k)       ->   P_k(n - |k|, m - sign k),

where P(n, m) is the dimension of the weight space (n, m) of the Verma module
and P_k(n, m) is the number of those of its PBW monomials that do NOT contain
the mode G^{sign k}_{-|k|}.  The second partition function is the one the
charged singular vectors force: the submodule such a vector generates is not
a Verma module, one odd mode being killed.
"""

import sys
from collections import defaultdict

import sympy

from verma import NS2, Module, h, q, t
from runns2 import h_rs, h_k, multiplicity, C

LOG = open("run-ns2exp.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def run(NMAX=2):
    say("=" * 74)
    say("ns(2): the exponents of the determinant against the two partition functions")
    say("=" * 74)
    alg = NS2(2 * NMAX + 6, C)
    M = Module(alg)

    # the PBW monomials of every weight space, kept as monomials so that the
    # restricted count P_k is available
    lev = sympy.Rational(0)
    mons = {}
    while lev <= NMAX + sympy.Rational(1, 2):
        for m in range(-2 * NMAX - 2, 2 * NMAX + 3):
            b = M.basis(lev, m)
            if b:
                mons[(lev, m)] = b
        lev += sympy.Rational(1, 2)

    def P(n, m):
        return len(mons.get((sympy.Rational(n), m), []))

    def Pk(n, m, k):
        """monomials of (n, m) not containing G^{sign k}_{-|k|}."""
        nm = "Gp" if k > 0 else "Gm"
        key = (nm, -abs(sympy.Rational(k)))
        out = 0
        for mo in mons.get((sympy.Rational(n), m), []):
            if key not in mo:
                out += 1
        return out

    bad = 0
    lev = sympy.Rational(1, 2)
    while lev <= NMAX:
        for m in sorted({mm for (nn, mm) in mons if nn == lev}):
            b, G = M.gram(lev, m)
            if not b:
                continue
            det = sympy.factor(sympy.simplify(sympy.together(sympy.expand(G.det()))))
            say("")
            say("level %s, charge %+d: dim = %d" % (lev, m, len(b)))
            if det == 0:
                say("   the determinant vanishes identically")
                continue
            seen = 0
            for r in range(1, 4 * NMAX + 3):
                for s in range(2, 4 * NMAX + 3, 2):
                    if sympy.Rational(r * s, 2) > lev:
                        continue
                    fac = sympy.numer(sympy.cancel(sympy.together(h - h_rs(r, s))))
                    e = multiplicity(det, fac)
                    want = P(lev - sympy.Rational(r * s, 2), m)
                    seen += e
                    flag = "ok" if e == want else "MISMATCH"
                    if e or want:
                        say("   (h-h_{%d,%d}): found %d, P(n-rs/2, m) = %d   %s"
                            % (r, s, e, want, flag))
                    if e != want:
                        bad += 1
            kk = sympy.Rational(1, 2)
            while kk <= lev:
                for sgn in (+1, -1):
                    k = sgn * kk
                    fac = sympy.numer(sympy.cancel(sympy.together(h - h_k(k))))
                    e = multiplicity(det, fac)
                    want = Pk(lev - kk, m - sgn, k)
                    seen += e
                    flag = "ok" if e == want else "MISMATCH"
                    if e or want:
                        say("   (h-h_{%s}):    found %d, P_k(n-|k|, m-sign k) = %d   %s"
                            % (k, e, want, flag))
                    if e != want:
                        bad += 1
                kk += 1
            say("   total degree in h accounted for: %d of %s"
                % (seen, sympy.Poly(sympy.numer(sympy.cancel(sympy.together(det))),
                                    h).degree()))
        lev += sympy.Rational(1, 2)
    say("")
    say("mismatches: %d" % bad)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
