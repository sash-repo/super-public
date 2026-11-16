"""
The table (201) of the Schwarz derivatives on the Moebius superstring,
for n = 0, 1, 2.

What is tested, for a finite Moebius contact diffeomorphism G = exp(e K^M_g) of
S^{1|n}_M with a generic even generating function g:

  1) the row n = 0: the classical Schwarzian, of the weight 2 -- the row in
     which the two sectors coincide, there being no odd coordinate to twist;
  2) the rows n = 1, 2 as printed: 1-cocycles of the weight (4-n)/2 in m, and
     of no other weight, and with no other numerical coefficient;
  3) why the table stops at n = 2 in this sector as well: the frame member in
     the twisted direction is not a square root of cD, whereas sqrt(t) D^M_zeta
     is -- so the answer downstairs is the contact one of Table (194) divided
     by sqrt(t), which is the transition function of the Moebius bundle.
"""
import sys

import sympy as sp

import radul as R
import mobius as M
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t, d_th,
                   VF, einv, epow)
from mobius import (ZETA, cD_field, D_M, K_f, Diffeo, Composite, multiplier,
                    anti, vf_eq, vf_scal, vf_add, S_M, cocycle)
from test_mobius import gen

OK = BAD = 0


def check(name, cond):
    global OK, BAD
    if cond:
        OK += 1
        print("  ok   %s" % name)
    else:
        BAD += 1
        print("  FAIL %s" % name)
    sys.stdout.flush()
    return cond


def S_bad(G, n, coef):
    """The row n of (201) with the coefficient 3/2 replaced by coef."""
    m = multiplier(G, n)
    mi = einv(m)
    mi2 = emul(mi, mi)
    cD = cD_field(n)
    if n == 0:
        A = emul(d_t(d_t(m)), mi)
        B = emul(emul(d_t(m), d_t(m)), mi2)
    elif n == 1:
        Dz = D_M(ZETA, n)
        A = emul(Dz(cD(m)), mi)
        B = emul(emul(Dz(m), cD(m)), mi2)
    else:
        Dz, Dth = D_M(ZETA, n), D_M(2, n)
        A = emul(Dz(Dth(m)), mi)
        B = emul(emul(Dz(m), Dth(m)), mi2)
    return escal(sp.Rational(1, 2), esub(A, escal(coef, B)))


def twisted(Sfun, w, n, G, H, power=1):
    """S(GH) = S(H) + m_H^w . (H^*(t)/t)^{power/2} . H^*(S(G)).

    The second factor is the multiplier of the Moebius bundle: u^2 = m_G t/t~,
    see eq. (115).  For power = 0 this is the untwisted identity.
    """
    GH = Composite(G, H)
    mH = multiplier(H, n)
    u = epow(escal(1 / t, H.star({(): t})), sp.Rational(power, 2))
    return eeq(Sfun(GH),
               eadd(Sfun(H), emul(emul(epow(mH, w), u), H.star(Sfun(G)))))


if __name__ == "__main__":
    print("=" * 70)
    print("the table (201)")
    print("=" * 70)

    for n in (0, 1, 2):
        w = sp.Rational(4 - n, 2)
        tw = 0 if n == 0 else 1
        print("\nS^M_%d, the weight (4-n)/2 = %s, the Moebius power %d"
              % (n, w, tw))
        M.setup(n)
        R.set_order(3)
        G = Diffeo(gen(n), e, n)
        H = Diffeo(gen(n, seed=3), d, n)
        S = lambda X, n=n: S_M(X, n)
        check("S^M_%d is not identically zero" % n, bool(S(G)))
        check("S^M_%d(GH) = S^M_%d(H) + m_H^{%s}(H^*t/t)^{%d/2} H^*(S^M_%d(G))"
              % (n, n, w, tw, n), twisted(S, w, n, G, H, tw))
        for bad in (w - 1, w + 1, sp.Integer(2)):
            if bad == w:
                continue
            check("   and of no other weight: %s fails" % bad,
                  not twisted(S, bad, n, G, H, tw))
        for p in (tw - 1, tw + 1, tw + 2, tw - 2):
            check("   and of no other power of the Moebius factor: %d/2 fails"
                  % p, not twisted(S, w, n, G, H, p))
        for coef in (sp.Integer(1), sp.Rational(1, 2), sp.Integer(2)):
            check("   the coefficient 3/2 is forced: %s fails" % coef,
                  not twisted(lambda X, n=n, c=coef: S_bad(X, n, c), w, n, G, H, tw))

    # the Moebius flows annihilate the n = 0 row, as in the contact sector
    print("\nthe annihilator, n = 0")
    M.setup(0)
    R.set_order(3)
    Gq = Diffeo({(): t ** 2 + t + sp.Rational(1, 3)}, e, 0)
    check("S^M_0 = 0 on the Moebius flows", not S_M(Gq, 0))

    # why there is no row n >= 3 here either: the twisted frame member
    print("\nthe frame in the twisted direction")
    for n in (1, 2, 3):
        M.setup(n)
        R.set_order(2)
        cD = cD_field(n)
        Dz = D_M(ZETA, n)
        Dzc = vf_scal(num(sp.sqrt(t)), Dz, n)
        check("n = %d: (D^M_zeta)^2 is not a multiple of cD" % n,
              not any(vf_eq(vf_scal(num(sp.Rational(1, 2)), anti(Dz, Dz, n), n),
                            vf_scal(num(c), cD, n), n)
                      for c in (1, 1 / t, 1 / t ** 2, t)))
        check("n = %d: (sqrt(t) D^M_zeta)^2 = cD, i.e., the frame of the cover"
              % n,
              vf_eq(vf_scal(num(sp.Rational(1, 2)), anti(Dzc, Dzc, n), n), cD, n))

    print("\n%d ok, %d FAILED" % (OK, BAD))
    sys.exit(1 if BAD else 0)
