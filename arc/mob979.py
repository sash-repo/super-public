"""
The Moebius table read in the frame of the two-sheeted cover.  Two questions:

  (a) the analog of the k^L(1|3) case of Radul for the Moebius table (201),
      which stops at n = 2;
  (b) the claim that, for n = 1, 2, the products sqrt(t) S^M_n
      obey the identity of Statement 9.10 with the second factor deleted.

The cover frame is

     ^D_1 := sqrt(t) D^M_zeta = D_{Theta_1},  where Theta_1 := sqrt(t) zeta,
     ^D_i := D^M_{theta_{i-1}}                for i >= 2,

in which every square is cD and the anticommutators vanish, see (202).
What is computed here:

  1) the cover-frame entries ^S_n, i.e., the rows of Table (194) with the
     D_{theta_i} replaced by the ^D_i, are sqrt(t) S^M_n for n = 1, 2 -- so
     the whole of the Moebius table is the contact table read upstairs, and
     the division by sqrt(t) IS the twist;
  2) the ^S_n obey the cocycle identity of the weight (4-n)/2 with NO second
     factor, whereas the S^M_n obey the twisted one -- which is (b);
  3) n = 3: eq. (196) built on the cover frame is a 1-cocycle of the weight
     1/2 with no second factor, and of no other weight and no other power of
     the Moebius multiplier -- which is (a): the analog of the k^L(1|3) case
     is the k^L(1|3) formula itself, read in the cover frame.

Every check is exact, over Q(t) with the flow parameters truncated.
"""
import sys

import sympy as sp

import radul as R
import mobius as M
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t,
                   VF, einv, epow)
from mobius import (ZETA, cD_field, D_M, Diffeo, Composite, multiplier,
                    vf_scal, vf_eq, anti, S_M)
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


# --------------------------------------------------------- the cover frame
def cover_frame(i, n):
    """^D_i: sqrt(t) D^M_zeta for i = 1, and D^M_{theta_{i-1}} otherwise."""
    if i == ZETA:
        return vf_scal(num(sp.sqrt(t)), D_M(ZETA, n), n)
    return D_M(i, n)


def cover_coord(G, i, n):
    """G^*(Theta_i), where Theta_1 = sqrt(t) zeta and Theta_i = theta_{i-1}."""
    if i == ZETA:
        return emul(epow(escal(1 / t, G.star({(): t})), sp.Rational(1, 2)),
                    escal(sp.sqrt(t), G.star(th(ZETA))))
    return G.star(th(i))


def S_cover(G, n):
    """The row n of Table (194) written in the cover frame ^D_i."""
    m = multiplier(G, n)
    mi = einv(m)
    mi2 = emul(mi, mi)
    cD = cD_field(n)
    D1 = cover_frame(ZETA, n)
    if n == 1:
        A = emul(D1(cD(m)), mi)
        B = emul(emul(D1(m), cD(m)), mi2)
    elif n == 2:
        D2 = cover_frame(2, n)
        A = emul(D1(D2(m)), mi)
        B = emul(emul(D1(m), D2(m)), mi2)
    else:
        raise ValueError("n = 1, 2 only")
    return escal(sp.Rational(1, 2), esub(A, escal(sp.Rational(3, 2), B)))


# ------------------------------------------------------- eq. (196) upstairs
def U_cover(G, n):
    im = [cover_coord(G, j, n) for j in range(1, n + 1)]
    return [[cover_frame(i, n)(im[j - 1]) for j in range(1, n + 1)]
            for i in range(1, n + 1)]


def amatrix(G, n):
    U = U_cover(G, n)
    mi = einv(multiplier(G, n))
    return [[emul(mi, U[j][i]) for j in range(n)] for i in range(n)]


def apply_row(v_row, a_row, n):
    a0 = eadd(*[emul(v_row[j], cover_frame(j + 1, n).a0) for j in range(n)])
    comp = [eadd(*[emul(v_row[j], cover_frame(j + 1, n).a[k]) for j in range(n)])
            for k in range(n)]
    v = VF(a0, comp)
    return [v(a_row[j]) for j in range(n)]


def sprod(row1, row2, n):
    out = {}
    for i in range(n):
        for S, c in row2[i].items():
            sign = -1 if len(S) % 2 else 1
            out = eadd(out, escal(sign, emul(row1[i], {S: c})))
    return out


def S3_cover(G, n=3, c=-1):
    """c . m^{3/2} (v_1(v_2), v_3), the frame being the cover one."""
    a = amatrix(G, n)
    w = apply_row(a[0], a[1], n)
    return escal(c, emul(epow(multiplier(G, n), sp.Rational(3, 2)),
                         sprod(w, a[2], n)))


# ------------------------------------------------------- the two identities
def ident(Sfun, w, n, G, H, power):
    """S(GH) = S(H) + m_H^w (H^*t/t)^{power/2} H^*(S(G))."""
    GH = Composite(G, H)
    mH = multiplier(H, n)
    u = epow(escal(1 / t, H.star({(): t})), sp.Rational(power, 2))
    return eeq(Sfun(GH),
               eadd(Sfun(H), emul(emul(epow(mH, w), u), H.star(Sfun(G)))))


def run_low(n):
    print("\n--- n = %d -------------------------------------------------" % n)
    M.setup(n)
    R.set_order(2)
    G = Diffeo(gen(n), e, n)
    H = Diffeo(gen(n, seed=3), d, n)
    w = sp.Rational(4 - n, 2)

    # the frame upstairs is a Clifford one, the one downstairs is not
    D1 = cover_frame(ZETA, n)
    sq1 = anti(D1, D1, n)
    sq1 = vf_scal(num(sp.Rational(1, 2)), sq1, n)
    check("(sqrt(t) D^M_zeta)^2 = cD", vf_eq(sq1, cD_field(n), n))
    if n >= 2:
        D2 = cover_frame(2, n)
        check("{sqrt(t) D^M_zeta, D^M_theta} = 0",
              vf_eq(anti(D1, D2, n), VF({}, [{} for _ in range(n)]), n))
        Dz = D_M(ZETA, n)
        check("   whereas {D^M_zeta, D^M_theta} != 0",
              not vf_eq(anti(Dz, D2, n), VF({}, [{} for _ in range(n)]), n))

    Sc, Sm = S_cover(G, n), S_M(G, n)
    check("^S_%d = sqrt(t) . S^M_%d   (the cover entry IS the printed one"
          " times sqrt(t))" % (n, n),
          eeq(Sc, escal(sp.sqrt(t), Sm)))

    f_cov = lambda X: S_cover(X, n)
    f_mob = lambda X: S_M(X, n)
    f_sqt = lambda X: escal(sp.sqrt(t), S_M(X, n))

    check("S^M_%d: the TWISTED identity, weight %s, power 1/2" % (n, w),
          ident(f_mob, w, n, G, H, 1))
    check("S^M_%d: the untwisted identity fails" % n,
          not ident(f_mob, w, n, G, H, 0))
    check("sqrt(t) S^M_%d: the UNTWISTED identity, weight %s  <-- (b)" % (n, w),
          ident(f_sqt, w, n, G, H, 0))
    check("sqrt(t) S^M_%d: the twisted identity fails" % n,
          not ident(f_sqt, w, n, G, H, 1))
    check("^S_%d: the UNTWISTED identity, weight %s" % (n, w),
          ident(f_cov, w, n, G, H, 0))
    for ww in (w + 1, w - 1, sp.Integer(2)):
        if ww == w:
            continue
        check("^S_%d: and of no other weight: %s fails" % (n, ww),
              not ident(f_cov, ww, n, G, H, 0))
    for p in (1, -1, 2):
        check("^S_%d: and of no other power of the multiplier: %d/2 fails"
              % (n, p), not ident(f_cov, w, n, G, H, p))


def run3():
    n = 3
    print("\n--- n = 3, eq. (196) in the cover frame ---------------------")
    M.setup(n)
    R.set_order(2)
    G = Diffeo(gen(n), e, n)
    H = Diffeo(gen(n, seed=3), d, n)
    m = multiplier(G, n)
    w = sp.Rational(1, 2)

    U = U_cover(G, n)
    gram = [[eadd(*[emul(U[i][j], U[i][l]) for i in range(n)])
             for l in range(n)] for j in range(n)]
    scalar = all(eeq(gram[j][l], gram[0][0] if j == l else {})
                 for j in range(n) for l in range(n))
    check("U^tU is scalar in the cover frame", scalar)
    check("U^tU = m . 1_3   (eq. (112))", scalar and eeq(gram[0][0], m))

    a = amatrix(G, n)
    ort = all(eeq(sprod(a[i], a[j], n),
                  emul(einv(m), num(1)) if i == j else {})
              for i in range(n) for j in range(n))
    check("(v_i, v_j) = m^{-1} delta_ij   (eq. (199))", ort)

    S = lambda X: S3_cover(X, n, -1)
    check("S^M_3 is not identically zero", bool(S(G)))
    check("S^M_3: the UNTWISTED identity, weight 1/2  <-- (a)",
          ident(S, w, n, G, H, 0))
    check("S^M_3: the twisted identity fails", not ident(S, w, n, G, H, 1))
    for p in (-1, 2, 3):
        check("S^M_3: and no other power of the multiplier: %d/2 fails" % p,
              not ident(S, w, n, G, H, p))
    for ww in (sp.Rational(3, 2), sp.Rational(-1, 2), sp.Integer(1)):
        check("S^M_3: and no other weight: %s fails" % ww,
              not ident(S, ww, n, G, H, 0))


if __name__ == "__main__":
    print("=" * 70)
    print("the Moebius table in the cover frame")
    print("=" * 70)
    run_low(1)
    run_low(2)
    run3()
    print("\n%d ok, %d FAILED" % (OK, BAD))
    sys.exit(1 if BAD else 0)
