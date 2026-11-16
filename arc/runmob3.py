"""
The frame formula eq. (196) in the Moebius sector, n = 3: Radul's formula in terms of
the frame v_i.

On S^{1|n}_M the natural frame (56) is not a Clifford frame: the twisted
member squares to (1/t)cD - (zeta/2t)D^M_zeta and does not anticommute with the
others.  What is one is the frame of the two-sheeted cover,

     ^D_1 := sqrt(t) D^M_zeta = D_{Theta_1},  where Theta_1 := sqrt(t) zeta,
     ^D_i := D^M_{theta_{i-1}}                for i >= 2,

and in it the whole of Subsection 9.6 goes through: U^tU = m . 1_n, the
orthogonality of the v_i, and eq. (196) as a 1-cocycle -- with the multiplier of
the Moebius bundle in front, exactly as in the rows n = 1, 2 of (201).
"""
import sys

import sympy as sp

import radul as R
import mobius as M
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t, VF,
                   einv, epow)
from mobius import (ZETA, cD_field, D_M, Diffeo, Composite, multiplier,
                    vf_scal)
from test_mobius import gen

OK = BAD = 0
N = 3


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


def cover_frame(i, n):
    """^D_i: the frame of the two-sheeted cover."""
    if i == ZETA:
        return vf_scal(num(sp.sqrt(t)), D_M(ZETA, n), n)
    return D_M(i, n)


def cover_coord(G, i, n):
    """G^*(Theta_i), where Theta_1 = sqrt(t) zeta and Theta_i = theta_{i-1}."""
    if i == ZETA:
        return emul(epow(escal(1 / t, G.star({(): t})), sp.Rational(1, 2)),
                    escal(sp.sqrt(t), G.star(th(ZETA))))
    return G.star(th(i))


def U_cover(G, n):
    """U_{ij} = ^D_i(G^* Theta_j)."""
    im = [cover_coord(G, j, n) for j in range(1, n + 1)]
    return [[cover_frame(i, n)(im[j - 1]) for j in range(1, n + 1)]
            for i in range(1, n + 1)]


def amatrix(G, n):
    """(a_ij) = U^{-1} = m^{-1} U^t, by eq. (112)."""
    U = U_cover(G, n)
    mi = einv(multiplier(G, n))
    return [[emul(mi, U[j][i]) for j in range(n)] for i in range(n)]


def apply_row(v_row, a_row, n):
    """v(w) coefficient-wise, v = sum_j v_j ^D_j."""
    a0 = eadd(*[emul(v_row[j], cover_frame(j + 1, n).a0) for j in range(n)])
    comp = [eadd(*[emul(v_row[j], cover_frame(j + 1, n).a[k]) for j in range(n)])
            for k in range(n)]
    v = VF(a0, comp)
    return [v(a_row[j]) for j in range(n)]


def sprod(row1, row2, n):
    """(sum a_i ^D_i, sum b_j ^D_j) = sum_i (-1)^{p(b_i)} a_i b_i."""
    out = {}
    for i in range(n):
        for S, c in row2[i].items():
            sign = -1 if len(S) % 2 else 1
            out = eadd(out, escal(sign, emul(row1[i], {S: c})))
    return out


def S3_M(G, n=N, c=-1):
    """c . m^{3/2} (v_1(v_2), v_3) in the cover frame."""
    a = amatrix(G, n)
    w = apply_row(a[0], a[1], n)
    return escal(c, emul(epow(multiplier(G, n), sp.Rational(3, 2)),
                         sprod(w, a[2], n)))


def twisted(Sfun, w, n, G, H, power):
    GH = Composite(G, H)
    mH = multiplier(H, n)
    u = epow(escal(1 / t, H.star({(): t})), sp.Rational(power, 2))
    return eeq(Sfun(GH),
               eadd(Sfun(H), emul(emul(epow(mH, w), u), H.star(Sfun(G)))))


if __name__ == "__main__":
    print("=" * 70)
    print("eq. (196) in the Moebius sector, n = 3")
    print("=" * 70)
    M.setup(N)
    R.set_order(2)
    G = Diffeo(gen(N), e, N)
    H = Diffeo(gen(N, seed=3), d, N)
    m = multiplier(G, N)

    U = U_cover(G, N)
    gram = [[eadd(*[emul(U[i][j], U[i][l]) for i in range(N)])
             for l in range(N)] for j in range(N)]
    scalar = all(eeq(gram[j][l], gram[0][0] if j == l else {})
                 for j in range(N) for l in range(N))
    check("U^tU is scalar in the cover frame", scalar)
    check("U^tU = m . 1_n   (eq. (112))", scalar and eeq(gram[0][0], m))

    a = amatrix(G, N)
    ort = all(eeq(sprod(a[i], a[j], N),
                  emul(einv(m), num(1)) if i == j else {})
              for i in range(N) for j in range(N))
    check("(v_i, v_j) = m^{-1} delta_ij   (eq. (199))", ort)

    S = lambda X: S3_M(X, N, -1)
    check("S_3 is not identically zero", bool(S(G)))
    for p in (1, 0, -1, 2):
        cond = twisted(S, sp.Rational(1, 2), N, G, H, p)
        check("S_3(GH) = S_3(H) + m_H^{1/2}(H^*t/t)^{%d/2} H^*(S_3(G))"
              % p if p else
              "   the untwisted identity fails", cond if p == 1 else not cond)
    for w in (sp.Rational(3, 2), sp.Rational(-1, 2)):
        check("   and of no other weight: %s fails" % w,
              not twisted(S, w, N, G, H, 1))

    print("\n%d ok, %d FAILED" % (OK, BAD))
    sys.exit(1 if BAD else 0)
