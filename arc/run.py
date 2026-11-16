"""
Radul's Sect. 5.2 (Stockholm Reports 1986-21, pp. 52-53), checked.

What is tested, for a finite contact diffeomorphism G = exp(e K_g) of the
superstring S^{1|n} with a generic even generating function g:

  1) his S_1 and S_2 are 1-cocycles of the weight 2 - n/2 he prints -- and of
     no other weight, and with no other numerical coefficient;
  2) the identities of his Comment 4) for the frame v_i;
  3) the constant in his S_3 = -3 m^{3/2} (v_1(v_2), v_3) alpha^{1/2}, got by
     linearizing and comparing with the pattern of S_1, S_2.

The conventions are those of eq. (54) of the text: the frame of the contact
distribution is D_{theta_i} = theta_i d_t + d_{theta_i}, D^2 = d_t.
"""
import sys
import sympy as sp
import radul as R
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t, d_th,
                   Diffeo, Composite, multiplier, U_matrix, einv, epow, D_op,
                   VF, clean)
from test_radul import gen, first_order

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


# ------------------------------------------------------------------ the frame
def amatrix(G, n):
    """(a_ij) = (D_{theta_i}(G^* theta_j))^{-1} = m^{-1} U^t, by (112)."""
    U = U_matrix(G, n, D_op)
    mi = einv(multiplier(G, n))
    return [[emul(mi, U[j][i]) for j in range(n)] for i in range(n)]


def vfield(a_row, n):
    """v = sum_j a_j D_{theta_j}, as a vector field."""
    a0 = eadd(*[emul(a_row[j], th(j + 1)) for j in range(n)])
    return VF(a0, [a_row[j] for j in range(n)])


def apply_to_row(v, a_row, n):
    """v(w) = sum_j v(a_j) D_j, coefficient-wise: returns the row (v(a_j))."""
    return [v(a_row[j]) for j in range(n)]


def sprod(row1, row2, n):
    """(sum a_i D_i, sum b_j D_j) = sum_i (-1)^{p(b_i)} a_i b_i,  eq. (336)."""
    out = {}
    for i in range(n):
        for S, c in row2[i].items():
            sign = -1 if len(S) % 2 else 1
            out = eadd(out, escal(sign, emul(row1[i], {S: c})))
    return out


# ------------------------------------------------------------- the Schwarzians
def S_radul(G, n):
    """Radul's S_n written in the frame D_{theta_i}, as the function part."""
    m = multiplier(G, n)
    mi = einv(m)
    mi2 = emul(mi, mi)
    if n == 1:
        A = emul(D_op(1)(d_t(m)), mi)
        B = emul(emul(D_op(1)(m), d_t(m)), mi2)
    elif n == 2:
        A = emul(D_op(1)(D_op(2)(m)), mi)
        B = emul(emul(D_op(1)(m), D_op(2)(m)), mi2)
    else:
        raise ValueError
    return escal(sp.Rational(1, 2), esub(A, escal(sp.Rational(3, 2), B)))


def S3_frame(G, n=3, c=-3):
    """c . m^{3/2} (v_1(v_2), v_3)."""
    a = amatrix(G, 3)
    v1 = vfield(a[0], 3)
    w = apply_to_row(v1, a[1], 3)                 # v_1(v_2)
    return escal(c, emul(epow(multiplier(G, 3), sp.Rational(3, 2)),
                         sprod(w, a[2], 3)))


def cocycle(Sfun, w, n, G, H):
    """S(GH) = S(H) + m_H^w H^*(S(G))."""
    GH = Composite(G, H)
    mH = multiplier(H, n)
    return eeq(Sfun(GH), eadd(Sfun(H), emul(epow(mH, w), H.star(Sfun(G)))))


if __name__ == "__main__":
    print("=" * 70)
    print("Radul, Sect. 5.2")
    print("=" * 70)

    # ---------------------------------------------------------- S_1 and S_2
    for n in (1, 2):
        print("\nS_%d, the weight 2 - n/2 = %s" % (n, sp.Rational(4 - n, 2)))
        R.setup(n)
        R.set_order(3)
        G = Diffeo(gen(n), e, n)
        H = Diffeo(gen(n, seed=3), d, n)
        S = lambda X: S_radul(X, n)
        check("S_%d is not identically zero" % n, bool(S(G)))
        check("S_%d(GH) = S_%d(H) + m_H^{%s} H^*(S_%d(G))"
              % (n, n, sp.Rational(4 - n, 2), n),
              cocycle(S, sp.Rational(4 - n, 2), n, G, H))
        for w in (sp.Rational(4 - n, 2) - 1, sp.Rational(4 - n, 2) + 1, 2):
            if w == sp.Rational(4 - n, 2):
                continue
            check("   and of no other weight: %s fails" % w,
                  not cocycle(S, w, n, G, H))
        # the coefficient 3/2 is forced
        for bad in (sp.Integer(1), sp.Rational(1, 2), sp.Integer(2)):
            def Sbad(X, n=n, bad=bad):
                m = multiplier(X, n)
                mi = einv(m)
                mi2 = emul(mi, mi)
                if n == 1:
                    A = emul(D_op(1)(d_t(m)), mi)
                    B = emul(emul(D_op(1)(m), d_t(m)), mi2)
                else:
                    A = emul(D_op(1)(D_op(2)(m)), mi)
                    B = emul(emul(D_op(1)(m), D_op(2)(m)), mi2)
                return escal(sp.Rational(1, 2), esub(A, escal(bad, B)))
            check("   the coefficient 3/2 is forced: %s fails" % bad,
                  not cocycle(Sbad, sp.Rational(4 - n, 2), n, G, H))

    # the second term of S_2 is (A_1 m)(A_2 m), not A_1A_2(m) m'
    print("\nthe second term of S_2")
    R.setup(2)
    R.set_order(3)
    G = Diffeo(gen(2), e, 2)
    H = Diffeo(gen(2, seed=3), d, 2)

    def S2_kir(X):
        m = multiplier(X, 2)
        mi = einv(m)
        A = emul(D_op(1)(D_op(2)(m)), mi)
        B = emul(emul(D_op(1)(D_op(2)(m)), d_t(m)), emul(mi, mi))
        return escal(sp.Rational(1, 2), esub(A, escal(sp.Rational(3, 2), B)))

    check("(D_1D_2 m) m' in place of (D_1 m)(D_2 m) is not a cocycle",
          not cocycle(S2_kir, sp.Integer(1), 2, G, H))

    # ------------------------------------------------------------- Comment 4)
    print("\nComment 4): the frame v_i,  n = 3")
    R.setup(3)
    R.set_order(3)
    G = Diffeo(gen(3), e, 3)
    n = 3
    m = multiplier(G, n)
    mi = einv(m)
    a = amatrix(G, n)
    v = [vfield(a[i], n) for i in range(n)]

    good = True
    for i in range(n):
        for j in range(n):
            good &= eeq(sprod(a[i], a[j], n), mi if i == j else {})
    check("(v_i, v_j) = m^{-1} delta_ij", good)

    vv = [[apply_to_row(v[i], a[j], n) for j in range(n)] for i in range(n)]
    check("v_i(v_j) = - v_j(v_i) for i != j",
          all(all(eeq(vv[i][j][k], escal(-1, vv[j][i][k])) for k in range(n))
              for i in range(n) for j in range(n) if i != j))
    check("v_i(v_i) = v_j(v_j)",
          all(all(eeq(vv[i][i][k], vv[0][0][k]) for k in range(n))
              for i in range(n)))

    # v_1(v_2) = (m^{-1}/2) v_2(m) v_1 - (m^{-1}/2) v_1(m) v_2
    #            + m (v_1(v_2), v_3) v_3
    c1 = escal(sp.Rational(1, 2), emul(mi, v[1](m)))
    c2 = escal(sp.Rational(-1, 2), emul(mi, v[0](m)))
    c3 = emul(m, sprod(vv[0][1], a[2], n))
    rhs = [eadd(emul(c1, a[0][k]), emul(c2, a[1][k]), emul(c3, a[2][k]))
           for k in range(n)]
    check("v_1(v_2) = (m^{-1}/2)(v_2(m) v_1 - v_1(m) v_2) + m (v_1(v_2), v_3) v_3",
          all(eeq(vv[0][1][k], rhs[k]) for k in range(n)))

    # v_1(v_1) = -(m^{-1}/2) sum_i v_i(m) v_i
    rhs = [eadd(*[emul(escal(sp.Rational(-1, 2), emul(mi, v[i](m))), a[i][k])
                  for i in range(n)]) for k in range(n)]
    check("v_1(v_1) = -(m^{-1}/2) sum_i v_i(m) v_i",
          all(eeq(vv[0][0][k], rhs[k]) for k in range(n)))

    # ------------------------------------------------------------------- S_3
    print("\nS_3 = c . m^{3/2} (v_1(v_2), v_3) alpha^{1/2}")
    S3 = lambda X: S3_frame(X, 3, -3)
    check("S_3 is not identically zero", bool(S3(G)))
    H = Diffeo(gen(3, seed=3), d, 3)
    check("S_3(GH) = S_3(H) + m_H^{1/2} H^*(S_3(G))",
          cocycle(S3, sp.Rational(1, 2), 3, G, H))
    for w in (sp.Rational(3, 2), sp.Integer(1), sp.Integer(0)):
        check("   and of no other weight: %s fails" % w,
              not cocycle(S3, w, 3, G, H))

    # the constant: the pattern of S_1, S_2 is  S_n = (1/2)(D_n(m)/m - ...)
    # with D_3 = D_1D_2D_3 d_t^{-1}, so the first order in e must be
    # (1/2) D_1D_2D_3 d_t^{-1}(2 d_t g) = D_1D_2D_3(g).
    g = gen(3)
    want = D_op(1)(D_op(2)(D_op(3)(g)))
    got = first_order(S3_frame(G, 3, 1))
    got = {S: sp.expand(sp.diff(c, e).subs({e: 0, d: 0})) for S, c in got.items()}
    got = clean({S: c for S, c in got.items() if c != 0})
    ratios = set()
    for S in set(list(got) + list(want)):
        x, y = got.get(S), want.get(S)
        if y in (None, 0):
            ratios.add(sp.nan if x else sp.Integer(0))
        else:
            ratios.add(sp.simplify(sp.cancel((x or 0) / y)))
    ratios.discard(sp.Integer(0))
    print("     linearization of m^{3/2}(v_1(v_2),v_3) / D_1D_2D_3(g) = %s"
          % sorted(ratios, key=str))
    check("the constant continuing the normalization of S_1, S_2 is -1 "
          "(Radul prints -3)",
          len(ratios) == 1 and sp.simplify(list(ratios)[0] + 1) == 0)

    print("\n%d ok, %d FAILED" % (OK, BAD))
