r"""
The vector (block) form of eq. (105) and of the formulas derived from it,
with gamma := (gamma_1, ...)^t and so on.

With  al := (al_1,...,al_n)^t,  be, ga, de  likewise (columns of odd elements
of C), th := (th_1,...,th_n)^t and E = (E_ij) in Mat(n; C_ev), the supermatrix
of eq. (105) and its supertranspose in the (1,1,n) format are

        / A    B    ga^t \                / A    C   -al^t \
    X = | C    D    de^t | ,      X^{st} = | B    D   -be^t | ,
        \ al   be   E    /                \ ga   de   E^t  /

and this file checks, over the Grassmann algebra of lfgrp.py (odd generators
al_i, be_i, th_i; even parameters A, B, C and the coordinate t):

 (1) that this block supertranspose is the entrywise one of eq. (67),
     i.e. that X^{st} B X = B holds for the solved point of lfgrp.osp_point;

 (2) that X^{st} B X = B is, in vector form, the system

        AD - BC - al^t be = 1,
        A de - C ga - E^t al = 0,      B de - D ga - E^t be = 0,
        E^{st} E + ga de^t - de ga^t = 1_n,

     which for n = 1 is exactly the printed system (sy);

 (3) that the solved system (108) in vector form,

        Delta := AD - BC = 1 + al^t be,
        ga^t = Delta^{-1} (B al - A be)^t E,   de^t = Delta^{-1} (D al - C be)^t E,
        E^{st} E = 1_n - N,   N = ga de^t - de ga^t   (symmetric),

     agrees entry by entry with the indexed form printed in the text;

 (4) that the vector form of the transformations (109), of the multiplier
     (111), of the superconformality condition (110) --- namely
     D_th(t~) = U th~ with D_th := (D_1,...,D_n)^t --- and of the generators
     (114) agree entry by entry with the indexed forms.

Run:  python3 vecform.py        (needs sympy; ../venv/bin/python also works)
"""
import sys

import sympy as sp

from lfgrp import G, const, gen, osp_point, Map, power


def zeros(n, m):
    return [[const(0)] * m for _ in range(n)]


def mat_mul(P, Q):
    n, k, m = len(P), len(Q), len(Q[0])
    out = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = const(0)
            for l in range(k):
                s = s + P[i][l] * Q[l][j]
            out[i][j] = s
    return out


def mat_sub(P, Q):
    return [[P[i][j] - Q[i][j] for j in range(len(P[0]))] for i in range(len(P))]


def mat_is(P, Q, tag, report):
    ok = all((P[i][j] - Q[i][j]).is_zero()
             for i in range(len(P)) for j in range(len(P[0])))
    report.append((tag, ok))
    return ok


def build(n):
    """the C-point of OSp(n|2) of lfgrp, returned in vector blocks."""
    M, den, ok_E, (A, B, C, Dg) = osp_point(n)
    ths = ['th%d' % i for i in range(1, n + 1)]
    al = [gen('al%d' % i) for i in range(1, n + 1)]
    be = [gen('be%d' % i) for i in range(1, n + 1)]
    Delta = const(1) + sum((al[i] * be[i] for i in range(n)), const(0))
    Dinv = Delta.inv()
    # rebuild E, ga, de exactly as osp_point does (its locals are not returned)
    O = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
    E = [row[:] for row in O]
    for _ in range(4):
        ga = [sum(((const(B) * al[i] - const(A) * be[i]) * E[i][j] for i in range(n)),
                  const(0)) * Dinv for j in range(n)]
        de = [sum(((Dg * al[i] - const(C) * be[i]) * E[i][j] for i in range(n)),
                  const(0)) * Dinv for j in range(n)]
        N = [[ga[i] * de[j] - de[i] * ga[j] for j in range(n)] for i in range(n)]
        S = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
        Npow = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
        for k in range(1, 2 * n + 2):
            Npow = mat_mul(Npow, N)
            if all(not Npow[i][j].d for i in range(n) for j in range(n)):
                break
            cf = sp.binomial(sp.Rational(1, 2), k) * (-1) ** k
            for i in range(n):
                for j in range(n):
                    S[i][j] = S[i][j] + Npow[i][j] * cf
        E = mat_mul(O, S)
    return dict(n=n, A=const(A), B=const(B), C=const(C), D=Dg, al=al, be=be,
                ga=ga, de=de, E=E, N=N, Delta=Delta, ths=ths, M=M, den=den,
                ok_E=ok_E)


def supermatrix(d):
    """X of eq. (105) written out entrywise, rows/cols ordered (1, 2, 3..n+2)."""
    n = d['n']
    X = zeros(n + 2, n + 2)
    X[0][0], X[0][1] = d['A'], d['B']
    X[1][0], X[1][1] = d['C'], d['D']
    for j in range(n):
        X[0][2 + j] = d['ga'][j]           # ga^t in the first row
        X[1][2 + j] = d['de'][j]           # de^t in the second row
        X[2 + j][0] = d['al'][j]           # al as a column
        X[2 + j][1] = d['be'][j]           # be as a column
        for i in range(n):
            X[2 + i][2 + j] = d['E'][i][j]
    return X


def st_block(d):
    r"""X^{st} written from the blocks, in the convention of eq. (67):
       / A   C  -al^t \
       | B   D  -be^t |
       \ ga  de  E^t  /                                                   """
    n = d['n']
    Y = zeros(n + 2, n + 2)
    Y[0][0], Y[0][1] = d['A'], d['C']
    Y[1][0], Y[1][1] = d['B'], d['D']
    for j in range(n):
        Y[0][2 + j] = d['al'][j] * (-1)
        Y[1][2 + j] = d['be'][j] * (-1)
        Y[2 + j][0] = d['ga'][j]
        Y[2 + j][1] = d['de'][j]
        for i in range(n):
            Y[2 + i][2 + j] = d['E'][j][i]
    return Y


def gram(n):
    B = zeros(n + 2, n + 2)
    B[0][1], B[1][0] = const(1), const(-1)
    for i in range(n):
        B[2 + i][2 + i] = const(1)
    return B


def check(n, report):
    d = build(n)
    A, B, C, D = d['A'], d['B'], d['C'], d['D']
    al, be, ga, de, E, N = d['al'], d['be'], d['ga'], d['de'], d['E'], d['N']
    Et = [[E[j][i] for j in range(n)] for i in range(n)]

    # ---- (1) the block supertranspose solves X^{st} B X = B
    X, Y, Bm = supermatrix(d), st_block(d), gram(n)
    mat_is(mat_mul(Y, mat_mul(Bm, X)), Bm, "n=%d  X^{st} B X = B (block st)" % n, report)

    # ---- (2) the unsolved system in vector form
    altbe = sum((al[i] * be[i] for i in range(n)), const(0))            # al^t be
    report.append(("n=%d  AD - BC - al^t be = 1" % n,
                   (A * D - B * C - altbe - const(1)).is_zero()))
    r1 = [A * de[j] - C * ga[j] - sum((Et[j][i] * al[i] for i in range(n)), const(0))
          for j in range(n)]
    r2 = [B * de[j] - D * ga[j] - sum((Et[j][i] * be[i] for i in range(n)), const(0))
          for j in range(n)]
    report.append(("n=%d  A de - C ga - E^t al = 0" % n,
                   all(x.is_zero() for x in r1)))
    report.append(("n=%d  B de - D ga - E^t be = 0" % n,
                   all(x.is_zero() for x in r2)))
    lhs = mat_mul(Et, E)
    Nv = [[ga[i] * de[j] - de[i] * ga[j] for j in range(n)] for i in range(n)]
    one = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
    mat_is([[lhs[i][j] + Nv[i][j] for j in range(n)] for i in range(n)], one,
           "n=%d  E^{st}E + ga de^t - de ga^t = 1_n" % n, report)

    # ---- (3) the solved system (108) in vector form
    Dinv = d['Delta'].inv()
    report.append(("n=%d  Delta = AD - BC = 1 + al^t be" % n,
                   (A * D - B * C - d['Delta']).is_zero()))
    #   ga^t = Delta^{-1} (B al - A be)^t E  :  j-th entry
    gav = [sum(((B * al[i] - A * be[i]) * E[i][j] for i in range(n)), const(0)) * Dinv
           for j in range(n)]
    dev = [sum(((D * al[i] - C * be[i]) * E[i][j] for i in range(n)), const(0)) * Dinv
           for j in range(n)]
    report.append(("n=%d  ga^t = Delta^{-1}(B al - A be)^t E" % n,
                   all((gav[j] - ga[j]).is_zero() for j in range(n))))
    report.append(("n=%d  de^t = Delta^{-1}(D al - C be)^t E" % n,
                   all((dev[j] - de[j]).is_zero() for j in range(n))))
    mat_is(Nv, N, "n=%d  N = ga de^t - de ga^t (= the indexed N)" % n, report)
    mat_is(Nv, [[Nv[j][i] for j in range(n)] for i in range(n)],
           "n=%d  N symmetric" % n, report)
    report.append(("n=%d  E^{st}E = 1_n - N" % n, d['ok_E']))

    # ---- (4) the transformations in vector form
    M, den, ths = d['M'], d['den'], d['ths']
    th = [gen(s) for s in ths]
    #   denominator  Ct + D + de^t th
    denv = C * const(sp.Symbol('t')) + D + sum((de[j] * th[j] for j in range(n)),
                                               const(0))
    report.append(("n=%d  denominator = Ct + D + de^t th" % n, (denv - den).is_zero()))
    numv = A * const(sp.Symbol('t')) + B + sum((ga[j] * th[j] for j in range(n)),
                                               const(0))
    report.append(("n=%d  t~ = (At + B + ga^t th)/(Ct + D + de^t th)" % n,
                   (numv * denv.inv() - M.f).is_zero()))
    phv = [(al[i] * const(sp.Symbol('t')) + be[i]
            + sum((E[i][j] * th[j] for j in range(n)), const(0))) * denv.inv()
           for i in range(n)]
    report.append(("n=%d  th~ = (al t + be + E th)/(Ct + D + de^t th)" % n,
                   all((phv[i] - M.ph[i]).is_zero() for i in range(n))))
    m, ok_D = M.multiplier_D()
    report.append(("n=%d  m_G = (Ct + D + de^t th)^{-2}" % n,
                   (m - power(denv, -2)).is_zero()))
    #   U := (D_i(th~_j)) ,  U^t U = m_G 1_n ,  and  D_th(t~) = U th~
    U = [[M.D(i, M.ph[j]) for j in range(n)] for i in range(n)]
    Ut = [[U[j][i] for j in range(n)] for i in range(n)]
    mat_is(mat_mul(Ut, U), [[m if i == j else const(0) for j in range(n)]
                            for i in range(n)], "n=%d  U^t U = m_G 1_n" % n, report)
    report.append(("n=%d  D_th(t~) = U th~  (vector form of (110))" % n,
                   all((M.D(i, M.f)
                        - sum((U[i][j] * M.ph[j] for j in range(n)), const(0))).is_zero()
                       for i in range(n))))
    return d


def check_generators(n, report):
    """the generators (114) in vector form: th -> O th and the supertranslation.

    The sign: K_th_i = th_i d_t - d_th_i, so exp(sum_i eps_i K_th_i) is
        t -> t + eps^t th,   th -> th - eps
    (the series stops after one term).  This satisfies (110); the variant with
    BOTH signs + does not -- although the multiplier
    is m_G = 1 for either, U being 1_n.
    """
    tt = sp.Symbol('t')
    ths = ['th%d' % i for i in range(1, n + 1)]
    eps = ['ep%d' % i for i in range(1, n + 1)]
    G.order = eps + ths
    th = [gen(s) for s in ths]
    ep = [gen(s) for s in eps]

    def supertransl(s_t, s_th):
        f = const(tt) + sum((ep[i] * th[i] for i in range(n)), const(0)) * s_t
        return Map(f, [th[i] + ep[i] * s_th for i in range(n)], ths)

    M = supertransl(+1, -1)                       # exp(sum eps_i K_th_i)
    m, okU = M.multiplier_D()
    report.append(("n=%d  supertranslation t -> t + eps^t th, th -> th - eps: "
                   "m_G = 1 and (110) holds" % n,
                   okU and (m - const(1)).is_zero() and M.superconformal()))
    M2 = supertransl(-1, +1)                      # the same with eps -> -eps
    report.append(("n=%d  ... the same for eps -> -eps  (the point al=0, be=eps)" % n,
                   M2.superconformal()))
    M3 = supertransl(+1, +1)                      # both signs +
    m3, okU3 = M3.multiplier_D()
    report.append(("n=%d  ... whereas both signs + fail (110), though m_G = 1" % n,
                   (not M3.superconformal()) and okU3 and (m3 - const(1)).is_zero()))
    # rotation  th -> O th  (a plane rotation, O in O(n))
    if n >= 2:
        ps = sp.Symbol('ps')
        O = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
        O[0][0] = const(sp.cos(ps)); O[0][1] = const(-sp.sin(ps))
        O[1][0] = const(sp.sin(ps)); O[1][1] = const(sp.cos(ps))
        ph = [sum((O[i][j] * th[j] for j in range(n)), const(0)) for i in range(n)]
        M = Map(const(tt), ph, ths)
        m, ok = M.multiplier_form()
        report.append(("n=%d  rotation th -> O th is contact, m_G = 1" % n,
                       ok and (m - const(1)).is_zero()))


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    report = []
    for n in ns:
        print("=" * 72)
        print("n = %d" % n)
        check(n, report)
        check_generators(n, report)
        for tag, ok in report:
            if tag.startswith("n=%d " % n):
                print("   %-52s %s" % (tag, "OK" if ok else "FAILED"))
    bad = [t for t, ok in report if not ok]
    print("=" * 72)
    print("%d checks, %d failures" % (len(report), len(bad)))
    for t in bad:
        print("   FAILED:", t)
    sys.exit(1 if bad else 0)
