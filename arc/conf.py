r"""
"Conformality" in the sense of [Be2, Ch. 3, Sec. 8.8a 2), p. 260] versus the
superconformality condition (110) of the text.

Page 260 defines: a Lie (super)algebra is CONFORMAL if it preserves a tensor B
up to a factor; the elements K_F of k^L(1|n) multiply alpha . bar alpha by
K_1(F) . K_1(F).  The group version of that is
G^*(alpha_1) = m_G alpha_1.  What is checked here is that (110) is exactly that
condition, no more and no less:

  (I)  THE IDENTITY (holds for EVERY change of coordinates t -> f, th_i -> ph_i,
       contact or not):

           D_j(f) - sum_i ph_i D_j(ph_i)  =  c_j + th_j m,

       where     m   := d_t f + sum_i ph_i d_t(ph_i)      [the dt-component of G^*(alpha_1)]
                 c_j := d_th_j f - sum_i ph_i d_th_j(ph_i) [its d th_j-component]

       Proof:  D_j = d_th_j + th_j d_t, and ph_i th_j = - th_j ph_i, so
       ph_i D_j(ph_i) = ph_i d_th_j(ph_i) - th_j ph_i d_t(ph_i).

  (II) Hence (110) [the left-hand side vanishes] <=> c_j = - th_j m for all j
       <=> G^*(alpha_1) = m alpha_1, alpha_1 = dt + sum th_i d th_i.  So (110) IS
       conformality in the sense of p. 260, with the factor m = m_G of eq. (111).

  (III) U := (D_i(ph_j)) satisfies U^t U = m 1_n, i.e. U = m^{1/2} O with O
       orthogonal: the odd directions undergo a CLASSICAL conformal map, factor
       m.  For n = 1 the group O(1) = {+-1} is discrete, so U itself is the
       factor: the FORM d th is multiplied by U, whence d th . bar d th by
       U bar U -- conformality of a genuine bilinear form, which is what
       [Be2, p. 261] means by "superconformal in the classical sense of the
       word", and there ber = m^{1/2} = +- U: the volume element transforms as
       d th.  For n = 2, ber = 1.

Run:  ../venv/bin/python conf.py        (needs sympy)
"""
import sys

import sympy as sp

from lfgrp import G, Map, const, gen, osp_point, power, mat_det, mat_inv, t

FAIL = []


def check(name, ok):
    print("   %-62s %s" % (name, "OK" if ok else "FAILED"))
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ the pieces

def dt_component(M):
    """m := d_t f + sum_i ph_i d_t(ph_i);  the dt-component of G^*(alpha_1)."""
    m = M.f.dt()
    for p in M.ph:
        m = m + p * p.dt()
    return m


def dth_component(M, j):
    """c_j := d_th_j f - sum_i ph_i d_th_j(ph_i); the d th_j-component."""
    c = M.f.dodd(M.ths[j])
    for p in M.ph:
        c = c - p * p.dodd(M.ths[j])
    return c


def scf_residual(M, j):
    """D_j(tilde t) - sum_i tilde th_i D_j(tilde th_i):  the left side of (110)."""
    r = M.D(j, M.f)
    for p in M.ph:
        r = r - p * M.D(j, p)
    return r


def Umat(M):
    n = len(M.ths)
    return [[M.D(i, M.ph[j]) for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- the test set

def osp(n):
    M, den, ok_E, _ = osp_point(n)
    return M, den, ok_E


def generators(n):
    """the five generators of eq. (114), plus two deliberately NON-contact maps."""
    ths = ['th%d' % i for i in range(1, n + 1)]
    eps = ['ep%d' % i for i in range(1, n + 1)]
    G.order = eps + ths
    th = [gen(x) for x in ths]
    ep = [gen(x) for x in eps]
    b, lam = sp.symbols('b lam')
    ps = sp.Symbol('ps')
    out = []

    out.append(("translation  t+b", Map(const(t + b), th, ths)))
    out.append(("dilation  (lam^2 t, lam th)",
                Map(const(lam ** 2 * t), [x * lam for x in th], ths)))
    out.append(("inversion  (-1/t, th/t)",
                Map(const(-1 / t), [x * (1 / t) for x in th], ths)))
    if n >= 2:                       # a genuine rotation needs two coordinates
        O = [[sp.Integer(1) if i == j else sp.Integer(0) for j in range(n)]
             for i in range(n)]
        O[0][0] = sp.cos(ps); O[0][1] = -sp.sin(ps)
        O[1][0] = sp.sin(ps); O[1][1] = sp.cos(ps)
        out.append(("rotation  th -> O th",
                    Map(const(t),
                        [sum((th[k] * O[j][k] for k in range(n)), const(0))
                         for j in range(n)], ths)))
    sup = sum((ep[i] * th[i] for i in range(n)), const(0))
    out.append(("supertranslation  (t + eps^t th, th - eps)",
                Map(const(t) + sup, [th[i] - ep[i] for i in range(n)], ths)))
    # the two maps that are NOT contact: the identity (I) must hold anyway
    out.append(("NOT contact: (t + eps^t th, th + eps)",
                Map(const(t) + sup, [th[i] + ep[i] for i in range(n)], ths)))
    out.append(("NOT contact: (t^2, th)", Map(const(t ** 2), th, ths)))
    return out


# ---------------------------------------------------------------------- checks

def check_identity(tag, M):
    """(I): the identity, unconditionally."""
    n = len(M.ths)
    m = dt_component(M)
    ok = True
    for j in range(n):
        lhs = scf_residual(M, j)
        rhs = dth_component(M, j) + gen(M.ths[j]) * m
        if not (lhs - rhs).is_zero():
            ok = False
    check("identity  D_j(f)-sum ph D_j(ph) = c_j + th_j m   [%s]" % tag, ok)


def check_conformality(tag, M, den=None):
    """(II): c_j = - th_j m  <=>  G^*(alpha_1) = m alpha_1  <=>  (110)."""
    n = len(M.ths)
    m = dt_component(M)
    conf = all((dth_component(M, j) + gen(M.ths[j]) * m).is_zero()
               for j in range(n))
    scf = all(scf_residual(M, j).is_zero() for j in range(n))
    check("G^*(alpha_1) = m alpha_1  (all components)      [%s]" % tag, conf)
    check("(110): D_j(t~) = sum_i th~_i D_j(th~_i)          [%s]" % tag, scf)
    check("the two are equivalent here                     [%s]" % tag,
          conf == scf)
    if den is not None:
        check("m = (Ct+D+de^t th)^{-2}, eq. (111)               [%s]" % tag,
              (m - (den * den).inv()).is_zero())
    return m


def check_CO(tag, M, m, sqrt_m=None):
    """(III): U^t U = m 1_n, and U = m^{1/2} O with O orthogonal."""
    n = len(M.ths)
    U = Umat(M)
    ok = True
    for i in range(n):
        for j in range(n):
            v = sum((U[k][i] * U[k][j] for k in range(n)), const(0))
            tgt = m if i == j else const(0)
            if not (v - tgt).is_zero():
                ok = False
    check("U^t U = m 1_n   (U is m^{1/2} times orthogonal)  [%s]" % tag, ok)
    if sqrt_m is not None:
        Oo = [[U[i][j] * sqrt_m.inv() for j in range(n)] for i in range(n)]
        ok = True
        for i in range(n):
            for j in range(n):
                v = sum((Oo[k][i] * Oo[k][j] for k in range(n)), const(0))
                tgt = const(1 if i == j else 0)
                if not (v - tgt).is_zero():
                    ok = False
        check("O := m^{-1/2} U is orthogonal                   [%s]" % tag, ok)


def check_ber(tag, M, m, sqrt_m):
    """n = 1: ber = U = m^{1/2}   ([Be2, p. 261]: vol transforms as d th)."""
    n = len(M.ths)
    b = M.ber()
    check("ber^2 = m^{2-n}                                 [%s]" % tag,
          (b * b - power(m, 2 - n)).is_zero())
    if n == 1:
        U = M.D(0, M.ph[0])
        check("n=1: ber = U = m^{1/2}  (vol transforms as d th) [%s]" % tag,
              (b - U).is_zero() and (U - sqrt_m).is_zero())
    if n == 2:
        check("n=2: ber = 1                                    [%s]" % tag,
              (b - const(1)).is_zero())


# ------------------------------------------------------------------------ main

if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [1, 2, 3]

    for n in ns:
        print("=" * 78)
        print("n = %d :  the general C-point of OSp(%d|2) acting by eq. (109)" % (n, n))
        M, den, ok_E = osp(n)
        check("E^t E = 1 - N  (the C-point is in OSp)", ok_E)
        check_identity("OSp point", M)
        m = check_conformality("OSp point", M, den)
        check_CO("OSp point", M, m, den.inv())
        check_ber("OSp point", M, m, den.inv())

        print("-" * 78)
        print("n = %d :  the generators of eq. (114), and two non-contact maps" % n)
        for tag, Mg in generators(n):
            check_identity(tag, Mg)
            mg = dt_component(Mg)
            conf = all((dth_component(Mg, j) + gen(Mg.ths[j]) * mg).is_zero()
                       for j in range(len(Mg.ths)))
            scf = all(scf_residual(Mg, j).is_zero() for j in range(len(Mg.ths)))
            check("conformal? %s   (110)? %s   -- and they agree"
                  % ("yes" if conf else "NO ", "yes" if scf else "NO "),
                  conf == scf)
            print("       m_G = %s,   U = %s" % (mg, Umat(Mg)))

    print("=" * 78)
    print("failures: %d" % len(FAIL))
    for f in FAIL:
        print("   ", f)
