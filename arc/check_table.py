"""
Independent verification of the central Table (61) of the text and of
the component formulas (48) and (50).

Step 0 (validation of the engine and of the text's formulas (6), (7),
        (11), (14)):
            [K_f, K_g] = K_{{f,g}_{k.b.}}          on R^L(n)
            [K^M_f, K^M_g] = K^M_{{f,g}_{R.b.}}    on R^M(n)
Step 1: is each *printed* component cocycle a cocycle?
Step 2: do the generating-function cocycles of Table (61) reproduce them,
        and with which constant?
"""

from fractions import Fraction as F
from itertools import combinations
from stringy import (mono, add, mul, smul, d_t, d_odd, parity, eq, show, VF)
from check_cocycles import (L_bracket, L_res, M_bracket, M_res, M_ops, Delta,
                            monomials)

# --------------------------------------------------------------- the fields

def K_L(n, f):
    """K_f for k^L(1|n), eq. (6)."""
    E = lambda h: {(a, S): len(S) * c for (a, S), c in h.items() if len(S)}
    nab = lambda h: {k: (2 - len(k[1])) * c for k, c in h.items() if (2 - len(k[1]))}
    comps = {'t': nab(f)}
    p = parity(f)
    for i in range(1, n + 1):
        ci = smul(-(-1) ** p, add(d_odd(i, f), smul(-1, mul(mono(0, (i,)), d_t(f)))))
        if ci:
            comps[i] = ci
    return VF(comps, tuple(range(1, n + 1)))

def K_M(n, f):
    """K^M_f = nabla(f) D + H^M_f + D(f) E,  eqs. (11), (12), (13).
    Odd coordinates: 0 = zeta, 1..n-1 = theta_i."""
    Dcal = M_ops(n)[0]
    nab = lambda h: {k: (2 - len(k[1])) * c for k, c in h.items() if (2 - len(k[1]))}
    p = parity(f)
    nf, Df = nab(f), Dcal(f)
    # nabla(f) * D  with D = d_t - zeta/(2t) d_zeta
    comps = {'t': nf, 0: smul(F(-1, 2), mul(nf, mono(-1, (0,))))}
    # H^M_f = (-1)^p ( (1/t) d_zeta f  d_zeta + sum d_theta_i f  d_theta_i )
    comps[0] = add(comps[0], smul((-1) ** p, mul(mono(-1), d_odd(0, f))))
    for i in range(1, n):
        comps[i] = smul((-1) ** p, d_odd(i, f))
    # D(f) * E  with E = zeta d_zeta + sum theta_i d_theta_i
    comps[0] = add(comps[0], mul(Df, mono(0, (0,))))
    for i in range(1, n):
        comps[i] = add(comps.get(i, {}), mul(Df, mono(0, (i,))))
    comps = {k: v for k, v in comps.items() if v}
    return VF(comps, tuple(range(0, n)))

def validate_fields():
    print("=" * 72)
    print("STEP 0.  [K_f, K_g] = K_{{f,g}} ?")
    for n in (1, 2, 3, 4):
        keys = monomials(n, range(-2, 3), tuple(range(1, n + 1)))
        bad = 0
        for kf in keys:
            for kg in keys:
                f, g = mono(*kf), mono(*kg)
                lhs = K_L(n, f).bracket(K_L(n, g))
                rhs = K_L(n, L_bracket(n, f, g))
                if not (lhs == rhs):
                    bad += 1
                    if bad == 1:
                        print("     first failure L n=%d: %s , %s" % (n, kf, kg))
                        print("       [K_f,K_g] =", lhs)
                        print("       K_{f,g}   =", rhs)
        print("  k^L(1|%d): %s" % (n, "OK" if bad == 0 else "%d failures" % bad))
    for n in (1, 2, 3, 4):
        keys = monomials(n, range(-2, 3), tuple(range(0, n)))
        bad = 0
        for kf in keys:
            for kg in keys:
                f, g = mono(*kf), mono(*kg)
                lhs = K_M(n, f).bracket(K_M(n, g))
                rhs = K_M(n, M_bracket(n, f, g))
                if not (lhs == rhs):
                    bad += 1
                    if bad == 1:
                        print("     first failure M n=%d: %s , %s" % (n, kf, kg))
                        print("       [K_f,K_g] =", lhs)
                        print("       K_{f,g}   =", rhs)
        print("  k^M(1|%d): %s" % (n, "OK" if bad == 0 else "%d failures" % bad))

# ---------------------------------------------- printed component cocycles

def kvdl_L4(kf, kg, alpha=0, beta=0, gamma=0):
    """(48) for k^L(1|4), monomials given by keys (a, S), S subset of 1..4.
    L_m = t^{m+1};  E^i_m = t^{m+1} th_i;  T^{ij}_m = t^m th_i th_j;
    F^i_m = t^m d_{th_i}(th1 th2 th3 th4);  S_m = t^{m-1} th1th2th3th4."""
    (a1, S1), (a2, S2) = kf, kg
    k1, k2 = len(S1), len(S2)
    if k1 == 0 and k2 == 0:
        m, nn = a1 - 1, a2 - 1
        return alpha * m * (m * m - 1) if m + nn == 0 else F(0)
    if k1 == 1 and k2 == 1:
        if S1 != S2:
            return F(0)
        m, nn = a1 - 1, a2 - 1
        return alpha * m * (m + 1) if m + nn + 1 == 0 else F(0)
    if k1 == 2 and k2 == 2:
        m, nn = a1, a2
        if S1 == S2:
            return -alpha * m if m + nn == 0 else F(0)
        if set(S1) & set(S2):
            return F(0)                       # e.g. th1th2 vs th1th3
        # complementary pairs: (i,j,k,l) in A_4
        sgn = _perm_sign(S1 + S2)
        return -beta * m * sgn if m + nn == 0 else F(0)
    if k1 == 3 and k2 == 3:
        if S1 != S2:
            return F(0)
        m, nn = a1, a2
        # F^i_m = t^m d_{th_i}(th1..th4) = +- t^m th_j th_k th_l
        s1 = _dsign(S1)
        return -alpha * s1 * s1 if m + nn + 1 == 0 else F(0)
    if k1 == 4 and k2 == 4:
        m, nn = a1 + 1, a2 + 1
        return alpha * F(1, 1) / m if (m + nn == 0 and m != 0) else F(0)
    if k1 == 0 and k2 == 4:
        m, nn = a1 - 1, a2 + 1
        return (gamma + beta * m) if m + nn == 0 else F(0)
    if k1 == 4 and k2 == 0:
        return -kvdl_L4(kg, kf, alpha, beta, gamma)     # c antisymmetric on even
    if k1 == 1 and k2 == 3:
        i = S1[0]
        if i in S2:
            return F(0)
        m, nn = a1 - 1, a2
        s2 = _dsign(S2)
        v = (gamma / F(2) + beta * (m + F(1, 2))) if m + nn + 1 == 0 else F(0)
        return v * s2
    if k1 == 3 and k2 == 1:
        return kvdl_L4(kg, kf, alpha, beta, gamma)      # symmetric on odd pairs
    return F(0)

def _perm_sign(S):
    """sign of the permutation taking S (a tuple of 4 distinct labels) to sorted order"""
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    return sgn

def _dsign(S3):
    """t^m d_{th_i}(th1th2th3th4) = s * t^m * (the sorted product of S3); returns s"""
    i = [j for j in (1, 2, 3, 4) if j not in S3][0]
    # d_{th_i}(th1 th2 th3 th4) = (-1)^{i-1} * product of the others in order
    return (-1) ** (i - 1)

def cocycle_test(keys, bracket, c):
    bad = []
    for kf in keys:
        for kg in keys:
            for kh in keys:
                f, g, h = mono(*kf), mono(*kg), mono(*kh)
                lhs = c(f, bracket(g, h))
                rhs = c(bracket(f, g), h) + (-1) ** (parity(f) * parity(g)) * c(g, bracket(f, h))
                if lhs != rhs:
                    bad.append((kf, kg, kh, lhs, rhs))
    return bad

def bilinear_from_keys(cfun):
    """extend a function of two monomial keys to functions"""
    def c(f, g):
        tot = F(0)
        for kf, cf in f.items():
            for kg, cg in g.items():
                tot += cf * cg * cfun(kf, kg)
        return tot
    return c

# ------------------------------------------- generating-function cocycles

def L4_ops():
    def Dth(i):
        return lambda f: add(mul(mono(0, (i,)), d_t(f)), d_odd(i, f))
    def Dinv(f):
        out = {}
        for (a, S), c in f.items():
            if a == -1:
                raise ZeroDivisionError("D_1^{-1} undefined on t^{-1}: %s" % show(f))
            out[(a + 1, S)] = out.get((a + 1, S), 0) + c / (a + 1)
        return {k: v for k, v in out.items() if v}
    return Dth, Dinv

def table_cocycles_L4():
    Dth, Dinv = L4_ops()
    def c1(f, g):                       # Res(f D_1^{-1} D_th1 D_th2 D_th3 D_th4 (g))
        h = g
        for i in (4, 3, 2, 1):
            h = Dth(i)(h)
        return L_res(4, mul(f, Dinv(h)))
    def c2(f, g):                       # Res(f (2-E)/t (g))
        h = {(a - 1, S): (2 - len(S)) * c for (a, S), c in g.items() if (2 - len(S))}
        return L_res(4, mul(f, h))
    def c3(f, g):                       # Res(f d_t(g))
        return L_res(4, mul(f, d_t(g)))
    return c1, c2, c3

def run_L4():
    print("=" * 72)
    print("STEP 1/2.  k^L'(1|4): the three cocycles")
    labels = (1, 2, 3, 4)
    keys = [k for k in monomials(4, range(-3, 4), labels)
            if not (k[0] == -1 and len(k[1]) == 4)]        # k^L'(1|4)
    small = [k for k in monomials(4, range(-2, 3), labels)
             if not (k[0] == -1 and len(k[1]) == 4)]
    for name, params in (("alpha", (1, 0, 0)), ("beta", (0, 1, 0)), ("gamma", (0, 0, 1))):
        c = bilinear_from_keys(lambda kf, kg, p=params: kvdl_L4(kf, kg, F(p[0]), F(p[1]), F(p[2])))
        bad = cocycle_test(small, lambda f, g: L_bracket(4, f, g), c)
        print("  printed (48), %-5s part: cocycle identity %s"
              % (name, "OK" if not bad else "FAILS (%d, e.g. %s: %s vs %s)"
                 % (len(bad), bad[0][:3], bad[0][3], bad[0][4])))
    c1, c2, c3 = table_cocycles_L4()
    for nm, cf in (("(1) Res(f D_1^{-1}D_th1..D_th4 g)", c1),
                   ("(2) Res(f (2-E)/t g)", c2),
                   ("(3) Res(f d_t g)", c3)):
        bad = cocycle_test(small, lambda f, g: L_bracket(4, f, g), lambda f, g: cf(f, g))
        print("  MainT %-34s: cocycle identity %s" % (nm, "OK" if not bad else
              "FAILS (%d, e.g. %s: %s vs %s)" % (len(bad), bad[0][:3], bad[0][3], bad[0][4])))
    # compare each MainT cocycle with the printed alpha/beta/gamma ones
    for nm, cf in (("(1)", c1), ("(2)", c2), ("(3)", c3)):
        for name, params in (("alpha", (1, 0, 0)), ("beta", (0, 1, 0)), ("gamma", (0, 0, 1))):
            ck = bilinear_from_keys(lambda kf, kg, p=params: kvdl_L4(kf, kg, F(p[0]), F(p[1]), F(p[2])))
            ratios, mism = {}, 0
            for kf in keys:
                for kg in keys:
                    v1 = cf(mono(*kf), mono(*kg))
                    v2 = ck(mono(*kf), mono(*kg))
                    if v1 == 0 and v2 == 0:
                        continue
                    if v1 == 0 or v2 == 0:
                        mism += 1
                        continue
                    ratios.setdefault(F(v1) / F(v2), []).append((kf, kg))
            if ratios or mism:
                print("    MainT %s  vs printed %-5s : ratios %s, one-sided zeros %d"
                      % (nm, name, {str(k): len(v) for k, v in ratios.items()}, mism))

if __name__ == "__main__":
    validate_fields()
    run_L4()
