"""
Are the cocycles c_IS and c_KvdL proportional in the k^M case, and what are the
constants c_IS = lambda * c_KvdL in the k^L case?

This script computes both, from scratch:
  * c_IS(f,g) := Res(f * s_n(g))  with s_n from Table (61) / the proof of
    Theorem 4.5.1;
  * c_KvdL(f,g) from the component formulas (48) / (50),
    rewritten in generating functions;
and verifies (a) that c_IS is a 2-cocycle for the contact/Ramond bracket,
(b) that c_IS = lambda_n * c_KvdL for a single constant lambda_n.
"""

from fractions import Fraction
from itertools import combinations
from fractions import Fraction as Rational
def SS(x):
    return Rational(x)
from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, show

# ------------------------------------------------------------------ L case

def L_ops(n):
    """operators on R^L(n): odd labels 1..n"""
    D1 = d_t
    def Dth(i):
        return lambda f: add(mul(mono(0, (i,)), d_t(f)), d_odd(i, f))
    def Kth(i):
        return lambda f: add(mul(mono(0, (i,)), d_t(f)), smul(-1, d_odd(i, f)))
    return D1, Dth, Kth

def L_bracket(n, f, g):
    """contact bracket (7): (2-E)(f) dg/dt - df/dt (2-E)(g) - (-1)^p(f) sum d_i f d_i g"""
    def nabla(h):
        out = {}
        for (a, S), c in h.items():
            out[(a, S)] = out.get((a, S), 0) + (2 - len(S)) * c
        return {k: v for k, v in out.items() if v}
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in range(1, n + 1):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, smul(-(-1) ** parity(f), s))

def L_res(n, f):
    return f.get((-1, tuple(range(1, n + 1))), Rational(0))

def L_sn(n):
    D1, Dth, Kth = L_ops(n)
    def compose(*ops):
        def h(f):
            for op in reversed(ops):
                f = op(f)
            return f
        return h
    if n == 0:
        return compose(D1, D1, D1)
    if n == 1:
        return compose(Dth(1), D1, D1)
    if n == 2:
        return compose(Dth(1), Dth(2), D1)
    if n == 3:
        return compose(Dth(1), Dth(2), Dth(3))
    raise ValueError

def L_cKvdL(n, f, g):
    """cocycle c(alpha=0,beta=0,gamma=1)... for n<4 the KvdL cocycle in
    generating functions, read off (48) with L_m = K_{t^{m+1}} etc."""
    # only used for n<=3: the single cocycle, alpha != 0 (the "Virasoro-like" one)
    raise NotImplementedError

# ------------------------------------------------------------------ M case
# odd labels: 0 = zeta, 1..n-1 = theta_i

def M_ops(n):
    half = Rational(1,2)
    def Dcal(f):                      # D = d_t - zeta/(2t) d_zeta
        return add(d_t(f), smul(-half, mul(mono(-1, (0,)), d_odd(0, f))))
    def Dz(f):                        # D^M_zeta = zeta d_t + (1/t) d_zeta
        return add(mul(mono(0, (0,)), d_t(f)), mul(mono(-1), d_odd(0, f)))
    def Kz(f):                        # K^M_zeta = zeta d_t - (1/t) d_zeta
        return add(mul(mono(0, (0,)), d_t(f)), smul(-1, mul(mono(-1), d_odd(0, f))))
    def Dth(i):                       # D^M_{theta_i}
        return lambda f: add(mul(mono(0, (i,)), d_t(f)), d_odd(i, f),
                             smul(-half, mul(mono(-1, (i, 0)), d_odd(0, f))))
    def Kth(i):
        return lambda f: add(mul(mono(0, (i,)), d_t(f)), smul(-1, d_odd(i, f)),
                             smul(-half, mul(mono(-1, (i, 0)), d_odd(0, f))))
    return Dcal, Dz, Kz, Dth, Kth

def Delta(f):
    """Delta = D^M_zeta o D^{-1}:  t^a phi(theta) -> zeta t^a phi(theta);
                                   t^a zeta phi(theta) -> t^a phi/(a+1/2)."""
    out = {}
    for (a, S), c in f.items():
        if 0 in S:
            T = tuple(j for j in S if j != 0)      # zeta is first in canonical order
            out[(a, T)] = out.get((a, T), 0) + c / (a + Rational(1,2))
        else:
            T = (0,) + S
            out[(a, T)] = out.get((a, T), 0) + c
    return {k: v for k, v in out.items() if v}

def M_bracket(n, f, g):
    """Ramond bracket (14)."""
    def nabla(h):
        out = {}
        for (a, S), c in h.items():
            out[(a, S)] = out.get((a, S), 0) + (2 - len(S)) * c
        return {k: v for k, v in out.items() if v}
    Dcal = M_ops(n)[0]
    res = add(mul(nabla(f), Dcal(g)), smul(-1, mul(Dcal(f), nabla(g))))
    s = mul(mono(-1), mul(d_odd(0, f), d_odd(0, g)))
    for i in range(1, n):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, smul(-(-1) ** parity(f) * -1, s)) if False else \
           add(res, smul((-1) ** parity(f), s))

def M_res(n, f):
    """Res = coefficient of  t^{-1} zeta theta_1...theta_{n-1}."""
    return f.get((-1, tuple(range(0, n))), Rational(0))

def M_sn(n):
    Dcal, Dz, Kz, Dth, Kth = M_ops(n)
    def compose(*ops):
        def h(f):
            for op in reversed(ops):
                f = op(f)
            return f
        return h
    if n == 1:
        return compose(Dz, Dcal, Dcal)
    if n == 2:
        return compose(Dz, Dth(1), Dcal)
    if n == 3:
        return compose(Dz, Dth(2), Dth(1))
    if n == 4:
        return compose(Delta, Dth(3), Dth(2), Dth(1))
    raise ValueError

def M_cIS(n, f, g, sign=True):
    s = M_sn(n)
    val = M_res(n, mul(f, s(g)))
    if sign:
        val = val * (-1) ** parity(f)
    return val

def M_cKvdL(n, f, g):
    """(50) rewritten in generating functions.
    f, g are monomials.  Returns the value of the KvdL cocycle (corrected signs)."""
    (a, S), = f.items().__iter__().__next__()[0], None  # placeholder
    return None

def kvdl_value(fkey, gkey, n):
    """Value of the (corrected) KvdL cocycle on two monomials of R^M(n),
    given as keys (a, S).  Formulas from the \\iffalse appendix, in which
    zeta is label 0 and theta_i are 1..n-1."""
    (a1, S1), (a2, S2) = fkey, gkey
    if S1 != S2:
        return Rational(0)
    z = 0 in S1
    th = tuple(j for j in S1 if j != 0)
    k = len(th)
    if not z and k == 0:                       # t^{m+1}, t^{n+1}
        m, nn = a1 - 1, a2 - 1
        return m * (m * m - 1) if m + nn == 0 else Rational(0)
    if z and k == 0:                           # t^{m+1} zeta
        m, nn = a1 - 1, a2 - 1
        return Rational(m * m) - Rational(1,4) if m + nn == 0 else Rational(0)
    if not z and k == 1:                       # t^{m+1} theta
        m, nn = a1 - 1, a2 - 1
        return m * (m + 1) if m + nn + 1 == 0 else Rational(0)
    if z and k == 1:                           # t^m theta zeta  (canonical order zeta theta = -theta zeta)
        m, nn = a1, a2
        v = -(Rational(m) - Rational(1,2)) if m + nn - 1 == 0 else Rational(0)
        return v                              # sign of reordering cancels (both sides reordered)
    if not z and k == 2:                       # t^m theta_i theta_j
        m, nn = a1, a2
        return -m if m + nn == 0 else Rational(0)
    if z and k == 2:                           # t^m theta_i theta_j zeta
        m, nn = a1, a2
        return Rational(-1) if m + nn == 0 else Rational(0)
    if not z and k == 3:                       # t^m theta_1 theta_2 theta_3
        m, nn = a1, a2
        return Rational(-1) if m + nn + 1 == 0 else Rational(0)
    if z and k == 3:                           # t^{m-1} th1 th2 th3 zeta
        m, nn = a1 + 1, a2 + 1
        return Rational(1) / (Rational(m) - Rational(1,2)) if m + nn - 1 == 0 else Rational(0)
    return Rational(0)

# ------------------------------------------------------------------- checks

def monomials(n, arange, odd_labels):
    out = []
    for a in arange:
        for r in range(len(odd_labels) + 1):
            for S in combinations(odd_labels, r):
                out.append((a, S))
    return out

def check_cocycle_identity(n, bracket, cfun, keys):
    """c(f,{g,h}) = c({f,g},h) + (-1)^{p(f)p(g)} c(g,{f,h})"""
    bad = 0
    for kf in keys:
        for kg in keys:
            for kh in keys:
                f, g, h = mono(*kf), mono(*kg), mono(*kh)
                lhs = cfun(f, bracket(g, h))
                rhs = cfun(bracket(f, g), h) + (-1) ** (parity(f) * parity(g)) * cfun(g, bracket(f, h))
                if lhs - rhs != 0:
                    bad += 1
                    if bad < 4:
                        print("   COCYCLE FAIL", kf, kg, kh, lhs, rhs)
    return bad

def run_M(n, arange=range(-3, 4)):
    print("=" * 70)
    print("k^M(1|%d):  odd coordinates zeta=0, theta_1..theta_%d" % (n, n - 1))
    labels = tuple(range(0, n))          # 0 = zeta, 1..n-1 = theta
    keys = monomials(n, arange, labels)
    # (a) is c_IS a cocycle for the Ramond bracket?
    br = lambda f, g: M_bracket(n, f, g)
    cIS = lambda f, g: M_cIS(n, f, g, sign=True)
    small = monomials(n, range(-2, 3), labels)
    bad = check_cocycle_identity(n, br, cIS, small)
    print("  cocycle identity for c_IS (with (-1)^p(f)):  %s" % ("OK" if bad == 0 else "%d failures" % bad))
    cIS0 = lambda f, g: M_cIS(n, f, g, sign=False)
    bad0 = check_cocycle_identity(n, br, cIS0, small)
    print("  cocycle identity for c_IS (no sign factor):  %s" % ("OK" if bad0 == 0 else "%d failures" % bad0))
    # (b) proportionality with the KvdL cocycle
    for use_sign in (True, False):
        ratios = {}
        mism = []
        for kf in keys:
            for kg in keys:
                v1 = M_cIS(n, mono(*kf), mono(*kg), sign=use_sign)
                v2 = kvdl_value(kf, kg, n)
                if v1 == 0 and v2 == 0:
                    continue
                if v2 == 0 or v1 == 0:
                    mism.append((kf, kg, v1, v2))
                    continue
                r = Rational(v1) / Rational(v2)
                ratios.setdefault(r, []).append((kf, kg))
        tag = "with (-1)^p(f)" if use_sign else "no sign factor"
        print("  c_IS/c_KvdL (%s): ratios %s ; one-sided-zero mismatches: %d"
              % (tag, {str(k): len(v) for k, v in ratios.items()}, len(mism)))
        for m in mism[:4]:
            print("     mismatch:", m)

def run_L(n, arange=range(-3, 4)):
    print("=" * 70)
    print("k^L(1|%d)" % n)
    labels = tuple(range(1, n + 1))
    keys = monomials(n, arange, labels)
    br = lambda f, g: L_bracket(n, f, g)
    sn = L_sn(n)
    def cIS(f, g, sign=False):
        v = L_res(n, mul(f, sn(g)))
        return v * (-1) ** parity(f) if sign else v
    small = monomials(n, range(-2, 3), labels)
    bad = check_cocycle_identity(n, br, lambda f, g: cIS(f, g, False), small)
    print("  cocycle identity for c_IS (no sign): %s" % ("OK" if bad == 0 else "%d failures" % bad))
    bad = check_cocycle_identity(n, br, lambda f, g: cIS(f, g, True), small)
    print("  cocycle identity for c_IS (with (-1)^p(f)): %s" % ("OK" if bad == 0 else "%d failures" % bad))
    # compare with (48) for alpha=1,beta=gamma=0 restricted to n<4:
    # L_m = K_{t^{m+1}}, E^i_m = K_{t^{m+1} theta_i}, T^{ij}_m = K_{t^m th_i th_j},
    # F^i_m = K_{t^m d_{theta_i}(th1..th4)},  S_m = K_{t^{m-1} th1..th4}
    def kvdl_L(kf, kg):
        (a1, S1), (a2, S2) = kf, kg
        if S1 != S2:
            return Rational(0)
        k = len(S1)
        if k == 0:
            m, nn = a1 - 1, a2 - 1
            return m * (m ** 2 - 1) if m + nn == 0 else Rational(0)
        if k == 1:
            m, nn = a1 - 1, a2 - 1
            return m * (m + 1) if m + nn + 1 == 0 else Rational(0)
        if k == 2:
            m, nn = a1, a2
            return -m if m + nn == 0 else Rational(0)
        if k == 3:                      # F^i_m in k^L(1|4) <-> t^m th_j th_k th_l
            m, nn = a1, a2
            return Rational(-1) if m + nn + 1 == 0 else Rational(0)
        if k == 4:
            m, nn = a1 + 1, a2 + 1
            return Rational(1) / m if m + nn == 0 and m != 0 else Rational(0)
        return Rational(0)
    for use_sign in (True, False):
        ratios, mism = {}, []
        for kf in keys:
            for kg in keys:
                v1 = cIS(mono(*kf), mono(*kg), use_sign)
                v2 = kvdl_L(kf, kg)
                if v1 == 0 and v2 == 0:
                    continue
                if v1 == 0 or v2 == 0:
                    mism.append((kf, kg, v1, v2)); continue
                ratios.setdefault(Rational(v1) / Rational(v2), []).append((kf, kg))
        tag = "with (-1)^p(f)" if use_sign else "no sign factor"
        print("  c_IS/c_KvdL (%s): ratios %s ; one-sided-zero mismatches: %d"
              % (tag, {str(k): len(v) for k, v in ratios.items()}, len(mism)))
        for m in mism[:4]:
            print("     mismatch:", m)

if __name__ == "__main__":
    for n in (1, 2, 3):
        run_L(n)
    for n in (1, 2, 3, 4):
        run_M(n)
