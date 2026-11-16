"""
The alpha-cocycle on k^L'(1|4) (cocycle (1) of Table (61)).

Claim of the text (Subsect. 2.8): putting D_1^{-1} FIRST in the product
    c(f,g) = Res(f D_1^{-1} D_th1 D_th2 D_th3 D_th4 (g))
makes the expression well defined on k^L'(1|4).

Here we check what the composite really is.  Two readings:
  (a) literal: apply D_th4, D_th3, D_th2, D_th1, then integrate in t;
  (b) as ONE operator, i.e. simplify the composite BEFORE evaluating -- exactly
      the device used for  Delta = D^M_zeta o D^{-1}  in the k^M(1|4) case.
Reading (b) is obtained here by carrying the exponent as a symbol and cancelling.
"""

import sympy as sp
from fractions import Fraction as F
from stringy import mono, add, mul, smul, d_t, d_odd, parity
from check_cocycles import L_bracket, L_res, monomials

a = sp.Symbol('a')
LAB = (1, 2, 3, 4)

# ---- symbolic-exponent functions: {(exponent, S): coeff}, exponent a sympy expr

def _sort(S):
    S = list(S); sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]; sgn = -sgn
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sgn

def sadd(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = sp.simplify(out.get(k, 0) + v)
    return {k: v for k, v in out.items() if v != 0}

def s_dt(f):
    return {(e - 1, S): sp.expand(c * e) for (e, S), c in f.items() if sp.expand(c * e) != 0}

def s_dth(j, f):
    out = {}
    for (e, S), c in f.items():
        if j in S:
            p = S.index(j)
            out[(e, S[:p] + S[p + 1:])] = out.get((e, S[:p] + S[p + 1:]), 0) + (-1) ** p * c
    return {k: v for k, v in out.items() if v != 0}

def s_thmul(j, f):
    out = {}
    for (e, S), c in f.items():
        T, sgn = _sort((j,) + S)
        if sgn:
            out[(e, T)] = out.get((e, T), 0) + sgn * c
    return {k: v for k, v in out.items() if v != 0}

def s_Dth(j, f):
    return sadd(s_thmul(j, s_dt(f)), s_dth(j, f))

def s_Dinv(f):
    out = {}
    for (e, S), c in f.items():
        out[(e + 1, S)] = out.get((e + 1, S), 0) + sp.cancel(sp.together(c / (e + 1)))
    return {k: sp.simplify(v) for k, v in out.items() if sp.simplify(v) != 0}

def composite(S):
    """the operator  D_1^{-1} D_th1 D_th2 D_th3 D_th4  applied to t^a theta_S,
    simplified as a single operator (reading (b))."""
    f = {(a, tuple(S)): sp.Integer(1)}
    for j in (4, 3, 2, 1):
        f = s_Dth(j, f)
    return s_Dinv(f)

# ---- numeric version of reading (b): substitute an integer exponent

_TABLE = {}
def DeltaL(f):
    out = {}
    for (m, S), c in f.items():
        if S not in _TABLE:
            _TABLE[S] = composite(S)
        for (e, T), cc in _TABLE[S].items():
            val = cc.subs(a, m)
            if val.is_infinite or val is sp.nan:
                raise ZeroDivisionError("still singular at a=%s, S=%s" % (m, S))
            if val != 0:
                key = (int(e.subs(a, m)), T)
                out[key] = out.get(key, 0) + F(str(val)) * c
    return {k: v for k, v in out.items() if v}

def c_alpha(f, g, sign=False):
    v = L_res(4, mul(f, DeltaL(g)))
    return v * ((-1) ** parity(f) if sign else 1)

# ------------------------------------------------------------------- checks

def deg(k):
    """degree of K_{t^a theta_S} in the standard grading (deg t = 2, deg th = 1)"""
    return 2 * k[0] + len(k[1]) - 2

def exhaustive_test(c, keys):
    bad = []
    for kf in keys:
        for kg in keys:
            for kh in keys:
                if deg(kf) + deg(kg) + deg(kh) != 0:
                    continue                       # identity is 0 = 0
                f, g, h = mono(*kf), mono(*kg), mono(*kh)
                lhs = c(f, L_bracket(4, g, h))
                rhs = c(L_bracket(4, f, g), h) + (-1) ** (parity(f) * parity(g)) * c(g, L_bracket(4, f, h))
                if lhs != rhs:
                    bad.append((kf, kg, kh, lhs, rhs))
    return bad

def literal(f, g, sign=False):
    def Dth(i):
        return lambda h: add(mul(mono(0, (i,)), d_t(h)), d_odd(i, h))
    def Dinv(h):
        out = {}
        for (e, S), c in h.items():
            if e == -1:
                raise ZeroDivisionError
            out[(e + 1, S)] = out.get((e + 1, S), 0) + c / (e + 1)
        return {k: v for k, v in out.items() if v}
    h = g
    for i in (4, 3, 2, 1):
        h = Dth(i)(h)
    v = L_res(4, mul(f, Dinv(h)))
    return v * ((-1) ** parity(f) if sign else 1)

if __name__ == "__main__":
    print("The composite D_1^{-1} D_th1 D_th2 D_th3 D_th4 as ONE operator:")
    for S in [(), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]:
        comp = composite(S)
        txt = " + ".join("(%s) t^{%s} th%s" % (sp.factor(c), e, "".join(map(str, T)))
                         for (e, T), c in sorted(comp.items(), key=str))
        print("   t^a th%-8s -->  %s" % ("".join(map(str, S)), txt))
    keys = [k for k in monomials(4, range(-3, 4), LAB) if not (k[0] == -1 and len(k[1]) == 4)]
    small = [k for k in monomials(4, range(-2, 3), LAB) if not (k[0] == -1 and len(k[1]) == 4)]
    print()
    for sign in (False, True):
        tag = "with (-1)^p(f)" if sign else "plain        "
        bad = exhaustive_test(lambda f, g: c_alpha(f, g, sign), small)
        print("  reading (b) [one operator], %s : %s" % (tag, "COCYCLE" if not bad
              else "not a cocycle (%d, e.g. %s: %s vs %s)" % (len(bad), bad[0][:3], bad[0][3], bad[0][4])))
        try:
            bad = exhaustive_test(lambda f, g: literal(f, g, sign), small)
            print("  reading (a) [literal],      %s : %s" % (tag, "COCYCLE" if not bad
                  else "not a cocycle (%d, e.g. %s: %s vs %s)" % (len(bad), bad[0][:3], bad[0][3], bad[0][4])))
        except ZeroDivisionError:
            print("  reading (a) [literal],      %s : undefined" % tag)
    print()
    print("  components of reading (b) (printed values of (48), alpha-part, in brackets):")
    R = range(-3, 4)
    c = lambda f, g: c_alpha(f, g, False)
    print("   c(L_m,L_-m)       [ m(m^2-1) ] :", ", ".join("%d:%s" % (m, c(mono(m + 1), mono(-m + 1))) for m in R))
    print("   c(E^1_m,E^1_-m-1) [ m(m+1)   ] :", ", ".join("%d:%s" % (m, c(mono(m + 1, (1,)), mono(-m, (1,)))) for m in R))
    print("   c(T^12_m,T^12_-m) [ -m       ] :", ", ".join("%d:%s" % (m, c(mono(m, (1, 2)), mono(-m, (1, 2)))) for m in R))
    print("   c(F^1_m,F^1_-m-1) [ -1       ] :", ", ".join("%d:%s" % (m, c(mono(m, (2, 3, 4)), mono(-m - 1, (2, 3, 4)))) for m in R))
    print("   c(S_m,S_-m)       [ 1/m      ] :", ", ".join("%d:%s" % (m, c(mono(m - 1, (1, 2, 3, 4)), mono(-m - 1, (1, 2, 3, 4)))) for m in R if m))
    print("   c(L_m,S_-m)       [ 0        ] :", ", ".join("%d:%s" % (m, c(mono(m + 1), mono(-m - 1, (1, 2, 3, 4)))) for m in R if m))  # t^{-1}th1234 is not in k^L'(1|4)
