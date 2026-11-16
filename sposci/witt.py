#!/usr/bin/env python3
r"""
witt.py -- exact (rational) checks of the last paragraph of the section
"Examples of spinor/oscillator representations" of the text, the one that carries
two open questions:

  "the restriction of the cocycle equation 76 from vect^L(1|2) to its
   subalgebra witt however embedded is trivial whereas the restriction of the
   cocycle (...) from svect^L_lambda(1|2) to its subalgebra witt is
   non-trivial"

The two cocycles are the ones inserted from [LShRC] (= aRC) as eqs.
equation 76 and equation 78.

Everything is done with Fraction arithmetic in a finite window of the
Z-grading; no floats, no external packages.

Model: vect^L(1|2) = derivations of C[t,t^{-1}][th1,th2].
A superfunction is a dict {(k, mask): Fraction}, k in Z the power of t,
mask a 2-bit subset of {th1, th2} (bit 0 = th1, bit 1 = th2), monomials
written in the order th1 th2.
A vector field is a triple (f_t, f_1, f_2) of superfunctions, D = sum f_x d_x.
"""

from fractions import Fraction as F
from itertools import product

TH = (1, 2)          # bit 0 -> theta_1, bit 1 -> theta_2
PAR = {0: 0, 1: 1, 2: 1, 3: 0}       # parity of the theta-monomial


# ---------------------------------------------------------------- functions

def add(a, b, c=1):
    r = dict(a)
    for m, v in b.items():
        r[m] = r.get(m, 0) + c * v
        if r[m] == 0:
            del r[m]
    return r


def smul(s, a):
    return {m: s * v for m, v in a.items() if s * v != 0}


def mono_mul(m1, m2):
    """product of two theta-monomials; returns (sign, mask) or None."""
    if m1 & m2:
        return None
    # th1 th2 order: moving the factors of m2 past those of m1
    sign = 1
    if (m1 & 2) and (m2 & 1):      # th2 (from m1) must pass th1 (from m2)
        sign = -1
    return sign, m1 | m2


def fmul(a, b):
    r = {}
    for (k1, s1), v1 in a.items():
        for (k2, s2), v2 in b.items():
            p = mono_mul(s1, s2)
            if p is None:
                continue
            sg, s = p
            key = (k1 + k2, s)
            r[key] = r.get(key, 0) + sg * v1 * v2
            if r[key] == 0:
                del r[key]
    return r


def d_t(a):
    r = {}
    for (k, s), v in a.items():
        if k != 0:
            r[(k - 1, s)] = r.get((k - 1, s), 0) + k * v
            if r[(k - 1, s)] == 0:
                del r[(k - 1, s)]
    return r


def d_th(i, a):
    """left derivative in theta_i, i = 0 or 1 (bit index)."""
    bit = 1 << i
    r = {}
    for (k, s), v in a.items():
        if not (s & bit):
            continue
        # sign = (-1)^(number of theta factors before theta_i)
        sign = -1 if (i == 1 and (s & 1)) else 1
        key = (k, s ^ bit)
        r[key] = r.get(key, 0) + sign * v
        if r[key] == 0:
            del r[key]
    return r


DER = (d_t, lambda a: d_th(0, a), lambda a: d_th(1, a))
DER_PAR = (0, 1, 1)


# ------------------------------------------------------------ vector fields

def vf_par(D):
    """parity of a homogeneous vector field, or None if it is zero."""
    p = None
    for x in range(3):
        for (k, s), v in D[x].items():
            q = (PAR[s] + DER_PAR[x]) % 2
            if p is None:
                p = q
            assert p == q, "inhomogeneous field"
    return p


def apply(D, h):
    """D(h) for a superfunction h."""
    r = {}
    for x in range(3):
        if D[x]:
            r = add(r, fmul(D[x], DER[x](h)))
    return r


def bracket(D, E):
    p, q = vf_par(D), vf_par(E)
    if p is None or q is None:
        return ({}, {}, {})
    sg = -1 if (p and q) else 1
    return tuple(add(apply(D, E[x]), apply(E, D[x]), -sg) for x in range(3))


def vf_is_zero(D):
    return all(not c for c in D)


def vf_add(D, E, c=1):
    return tuple(add(D[x], E[x], c) for x in range(3))


def vf_smul(s, D):
    return tuple(smul(s, D[x]) for x in range(3))


def mono_vf(k, mask, x):
    """the basis field t^k * theta^mask * d_x."""
    D = [{}, {}, {}]
    D[x] = {(k, mask): F(1)}
    return tuple(D)


# ------------------------------------------------- the cocycle equation 76
#
#  c(t^k th1 d_th1, t^l th2 d_th2) =  k delta_{k,-l}
#  c(t^k th1 d_th2, t^l th2 d_th1) = -k delta_{k,-l}
#  c(t^k th1 th2 d_th1, t^l d_th2) = -k delta_{k,-l}
#  c(t^k th1 th2 d_th2, t^l d_th1) =  k delta_{k,-l}
#  all other pairs of basis fields: 0.

TAB = {
    ((1, 1), (2, 2)): 1,     # (mask, x) pairs, x: 1 -> d_th1, 2 -> d_th2
    ((1, 2), (2, 1)): -1,
    ((3, 1), (0, 2)): -1,
    ((3, 2), (0, 1)): 1,
}


def c_basis(k, m1, x1, l, m2, x2):
    """the cocycle on two basis fields t^k theta^m1 d_x1, t^l theta^m2 d_x2."""
    if k + l != 0:
        return F(0)
    key = ((m1, x1), (m2, x2))
    if key in TAB:
        return F(TAB[key] * k)
    key = ((m2, x2), (m1, x1))
    if key in TAB:
        # super-skew: c(a,b) = -(-1)^{p(a)p(b)} c(b,a)
        p1 = (PAR[m1] + DER_PAR[x1]) % 2
        p2 = (PAR[m2] + DER_PAR[x2]) % 2
        sg = -1 if not (p1 and p2) else 1
        return F(sg * TAB[key] * l)
    return F(0)


def coc(D, E):
    """bilinear extension of c to arbitrary vector fields."""
    tot = F(0)
    for x1 in range(3):
        for (k, m1), v1 in D[x1].items():
            for x2 in range(3):
                for (l, m2), v2 in E[x2].items():
                    tot += v1 * v2 * c_basis(k, m1, x1, l, m2, x2)
    return tot


# ----------------------------------------------------------------- checks

def basis_window(K):
    for k in range(-K, K + 1):
        for mask in range(4):
            for x in range(3):
                yield mono_vf(k, mask, x)


def check_cocycle_identity(K):
    """c(f,{g,h}) = c({f,g},h) + (-1)^{p(f)p(g)} c(g,{f,h}) on the window."""
    B = list(basis_window(K))
    bad = 0
    for f, g, h in product(B, repeat=3):
        pf, pg = vf_par(f), vf_par(g)
        lhs = coc(f, bracket(g, h))
        rhs = coc(bracket(f, g), h)
        rhs += (-1 if (pf and pg) else 1) * coc(g, bracket(f, h))
        if lhs != rhs:
            bad += 1
    return bad, len(B)


def witt_vect(a, m):
    """D^a_m = t^{m+1} d_t + a(m+1) t^m (th1 d_th1 + th2 d_th2)."""
    D = mono_vf(m + 1, 0, 0)
    if a:
        D = vf_add(D, vf_smul(F(a) * (m + 1), mono_vf(m, 1, 1)))
        D = vf_add(D, vf_smul(F(a) * (m + 1), mono_vf(m, 2, 2)))
    return D


def check_witt_in_vect(a, M):
    """[D^a_m, D^a_n] = (n-m) D^a_{m+n}, and c(D^a_m, D^a_{-m})."""
    ok = True
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            lhs = bracket(witt_vect(a, m), witt_vect(a, n))
            rhs = vf_smul(F(n - m), witt_vect(a, m + n))
            if not vf_is_zero(vf_add(lhs, rhs, -1)):
                ok = False
    vals = {m: coc(witt_vect(a, m), witt_vect(a, -m)) for m in range(-M, M + 1)}
    return ok, vals


# ----------------------- svect^L_lambda(1|2): LL_m and the cocycle equation 78

def LL(lam, m):
    """LL_m = t^m (t d_t + (lam+m+1)/2 (th1 d_th1 + th2 d_th2))."""
    D = mono_vf(m + 1, 0, 0)
    h = F(lam + m + 1, 1) / 2 if not isinstance(lam, F) else (lam + m + 1) / 2
    D = vf_add(D, vf_smul(h, mono_vf(m, 1, 1)))
    D = vf_add(D, vf_smul(h, mono_vf(m, 2, 2)))
    return D


def check_LL(lam, M):
    """LL is a Witt subalgebra of svect^L_lambda(1|2); div(t^lam LL_m) = 0."""
    ok = True
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            lhs = bracket(LL(lam, m), LL(lam, n))
            rhs = vf_smul(F(n - m), LL(lam, m + n))
            if not vf_is_zero(vf_add(lhs, rhs, -1)):
                ok = False
    # the defining condition of svect^L_lambda: lam * f = -t * Div D, eq. equation 50
    div_ok = True
    for m in range(-M, M + 1):
        D = LL(lam, m)
        # Div D = d_t(f_t) - d_th1(f_1) - d_th2(f_2)   (signs: p(th)=1)
        div = add(add(d_t(D[0]), d_th(0, D[1]), -1), d_th(1, D[2]), -1)
        lhs = smul(F(lam), D[0])
        rhs = smul(F(-1), fmul({(1, 0): F(1)}, div))
        if add(lhs, rhs, -1):
            div_ok = False
    return ok, div_ok


if __name__ == "__main__":
    print("=== vect^L(1|2), cocycle equation 76 ===")
    bad, nb = check_cocycle_identity(2)
    print(f"  cocycle identity on {nb}^3 triples (|k|<=2): {bad} failures")

    for a in (0, 1, 2, F(1, 3)):
        ok, vals = check_witt_in_vect(a, 4)
        line = ", ".join(f"m={m}: {vals[m]}" for m in (1, 2, 3, 4))
        print(f"  a={a}: Witt? {ok};  c(D_m, D_-m) -> {line}")
        # compare with -2a^2 (m^3 - m)
        pred = all(vals[m] == -2 * F(a) ** 2 * (m ** 3 - m) for m in vals)
        print(f"        equals -2a^2 (m^3-m) for all |m|<=4: {pred}")

    print()
    print("=== svect^L_lambda(1|2), the subalgebra {LL_m} ===")
    for lam in (F(1, 3), F(1, 2), F(2, 5), 0):
        ok, div_ok = check_LL(lam, 4)
        cvals = {m: F(1, 2) * m * (m ** 2 - (F(lam) + 1) ** 2) for m in range(1, 5)}
        print(f"  lambda={lam}: Witt? {ok}; inside svect^L_lambda? {div_ok}")
        print("        c(LL_m,LL_-m) = 1/2 m(m^2-(lam+1)^2) -> "
              + ", ".join(f"m={m}: {cvals[m]}" for m in cvals))
        # class in H^2: a m^3 + b m is cohomologous to a (m^3 - m); a = 1/2 here
        print(f"        class = (1/2)(m^3-m), i.e. non-trivial: "
              f"{F(1,2) != 0}")


# ------------- the 2-parameter family of witt's inside svect^L_lambda(1|2)
#
# Added to prove the claim on the sentence
# "witt can be embedded in svect^L_lambda(1|2) only using formulas equation 71".
# It cannot: the elements
#
#   D_m := LL_m + (alpha + beta m) t^m (th1 d_th1 - th2 d_th2),   alpha, beta in C
#
# span a witt for every (alpha, beta) -- the traceless part th1 d_th1 - th2 d_th2
# does not change the divergence, so D_m stays inside svect^L_lambda(1|2) --
# and the restriction of equation 78 to it is
#
#   c(D_m, D_n) = [ (1/2 - 2 beta^2) m^3 + (2 alpha^2 - (1/2)(lambda+1)^2) m ]
#                 delta_{m,-n},
#
# whose class in H^2(witt) is (1/2 - 2 beta^2)(m^3 - m): TRIVIAL exactly for
# beta = +- 1/2.  (Here c(LL_m, t^n T) = 0, i.e. the table equation 78 is
# taken to be complete, and m delta_{m,-n} is a coboundary.)

def TT(m):
    """t^m (th1 d_th1 - th2 d_th2), the traceless diagonal element."""
    return vf_add(mono_vf(m, 1, 1), mono_vf(m, 2, 2), -1)


def DD(lam, alpha, beta, m):
    return vf_add(LL(lam, m), vf_smul(F(alpha) + F(beta) * m, TT(m)))


def check_DD(lam, alpha, beta, M):
    """witt relations, membership in svect^L_lambda, and the restricted cocycle."""
    ok = True
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            lhs = bracket(DD(lam, alpha, beta, m), DD(lam, alpha, beta, n))
            rhs = vf_smul(F(n - m), DD(lam, alpha, beta, m + n))
            if not vf_is_zero(vf_add(lhs, rhs, -1)):
                ok = False
    div_ok = True
    for m in range(-M, M + 1):
        D = DD(lam, alpha, beta, m)
        div = add(add(d_t(D[0]), d_th(0, D[1]), -1), d_th(1, D[2]), -1)
        if add(smul(F(lam), D[0]), smul(F(-1), fmul({(1, 0): F(1)}, div)), -1):
            div_ok = False
    # the restriction of equation 78:  c(LL,LL) + (alpha+beta m)(alpha-beta m) c(T,T)
    vals = {}
    for m in range(1, M + 1):
        vals[m] = (F(1, 2) * m * (m ** 2 - (F(lam) + 1) ** 2)
                   + (F(alpha) + F(beta) * m) * (F(alpha) - F(beta) * m) * 2 * m)
    return ok, div_ok, vals


if __name__ == "__main__":
    print()
    print("=== witt's inside svect^L_lambda(1|2): the family D_m ===")
    for lam in (F(1, 3), F(1, 2), 0):
        for alpha, beta in ((0, 0), (0, F(1, 2)), (0, -F(1, 2)), (1, F(1, 2)),
                            (1, 0), (F(2, 3), F(1, 3))):
            ok, div_ok, vals = check_DD(lam, alpha, beta, 4)
            # the class: coefficient of m^3 (m delta_{m,-n} is a coboundary)
            cub = F(1, 2) - 2 * F(beta) ** 2
            print("  lambda=%-4s alpha=%-4s beta=%-4s witt? %s  in svect? %s"
                  "  m^3-coeff = %-5s  class %s"
                  % (lam, alpha, beta, ok, div_ok, cub,
                     "TRIVIAL" if cub == 0 else "non-trivial"))
            assert ok and div_ok
            for m in vals:
                pred = cub * m ** 3 + (2 * F(alpha) ** 2
                                       - F(1, 2) * (F(lam) + 1) ** 2) * m
                assert vals[m] == pred, (m, vals[m], pred)
    print("  (values of c(D_m,D_-m) match (1/2-2beta^2)m^3 + (2alpha^2 -"
          " (lambda+1)^2/2) m for all |m| <= 4)")
