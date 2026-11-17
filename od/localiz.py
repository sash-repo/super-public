#!/usr/bin/env python3
"""
Localization of a supercommutative superalgebra: exact check of everything that
goes into Subsect. 3.3 of "On odd parameters in geometry" (localization).

Two models, both with exact rational arithmetic.

MODEL U (decisive: inverses really exist, so every candidate relation can be
compared with genuine equality of fractions)

    A = Q[x]/(x^3) (x) Lambda[xi_1,xi_2,xi_3],   p = (x, xi_1, xi_2, xi_3),
    S = A \\ p = the units of A   (A/p = Q, so p is prime in the sense of (primeID)).

MODEL T (shows that the auxiliary factor t of the equivalence relation is not
vacuous: it contains a non-unit s and a zero-divisor killed by it)

    A = Q[x,y]/(xy) (x) Lambda[xi_1,xi_2,xi_3],  p = (x, xi_1, xi_2, xi_3),
    S = A \\ p  contains y, and y*x = 0.

Notation.  sigma = the parity automorphism (id on A_ev, -id on A_od); sbar := sigma(s).

What is established:

 (1) supercommutativity; every odd a has a^2 = 0.  Hence an odd element of a
     multiplicative S drags 0 into S and A_S = 0: in a NON-ZERO localization S
     has no odd elements.  Every odd element of A lies in every prime ideal, so
     the elements of S_p that are not even are NON-HOMOGENEOUS -- and for those
     the symbol p(s) of the display (2eq) is undefined.
 (2) s sbar = sbar s = (s_ev)^2, which is EVEN, hence CENTRAL; and s in S_p
     implies (s_ev)^2 in S_p.  So s^{-1} = sbar (s sbar)^{-1} = (s sbar)^{-1} sbar.
 (3) Consequently  r s^{-1} = (r sbar)(s sbar)^{-1}  and  s^{-1} r = (s sbar)^{-1}(sbar r):
     BOTH sets of fractions reduce to fractions with even = central denominators,
     for which left and right coincide literally.  <-- the 1-line proof of the
     boldfaced statement of Subsec. 3.3.
 (4) The naive cross-multiplication  r_1 s_2 = r_2 s_1  ((2eq) with all the
     signs equal to +1, as they are for even denominators) is NOT the equivalence
     of right fractions when the denominators are non-homogeneous: already
     (r,s) and (r u, s u) violate it.  The correct relation is the one on the
     normalised pairs, and it is verified here against genuine equality.
 (5) (addMult): with a_i := r_i sbar_i (right) resp. sbar_i r_i (left) and
     e_i := s_i sbar_i,
        a_1/e_1 + a_2/e_2 = (a_1 e_2 + a_2 e_1)/(e_1 e_2),
        (a_1/e_1)(a_2/e_2) = (a_1 a_2)/(e_1 e_2),
     with no signs at all; checked against the true sum and product in A.
 (6) Phi: (s,r) |-> (sbar r, s sbar) is the isomorphism from left fractions to
     right fractions; Psi: (r,s) |-> (s sbar, r sbar) is its inverse.  This is the
     answer to A. Lebedev's question.  Phi is NOT (s,r)|->(+-r,s).
 (7) For HOMOGENEOUS s (which, by (1), must be even) Phi is the identity
     (s,r) |-> (r,s) and all the parity signs of (2eq) equal +1.
"""

from fractions import Fraction
from itertools import combinations
import random

NXI = 3


# ============================================================== the algebra ==
# key = (pol, odd);  pol = ('1',0) | ('x',i) | ('y',j);  odd = increasing tuple.
# Model U truncates x^3 = 0 and has no y; model T has x*y = 0 and no truncation.

ONEPOL = ('1', 0)


class Ring:
    def __init__(self, name, xtrunc=None, xy_zero=False, vars_=('x',)):
        self.name = name
        self.xtrunc = xtrunc          # x^xtrunc = 0 (None: no truncation)
        self.xy_zero = xy_zero        # x*y = 0
        self.vars = vars_

    def polmul(self, a, b):
        if a == ONEPOL:
            return b
        if b == ONEPOL:
            return a
        if a[0] != b[0]:
            return None if self.xy_zero else None
        k = a[1] + b[1]
        if self.xtrunc is not None and k >= self.xtrunc:
            return None
        return (a[0], k)


U = Ring("U: Q[x]/(x^3) (x) Lambda[xi_1,xi_2,xi_3]", xtrunc=3, vars_=('x',))
T = Ring("T: Q[x,y]/(xy) (x) Lambda[xi_1,xi_2,xi_3]", xy_zero=True, vars_=('x', 'y'))


def clean(a):
    return {k: v for k, v in a.items() if v != 0}


def zero():
    return {}


def one():
    return {(ONEPOL, ()): Fraction(1)}


def polgen(v, k=1):
    return {((v, k), ()): Fraction(1)}


def gen_xi(i):
    return {(ONEPOL, (i,)): Fraction(1)}


def add(a, b):
    c = dict(a)
    for k, v in b.items():
        c[k] = c.get(k, Fraction(0)) + v
    return clean(c)


def neg(a):
    return {k: -v for k, v in a.items()}


def sub(a, b):
    return add(a, neg(b))


def smul(c, a):
    c = Fraction(c)
    return clean({k: c * v for k, v in a.items()})


def _merge_odd(o1, o2):
    if set(o1) & set(o2):
        return None, None
    lst = list(o1) + list(o2)
    sign = 1
    for i in range(len(lst)):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return tuple(lst), sign


def mul(R, a, b):
    c = {}
    for (pa, oa), va in a.items():
        for (pb, ob), vb in b.items():
            p = R.polmul(pa, pb)
            if p is None:
                continue
            o, sign = _merge_odd(oa, ob)
            if o is None:
                continue
            c[(p, o)] = c.get((p, o), Fraction(0)) + sign * va * vb
    return clean(c)


def part(a, p):
    return {k: v for k, v in a.items() if len(k[1]) % 2 == p}


def ev(a):
    return part(a, 0)


def od(a):
    return part(a, 1)


def sigma(a):
    return clean({k: (v if len(k[1]) % 2 == 0 else -v) for k, v in a.items()})


def const(a):
    return a.get((ONEPOL, ()), Fraction(0))


def is_hom(a):
    return len({len(k[1]) % 2 for k in a}) <= 1


def par(a):
    assert a and is_hom(a)
    return len(next(iter(a))[1]) % 2


def eq(a, b):
    return clean(a) == clean(b)


def show(a):
    if not a:
        return "0"
    out = []
    for (p, o), v in sorted(a.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][1])):
        t = []
        if v != 1 or (p == ONEPOL and not o):
            t.append(str(v))
        if p != ONEPOL:
            t.append(p[0] + ("^%d" % p[1] if p[1] > 1 else ""))
        for i in o:
            t.append("xi%d" % (i + 1))
        out.append("*".join(t) if t else "1")
    return " + ".join(out).replace("+ -", "- ")


# ------------------------------------------------ inverses (model U only) ----

def inv_U(s):
    """s = c(1+n) with n nilpotent  =>  s^{-1} = c^{-1} sum_k (-n)^k, a finite sum."""
    c = const(s)
    assert c != 0, "not a unit"
    n = smul(Fraction(1, 1) / c, sub(s, smul(c, one())))
    res, term = one(), one()
    for _ in range(12):                      # nilpotency degree is well below 12
        term = mul(U, term, neg(n))
        if not term:
            break
        res = add(res, term)
    return smul(Fraction(1, 1) / c, res)


# ------------------------------------------------------------ random data ----

def keys_of(R):
    pols = [ONEPOL]
    if R is U:
        pols += [('x', 1), ('x', 2)]
    else:
        pols += [('x', 1), ('x', 2), ('y', 1), ('y', 2)]
    return [(p, o) for p in pols
            for k in range(NXI + 1) for o in combinations(range(NXI), k)]


KEYS = {id(U): keys_of(U), id(T): keys_of(T)}


def rnd(R, rng, density=0.3, cmax=2, parity=None):
    a = {}
    for k in KEYS[id(R)]:
        if parity is not None and len(k[1]) % 2 != parity:
            continue
        if rng.random() < density:
            a[k] = Fraction(rng.randint(-cmax, cmax))
    return clean(a)


def rnd_unit(rng, **kw):
    s = rnd(U, rng, **kw)
    c = rng.choice([1, -1, 2, -3])
    return add(sub(s, smul(const(s), one())), smul(c, one()))


def rnd_hom_unit(rng, **kw):
    """A homogeneous element of S in model U: necessarily even."""
    s = rnd(U, rng, parity=0, **kw)
    c = rng.choice([1, -1, 2])
    return add(sub(s, smul(const(s), one())), smul(c, one()))


def in_S_T(a):
    """Model T: a is outside p iff its image in A/p = Q[y] is non-zero."""
    return any((not k[1]) and k[0][0] in ('1', 'y') for k in a)


def rnd_S_T(rng, **kw):
    while True:
        s = rnd(T, rng, **kw)
        s = add(s, smul(rng.choice([1, -1, 2]),
                        rng.choice([one(), polgen('y', 1), polgen('y', 2)])))
        if in_S_T(s):
            return s


# ================================================== model U: decisive tests ==

def frac_R(r, s):
    """the actual element r s^{-1} of A."""
    return mul(U, r, inv_U(s))


def frac_L(s, r):
    """the actual element s^{-1} r of A."""
    return mul(U, inv_U(s), r)


def norm_R(r, s):
    """r s^{-1} = (r sbar)(s sbar)^{-1}: even denominator."""
    sb = sigma(s)
    return mul(U, r, sb), mul(U, s, sb)


def norm_L(s, r):
    """s^{-1} r = (s sbar)^{-1}(sbar r): even denominator."""
    sb = sigma(s)
    return mul(U, s, sb), mul(U, sb, r)


def Phi(s, r):
    """left class (s,r) = s^{-1} r  |-->  right class (sbar r, s sbar)."""
    sb = sigma(s)
    return mul(U, sb, r), mul(U, s, sb)


def Psi(r, s):
    """right class (r,s) = r s^{-1}  |-->  left class (s sbar, r sbar)."""
    sb = sigma(s)
    return mul(U, s, sb), mul(U, r, sb)


def run_U(rng, n=250):
    res = {}

    def bump(k, cond):
        res[k] = res.get(k, 0) + (0 if cond else 1)

    diff_LR = 0
    naive_wrong = 0
    naive_witness = None
    for _ in range(n):
        s, u = rnd_unit(rng), rnd_unit(rng)
        s2 = rnd_unit(rng)
        r, r2 = rnd(U, rng), rnd(U, rng)
        sb = sigma(s)

        # (1) odd squares, supercommutativity
        a_od = rnd(U, rng, parity=1)
        bump("odd^2 = 0", not mul(U, a_od, a_od))
        b_ev = rnd(U, rng, parity=0)
        bump("even is central", eq(mul(U, b_ev, r), mul(U, r, b_ev)))

        # (2) s sbar = (s_ev)^2, even, and s^{-1} = sbar (s sbar)^{-1}
        e = mul(U, s, sb)
        bump("s sbar = sbar s = (s_ev)^2", eq(e, mul(U, ev(s), ev(s)))
             and eq(e, mul(U, sb, s)))
        bump("s sbar is even", not od(e))
        bump("s^{-1} = sbar (s sbar)^{-1}",
             eq(inv_U(s), mul(U, sb, inv_U(e))) and eq(inv_U(s), mul(U, inv_U(e), sb)))

        # (3) reduction to even denominators
        a1, e1 = norm_R(r, s)
        bump("r s^{-1} = (r sbar)(s sbar)^{-1}", eq(frac_R(r, s), mul(U, a1, inv_U(e1))))
        E1, A1 = norm_L(s, r)
        bump("s^{-1} r = (s sbar)^{-1}(sbar r)", eq(frac_L(s, r), mul(U, inv_U(E1), A1)))
        if not eq(frac_R(r, s), frac_L(s, r)):
            diff_LR += 1

        # (4) the naive cross-multiplication vs the truth, on the pair (r,s), (ru,su)
        ru, su = mul(U, r, u), mul(U, s, u)
        true_eq = eq(frac_R(r, s), frac_R(ru, su))
        naive = eq(mul(U, r, su), mul(U, ru, s))
        bump("(r,s) and (ru,su) ARE equal as fractions", true_eq)
        if naive != true_eq:
            naive_wrong += 1
            if naive_witness is None:
                naive_witness = (r, s, u)

        # the corrected relation, on the same pair and on random pairs
        b1, f1 = norm_R(r, s)
        b2, f2 = norm_R(ru, su)
        bump("corrected relation on (r,s),(ru,su)",
             eq(mul(U, b1, f2), mul(U, b2, f1)))
        c1, g1 = norm_R(r, s)
        c2, g2 = norm_R(r2, s2)
        bump("corrected relation <=> genuine equality",
             eq(mul(U, c1, g2), mul(U, c2, g1)) == eq(frac_R(r, s), frac_R(r2, s2)))

        # (5) (addMult)
        A_, E_ = norm_R(r, s)
        B_, F_ = norm_R(r2, s2)
        s_num, s_den = add(mul(U, A_, F_), mul(U, B_, E_)), mul(U, E_, F_)
        p_num, p_den = mul(U, A_, B_), mul(U, E_, F_)
        bump("(addMult) addition, right fractions",
             eq(mul(U, s_num, inv_U(s_den)), add(frac_R(r, s), frac_R(r2, s2))))
        bump("(addMult) multiplication, right fractions",
             eq(mul(U, p_num, inv_U(p_den)), mul(U, frac_R(r, s), frac_R(r2, s2))))
        LA, LE = norm_L(s, r)[1], norm_L(s, r)[0]
        LB, LF = norm_L(s2, r2)[1], norm_L(s2, r2)[0]
        ls_num = add(mul(U, LF, LA), mul(U, LE, LB))
        bump("(addMult) addition, left fractions",
             eq(mul(U, inv_U(mul(U, LE, LF)), ls_num),
                add(frac_L(s, r), frac_L(s2, r2))))
        bump("(addMult) multiplication, left fractions",
             eq(mul(U, inv_U(mul(U, LE, LF)), mul(U, LA, LB)),
                mul(U, frac_L(s, r), frac_L(s2, r2))))

        # (6) Phi and Psi
        pn, pd = Phi(s, r)
        bump("Phi(s,r) represents s^{-1} r", eq(frac_R(pn, pd), frac_L(s, r)))
        qn, qd = Psi(r, s)
        bump("Psi(r,s) represents r s^{-1}", eq(frac_L(qn, qd), frac_R(r, s)))
        bump("Psi o Phi = id", eq(frac_L(*Psi(*Phi(s, r))), frac_L(s, r)))
        # additivity / multiplicativity of Phi
        lsum = add(frac_L(s, r), frac_L(s2, r2))
        lprod = mul(U, frac_L(s, r), frac_L(s2, r2))
        bump("Phi additive", eq(add(frac_R(*Phi(s, r)), frac_R(*Phi(s2, r2))), lsum))
        bump("Phi multiplicative",
             eq(mul(U, frac_R(*Phi(s, r)), frac_R(*Phi(s2, r2))), lprod))

        # (7) homogeneous denominators
        h = rnd_hom_unit(rng)
        bump("homogeneous s in S is even", par(h) == 0 if h else True)
        bump("for even s: s^{-1} r = r s^{-1}", eq(frac_L(h, r), frac_R(r, h)))

    return res, diff_LR, naive_wrong, naive_witness, n


# ============================================= model T: the role of t ========

TCAND = [one(), polgen('y', 1), polgen('y', 2), polgen('y', 3)]


def killed_T(d):
    return (not d) or any(not mul(T, t, d) for t in TCAND)


def run_T(rng, n=150):
    """The corrected relation, with t, is an equivalence relation, and t is needed."""
    bad = 0
    used_t = 0
    for _ in range(n):
        s, s2, u = rnd_S_T(rng), rnd_S_T(rng), rnd_S_T(rng)
        r, r2 = rnd(T, rng), rnd(T, rng)

        def nrm(r, s):
            sb = sigma(s)
            return mul(T, r, sb), mul(T, s, sb)

        def rel(f, g):
            (a1, e1), (a2, e2) = nrm(*f), nrm(*g)
            return killed_T(sub(mul(T, a1, e2), mul(T, a2, e1)))

        f = (r, s)
        g = (r2, s2)
        fu = (mul(T, r, u), mul(T, s, u))
        if not rel(f, f) or not rel(g, g):
            bad += 1                                    # reflexive
        if rel(f, g) != rel(g, f):
            bad += 1                                    # symmetric
        if not rel(f, fu):
            bad += 1                                    # r/s = ru/su
        (a1, e1), (a2, e2) = nrm(*f), nrm(*fu)
        d = sub(mul(T, a1, e2), mul(T, a2, e1))
        if d and killed_T(d):
            used_t += 1                                 # t was genuinely needed
    return bad, used_t, n


def dl_sign_table():
    rows = []
    for ps1 in (0, 1):
        for ps2 in (0, 1):
            for pi in (0, 1):
                pr1, pr2 = (pi + ps1) % 2, (pi + ps2) % 2
                rows.append((ps1, ps2, pi, (ps1 * pr2) % 2, (ps1 * ps2) % 2,
                             (ps1 * pr2 + ps2 * pr1) % 2))
    return rows


def main():
    rng = random.Random(20260808)
    print("=" * 78)
    print("MODEL " + U.name)
    print("p = (x, xi_1, xi_2, xi_3);  S = A \\ p = the units of A")
    print("=" * 78)
    res, diff_LR, naive_wrong, wit, n = run_U(rng)
    for k in sorted(res):
        print("  %-46s : %d failures / %d" % (k, res[k], n))

    print("\n  left fraction s^{-1} r  DIFFERS from  r s^{-1} : %d of %d samples"
          % (diff_LR, n))
    print("  naive cross-multiplication r_1 s_2 = r_2 s_1 gives the WRONG answer")
    print("  on %d of %d samples of the pair (r,s), (ru,su)" % (naive_wrong, n))
    if wit:
        r, s, u = wit
        print("    e.g.  s = %s" % show(s))
        print("          u = %s" % show(u))
        print("          r = %s" % show(r))
        print("      r(su) - (ru)s = %s"
              % show(sub(mul(U, r, mul(U, s, u)), mul(U, mul(U, r, u), s))))
        print("      ... non-zero, although r s^{-1} = (ru)(su)^{-1} genuinely holds.")

    s = add(one(), gen_xi(0))
    r = gen_xi(1)
    print("\n  smallest witness by hand: s = 1 + xi_1 (a unit, NOT homogeneous)")
    print("    s^{-1}   = %s     (= sbar/(s sbar), s sbar = %s)"
          % (show(inv_U(s)), show(mul(U, s, sigma(s)))))
    print("    s^{-1} xi_2 = %s" % show(frac_L(s, r)))
    print("    xi_2 s^{-1} = %s" % show(frac_R(r, s)))
    print("    Phi(s, xi_2) = (%s, %s)" % (show(Phi(s, r)[0]), show(Phi(s, r)[1])))

    print("\n" + "=" * 78)
    print("MODEL " + T.name)
    print("p = (x, xi_1, xi_2, xi_3);  S = A \\ p  contains the non-unit y, y*x = 0")
    print("=" * 78)
    bad, used_t, nT = run_T(rng)
    print("  corrected relation is an equivalence relation : %d failures / %d" % (bad, nT))
    print("  samples where a non-trivial t was needed      : %d / %d" % (used_t, nT))

    print("\n" + "=" * 78)
    print("The parity signs of the display (2eq)")
    print("=" * 78)
    print("  (pi := p(r_i)+p(s_i) = the common parity of the two fractions)")
    print("  p(s1) p(s2)  pi | (2eq) exp | cross-mult exp | (2eq) symmetric?")
    for ps1, ps2, pi, dl, cr, sym in dl_sign_table():
        print("    %d     %d     %d  |     %d     |       %d        |   %s"
              % (ps1, ps2, pi, dl, cr, "yes" if sym == 0 else "NO"))
    print("  The two exponents differ, and (2eq) loses its symmetry, only on rows")
    print("  with an ODD denominator -- which by (1) never occur.  On every row")
    print("  with p(s1) = p(s2) = 0 all the signs are +1.")


if __name__ == "__main__":
    main()
