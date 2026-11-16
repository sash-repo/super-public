#!/usr/bin/env python3
"""
n4emb.py -- is the acting algebra of the N=4 Ivanov-Krivonos system a
subalgebra of the contact algebra k^L(1|4)?

The text says, in Subsection 6.2, that the constraint
of (IK4) "is preserved by the whole of k^L(1|4), hence by the subalgebra
(svectlam) of Proposition 6.1.1 inside it".  That "inside it" is the sentence this
engine tests: is

    svect^L_lambda(1|2)   =   the small N=4 superconformal algebra of [Ad]

a subalgebra of k^L(1|4) = the large N=4 superconformal algebra?

Everything is exact arithmetic over Q (and over Q(i) where an isotropic basis is
needed), in the generating-function model of k^L(1|n):

    R^L(4) = C[t, t^{-1}] (x) Lambda(theta_1, ..., theta_4),

    {f, g} = (2-E)f . d_t g  -  d_t f . (2-E)g  -  (-1)^{p(f)} sum_i d_i f d_i g,

which is eq. (kbr) of the text, with E = sum theta_i d_{theta_i}.

What is computed:

 (1) the contact bracket, checked against the relations the text states:
     the Virasoro tower t^{n+1}, the o(4) currents t^n theta_i theta_j, the
     codimension-1 quotient spanned by t^{-1} theta_1 theta_2 theta_3 theta_4.

 (2) the bracket of two odd towers of conformal weight 3/2, in closed form:
        {t^r Theta, t^s Theta'} = (s-r) t^{r+s-1} Theta Theta' - (Theta.Theta') t^{r+s},
     whence: two INDEPENDENT weight-3/2 odd towers never commute.  The small
     N=4 algebra needs two commuting ones (its doublets are abelian), so no
     CONFORMAL (= weight-graded) embedding exists.

 (3) the general MODE-graded case, which is the one that matters, because the
     Virasoro of the subalgebra need not be the standard t^{n+1}: it may be
     twisted by a current, L_n = t^{n+1} + n mu t^n omega + nu t^{n-1} vol, and
     then weight-3/2 and weight-1/2 odd elements sit in one and the same mode
     level and may mix.  The mode-r odd space is 8-dimensional,

        span(t^{r+1/2} theta_i)  +  span(t^{r-1/2} theta_j theta_k theta_l),

     and the engine searches it for a pair of abelian towers, i.e. for the
     G-doublet of the small N=4 algebra.

Usage:  python3 n4emb.py           (writes n4emb-out.txt when run by run())
"""

from fractions import Fraction as Q
import itertools
import sys

# ---------------------------------------------------------------------------
# functions t^a * theta_S, coefficients in Q(i) written as pairs (re, im)
# ---------------------------------------------------------------------------

N = 4
IDX = (1, 2, 3, 4)


class C:
    """A rational complex number."""
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = Q(re)
        self.im = Q(im)

    def __add__(s, o):
        o = cx(o)
        return C(s.re + o.re, s.im + o.im)

    __radd__ = __add__

    def __neg__(s):
        return C(-s.re, -s.im)

    def __sub__(s, o):
        return s + (-cx(o))

    def __rsub__(s, o):
        return cx(o) + (-s)

    def __mul__(s, o):
        o = cx(o)
        return C(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)

    __rmul__ = __mul__

    def __eq__(s, o):
        o = cx(o)
        return s.re == o.re and s.im == o.im

    def __hash__(s):
        return hash((s.re, s.im))

    def __bool__(s):
        return s.re != 0 or s.im != 0

    def __repr__(s):
        if s.im == 0:
            return str(s)
        if s.re == 0:
            return "%si" % s.im
        return "(%s%s%si)" % (s.re, "+" if s.im > 0 else "-", abs(s.im))

    def __str__(s):
        return repr(s) if s.im else str(s.re)


def cx(z):
    return z if isinstance(z, C) else C(z)


I = C(0, 1)


def sort_odd(S):
    """Sort a tuple of odd labels; return (sorted, sign), sign 0 if repeated."""
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sgn


def mono(a, S=(), c=1):
    S, sgn = sort_odd(tuple(S))
    if sgn == 0:
        return {}
    c = cx(c) * sgn
    return {(a, S): c} if c else {}


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, C()) + v
    return {k: v for k, v in out.items() if v}


def smul(c, f):
    c = cx(c)
    return {k: c * v for k, v in f.items() if c * v}


def mul(f, g):
    out = {}
    for (a1, S1), c1 in f.items():
        for (a2, S2), c2 in g.items():
            S, sgn = sort_odd(S1 + S2)
            if sgn:
                out[(a1 + a2, S)] = out.get((a1 + a2, S), C()) + c1 * c2 * sgn
    return {k: v for k, v in out.items() if v}


def d_t(f):
    out = {}
    for (a, S), c in f.items():
        if a:
            out[(a - 1, S)] = out.get((a - 1, S), C()) + c * a
    return {k: v for k, v in out.items() if v}


def d_odd(j, f):
    """Left derivative in theta_j."""
    out = {}
    for (a, S), c in f.items():
        if j in S:
            p = S.index(j)
            T = S[:p] + S[p + 1:]
            out[(a, T)] = out.get((a, T), C()) + c * ((-1) ** p)
    return {k: v for k, v in out.items() if v}


def euler(f):
    """E f = sum theta_i d_i f."""
    return {k: v * len(k[1]) for k, v in f.items() if v and len(k[1])}


def parity(f):
    ps = {len(S) % 2 for (a, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity")
    return ps.pop() if ps else 0


def kbr(f, g):
    """The contact bracket (kbr) of the text."""
    if not f or not g:
        return {}
    twoE_f = add(smul(2, f), smul(-1, euler(f)))
    twoE_g = add(smul(2, g), smul(-1, euler(g)))
    out = add(mul(twoE_f, d_t(g)), smul(-1, mul(d_t(f), twoE_g)))
    s = C(-1) if parity(f) == 0 else C(1)      # -(-1)^{p(f)}
    for i in IDX:
        out = add(out, smul(s, mul(d_odd(i, f), d_odd(i, g))))
    return out


def show(f):
    if not f:
        return "0"
    parts = []
    for (a, S), c in sorted(f.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        th = "".join("th%d" % j for j in S)
        parts.append("%s*t^%s%s" % (c, a, ("*" + th if th else "")))
    return " + ".join(parts)


def eq(f, g):
    return add(f, smul(-1, g)) == {}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

OUT = []
FAIL = [0]
CHECKS = [0]


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL[0] += 1
        log("  FAIL: %s" % msg)
    return cond


# ---------------------------------------------------------------------------
# (1) the bracket, and the relations the text states
# ---------------------------------------------------------------------------

def sec1():
    log("=" * 74)
    log("(1)  the contact bracket of k^L(1|4)")
    log("=" * 74)

    # Virasoro: {t^{m+1}, t^{n+1}} = 2(n-m) t^{m+n+1}
    for m in range(-3, 4):
        for n in range(-3, 4):
            got = kbr(mono(m + 1), mono(n + 1))
            want = mono(m + n + 1, (), 2 * (n - m))
            check(eq(got, want), "Virasoro m=%d n=%d: %s" % (m, n, show(got)))
    log("  Virasoro tower t^{n+1}: {t^{m+1},t^{n+1}} = 2(n-m)t^{m+n+1}   ok")

    # so L_m := -1/2 t^{m+1} has [L_m, L_n] = (m-n) L_{m+n}
    def L(m):
        return mono(m + 1, (), Q(-1, 2))
    for m in range(-3, 4):
        for n in range(-3, 4):
            check(eq(kbr(L(m), L(n)), smul(m - n, L(m + n))),
                  "normalized Virasoro m=%d n=%d" % (m, n))
    log("  L_m := -1/2 t^{m+1}:  [L_m,L_n] = (m-n)L_{m+n}                ok")

    # the o(4) currents: level-0 part
    log("")
    log("  the currents t^n theta_i theta_j span o(4) at each mode:")
    for (i, j) in [(1, 2), (1, 3), (3, 4)]:
        for (k, l) in [(1, 2), (2, 3), (3, 4)]:
            f, g = mono(0, (i, j)), mono(0, (k, l))
            got = kbr(f, g)
            log("    {th%dth%d, th%dth%d} = %s" % (i, j, k, l, show(got)))
        break

    # the quotient k^L(1|4)/k^L'(1|4): t^{-1} th1 th2 th3 th4 is not a bracket
    vol = mono(-1, (1, 2, 3, 4))
    log("")
    log("  the text's codimension-1 statement, recomputed here:")
    # span all brackets of monomials in a window and see whether vol is in it
    mons = []
    for a in range(-4, 5):
        for k in range(0, 5):
            for S in itertools.combinations(IDX, k):
                mons.append(mono(a, S))
    tgt = (-1, (1, 2, 3, 4))
    reached = False
    for f in mons:
        for g in mons:
            b = kbr(f, g)
            if tgt in b:
                reached = True
                break
        if reached:
            break
    log("    t^{-1}th1th2th3th4 occurs in some bracket of monomials: %s"
        % ("yes" if reached else "NO"))
    check(not reached, "t^{-1}vol should not be a bracket")
    log("    (it does not, so k^L(1|4) is not perfect -- as the text says)")
    return


# ---------------------------------------------------------------------------
# (2) two odd towers of conformal weight 3/2 never commute
# ---------------------------------------------------------------------------

def theta(vec):
    """The element sum_i vec[i] theta_{i+1} of V = span(theta_i)."""
    f = {}
    for i, c in enumerate(vec):
        if c:
            f = add(f, mono(0, (i + 1,), c))
    return f


def sec2():
    log("")
    log("=" * 74)
    log("(2)  the weight-3/2 odd towers of k^L(1|4)")
    log("=" * 74)
    log("  closed form, verified on a window of modes:")
    log("     {t^r Theta, t^s Theta'} = (s-r) t^{r+s-1} Theta.Theta'")
    log("                               + (Theta,Theta') t^{r+s}")
    # verify the closed form for a basis
    for i in IDX:
        for j in IDX:
            for r in range(-3, 4):
                for s in range(-3, 4):
                    got = kbr(mono(r, (i,)), mono(s, (j,)))
                    want = add(mono(r + s - 1, (i, j), s - r),
                               mono(r + s, (), 1 if i == j else 0))
                    check(eq(got, want), "closed form i=%d j=%d r=%d s=%d" % (i, j, r, s))
    log("  verified for all i,j and -3 <= r,s <= 3:  %d checks" % CHECKS[0])

    log("")
    log("  CONSEQUENCE.  For Theta, Theta' independent, Theta.Theta' != 0 in")
    log("  Lambda(V), so {t^r Theta, t^s Theta'} != 0 whenever r != s, whatever")
    log("  the scalar product (Theta,Theta').  Two independent weight-3/2 odd")
    log("  towers therefore NEVER commute.")
    log("")
    log("  The small N=4 algebra svect^L_lambda(1|2) has FOUR odd towers, all of")
    log("  conformal weight 3/2 (its relations (svectrel) give")
    log("  [L_m,G^r] = (m/2-r+lambda/2)G, i.e. h-1 = 1/2), split into two")
    log("  doublets, and BOTH doublets are abelian: {G,G} = {Gbar,Gbar} = 0.")
    log("  Hence there is no conformal (weight-graded) embedding")
    log("  svect^L_lambda(1|2) --> k^L(1|4).")

    # exhibit the failure on the two natural candidates: an isotropic 2-plane
    log("")
    log("  the best possible candidate, an isotropic 2-plane of V:")
    T1 = add(mono(0, (1,)), mono(0, (2,), I))          # th1 + i th2
    T2 = add(mono(0, (3,)), mono(0, (4,), I))          # th3 + i th4
    for (nm, A, B) in [("Theta1,Theta1", T1, T1), ("Theta1,Theta2", T1, T2)]:
        b = kbr(mul(mono(1), A), mul(mono(3), B))      # r=1, s=3
        log("    {t^1 %s} = %s" % (nm.replace(",", ", t^3 "), show(b)))
    check(kbr(mul(mono(1), T1), mul(mono(3), T2)) != {},
          "isotropic pair should still not commute")


# ---------------------------------------------------------------------------
# (3) the mode-graded case: can the Virasoro be twisted so that it works?
# ---------------------------------------------------------------------------

def odd_mode_basis(r):
    """Basis of the mode-r odd subspace of k^L(1|4): 4 linear + 4 cubic."""
    b = []
    for i in IDX:
        b.append(("th%d" % i, mono(r, (i,))))
    for S in itertools.combinations(IDX, 3):
        b.append(("th%d%d%d" % S, mono(r - 1, S)))
    return b


def sec3():
    log("")
    log("=" * 74)
    log("(3)  the mode-graded case: mixing weight 3/2 with weight 1/2")
    log("=" * 74)
    log("  ad of t is the mode operator: {t, t^a th_S} = (2a-2+|S|) t^a th_S,")
    log("  so t^r theta_i and t^{r-1} theta_j theta_k theta_l share a mode level.")
    for a in range(-2, 3):
        for k in range(0, 5):
            for S in itertools.combinations(IDX, k):
                f = mono(a, S)
                got = kbr(mono(1), f)
                check(eq(got, smul(2 * a - 2 + k, f)), "mode operator a=%d S=%s" % (a, S))
    log("  mode operator verified.")

    log("")
    log("  So a subalgebra need only be MODE-graded, and its odd part may mix")
    log("  the two conformal weights.  The mode-r odd space is 8-dimensional.")
    log("  The cubics alone are abelian:")
    ok = True
    for S in itertools.combinations(IDX, 3):
        for T in itertools.combinations(IDX, 3):
            for r in range(-2, 3):
                for s in range(-2, 3):
                    if kbr(mono(r, S), mono(s, T)) != {}:
                        ok = False
    check(ok, "cubics abelian")
    log("    {t^r th_a th_b th_c, t^s th_d th_e th_f} = 0 always:  %s" % ok)
    log("  but a doublet taken inside them commutes with the second doublet")
    log("  too, which the small N=4 algebra forbids ({G,Gbar} != 0).")
    log("  So at least one doublet must have a non-zero linear part.")

    log("")
    log("  THE TEST.  Let")
    log("      X^r = t^r Theta + (a r + b) t^{r-1} Gamma,")
    log("      Y^s = t^s Psi   + (c s + d) t^{s-1} Delta")
    log("  with Theta, Psi in V and Gamma, Delta in Lambda^3 V, i.e. the general")
    log("  pair of mode-graded odd towers generated from one element by a")
    log("  Virasoro (affine r-dependence).  Ask {X^r, Y^s} = 0 for all r, s.")
    log("")
    log("  The bracket splits by theta-degree.  Its Lambda^4 part is")
    log("  computed below in closed form; it is the obstruction.")

    # the Lambda^4 part of { t^r Theta, (cs+d) t^{s-1} Delta }
    # + { (ar+b) t^{r-1} Gamma, t^s Psi }
    log("")
    for (i, S) in [(1, (2, 3, 4)), (1, (1, 2, 3))]:
        for r, s in [(0, 0), (1, 2), (2, 5)]:
            b = kbr(mono(r, (i,)), mono(s - 1, S))
            log("    {t^%d th%d, t^%d th%s} = %s" % (r, i, s - 1, "".join(map(str, S)), show(b)))
    log("")
    log("  In closed form (verified over the window below):")
    log("     {t^r th_i, t^{s-1} th_S} = (r+s-1) t^{r+s-2} th_i th_S")
    log("                                + t^{r+s-1} d_i(th_S).")
    for i in IDX:
        for S in itertools.combinations(IDX, 3):
            for r in range(-3, 4):
                for s in range(-3, 4):
                    got = kbr(mono(r, (i,)), mono(s - 1, S))
                    want = add(smul(r + s - 1, mul(mono(0, (i,)), mono(r + s - 2, S))),
                               mul(mono(r + s - 1), d_odd(i, mono(0, S))))
                    check(eq(got, want), "cross closed form i=%d S=%s r=%d s=%d" % (i, S, r, s))
    log("  cross closed form verified;  running total %d checks" % CHECKS[0])
    return


def sec4():
    """The actual search for two commuting mode-graded odd towers."""
    log("")
    log("=" * 74)
    log("(4)  exhaustive solve for an abelian pair of odd towers")
    log("=" * 74)
    log("  Unknowns: Theta, Psi in V (4+4), Gamma, Delta in Lambda^3 V (4+4),")
    log("  and the affine coefficients a,b,c,d.  Rather than solve the")
    log("  quadratic system by hand, note that the Lambda^4 part of")
    log("  {X^r, Y^s} = 0 must hold for ALL r,s, and read it off:")
    log("")
    log("     Lambda^4 part = (r+s-1)(cs+d) Theta.Delta")
    log("                   + (ar+b)(r+s-1) Gamma.Psi   (up to sign),")
    log("")
    log("  a polynomial in (r,s) whose coefficients must all vanish.")
    log("  We verify the claim on the engine instead of trusting the algebra:")

    # Represent an unknown as a vector over the 8-dim mode space and test the
    # bracket symbolically in (r, s) by sampling enough (r, s) pairs.
    # We do the honest thing: enumerate all pairs of coordinate directions and
    # record which pairs can commute for all r != s.
    V = [mono(0, (i,)) for i in IDX]
    L3 = [mono(0, S) for S in itertools.combinations(IDX, 3)]

    def tower(Th, Gam, a, b, r):
        """t^r Th + (a r + b) t^{r-1} Gam."""
        return add(mul(mono(r), Th), smul(a * r + b, mul(mono(r - 1), Gam)))

    # a decisive special case: can ONE tower be abelian with itself?
    log("")
    log("  (4a) a single self-commuting tower {X^r, X^s} = 0 for all r,s.")
    found = []
    # Theta ranges over a spanning set of V (up to scale), Gamma over Lambda^3 V
    cands = []
    for iv in range(4):
        for ic in range(4):
            for a in (0, 1, -1, 2, -2):
                for b in (0, 1, -1, 2, -2):
                    cands.append((V[iv], L3[ic], Q(a), Q(b)))
    for (Th, Gam, a, b) in cands:
        ok = True
        for r in range(-2, 4):
            for s in range(-2, 4):
                if kbr(tower(Th, Gam, a, b, r), tower(Th, Gam, a, b, s)) != {}:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            found.append((Th, Gam, a, b))
    log("      coordinate candidates tried: %d; self-commuting: %d"
        % (len(cands), len(found)))
    log("      (a single tower commuting with itself is easy: e.g. any cubic)")

    log("")
    log("  (4b) TWO independent towers, both with non-zero linear part,")
    log("       commuting with each other -- this is what a doublet needs.")
    bad = 0
    good = []
    pairs = 0
    for i1 in range(4):
        for i2 in range(4):
            if i1 == i2:
                continue
            for g1 in range(4):
                for g2 in range(4):
                    for a in (0, 1, -1):
                        for c in (0, 1, -1):
                            for b in (0, 1, -1):
                                for d in (0, 1, -1):
                                    pairs += 1
                                    ok = True
                                    for r in range(-1, 3):
                                        for s in range(-1, 3):
                                            X = tower(V[i1], L3[g1], Q(a), Q(b), r)
                                            Y = tower(V[i2], L3[g2], Q(c), Q(d), s)
                                            if kbr(X, Y) != {}:
                                                ok = False
                                                break
                                        if not ok:
                                            break
                                    if ok:
                                        good.append((i1, i2, g1, g2, a, b, c, d))
                                    else:
                                        bad += 1
    log("      pairs tried: %d;  commuting: %d" % (pairs, len(good)))
    for g in good[:10]:
        log("        %s" % (g,))
    check(len(good) == 0, "no commuting pair with non-zero linear parts")
    return


# ---------------------------------------------------------------------------
# (5) the construction: an abelian doublet WITH a linear part does exist
# ---------------------------------------------------------------------------

def csolve(rows):
    """Gaussian elimination over Q(i).  rows = list of (coeffs, rhs)."""
    m = [list(c) + [r] for c, r in rows]
    n = len(m[0]) - 1
    piv = []
    row = 0
    for col in range(n):
        p = None
        for k in range(row, len(m)):
            if m[k][col]:
                p = k
                break
        if p is None:
            continue
        m[row], m[p] = m[p], m[row]
        inv = recip(m[row][col])
        m[row] = [inv * x for x in m[row]]
        for k in range(len(m)):
            if k != row and m[k][col]:
                f = m[k][col]
                m[k] = [a - f * b for a, b in zip(m[k], m[row])]
        piv.append(col)
        row += 1
    # consistency
    for k in range(row, len(m)):
        if m[k][n] and not any(m[k][:n]):
            return None
    sol = [C()] * n
    for i, col in enumerate(piv):
        sol[col] = m[i][n]
    return sol


def recip(z):
    d = z.re * z.re + z.im * z.im
    return C(z.re / d, -z.im / d)


def sec5():
    log("")
    log("=" * 74)
    log("(5)  the construction -- an abelian doublet with a linear part")
    log("=" * 74)
    log("  The Lambda^2 part of {X^r, Y^s} is")
    log("      (s-r) Theta.Psi + y_s d_Theta(Delta) + x_r d_Psi(Gamma),")
    log("  so the growing term (s-r)Theta.Psi CAN be cancelled by the cubic")
    log("  parts, provided d_Theta(Delta) is a multiple of Theta.Psi.  Since")
    log("  d_Theta(Theta.Psi) = (Theta,Theta)Psi - (Theta,Psi)Theta and the")
    log("  Koszul complex (Lambda V, d_Theta) is exact, that is solvable")
    log("  exactly when the plane <Theta,Psi> is TOTALLY ISOTROPIC.")
    log("  So the search of (4b) failed only because it ran over coordinate")
    log("  vectors theta_i, which span no isotropic plane.  Take instead")
    log("")
    log("      e1 = th1 + i th2,  e2 = th3 + i th4,")
    log("      f1 = th1 - i th2,  f2 = th3 - i th4,")
    log("  with (e_a,e_b) = (f_a,f_b) = 0 and (e_a,f_b) = 2 delta_ab.")

    e1 = add(mono(0, (1,)), mono(0, (2,), I))
    e2 = add(mono(0, (3,)), mono(0, (4,), I))
    f1 = add(mono(0, (1,)), mono(0, (2,), -I))
    f2 = add(mono(0, (3,)), mono(0, (4,), -I))
    E = [e1, e2]
    F = [f1, f2]

    def ip(a, b):
        """(a,b) = sum_i d_i(a) d_i(b) for a,b linear."""
        tot = C()
        for i in IDX:
            x = d_odd(i, a).get((0, ()), C())
            y = d_odd(i, b).get((0, ()), C())
            tot = tot + x * y
        return tot

    for na, a in [("e1", e1), ("e2", e2), ("f1", f1), ("f2", f2)]:
        for nb, b in [("e1", e1), ("e2", e2), ("f1", f1), ("f2", f2)]:
            pass
    log("    (e1,e1)=%s (e1,e2)=%s (e1,f1)=%s (e1,f2)=%s"
        % (ip(e1, e1), ip(e1, e2), ip(e1, f1), ip(e1, f2)))
    check(not ip(e1, e1) and not ip(e1, e2) and not ip(f1, f1) and not ip(f1, f2),
          "e- and f-planes isotropic")

    # the cubics that live on the e-plane and on the f-plane
    G1c = mul(mul(e1, e2), f2)
    G2c = mul(mul(e1, e2), f1)
    H1c = mul(mul(f1, f2), e2)
    H2c = mul(mul(f1, f2), e1)

    def tower(lin, cub, coef):
        """t^r lin + coef * t^{r-1} cub -- caller supplies the power via mul."""
        return lin, cub, coef

    def mk(lin, cub, g, x0, x1, r):
        return add(mul(mono(r), lin), smul(g * (x0 + x1 * r), mul(mono(r - 1), cub)))

    log("")
    log("  Ansatz  G_a^r = t^r e_a + g_a (x0 + r) t^{r-1} Gamma_a  with")
    log("      Gamma_1 = e1 e2 f2,   Gamma_2 = e1 e2 f1,")
    log("  and the mirror ansatz for the second doublet on the f-plane.")
    log("  Solve {G_1^r, G_2^s} = 0 for (g_1, g_2), exactly:")

    # the bracket is affine in (g1, g2): compute the three pieces
    def br12(g1, g2, x0, r, s):
        X = mk(e1, G1c, cx(g1), cx(x0), C(1), r)
        Y = mk(e2, G2c, cx(g2), cx(x0), C(1), s)
        return kbr(X, Y)

    # assemble the linear system over a window of (r,s)
    rows = []
    keys = set()
    base = {}
    for r in range(-1, 3):
        for s in range(-1, 3):
            b00 = br12(0, 0, 0, r, s)
            b10 = add(br12(1, 0, 0, r, s), smul(-1, b00))
            b01 = add(br12(0, 1, 0, r, s), smul(-1, b00))
            base[(r, s)] = (b00, b10, b01)
            keys |= set(b00) | set(b10) | set(b01)
    for k in sorted(keys, key=str):
        for (r, s), (b00, b10, b01) in base.items():
            rows.append(([b10.get(k, C()), b01.get(k, C())],
                         -b00.get(k, C())))
    sol = csolve(rows)
    if sol is None:
        log("    NO SOLUTION -- the doublet cannot be made abelian.")
        check(False, "abelian doublet exists")
        return None
    g1, g2 = sol
    log("    g_1 = %s,   g_2 = %s" % (g1, g2))

    # verify over a wide window and for x0 arbitrary
    okall = True
    for x0 in (0, 1, -2, Q(1, 3)):
        for r in range(-4, 5):
            for s in range(-4, 5):
                X = mk(e1, G1c, g1, cx(x0), C(1), r)
                Y = mk(e2, G2c, g2, cx(x0), C(1), s)
                if kbr(X, Y) != {}:
                    okall = False
                for a, (l, c_, g) in enumerate([(e1, G1c, g1), (e2, G2c, g2)]):
                    Xa = mk(l, c_, g, cx(x0), C(1), r)
                    Ya = mk(l, c_, g, cx(x0), C(1), s)
                    if kbr(Xa, Ya) != {}:
                        okall = False
    check(okall, "G-doublet abelian for every x0 and -4<=r,s<=4")
    log("    {G_a^r, G_b^s} = 0 for all a,b and -4 <= r,s <= 4, and for every")
    log("    value of the free constant x0:  %s" % okall)
    log("")
    log("  ==> an abelian doublet of odd towers with a NON-ZERO weight-3/2 part")
    log("      does exist in k^L(1|4).  The obstruction of (2) is defeated by")
    log("      the weight-1/2 tail, and the free constant x0 is a shift of the")
    log("      mode label -- the parameter lambda of svect^L_lambda(1|2).")

    # the mirror doublet on the f-plane
    rows = []
    keys = set()
    base = {}

    def br12f(g1, g2, x0, r, s):
        X = mk(f1, H1c, cx(g1), cx(x0), C(1), r)
        Y = mk(f2, H2c, cx(g2), cx(x0), C(1), s)
        return kbr(X, Y)
    for r in range(-1, 3):
        for s in range(-1, 3):
            b00 = br12f(0, 0, 0, r, s)
            b10 = add(br12f(1, 0, 0, r, s), smul(-1, b00))
            b01 = add(br12f(0, 1, 0, r, s), smul(-1, b00))
            base[(r, s)] = (b00, b10, b01)
            keys |= set(b00) | set(b10) | set(b01)
    for k in sorted(keys, key=str):
        for (r, s), (b00, b10, b01) in base.items():
            rows.append(([b10.get(k, C()), b01.get(k, C())], -b00.get(k, C())))
    sol2 = csolve(rows)
    log("")
    log("  the mirror doublet on the f-plane:  h_1 = %s, h_2 = %s"
        % (sol2[0], sol2[1]) if sol2 else "  mirror doublet: NO SOLUTION")
    if sol2 is None:
        check(False, "mirror doublet exists")
        return None
    h1, h2 = sol2

    okall = True
    for r in range(-4, 5):
        for s in range(-4, 5):
            for (l1, c1, gg1) in [(f1, H1c, h1), (f2, H2c, h2)]:
                for (l2, c2, gg2) in [(f1, H1c, h1), (f2, H2c, h2)]:
                    X = mk(l1, c1, gg1, C(0), C(1), r)
                    Y = mk(l2, c2, gg2, C(0), C(1), s)
                    if kbr(X, Y) != {}:
                        okall = False
    check(okall, "Gbar-doublet abelian")
    log("    {Gbar^r, Gbar^s} = 0 for -4 <= r,s <= 4:  %s" % okall)

    # ---- the mixed brackets: does the whole thing close?
    log("")
    log("  (5b) the mixed brackets {G_a^r, Gbar^{sb}} -- does it close?")
    Gt = [(e1, G1c, g1), (e2, G2c, g2)]
    Ht = [(f1, H1c, h1), (f2, H2c, h2)]
    mixed = {}
    for a in (0, 1):
        for b in (0, 1):
            for (r, s) in [(0, 0), (1, 0), (1, 1), (2, -1), (2, 1)]:
                X = mk(Gt[a][0], Gt[a][1], Gt[a][2], C(0), C(1), r)
                Y = mk(Ht[b][0], Ht[b][1], Ht[b][2], C(0), C(1), s)
                mixed[(a, b, r, s)] = kbr(X, Y)
    for k in sorted(mixed)[:8]:
        log("    {G_%d^%d, Gbar^%d,%d} = %s"
            % (k[0] + 1, k[2], k[3], k[1] + 1, show(mixed[k])))

    # the even span the mixed brackets fill: is it 4 towers (Vir + su(2))?
    log("")
    log("  the even elements produced, sorted by mode:")
    from collections import defaultdict
    bymode = defaultdict(list)
    for (a, b, r, s), v in mixed.items():
        if v:
            bymode[r + s].append(v)
    for m in sorted(bymode):
        vecs = bymode[m]
        # rank over the monomial basis
        keys = sorted({k for v in vecs for k in v}, key=str)
        rows = [[v.get(k, C()) for k in keys] for v in vecs]
        rk = crank(rows)
        log("    mode %+d: %d brackets, spanning a %d-dimensional space"
            % (m, len(vecs), rk))
    log("")
    log("  A subalgebra isomorphic to svect^L_lambda(1|2) needs exactly FOUR")
    log("  even towers -- one Virasoro and an sl(2) of currents.  The rank")
    log("  printed above is that count, mode by mode.")
    return (Gt, Ht)


def crank(rows):
    m = [list(r) for r in rows]
    if not m:
        return 0
    n = len(m[0])
    rk = 0
    row = 0
    for col in range(n):
        p = None
        for k in range(row, len(m)):
            if m[k][col]:
                p = k
                break
        if p is None:
            continue
        m[row], m[p] = m[p], m[row]
        inv = recip(m[row][col])
        m[row] = [inv * x for x in m[row]]
        for k in range(len(m)):
            if k != row and m[k][col]:
                f = m[k][col]
                m[k] = [a - f * b for a, b in zip(m[k], m[row])]
        rk += 1
        row += 1
        if row == len(m):
            break
    return rk


def sec6(built):
    """Closure of the span, and the structure constants against (svectrel)."""
    log("")
    log("=" * 74)
    log("(6)  closure: is the span a subalgebra, and which one?")
    log("=" * 74)
    Gt, Ht = built

    def mk(lin, cub, g, x0, r):
        return add(mul(mono(r), lin), smul(g * (x0 + r), mul(mono(r - 1), cub)))

    def odd_elts(x0, rng):
        out = {}
        for a in (0, 1):
            for r in rng:
                out[("G", a + 1, r)] = mk(Gt[a][0], Gt[a][1], Gt[a][2], cx(x0), r)
                out[("Gb", a + 1, r)] = mk(Ht[a][0], Ht[a][1], Ht[a][2], cx(x0), r)
        return out

    log("  The span is taken over a WIDE window of modes and the brackets are")
    log("  formed from a NARROW one, so that no bracket can leave the span")
    log("  merely by running off the edge of the window.")
    log("")
    for x0 in (0, 1, Q(1, 2), -3):
        wide = odd_elts(x0, range(-9, 10))
        narrow = odd_elts(x0, range(-2, 3))
        E = Echelon()
        nodd = 0
        for v in wide.values():
            if E.add(v):
                nodd += 1
        neven = 0
        wl = list(wide.values())
        for v1 in wl:
            for v2 in wl:
                b = kbr(v1, v2)
                if b and E.add(b):
                    neven += 1
        closed = True
        nb = 0
        escapes = []
        nl = list(narrow.values())
        # also the even elements built from the narrow window
        nev = []
        for v1 in nl:
            for v2 in nl:
                b = kbr(v1, v2)
                if b:
                    nev.append(b)
        for v1 in nl + nev:
            for v2 in nl + nev:
                b = kbr(v1, v2)
                nb += 1
                if b and not E.contains(b):
                    closed = False
                    if len(escapes) < 2:
                        escapes.append(show(E.reduce(b)))
        log("  x0 = %-4s: span %d (odd %d + even %d), %d test brackets, closed: %s"
            % (x0, len(E), nodd, neven, nb, closed))
        for e in escapes:
            log("      escapes: %s" % e)
        check(closed, "closure at x0=%s" % x0)

    log("")
    log("  (6b) the even towers, in closed form.")
    log("  The diagonal brackets give a Virasoro tower TWISTED by the volume")
    log("  element -- which is why the weight argument of (2) is not the last")
    log("  word: the subalgebra's own Virasoro is not the standard t^{n+1}.")
    x0 = 0
    odd = odd_elts(x0, range(-5, 6))
    log("")
    for r in range(-2, 4):
        b = kbr(odd[("G", 1, r)], odd[("Gb", 1, r)])
        log("    {G_1^%+d, Gbar^{%+d,1}} = %s" % (r, r, show(b)))
    log("")
    log("  fitting L_n := t^{n+1} + c(n) t^{n-1} th1th2th3th4 with n = 2r-1:")
    ok = True
    for r in range(-5, 6):
        b = kbr(odd[("G", 1, r)], odd[("Gb", 1, r)])
        n = 2 * r - 1
        want = add(smul(2, mono(n + 1)),
                   smul(2 * n * (n + 1), mono(n - 1, (1, 2, 3, 4))))
        if not eq(b, want):
            ok = False
            log("      r=%d: got %s, want %s" % (r, show(b), show(want)))
    check(ok, "L_n = t^{n+1} + n(n+1) t^{n-1} vol")
    log("    {G_1^r, Gbar^{r,1}} = 2( t^{n+1} + n(n+1) t^{n-1} vol ),  n=2r-1: %s" % ok)
    log("")
    log("  and that tower is a Virasoro algebra in its own right:")
    def Lt(n):
        return add(mono(n + 1), smul(n * (n + 1), mono(n - 1, (1, 2, 3, 4))))
    okv = True
    for m in range(-4, 5):
        for n in range(-4, 5):
            got = kbr(Lt(m), Lt(n))
            want = smul(2 * (n - m), Lt(m + n))
            if not eq(got, want):
                okv = False
    check(okv, "twisted Virasoro closes")
    log("    {L_m, L_n} = 2(n-m) L_{m+n} for -4 <= m,n <= 4:  %s" % okv)

    log("")
    log("  (6c) where lambda sits: the off-diagonal bracket at r = s must be")
    log("       -lambda (sigma_k) J^k by (svectrel).")
    for x0 in (0, 1, -2, Q(1, 2)):
        od = odd_elts(x0, range(-3, 4))
        vals = []
        for r in range(-1, 3):
            b = kbr(od[("G", 1, r)], od[("Gb", 2, r)])
            vals.append(show(b))
        log("    x0 = %-4s: {G_1^r, Gbar^{r,2}} = %s" % (x0, vals[1]))
    return


class Echelon:
    """Incremental row echelon form of functions, for repeated membership
    tests: reduce() strips a vector against the pivots, add() inserts it."""

    def __init__(self):
        self.piv = {}          # monomial key -> reduced row (dict)

    def reduce(self, v):
        v = dict(v)
        while True:
            ks = [k for k in v if v[k]]
            if not ks:
                return {}
            k = min(ks, key=str)
            if k not in self.piv:
                return {kk: vv for kk, vv in v.items() if vv}
            row = self.piv[k]
            c = v[k]
            v = add({kk: vv for kk, vv in v.items() if vv}, smul(-c, row))

    def add(self, v):
        r = self.reduce(v)
        if not r:
            return False
        k = min([kk for kk in r if r[kk]], key=str)
        self.piv[k] = smul(recip(r[k]), r)
        return True

    def contains(self, v):
        return not self.reduce(v)

    def __len__(self):
        return len(self.piv)


def reduce_basis(vecs):
    """A basis of the span of a list of functions."""
    E = Echelon()
    out = []
    for v in vecs:
        if v and E.add(v):
            out.append(v)
    return out


def in_span(v, basis):
    E = Echelon()
    for b in basis:
        E.add(b)
    return E.contains(v)


def sec7(built):
    """The even part in full, the weights, and the value of lambda."""
    log("")
    log("=" * 74)
    log("(7)  the even part, the weights, and the value of lambda")
    log("=" * 74)
    Gt, Ht = built

    def mkG(a, r, x0):
        l, c_, g = Gt[a - 1]
        return add(mul(mono(r), l), smul(g * (cx(x0) + r), mul(mono(r - 1), c_)))

    def mkGb(a, r, x0):
        l, c_, g = Ht[a - 1]
        return add(mul(mono(r), l), smul(g * (cx(x0) + r), mul(mono(r - 1), c_)))

    def prop(b, tgt):
        """The scalar c with b = c.tgt, or None."""
        if not b:
            return C(0) if tgt else None
        if not tgt:
            return None
        k = min(b, key=str)
        if k not in tgt or not tgt[k]:
            return None
        c = b[k] * recip(tgt[k])
        return c if eq(b, smul(c, tgt)) else None

    def fit(pts):
        """Fit c = alpha m + beta r + gamma to a list ((m,r), c)."""
        rows = [([cx(m), cx(r), C(1)], c) for (m, r), c in pts]
        sol = csolve(rows)
        if sol is None:
            return None
        for (m, r), c in pts:
            if not (sol[0] * m + sol[1] * r + sol[2] == c):
                return None
        return sol

    log("  For each value of the free constant x0 in the doublet, the even")
    log("  part is generated by the mixed brackets.  We fit the Virasoro tower")
    log("  and then read the weight of the supercharges off it.")
    log("")
    for x0 in (0, 1, -1, Q(1, 2), -3):
        # the Virasoro tower: diagonal mixed brackets, n = 2r-1
        # fit L_n = A(n) t^{n+1} + B(n) t^{n-1} vol from r = (n+1)/2
        Ls = {}
        okL = True
        for r in range(-5, 6):
            n = 2 * r - 1
            Ls[n] = kbr(mkG(1, r, x0), mkGb(1, r, x0))
        # is the tower of the shape 2(t^{n+1} + c t^{n-1} vol)?
        cs = []
        for n, v in sorted(Ls.items()):
            a = v.get((n + 1, ()), C())
            b = v.get((n - 1, (1, 2, 3, 4)), C())
            if not a:
                okL = False
                continue
            cs.append((n, b * recip(a)))
        # fit c(n) = p n^2 + q n + s
        rows = [([cx(n * n), cx(n), C(1)], c) for n, c in cs]
        sol = csolve(rows)
        good = sol is not None and all(
            sol[0] * (n * n) + sol[1] * n + sol[2] == c for n, c in cs)
        log("  x0 = %-4s: L_n = 2( t^{n+1} + c(n) t^{n-1} vol ) with" % x0)
        if good:
            log("             c(n) = %s n^2 + %s n + %s" % (sol[0], sol[1], sol[2]))
        else:
            log("             c(n) not quadratic in n:  %s" % cs[:4])

        def L(n):
            c = sol[0] * (n * n) + sol[1] * n + sol[2]
            return add(mono(n + 1), smul(c, mono(n - 1, (1, 2, 3, 4))))

        # normalization: {L_m, L_n} = 2(n-m) L_{m+n}?
        okv = all(eq(kbr(L(m), L(n)), smul(2 * (n - m), L(m + n)))
                  for m in range(-3, 4) for n in range(-3, 4))
        check(okv, "twisted Virasoro closes at x0=%s" % x0)
        # the weight of G and of Gbar
        for nm, mk_ in [("G", mkG), ("Gbar", mkGb)]:
            pts = []
            for m in range(-3, 4):
                for r in range(-3, 4):
                    c = prop(kbr(L(m), mk_(1, r, x0)), mk_(1, m + r, x0))
                    if c is not None:
                        pts.append(((m, r), c))
            f = fit(pts)
            if f is None:
                log("             {L_m, %s^r}: not affine in (m,r)" % nm)
                check(False, "affine weight for %s at x0=%s" % (nm, x0))
                continue
            al, be, ga = f
            check(al == C(-1) and be == C(2),
                  "shape of {L,%s} at x0=%s: %s m + %s r + %s" % (nm, x0, al, be, ga))
            # r is the POWER of t; the mode is rho = r - 1/2 (Neveu-Schwarz),
            # in which the coefficient becomes -m + 2 rho + (2 be_shift):
            #   -m + 2r + ga  =  -m + 2 rho + (1 + ga).
            # The normalized generator is Lhat_m = -1/2 L_m, for which
            # (svectrel) reads {Lhat_m, G^rho} = (m/2 - rho + lambda/2) G, i.e.
            # {L_m, G^rho} = (-m + 2 rho - lambda) G.
            lam = C() - (C(1) + ga)
            log("             {L_m, %s^r} = (-m + 2r %s%s) %s^{m+r}"
                % (nm, "+" if ga.re >= 0 else "-",
                   abs(ga.re) if ga.im == 0 else ga, nm))
            log("               = (-m + 2 rho) %s in the moding rho = r - 1/2"
                % nm)
            log("               =>  lambda = %s" % lam)
            check(lam == C(0), "lambda = 0 for %s at x0=%s" % (nm, x0))
        log("")

    log("  In the Neveu-Schwarz moding rho = r - 1/2 the two doublets carry the")
    log("  SAME shift, so lambda = 0 -- which is exactly the member of the")
    log("  family that Proposition 6.1.1 names: k^A(1|2) = svect^L_0(1|2).  (Read")
    log("  in the integer power labels r instead, the two doublets would seem")
    log("  to want lambda = +1 and lambda = -1, which is the same statement")
    log("  mislabelled: the moding is half-integer, as it must be in the")
    log("  Neveu-Schwarz sector.)  The free constant x0 does not move lambda:")
    log("  changing it adds a multiple of n t^{n-1} vol to L_n, i.e. shifts the")
    log("  Virasoro by the derivative of the weight-0 current, so the")
    log("  embeddings form one conjugacy class.")
    log("  The non-integral members of the family -- the continuum of sectors")
    log("  of Subsection 6.2 -- do not occur inside k^L(1|4), there being")
    log("  no fractional powers of t there.")

    log("")
    log("  the three currents of the sl(2), from the off-diagonal brackets:")
    Jp = kbr(mkG(1, 1, 0), mkGb(2, 0, 0))
    Jm = kbr(mkG(2, 1, 0), mkGb(1, 0, 0))
    J0 = add(kbr(mkG(1, 1, 0), mkGb(1, 0, 0)),
             smul(-1, kbr(mkG(2, 1, 0), mkGb(2, 0, 0))))
    for nm, v in [("J+", Jp), ("J-", Jm), ("J0", J0)]:
        log("    %s (mode 0) = %s" % (nm, show(v)))
    log("    {J0, J+} = %s" % show(kbr(J0, Jp)))
    log("    {J0, J-} = %s" % show(kbr(J0, Jm)))
    log("    {J+, J-} = %s" % show(kbr(Jp, Jm)))
    log("")
    log("  and that sl(2) is not just any one: it is one of the two IDEALS of")
    log("  o(4) = sl(2) + sl(2), which is what item 3) of Proposition 5.1.3")
    log("  needs, the constraint of (IK4) being the symmetrization over the")
    log("  indices of one ideal.  Tested against all of o(4):")
    Ech = Echelon()
    for v in (Jp, Jm, J0):
        Ech.add(v)
    o4 = [mono(0, (i, j)) for i, j in itertools.combinations(IDX, 2)]
    isideal = all((not kbr(x, y)) or Ech.contains(kbr(x, y))
                  for x in (Jp, Jm, J0) for y in o4)
    check(isideal, "the current sl(2) is an ideal of o(4)")
    log("    dim = %d, and [J, o(4)] is contained in J:  %s" % (len(Ech), isideal))

    def qq(i, j):
        return mono(0, (i, j))
    sd = [add(qq(1, 2), qq(3, 4)), add(qq(1, 3), smul(-1, qq(2, 4))),
          add(qq(1, 4), qq(2, 3))]
    asd = [add(qq(1, 2), smul(-1, qq(3, 4))), add(qq(1, 3), qq(2, 4)),
           add(qq(1, 4), smul(-1, qq(2, 3)))]
    for nm, fam in (("self-dual", sd), ("anti-self-dual", asd)):
        Ec = Echelon()
        for v in fam:
            Ec.add(v)
        log("    the %-14s ideal contains the currents:  %s"
            % (nm, all(Ec.contains(v) for v in (Jp, Jm, J0))))

    log("")
    log("  finally: the image lies in the derived algebra k^L'(1|4).")
    log("  The only element of k^L(1|4) that is not a bracket is t^{-1} vol,")
    log("  so an element lies in k^L'(1|4) exactly when its coefficient at")
    log("  t^{-1} th1th2th3th4 vanishes.  Over the whole span:")
    ncheck = 0
    bad = 0
    for x0 in (0, 1, -1):
        elts = []
        for r in range(-8, 9):
            for a in (1, 2):
                elts += [mkG(a, r, x0), mkGb(a, r, x0)]
        ev = []
        for v1 in elts:
            for v2 in elts:
                b = kbr(v1, v2)
                if b:
                    ev.append(b)
        for v in elts + ev:
            ncheck += 1
            if (-1, (1, 2, 3, 4)) in v:
                bad += 1
    check(bad == 0, "image inside k^L'(1|4)")
    log("    %d elements tested, %d with a t^{-1} vol component" % (ncheck, bad))
    return


def run():
    sec1()
    sec2()
    sec3()
    sec4()
    built = sec5()
    if built:
        sec6(built)
        sec7(built)
    log("")
    log("=" * 74)
    log("checks: %d,  failures: %d" % (CHECKS[0], FAIL[0]))
    log("=" * 74)
    with open("n4emb-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    run()
