#!/usr/bin/env python3
"""
n4dyn.py -- step 2) of the N=4 programme: the DYNAMICAL equation of (IK4)
fixes the scalar mu(f) left free by the constraint.

Proposition 6.2.3 of the text gives the law

    delta_f psi^b = K_f(psi^b) - Lam^b_d(f) psi^d + mu(f) psi^b            (72)

and says that the constraint alone leaves mu free: any cD-chiral mu will do,
mu being a genuine 1-cocycle ambiguity.  At N=2 the ambiguity lambda(f) of
(deltaV) was fixed by the equation (IK2); here it must be fixed by the second
line of (IK4).  This engine does that, from scratch and in exact arithmetic
over Q(i).

THE SET-UP.  The two halves together form the superspace of sdim (2|8),

    x^-, theta^-_1..theta^-_4   and   x^+, theta^+_1..theta^+_4,

on which the quaternionic superfield q_g^b of (IK4q) lives:  its LOWER index g
is the doublet index of the plus half, its UPPER index b that of the minus
half.  The constraints of the first line of (IK4) are

    cD_-^{(a} q_g^{b)} = 0            (the minus half, on the upper index)
    bD_{+(a} q_{g)}^b  = 0            (the plus half,  on the lower index)

and the dynamics is the second line,

    E1:  cD_-^a (q cD_+^g q^{-1})_g^b = 0
    E2:  cD_+^a (q^{-1} bD_{-g} q)_b^g = -4 i m (bar q)_b^a .

WHAT IS COMPUTED.

 (1) the two-half algebra itself: the contact field K_f of a MINUS-half
     generating function f as a derivation of the whole of O(2|8), checked
     against K_f(g) = {f,g} + 2(d_t f)g of n4shift on minus-half functions,
     and checked to commute with every plus-half derivative.

 (2) the two multipliers and the identity that drives everything:

        Lam^a_b(f) + bLam_b^a(f) = -2 delta^a_b d_t f,                    (*)

     which is {cD^a, bD_b} = -4 delta d_t halved -- the N=4 form of
     lambda + bar lambda = -2 d_t f of Proposition 5.1.4.

 (3) the candidate scalar

        mu(f) := tr Lam(f) + 2 d_t f = i (D_1 D_2 + D_3 D_4) f,

     the u(1)-charge of the multiplier, and the identity

        cD^1 mu = bD_2 Xi(f),   cD^2 mu = -bD_1 Xi(f),

     so that mu is cD-chiral -- which item 3) of Proposition 6.2.3 demands of it --
     EXACTLY on the subalgebra k^Xi on which the constraint survives at all.

 (4) explicit q's satisfying both constraints, built as sums A_g(+) psi^b(-)
     of solutions of the two one-half constraint systems (the joint solution
     space is that tensor product), with an invertible constant term.

 (5) THE POINT, part one: mu drops out of the left-hand side of E2.  Indeed
     the mu-part of the law changes the current q^{-1} bD_g q by bD_g mu . 1
     alone, and cD_+^a kills it, mu being a minus-half function.  This is the
     N=4 form of "cD^+ lambda = 0" in the proof of Proposition 5.1.4.

 (6) THE POINT, part two: mu is then read off the right-hand side.  With the
     mu = 0 law the two sides of E2 transform by

        LHS:  K_f + (Lam^T + 2 d_t f) . ,       RHS:  K_f + (Lam^T - tr Lam) .

     so they agree if and only if mu = tr Lam + 2 d_t f.  The engine does NOT
     assume this: the right-hand side -4im bar q is INVERTIBLE, so its
     transport matrix is divided out rather than guessed, and mu is then the
     difference of the two transports -- which has to come out SCALAR, and
     does.  Invertibility is also what makes mu unique.

 (7) the first dynamical equation E1 is covariant for every mu: the current
     q cD_+^g q^{-1} is invariant under q -> q(1+B) whenever cD_+ B = 0.  So
     it is E2, and only E2, that fixes the shift.

 (8) the whole law, with the mu found, carries E2 to a multiple of itself and
     preserves both constraints.

Usage:  python3 n4dyn.py           (writes n4dyn-out.txt)
"""

import itertools
import sys

from fractions import Fraction as Q

import n4emb
from n4emb import C, I, cx, sort_odd
import n4shift
from n4shift import cnullspace, crank, recip

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
    sys.stdout.flush()


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL[0] += 1
        log("  FAIL: %s" % msg)
    return cond


# ---------------------------------------------------------------------------
# (0) the algebra O(2|8):  keys (a, b, S),  a = deg x^-, b = deg x^+,
#     S a sorted tuple of odd labels, 1..4 minus half, 5..8 plus half.
# ---------------------------------------------------------------------------

MI = (1, 2, 3, 4)          # the odd labels of the minus half
PL = (5, 6, 7, 8)          # ... of the plus half


def mono(a=0, b=0, S=(), c=1):
    S, sgn = sort_odd(tuple(S))
    if sgn == 0:
        return {}
    c = cx(c) * sgn
    return {(a, b, S): c} if c else {}


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, C()) + v
    return {k: v for k, v in out.items() if v}


def smul(c, f):
    c = cx(c)
    return {k: c * v for k, v in f.items() if c * v}


def sub(f, g):
    return add(f, smul(-1, g))


def mul(f, g):
    out = {}
    for (a1, b1, S1), c1 in f.items():
        for (a2, b2, S2), c2 in g.items():
            S, sgn = sort_odd(S1 + S2)
            if sgn:
                k = (a1 + a2, b1 + b2, S)
                out[k] = out.get(k, C()) + c1 * c2 * sgn
    return {k: v for k, v in out.items() if v}


def mulmany(*fs):
    out = mono(0, 0, (), 1)
    for f in fs:
        out = mul(out, f)
    return out


def d_xm(f):
    out = {}
    for (a, b, S), c in f.items():
        if a:
            k = (a - 1, b, S)
            out[k] = out.get(k, C()) + c * a
    return {k: v for k, v in out.items() if v}


def d_xp(f):
    out = {}
    for (a, b, S), c in f.items():
        if b:
            k = (a, b - 1, S)
            out[k] = out.get(k, C()) + c * b
    return {k: v for k, v in out.items() if v}


def d_odd(j, f):
    """Left derivative in theta_j, j in 1..8."""
    out = {}
    for (a, b, S), c in f.items():
        if j in S:
            p = S.index(j)
            T = S[:p] + S[p + 1:]
            k = (a, b, T)
            out[k] = out.get(k, C()) + c * ((-1) ** p)
    return {k: v for k, v in out.items() if v}


def parity(f):
    ps = {len(S) % 2 for (a, b, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity: %s" % show(f))
    return ps.pop() if ps else 0


def eq(f, g):
    return sub(f, g) == {}


def show(f, cap=6):
    if not f:
        return "0"
    parts = []
    items = sorted(f.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
    for (a, b, S), c in items[:cap]:
        th = "".join(("tm%d" % j) if j <= 4 else ("tp%d" % (j - 4)) for j in S)
        parts.append("%s*xm^%s*xp^%s%s" % (c, a, b, ("*" + th if th else "")))
    s = " + ".join(parts)
    if len(items) > cap:
        s += " + ...(%d more)" % (len(items) - cap)
    return s


def nterms(f):
    return len(f)


# --- the odd derivatives of each half ---------------------------------------


def Dm(i, f):
    """D_{theta^-_i} = d_i - theta^-_i d_{x^-},  i = 1..4."""
    return sub(d_odd(i, f), mul(mono(0, 0, (i,)), d_xm(f)))


def Dp(i, f):
    """D_{theta^+_i} = d_{i+4} - theta^+_i d_{x^+},  i = 1..4."""
    return sub(d_odd(i + 4, f), mul(mono(0, 0, (i + 4,)), d_xp(f)))


def cDm(a, f):
    j = 2 * a - 1
    return sub(Dm(j, f), smul(I, Dm(j + 1, f)))


def bDm(a, f):
    j = 2 * a - 1
    return add(Dm(j, f), smul(I, Dm(j + 1, f)))


def cDp(a, f):
    j = 2 * a - 1
    return sub(Dp(j, f), smul(I, Dp(j + 1, f)))


def bDp(a, f):
    j = 2 * a - 1
    return add(Dp(j, f), smul(I, Dp(j + 1, f)))


EPSU = {(1, 2): C(1), (2, 1): C(-1), (1, 1): C(), (2, 2): C()}


# --- embedding the one-half model of n4emb ----------------------------------


def emb_m(f):
    """A minus-half function of n4emb, {(a,S):c}, as an element of O(2|8)."""
    return {(a, 0, S): c for (a, S), c in f.items()}


def emb_p(f):
    """A plus-half function of n4emb as an element of O(2|8)."""
    return {(0, a, tuple(j + 4 for j in S)): c for (a, S), c in f.items()}


# ---------------------------------------------------------------------------
# (1) the contact field of a minus-half f as a derivation of O(2|8)
# ---------------------------------------------------------------------------


def Kcoeffs(f):
    """(A, B_1..B_4) with K_f = A d_{x^-} + sum_i B_i d_{theta^-_i}.

    The coefficients are read off K_f(x) and K_f(theta_i) in the ONE-half
    model of n4shift, i.e. off K_f(g) = {f,g} + 2 (d_t f) g.
    """
    A = emb_m(n4shift.Kf(f, n4emb.mono(1)))
    B = [emb_m(n4shift.Kf(f, n4emb.mono(0, (i,)))) for i in MI]
    return A, B


def Kf(f, g, coeffs=None):
    """The contact field of the minus-half generating function f, acting on
    the whole of O(2|8) as a superderivation."""
    A, B = coeffs if coeffs is not None else Kcoeffs(f)
    out = mul(A, d_xm(g))
    for n, i in enumerate(MI):
        out = add(out, mul(B[n], d_odd(i, g)))
    return out


def EMB(f):
    """A generating function, given in either model, as an element of O(2|8)."""
    for k in f:
        return f if len(k) == 3 else emb_m(f)
    return {}


def Lam(f, a, b):
    """Lam^a_b(f) = 1/2 cD^a bD_b f, in the minus half."""
    return smul(Q(1, 2), cDm(a, bDm(b, EMB(f))))


def LamM(f):
    """The gl(2)-block Lam(f) as a 2x2 matrix, rows a, columns b."""
    return [[Lam(f, 1, 1), Lam(f, 1, 2)], [Lam(f, 2, 1), Lam(f, 2, 2)]]


def LamBar(f, a, b):
    """bLam_a^b(f) = 1/2 bD_a cD^b f."""
    return smul(Q(1, 2), bDm(a, cDm(b, EMB(f))))


def Xi(f):
    return smul(Q(1, 2), cDm(1, cDm(2, EMB(f))))


def XiBar(f):
    return smul(Q(1, 2), bDm(1, bDm(2, EMB(f))))


def trLam(f):
    return add(Lam(f, 1, 1), Lam(f, 2, 2))


def mu_of(f):
    """The candidate scalar: mu(f) = tr Lam(f) + 2 d_t f."""
    return add(trLam(f), smul(2, d_xm(EMB(f))))


# ---------------------------------------------------------------------------
# matrices of superfields
# ---------------------------------------------------------------------------


def mzero():
    return [[{}, {}], [{}, {}]]


def mid(c=1):
    return [[mono(0, 0, (), c), {}], [{}, mono(0, 0, (), c)]]


def mscal(u):
    """The scalar matrix u . 1."""
    return [[u, {}], [{}, u]]


def madd(*Ms):
    return [[add(*[M[i][j] for M in Ms]) for j in (0, 1)] for i in (0, 1)]


def msub(P, R):
    return [[sub(P[i][j], R[i][j]) for j in (0, 1)] for i in (0, 1)]


def msmul(c, M):
    return [[smul(c, M[i][j]) for j in (0, 1)] for i in (0, 1)]


def mmul(P, R):
    return [[add(mul(P[i][0], R[0][j]), mul(P[i][1], R[1][j]))
             for j in (0, 1)] for i in (0, 1)]


def mtrans(M):
    return [[M[0][0], M[1][0]], [M[0][1], M[1][1]]]


def mtr(M):
    return add(M[0][0], M[1][1])


def mdet(M):
    """Only used on matrices with EVEN entries."""
    return sub(mul(M[0][0], M[1][1]), mul(M[0][1], M[1][0]))


def madj(M):
    """The adjugate; linear in M, and equal to eps M^T eps^{-1}."""
    return [[M[1][1], smul(-1, M[0][1])], [smul(-1, M[1][0]), M[0][0]]]


def mapply(op, M):
    return [[op(M[i][j]) for j in (0, 1)] for i in (0, 1)]


def meq(P, R):
    return all(eq(P[i][j], R[i][j]) for i in (0, 1) for j in (0, 1))


def body(f):
    """The (x-independent, theta-free) constant term."""
    return f.get((0, 0, ()), C())


def inv_scalar(u, ord_max=12):
    """1/u for u = c (1 + nilpotent), c a non-zero constant."""
    c = body(u)
    if not c:
        raise ValueError("no invertible body: %s" % show(u))
    n = smul(recip(c), sub(u, mono(0, 0, (), c)))    # u/c - 1, nilpotent
    out = mono(0, 0, (), 1)
    term = mono(0, 0, (), 1)
    for _ in range(ord_max):
        term = smul(-1, mul(term, n))
        if not term:
            break
        out = add(out, term)
    return smul(recip(c), out)


def minv(M):
    """The inverse of a 2x2 matrix of EVEN superfields with invertible body."""
    d = mdet(M)
    di = inv_scalar(d)
    return [[mul(di, x) for x in row] for row in madj(M)]


# ---------------------------------------------------------------------------
# (2) preliminaries: the derivation, the multipliers, the identity (*)
# ---------------------------------------------------------------------------


def test_fs(lo=-1, hi=2):
    """A supply of minus-half generating functions, from n4emb's model."""
    out = []
    for a in range(lo, hi + 1):
        for r in range(0, 5):
            for S in itertools.combinations(n4emb.IDX, r):
                out.append(n4emb.mono(a, S))
    return out


def two_half_monos(loa, hia, lob, hib, rmax=8):
    keys = []
    for a in range(loa, hia + 1):
        for b in range(lob, hib + 1):
            for r in range(0, rmax + 1):
                for S in itertools.combinations(MI + PL, r):
                    keys.append((a, b, S))
    return keys


def sec1():
    log("")
    log("=" * 74)
    log("(1)  the two-half algebra O(2|8) and the contact field K_f")
    log("=" * 74)
    log("    keys (a,b,S): a = deg x^-, b = deg x^+, S in {1..4} (minus half)")
    log("    and {5..8} (plus half).  K_f of a MINUS-half generating function f")
    log("    is the derivation A d_{x^-} + sum_i B_i d_{theta^-_i} whose")
    log("    coefficients are K_f(x) and K_f(theta_i) of n4shift.")
    log("")

    fs = test_fs(-1, 2)
    gs = test_fs(-2, 2)

    # (a) K_f agrees with {f,g} + 2 (d_t f) g on minus-half functions
    bad = 0
    for f in fs[:60]:
        co = Kcoeffs(f)
        for g in gs[:60]:
            want = emb_m(n4shift.Kf(f, g))
            got = Kf(f, emb_m(g), co)
            if not eq(got, want):
                bad += 1
    check(bad == 0,
          "K_f as a derivation of O(2|8) restricts to n4shift's K_f (%d pairs)"
          % (60 * 60))
    log("      K_f | O(1|4) = {f,.} + 2(d_t f).  on %d pairs: %s"
        % (60 * 60, "ok" if bad == 0 else "%d MISMATCHES" % bad))

    # (b) K_f is a superderivation of the whole of O(2|8)
    testel = [mono(1, 0, ()), mono(0, 1, ()), mono(-1, 2, (1, 5)),
              mono(0, 0, (2, 3, 7)), mono(2, -1, (5, 6)), mono(0, 0, (4, 8)),
              mono(1, 1, (1, 2, 3, 4)), mono(0, 0, (5, 6, 7, 8))]
    bad = 0
    ntest = 0
    for f in fs[:24]:
        co = Kcoeffs(f)
        pf = n4emb.parity(f)
        for g in testel:
            for h in testel:
                ntest += 1
                lhs = Kf(f, mul(g, h), co)
                pg = parity(g)
                rhs = add(mul(Kf(f, g, co), h),
                          smul((-1) ** (pf * pg), mul(g, Kf(f, h, co))))
                if not eq(lhs, rhs):
                    bad += 1
    check(bad == 0, "K_f is a superderivation of O(2|8) (%d Leibniz triples)"
          % ntest)
    log("      Leibniz on O(2|8): %d triples, %s" % (ntest, "ok" if not bad
                                                     else "%d FAIL" % bad))

    # (c) K_f commutes with every plus-half derivative
    bad = 0
    ntest = 0
    for f in fs[:24]:
        co = Kcoeffs(f)
        pf = n4emb.parity(f)
        for g in testel:
            for a in (1, 2):
                for op in (cDp, bDp):
                    ntest += 1
                    lhs = Kf(f, op(a, g), co)
                    rhs = smul((-1) ** pf, op(a, Kf(f, g, co)))
                    if not eq(lhs, rhs):
                        bad += 1
    check(bad == 0, "[K_f, cD_+^a] = [K_f, bD_{+a}] = 0 (%d checks)" % ntest)
    log("      [K_f, plus-half derivatives] = 0: %d checks, %s"
        % (ntest, "ok" if not bad else "%d FAIL" % bad))

    # (d) the multiplier in the minus half, and the identity (*)
    def commutator(f, a, g, co, pf):
        return sub(Kf(f, cDm(a, g), co),
                   smul((-1) ** pf, cDm(a, Kf(f, g, co))))

    def multipl(f, a, g):
        r = {}
        for b in (1, 2):
            r = add(r, mul(Lam(f, a, b), cDm(b, g)))
            if EPSU[(a, b)]:
                r = add(r, smul(EPSU[(a, b)], mul(Xi(f), bDm(b, g))))
        return r

    # the overall sign of the convention is DETERMINED, not chosen
    signs = set()
    for f in fs[:12]:
        co = Kcoeffs(f)
        pf = n4emb.parity(f)
        for a in (1, 2):
            for g in testel:
                lh = commutator(f, a, g, co, pf)
                rh = multipl(f, a, g)
                if not lh and not rh:
                    continue
                if eq(lh, rh):
                    signs.add(1)
                elif eq(lh, smul(-1, rh)):
                    signs.add(-1)
                else:
                    signs.add(0)
    check(signs in ({1}, {-1}),
          "the multiplier holds with one overall sign, determined here")
    SGN = list(signs)[0] if signs in ({1}, {-1}) else 1
    bad = 0
    ntest = 0
    for f in fs:
        co = Kcoeffs(f)
        pf = n4emb.parity(f)
        for a in (1, 2):
            for g in testel:
                ntest += 1
                if not eq(commutator(f, a, g, co, pf),
                          smul(SGN, multipl(f, a, g))):
                    bad += 1
    check(bad == 0,
          "[K_f, cD^a] = %s(Lam^a_b cD^b + Xi eps^{ab} bD_b) on O(2|8) (%d)"
          % ("" if SGN == 1 else "-", ntest))
    log("      the multiplier (71) on O(2|8): %d checks, %s (overall sign %+d,"
        % (ntest, "ok" if not bad else "%d FAIL" % bad, SGN))
    log("       determined by the engine, not chosen)")

    # (e) Lam + bLam^T = -2 d_t f
    bad = 0
    for f in fs:
        ff = emb_m(f)
        for a in (1, 2):
            for b in (1, 2):
                want = smul(-2, d_xm(ff)) if a == b else {}
                got = add(Lam(f, a, b), LamBar(f, b, a))
                if not eq(got, want):
                    bad += 1
    check(bad == 0, "Lam^a_b + bLam_b^a = -2 delta^a_b d_t f")
    log("      (*)  Lam^a_b + bLam_b^a = -2 delta^a_b d_t f : ok on %d f"
        % len(fs))
    log("           -- the N=4 form of lambda + bar lambda = -2 d_t f")
    return fs


# ---------------------------------------------------------------------------
# (3) the candidate mu and its chirality
# ---------------------------------------------------------------------------


def sec2(fs):
    log("")
    log("=" * 74)
    log("(2)  the candidate scalar  mu(f) = tr Lam(f) + 2 d_t f")
    log("=" * 74)

    bad = 0
    for f in fs:
        ff = emb_m(f)
        want = smul(I, add(Dm(1, Dm(2, ff)), Dm(3, Dm(4, ff))))
        if not eq(mu_of(f), want):
            bad += 1
    check(bad == 0, "mu = tr Lam + 2 d_t f = i (D_1D_2 + D_3D_4) f")
    log("      mu(f) = i (D_1 D_2 + D_3 D_4) f -- the u(1)-charge of the")
    log("      multiplier, tr Lam = -2 d_t f + i(D_1D_2+D_3D_4) f being item 1)")
    log("      of Proposition 6.2.3.")

    bad1 = bad2 = 0
    for f in fs:
        m = mu_of(f)
        if not eq(cDm(1, m), bDm(2, Xi(f))):
            bad1 += 1
        if not eq(cDm(2, m), smul(-1, bDm(1, Xi(f)))):
            bad2 += 1
    check(bad1 == 0, "cD^1 mu = bD_2 Xi")
    check(bad2 == 0, "cD^2 mu = -bD_1 Xi")
    log("")
    log("      cD^1 mu(f) =  bD_2 Xi(f),   cD^2 mu(f) = -bD_1 Xi(f)")
    log("      on all %d test functions.  Hence mu is cD-chiral EXACTLY on"
        % len(fs))
    log("      k^Xi = {Xi = 0} -- which is where the constraint lives at all,")
    log("      and item 3) of Proposition 6.2.3 demands cD^a mu = 0.  So the scalar the")
    log("      dynamics will ask for is admissible for the constraint, and it")
    log("      is admissible for no larger algebra.")


# ---------------------------------------------------------------------------
# (4) superfields satisfying the constraints
# ---------------------------------------------------------------------------


def half_solutions(lo, hi, which, nilpotent_only=True):
    """A basis of the EVEN doublets satisfying the one-half constraint.

    which = 'minus': cD^{(a} psi^{b)} = 0, the upper index, operators cD.
    which = 'plus' : bD_{(a} A_{g)} = 0,  the lower index, operators bD.

    Returned as pairs (u_1, u_2) of elements of O(2|8) supported on that half.
    """
    if which == "minus":
        ops = [lambda a, g: cDm(a, g)]
        odds = MI
        mk = lambda a, S: mono(a, 0, S)
        deg = lambda k: k[0]
    else:
        ops = [lambda a, g: bDp(a, g)]
        odds = PL
        mk = lambda a, S: mono(0, a, S)
        deg = lambda k: k[1]
    D = ops[0]

    # domain: even functions, degrees lo..hi
    keys = []
    for a in range(lo, hi + 1):
        for r in (0, 2, 4):
            for S in itertools.combinations(odds, r):
                keys.append((a, S))
    ncols = 2 * len(keys)

    # codomain: odd functions, degrees lo-1..hi
    okeys = []
    for a in range(lo - 1, hi + 1):
        for r in (1, 3):
            for S in itertools.combinations(odds, r):
                okeys.append((a, S))
    oidx = {k: j for j, k in enumerate(okeys)}
    nrow = 3 * len(okeys)

    M = [[C() for _ in range(ncols)] for _ in range(nrow)]

    def stamp(row_block, comp, f):
        for (a, b, S), c in f.items():
            k = (a if which == "minus" else b, S)
            M[row_block * len(okeys) + oidx[k]][comp] += c

    for j, k in enumerate(keys):
        g = mk(k[0], k[1])
        d1, d2 = D(1, g), D(2, g)
        # component (1,1): 2 D^1 u^1 ;  (1,2): D^1 u^2 + D^2 u^1 ; (2,2): 2 D^2 u^2
        stamp(0, 2 * j, smul(2, d1))                 # u^1 in (11)
        stamp(1, 2 * j, d2)                          # u^1 in (12)
        stamp(1, 2 * j + 1, d1)                      # u^2 in (12)
        stamp(2, 2 * j + 1, smul(2, d2))             # u^2 in (22)

    sols = []
    for vec in cnullspace(M):
        u1, u2 = {}, {}
        for j, k in enumerate(keys):
            if vec[2 * j]:
                u1 = add(u1, smul(vec[2 * j], mk(k[0], k[1])))
            if vec[2 * j + 1]:
                u2 = add(u2, smul(vec[2 * j + 1], mk(k[0], k[1])))
        if u1 or u2:
            if nilpotent_only and (body_free(u1) or body_free(u2)):
                continue
            sols.append((u1, u2))
    return sols


def body_free(f):
    """True if f has a theta-free term (which would spoil nilpotency)."""
    return any(S == () for (a, b, S) in f)


def sec3():
    log("")
    log("=" * 74)
    log("(3)  superfields q satisfying BOTH constraints of the first line")
    log("=" * 74)
    log("    the joint solution space is the tensor product of the two")
    log("    one-half solution spaces: q_g^b = sum_k A_g^{(k)}(+) psi^{(k)b}(-)")
    log("    solves both, the two constraint operators living on different")
    log("    halves and different indices.")
    log("")
    mins = half_solutions(-1, 2, "minus")
    plus = half_solutions(-1, 2, "plus")
    log("      minus-half solutions cD^{(a}psi^{b)}=0, even, theta-full,")
    log("      degrees -1..2:  %d basis elements" % len(mins))
    log("      plus-half solutions bD_{(a}A_{g)}=0, likewise: %d" % len(plus))
    check(len(mins) > 0 and len(plus) > 0, "both constraint systems solvable")

    # a couple of check that they really are solutions
    bad = 0
    for (u1, u2) in mins:
        if not eq(cDm(1, u1), {}) or not eq(cDm(2, u2), {}):
            bad += 1
        if not eq(add(cDm(1, u2), cDm(2, u1)), {}):
            bad += 1
    for (u1, u2) in plus:
        if not eq(bDp(1, u1), {}) or not eq(bDp(2, u2), {}):
            bad += 1
        if not eq(add(bDp(1, u2), bDp(2, u1)), {}):
            bad += 1
    check(bad == 0, "the computed bases do satisfy their constraints")
    log("      both bases verified against their constraints.")
    return mins, plus


def build_q(mins, plus, coeffs, q0=None):
    """q = q0 + sum_k c_k A^{(k)} (x) psi^{(k)}, a 2x2 matrix of O(2|8).

    coeffs is a list of (k_plus, k_minus, c).
    """
    q = [[mono(0, 0, (), 1), {}], [{}, mono(0, 0, (), 1)]] if q0 is None \
        else [[dict(q0[i][j]) for j in (0, 1)] for i in (0, 1)]
    for (kp, km, c) in coeffs:
        A = plus[kp % len(plus)]
        ps = mins[km % len(mins)]
        for g in (0, 1):
            for b in (0, 1):
                q[g][b] = add(q[g][b], smul(c, mul(A[g], ps[b])))
    return q


def check_constraints(q, tag=""):
    """Both lines of the constraint, for the matrix q."""
    bad = 0
    for g in (0, 1):                              # minus half, upper index
        u = (q[g][0], q[g][1])
        if not eq(cDm(1, u[0]), {}):
            bad += 1
        if not eq(cDm(2, u[1]), {}):
            bad += 1
        if not eq(add(cDm(1, u[1]), cDm(2, u[0])), {}):
            bad += 1
    for b in (0, 1):                              # plus half, lower index
        v = (q[0][b], q[1][b])
        if not eq(bDp(1, v[0]), {}):
            bad += 1
        if not eq(bDp(2, v[1]), {}):
            bad += 1
        if not eq(add(bDp(1, v[1]), bDp(2, v[0])), {}):
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# (5) the two dynamical equations and their variations
# ---------------------------------------------------------------------------


def current_minus(q, qi):
    """C_g = q^{-1} bD_{-g} q, a pair of 2x2 matrices (g = 1, 2)."""
    return [mmul(qi, mapply(lambda f: bDm(g, f), q)) for g in (1, 2)]


def lhs2(q, qi):
    """LHS of E2:  L_b^a = cD_+^a ( sum_g (q^{-1} bD_{-g} q)_b^g ).

    Returned as a 2x2 matrix, row b (minus-type), column a (plus-type).
    """
    Cs = current_minus(q, qi)
    c = [add(Cs[0][b][0], Cs[1][b][1]) for b in (0, 1)]   # c_b = (C_g)_b^g
    return [[cDp(a, c[b]) for a in (1, 2)] for b in (0, 1)]


EPSL = {(1, 1): C(), (1, 2): C(1), (2, 1): C(-1), (2, 2): C()}   # eps_{ab}
EPSH = {(1, 1): C(), (1, 2): C(1), (2, 1): C(-1), (2, 2): C()}   # eps^{ab}


def qbar(q):
    """bar q_b^a = - eps_{bg} eps^{ad} q_d^g, of (IK4q).

    q is stored as q[g-1][b-1], g the lower (plus-type) index, b the upper
    (minus-type) one; bar q is stored as bq[b-1][a-1], b minus-type lower,
    a plus-type upper -- the index layout of the LHS of E2.
    """
    out = mzero()
    for b in (1, 2):
        for a in (1, 2):
            v = {}
            for g in (1, 2):
                for d in (1, 2):
                    c = EPSL[(b, g)] * EPSH[(a, d)]
                    if c:
                        v = add(v, smul(-c, q[d - 1][g - 1]))
            out[b - 1][a - 1] = v
    return out


M4IM = C(0, -4)          # the constant -4 i m of (IK4), with m = 1


def rhs2(q):
    """RHS of E2: -4 i m (bar q)_b^a, rows b (minus), columns a (plus)."""
    return msmul(M4IM, qbar(q))


def cur1(q, qi):
    """j^b = sum_g (q cD_+^g q^{-1})_g^b, the current of the first equation.

    q has rows g (plus-type), columns b (minus-type); q^{-1} has rows b,
    columns g; so J^g := q (cD_+^g q^{-1}) has rows and columns of PLUS type,
    and the equation contracts its row index with the label g.
    """
    out = [{}, {}]
    for g in (1, 2):
        Jg = mmul(q, mapply(lambda f: cDp(g, f), qi))
        for b in (0, 1):
            out[b] = add(out[b], Jg[g - 1][b])
    return out


def lhs1(q, qi):
    """LHS of E1: cD_-^a j^b, rows a (minus), columns b (plus)."""
    j = cur1(q, qi)
    return [[cDm(a, j[b]) for b in (0, 1)] for a in (1, 2)]


EPSF = mono(0, 0, (9,))      # the odd constant multiplying an odd variation


def W(pf, x):
    """An odd f is applied through the odd constant eps, as always for an odd
    symmetry: eps.delta_f is an even variation, and only then is q + eps.delta q
    an even superfield again.  For even f this is the identity."""
    return mul(EPSF, x) if pf else x


def WM(pf, M):
    return mapply(lambda x: W(pf, x), M)


def var_q(f, q, co, B):
    """delta_f q = eps^{p(f)} ( K_f(q) + q B )."""
    return WM(n4emb.parity(f) if not is_two(f) else parity(f),
              madd(mapply(lambda g: Kf(f, g, co), q), mmul(q, B)))


def is_two(f):
    for k in f:
        return len(k) == 3
    return True


def Bmat(f, mu=None, sign=-1):
    """B = sign * Lam^T + mu . 1  (sign = -1 by Proposition 6.2.3)."""
    L = LamM(f)
    B = msmul(sign, mtrans(L))
    if mu is not None:
        B = madd(B, mscal(mu))
    return B


# --- exact linearizations (delta is a first-order variation of q) ------------


def var_qi(q, qi, dq):
    """delta(q^{-1}) = -q^{-1} (delta q) q^{-1}."""
    return msmul(-1, mmul(qi, mmul(dq, qi)))


def var_lhs2(q, qi, dq):
    """The variation of LHS of E2 induced by delta q = dq."""
    dqi = var_qi(q, qi, dq)
    dc = [{}, {}]
    for g in (1, 2):
        dC = madd(mmul(dqi, mapply(lambda f: bDm(g, f), q)),
                  mmul(qi, mapply(lambda f: bDm(g, f), dq)))
        for b in (0, 1):
            dc[b] = add(dc[b], dC[b][g - 1])
    return [[cDp(a, dc[b]) for a in (1, 2)] for b in (0, 1)]


def var_rhs2(dq):
    """bar q is LINEAR in q, so the variation is the same formula on dq."""
    return rhs2(dq)


def var_lhs1(q, qi, dq):
    """The variation of LHS of E1 induced by delta q = dq."""
    dqi = var_qi(q, qi, dq)
    dj = [{}, {}]
    for g in (1, 2):
        dJ = madd(mmul(dq, mapply(lambda f: cDp(g, f), qi)),
                  mmul(q, mapply(lambda f: cDp(g, f), dqi)))
        for b in (0, 1):
            dj[b] = add(dj[b], dJ[g - 1][b])
    return [[cDm(a, dj[b]) for b in (0, 1)] for a in (1, 2)]


# ---------------------------------------------------------------------------
# (4) the generating functions to test on
# ---------------------------------------------------------------------------


def acting_algebra(lo=-1, hi=2, kappa=0):
    """The generators (KAemb) of Proposition 6.1.2 at kappa = 0 -- the image
    of svect^L_0(1|2) inside k^L(1|4), which is the acting algebra of (IK4)."""
    return [(nm, g) for nm, g in n4shift.KAgens(Q(kappa), lo, hi)]


def stabilizer(lo=-1, hi=2):
    """A basis of k^Xi cap k^bXi = the f preserving BOTH isotropic planes,
    eq. (KAplus) of the text."""
    out = []
    for n, f in n4shift.kerXi(lo, hi, [n4shift.Xi, n4shift.XiBar]):
        out.append(("h_%d" % n, f))
    return out


def sample_qs(mins, plus, nq=3):
    """A few q's satisfying both constraints, with an invertible body."""
    out = []
    combos = [
        [(0, 0, C(1)), (1, 2, C(0, 1)), (2, 1, C(-1))],
        [(3, 1, C(1)), (0, 4, C(2)), (2, 3, C(0, -1)), (1, 0, C(1))],
        [(5, 2, C(1, 1)), (4, 5, C(-2)), (3, 3, C(1))],
        [(1, 1, C(1))],
    ]
    for c in combos[:nq]:
        q = build_q(mins, plus, c)
        d = mdet(q)
        if body(d):
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# (5) THE POINT: the equation fixes mu
# ---------------------------------------------------------------------------


def sec4(mins, plus):
    log("")
    log("=" * 74)
    log("(4)  the second dynamical equation of (IK4) fixes the scalar mu")
    log("=" * 74)
    log("    E2:   cD_+^a (q^{-1} bD_{-g} q)_b^g  =  -4 i m (bar q)_b^a")
    log("")
    log("    Both sides are matrices with row index b (minus-type) and column")
    log("    index a (plus-type).  Under the minus-half transformation")
    log("        delta_f q = K_f(q) + q B,     B = -Lam^T + mu . 1,")
    log("    each side must transform by K_f + one and the SAME matrix; that")
    log("    is what covariance means, and it is what fixes mu.")
    log("")

    qs = sample_qs(mins, plus)
    log("      %d test superfields q, all satisfying both constraints:" % len(qs))
    for i, q in enumerate(qs):
        bad = check_constraints(q)
        check(bad == 0, "q%d satisfies both constraints" % i)
        log("        q%d: %d terms, det body = %s, constraint violations %d"
            % (i, sum(nterms(q[g][b]) for g in (0, 1) for b in (0, 1)),
               body(mdet(q)), bad))

    fs = [(nm, f) for nm, f in acting_algebra(-1, 2) + stabilizer(-1, 2) if f]
    log("      %d generating functions f: the image of (KAemb) at kappa = 0"
        % len(fs))
    log("      together with a basis of k^Xi cap k^bXi = (KAplus).")

    # ---- (a) the variation of the current, in closed form -----------------
    log("")
    log("    (a) the current C_g := q^{-1} bD_{-g} q varies by")
    log("          delta C_g = K_f(C_g) + [C_g, B] - bLam_g^d C_d + bD_g B,")
    log("        an identity, not an ansatz: the four terms are the two halves")
    log("        of the Leibniz rule, the multiplier of bD_{-g}, and the")
    log("        derivative of the shift.")
    def cur_var(f, q, qi, co, B, g, sgn):
        """The claimed closed form of delta C_g."""
        pf = n4emb.parity(f)
        Bw = WM(pf, B)
        C = [mmul(qi, mapply(lambda x: bDm(h, x), q)) for h in (1, 2)]
        out = WM(pf, mapply(lambda x: Kf(f, x, co), C[g - 1]))
        out = madd(out, msub(mmul(C[g - 1], Bw), mmul(Bw, C[g - 1])))
        for d in (1, 2):
            out = madd(out, msmul(sgn, mmul(mscal(W(pf, LamBar(f, g, d))),
                                            C[d - 1])))
        return madd(out, mapply(lambda x: bDm(g, x), Bw))

    def cur_got(f, q, qi, co, B, g):
        dq = var_q(f, q, co, B)
        dqi = var_qi(q, qi, dq)
        return madd(mmul(dqi, mapply(lambda x: bDm(g, x), q)),
                    mmul(qi, mapply(lambda x: bDm(g, x), dq)))

    # the sign of the multiplier term is determined, not chosen
    f0 = fs[0][1]
    q0 = qs[0]
    qi0 = minv(q0)
    B0 = Bmat(f0, mu=mu_of(f0))
    SG = None
    for s in (-1, 1):
        if all(meq(cur_got(f0, q0, qi0, Kcoeffs(f0), B0, g),
                   cur_var(f0, q0, qi0, Kcoeffs(f0), B0, g, s))
               for g in (1, 2)):
            SG = s
    check(SG is not None, "the closed form of delta C_g holds with one sign")
    if SG is None:
        SG = -1
    bad = n = 0
    for (nm, f) in fs:
        co = Kcoeffs(f)
        B = Bmat(f, mu=mu_of(f))
        for q in qs:
            qi = minv(q)
            for g in (1, 2):
                n += 1
                if not meq(cur_got(f, q, qi, co, B, g),
                           cur_var(f, q, qi, co, B, g, SG)):
                    bad += 1
    check(bad == 0, "delta C_g = K_f C_g + [C_g,B] %s bLam_g^d C_d + bD_g B (%d)"
          % ("-" if SG < 0 else "+", n))
    log("        with the sign %+d before bLam (determined here, not chosen):"
        % SG)
    log("        %d checks, %s" % (n, "ok" if not bad else "%d FAIL" % bad))

    # ---- (b) the two transports -------------------------------------------
    ok_lhs = ok_rhs = ok_mu = 0
    nf = 0
    first = True
    for (nm, f) in fs:
        pf = n4emb.parity(f)
        co = Kcoeffs(f)
        T_l = madd(mtrans(LamM(f)), mscal(smul(2, d_xm(EMB(f)))))
        T_r = madd(mtrans(LamM(f)), mscal(smul(-1, trLam(f))))
        for q in qs:
            nf += 1
            qi = minv(q)
            L = lhs2(q, qi)
            R = rhs2(q)
            B0 = Bmat(f, mu=None)                      # the mu = 0 law
            dq0 = var_q(f, q, co, B0)
            X = msub(var_lhs2(q, qi, dq0),
                     WM(pf, mapply(lambda g: Kf(f, g, co), L)))
            Y = msub(var_rhs2(dq0),
                     WM(pf, mapply(lambda g: Kf(f, g, co), R)))
            if meq(X, mmul(WM(pf, T_l), L)):
                ok_lhs += 1
            # the RHS is invertible, so ITS transport is unique and is DIVIDED
            # out rather than guessed
            BR = mmul(Y, minv(R))
            if meq(BR, WM(pf, T_r)):
                ok_rhs += 1
            d = msub(WM(pf, T_l), BR)
            if eq(d[0][1], {}) and eq(d[1][0], {}) and eq(d[0][0], d[1][1]) \
                    and eq(d[0][0], W(pf, mu_of(f))):
                ok_mu += 1
            if first and pf == 0 and mu_of(f):
                first = False
                log("")
                log("    (b) a specimen, f = %s, q0:" % nm)
                log("        transport of the LHS (identity)  : Lam^T + 2 d_t f")
                log("        transport of the RHS (divided out): %s"
                    % show(mtr(BR), 3))
                log("        their difference is SCALAR:")
                log("          d_1^1 = %s" % show(d[0][0], 4))
                log("          d_2^2 = %s" % show(d[1][1], 4))
                log("          off-diagonal: %s , %s"
                    % (show(d[0][1], 2), show(d[1][0], 2)))
                log("          mu(f) = tr Lam + 2 d_t f = %s"
                    % show(mu_of(f), 4))
    check(ok_lhs == nf, "the LHS transports by Lam^T + 2 d_t f (%d/%d)"
          % (ok_lhs, nf))
    check(ok_rhs == nf, "the RHS transports by Lam^T - tr Lam (%d/%d)"
          % (ok_rhs, nf))
    check(ok_mu == nf,
          "their difference is scalar and equals tr Lam + 2 d_t f (%d/%d)"
          % (ok_mu, nf))
    log("")
    log("      over %d pairs (f, q), both parities:" % nf)
    log("        the LHS transports by  Lam^T + 2 d_t f      : %d" % ok_lhs)
    log("        the RHS transports by  Lam^T - tr Lam       : %d" % ok_rhs)
    log("        the RHS transport is obtained by DIVISION (the right-hand")
    log("        side -4im bar q being invertible), so it is solved for, not")
    log("        assumed; the difference of the two is scalar -- which is")
    log("        itself a non-trivial fact, the traceless parts having to")
    log("        agree -- and equals mu = tr Lam + 2 d_t f : %d" % ok_mu)
    log("")
    log("      Hence the two sides of E2 transform alike if and only if")
    log("            mu(f) = tr Lam(f) + 2 d_t f = i (D_1D_2 + D_3D_4) f.")
    return qs, fs


def sec5(qs, fs):
    log("")
    log("=" * 74)
    log("(5)  mu drops out of the LEFT-hand side")
    log("=" * 74)
    log("    the mu-part of the law changes the current q^{-1} bD_{-g} q by")
    log("    (bD_g mu) . 1 alone -- a minus-half function -- and cD_+^a kills")
    log("    it.  This is why the RHS alone can fix mu, and it is the N=4 form")
    log("    of cD^+ lambda = 0 in the proof of Proposition 5.1.4.")
    log("")
    tests = [n4emb.mono(1, (1, 2)), n4emb.mono(0, ()), n4emb.mono(-1, (3,)),
             n4emb.mono(2, (1, 2, 3, 4)), n4emb.mono(0, (1,))]
    bad = n = 0
    for q in qs:
        qi = minv(q)
        for mu in tests:
            pm = n4emb.parity(mu)
            m = W(pm, EMB(mu))
            dq = mmul(q, mscal(m))                    # delta_mu q = mu q
            n += 1
            if not meq(var_lhs2(q, qi, dq), mzero()):
                bad += 1
    check(bad == 0, "delta_mu (LHS of E2) = 0 for every minus-half mu (%d)" % n)
    log("      delta_mu(LHS) = 0 on %d pairs (q, mu), chiral or not: %s"
        % (n, "ok" if not bad else "%d FAIL" % bad))

    bad = n = 0
    for q in qs:
        qi = minv(q)
        for mu in tests:
            pm = n4emb.parity(mu)
            m = W(pm, EMB(mu))
            dq = mmul(q, mscal(m))
            dqi = var_qi(q, qi, dq)
            for g in (1, 2):
                n += 1
                dC = madd(mmul(dqi, mapply(lambda f: bDm(g, f), q)),
                          mmul(qi, mapply(lambda f: bDm(g, f), dq)))
                if not meq(dC, mscal(bDm(g, m))):
                    bad += 1
    check(bad == 0, "delta_mu (q^{-1} bD_g q) = (bD_g mu) . 1 (%d)" % n)
    log("      delta_mu(q^{-1} bD_{-g} q) = (bD_g mu).1 on %d checks: %s"
        % (n, "ok" if not bad else "%d FAIL" % bad))
    log("      -- so the whole mu-dependence of the left-hand side is a")
    log("      minus-half function under cD_+^a, i.e. nothing.")


def sec6(qs, fs):
    log("")
    log("=" * 74)
    log("(6)  the FIRST dynamical equation is covariant for every mu")
    log("=" * 74)
    log("    E1:  cD_-^a (q cD_+^g q^{-1})_g^b = 0.  The current q cD_+^g q^{-1}")
    log("    is unchanged by q -> q(1 + B) whenever cD_+ B = 0, which every B")
    log("    built from a minus-half f is.  So E1 says nothing about the shift,")
    log("    and it is E2, and E2 alone, that fixes mu.")
    log("")
    bad = n = 0
    for q in qs:
        qi = minv(q)
        for (nm, f) in fs[:8]:
            pf = n4emb.parity(f)
            co = Kcoeffs(f)
            for mu in (None, mu_of(f)):
                B = WM(pf, Bmat(f, mu=mu))
                dq_shift = mmul(q, B)                  # the shift part alone
                n += 1
                if not meq(var_lhs1(q, qi, dq_shift), mzero()):
                    bad += 1
    check(bad == 0, "the shift does not act on E1 at all (%d checks)" % n)
    log("      delta_B(LHS of E1) = 0 for every B, with and without mu:")
    log("      %d checks, %s" % (n, "ok" if not bad else "%d FAIL" % bad))

    def e1_var(q, qi, f, co, pf, s1, s2):
        E1 = lhs1(q, qi)
        out = msmul(s1, WM(pf, mapply(lambda g: Kf(f, g, co), E1)))
        for a in (1, 2):
            for b in (0, 1):
                for c in (1, 2):
                    out[a - 1][b] = add(out[a - 1][b],
                                        smul(s1 * s2,
                                             mul(W(pf, Lam(f, a, c)),
                                                 E1[c - 1][b])))
        return out

    # the two signs are determined per parity, not chosen: the pair must work
    # for EVERY f of that parity, degenerate cases (where all four pairs fit)
    # being intersected away
    SIG = {}
    for pf in (0, 1):
        cand = [(nm, f) for nm, f in fs if n4emb.parity(f) == pf][:12]
        fits = {(s1, s2) for s1 in (1, -1) for s2 in (1, -1)}
        for nm, f in cand:
            co = Kcoeffs(f)
            for q in qs[:2]:
                qi = minv(q)
                dq = var_q(f, q, co, Bmat(f, mu=mu_of(f)))
                got = var_lhs1(q, qi, dq)
                fits &= {(s1, s2) for (s1, s2) in fits
                         if meq(got, e1_var(q, qi, f, co, pf, s1, s2))}
        if len(fits) == 1:
            SIG[pf] = list(fits)[0]
    check(len(SIG) == 2, "the transport of E1 has one sign pair per parity")
    bad = n = 0
    for q in qs[:2]:
        qi = minv(q)
        for (nm, f) in fs[:12]:
            pf = n4emb.parity(f)
            if pf not in SIG:
                continue
            co = Kcoeffs(f)
            dq = var_q(f, q, co, Bmat(f, mu=mu_of(f)))
            n += 1
            s1, s2 = SIG[pf]
            if not meq(var_lhs1(q, qi, dq), e1_var(q, qi, f, co, pf, s1, s2)):
                bad += 1
    check(bad == 0,
          "E1 transforms into a multiple of itself, i.e. is covariant (%d)" % n)
    log("      the whole law carries E1^a_b to K_f(E1^a_b) - Lam^a_c E1^c_b --")
    log("      the multiplier of cD_-^a and nothing else, with no parity sign;")
    log("      the sign pair found for each parity: even f %s, odd f %s"
        % (SIG.get(0), SIG.get(1)))
    log("      %d checks, %s" % (n, "ok" if not bad else "%d FAIL" % bad))


def sec7(qs, fs):
    log("")
    log("=" * 74)
    log("(7)  the law with the mu found, and its uniqueness")
    log("=" * 74)
    log("    delta_f q_g^b = K_f(q_g^b) - Lam^b_d(f) q_g^d + mu(f) q_g^b,")
    log("               mu(f) = tr Lam(f) + 2 d_t f.")
    log("")
    ok = n = ok_c = 0
    for (nm, f) in fs:
        pf = n4emb.parity(f)
        co = Kcoeffs(f)
        B = Bmat(f, mu=mu_of(f))
        T = WM(pf, madd(mtrans(LamM(f)), mscal(smul(2, d_xm(EMB(f))))))
        for q in qs:
            qi = minv(q)
            L, R = lhs2(q, qi), rhs2(q)
            dq = var_q(f, q, co, B)
            n += 1
            vL = msub(var_lhs2(q, qi, dq),
                      WM(pf, mapply(lambda g: Kf(f, g, co), L)))
            vR = msub(var_rhs2(dq),
                      WM(pf, mapply(lambda g: Kf(f, g, co), R)))
            if meq(vL, mmul(T, L)) and meq(vR, mmul(T, R)):
                ok += 1
            if check_constraints(dq) == 0:
                ok_c += 1
    check(ok == n, "both sides of E2 transform by K_f + (Lam^T + 2 d_t f) (%d/%d)"
          % (ok, n))
    check(ok_c == n, "the full law preserves both constraints (%d/%d)"
          % (ok_c, n))
    log("      over %d pairs (f, q): both sides of E2 transform by" % n)
    log("        K_f + (Lam^T + 2 d_t f) .    -- %d" % ok)
    log("      and the variation itself satisfies both constraints -- %d"
        % ok_c)
    log("")
    log("    Uniqueness.  Suppose mu' = mu + nu also worked.  Both sides would")
    log("    then transport by matrices differing by nu . 1, and on a SOLUTION")
    log("    the two sides are equal and invertible, being -4im bar q; so")
    log("    nu . bar q = 0 forces nu = 0.  The engine checks the")
    log("    invertibility, and that a non-zero nu displaces the RHS by")
    log("    exactly nu . RHS while leaving the LHS where it was.")
    bad = n = 0
    nus = [n4emb.mono(0, ()), n4emb.mono(1, ()), n4emb.mono(0, (1, 2))]
    for (nm, f) in fs[:6]:
        pf = n4emb.parity(f)
        co = Kcoeffs(f)
        for q in qs:
            qi = minv(q)
            R = rhs2(q)
            check(bool(body(mdet(R))), "the RHS is invertible")
            L = lhs2(q, qi)
            T = WM(pf, madd(mtrans(LamM(f)),
                            mscal(smul(2, d_xm(EMB(f))))))
            for nu in nus:
                if n4emb.parity(nu) != pf:
                    continue
                nn = EMB(nu)
                B = Bmat(f, mu=add(mu_of(f), nn))
                dq = var_q(f, q, co, B)
                vR = msub(var_rhs2(dq),
                          WM(pf, mapply(lambda g: Kf(f, g, co), R)))
                vL = msub(var_lhs2(q, qi, dq),
                          WM(pf, mapply(lambda g: Kf(f, g, co), L)))
                n += 1
                if not (meq(vL, mmul(T, L)) and
                        meq(vR, madd(mmul(T, R),
                                     mmul(mscal(W(pf, nn)), R)))):
                    bad += 1
    check(bad == 0, "a non-zero nu breaks E2 by exactly nu . RHS (%d)" % n)
    log("      a wrong scalar nu displaces the RHS by nu . RHS and the LHS by")
    log("      nothing: %d checks, %s" % (n, "ok" if not bad else "%d FAIL" % bad))



def sec8():
    log("")
    log("=" * 74)
    log("(8)  where mu actually lives: the u(1)-charge, and nothing else")
    log("=" * 74)
    log("    mu = tr Lam + 2 d_t f is the u(1)-part of the multiplier.  On the")
    log("    algebra that ACTS -- the image of (KAemb) at kappa = 0 -- it")
    log("    vanishes identically; it is felt only on the one extra dimension")
    log("    C.h of (KAplus), h = theta_1theta_2 + theta_3theta_4.")
    log("")
    bad = 0
    gens = [(nm, f) for nm, f in acting_algebra(-3, 3) if f]
    for nm, f in gens:
        if mu_of(f):
            bad += 1
    check(bad == 0, "mu = 0 on all %d generators of (KAemb) at kappa = 0"
          % len(gens))
    log("      mu = 0 on all %d generators of (KAemb), modes -3..3: %s"
        % (len(gens), "ok" if not bad else "%d NON-ZERO" % bad))

    h = add(mono(0, 0, (1, 2)), mono(0, 0, (3, 4)))
    hm = sub(mono(0, 0, (1, 2)), mono(0, 0, (3, 4)))
    check(eq(mu_of(h), mono(0, 0, (), C(0, -2))), "mu(h) = -2i")
    check(eq(mu_of(hm), {}), "mu(theta_1theta_2 - theta_3theta_4) = 0")
    log("      mu(h) = %s,   mu(theta_1theta_2 - theta_3theta_4) = %s"
        % (show(mu_of(h)), show(mu_of(hm))))
    log("      -- the self-dual direction, which is the u(1)-current adjoined")
    log("      in (KAplus), against the anti-self-dual one, which is the")
    log("      sl(2) of item 4) of Proposition 6.1.2.")
    log("")
    log("    Consequence for the dilaton.  Write B = -Lam^T + mu . 1.  Then")
    log("        (1/2) tr B = d_t f + (1/2) mu,")
    log("    by (*) alone; so the trace part of q -- which is exp(-u) -- obeys")
    log("        delta_f u = K_f(u) - d_t f + (1/2) mu(f),")
    log("    the N=1 dilaton law of Theorem 4.2.1 plus the u(1)-charge,")
    log("    exactly as the real and imaginary parts split at N=2 in Statement")
    log("    Proposition 5.1.4.  On the acting algebra mu = 0 and the dilaton law is")
    log("    verbatim the one of N <= 2.")
    bad = 0
    fs = test_fs(-2, 2)
    for f in fs:
        B = Bmat(f, mu=mu_of(f))
        want = add(d_xm(EMB(f)), smul(Q(1, 2), mu_of(f)))
        if not eq(smul(Q(1, 2), mtr(B)), want):
            bad += 1
    check(bad == 0, "(1/2) tr B = d_t f + (1/2) mu, on %d f" % len(fs))
    log("")
    log("      (1/2) tr B = d_t f + (1/2) mu on all %d test functions: %s"
        % (len(fs), "ok" if not bad else "%d FAIL" % bad))
    log("      Since mu vanishes on the acting algebra, the whole 1-cocycle")
    log("      ambiguity of item 4) of Proposition 6.2.3 is resolved, and resolved to")
    log("      ZERO there: delta_f q = K_f(q) - q Lam(f)^T is the law, and by")
    log("      that same item 4) it is an honest action, not a projective one.")



def run():
    log("=" * 74)
    log("n4dyn.py -- the dynamical equation of (IK4) fixes the shift mu")
    log("=" * 74)
    fs = sec1()
    sec2(fs)
    mins, plus = sec3()
    qs, gens = sec4(mins, plus)
    sec5(qs, gens)
    sec6(qs, gens)
    sec7(qs, gens)
    sec8()
    log("")
    log("=" * 74)
    log("checks: %d,  failures: %d" % (CHECKS[0], FAIL[0]))
    log("=" * 74)
    with open("n4dyn-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    run()
