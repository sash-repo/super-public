#!/usr/bin/env python3
"""
n4stress.py -- step 3) of the N=4 programme: the STRESS TENSOR of (IK4) and
the value of the classical anomaly.

Steps 1) and 2) settled the transformation law of the field: by Statement
Proposition 6.2.3 the constraint forces the sl(2)-part of the shift and leaves a
scalar mu(f) free, and by Proposition 6.2.4 the dynamical equation fixes that
scalar to be the u(1)-charge of the multiplier -- which VANISHES on the
algebra that acts.  So the law is (deltaq),

    delta_f q_g^b = K_f(q_g^b) - Lam^b_d(f) q_g^d,                      (78)

an honest action of svect^L_0(1|2), and the anomaly, if any, has to come from
the stress tensor.  This engine builds it, and reads the number off.

WHAT IS COMPUTED.

 (1) THE WEIGHT.  For k^L(1|N) the contact field has divergence

        div K_f = (2 - N) d_t f,

     computed here from scratch for N = 0, 1, 2, 3, 4 in a general-N contact
     model.  Hence the residue pairing <g, rho> = Res(g rho) is invariant
     exactly for densities of weight 2 - N/2 -- which is 2, 3/2, 1 for
     N = 0, 1, 2, i.e. the weights of the T of Proposition 2.3.4, of the
     bold T of Theorem 4.2.1 and of the bold T of Theorem 5.3.1 -- and 0 for N = 4.
     So at N = 4 the coadjoint module of k^L(1|4) is the module of FUNCTIONS,
     and the dilaton, of weight 0, already sits in it.

 (2) THE OPERATOR OF THE RECIPE IS THE IDENTITY ON THE ALGEBRA THAT ACTS.
     The recipe of item 1) of Theorem 5.3.1 is

        bold T = Psi^{-1} ( D_1 ... D_N d_t^{2-N} ) Psi,

     and at N = 4 it asks for d_t^{-2}.  But on the image of (KAemb) at
     kappa = 0 -- and on the whole stabilizer k^Xi cap k^bXi of (KAplus),
     and at no other value of kappa, and not on k^L(1|4) --

        D_1 D_2 D_3 D_4 = d_t^2,

     because f = g(t) + Theta h(t) satisfies it precisely when h = g'', and
     the twist of (KAemb) at kappa = 0 is exactly Theta d_t^2 of the untwisted
     tower.  So the operator of the recipe is the identity there: the recipe
     does not degenerate, it TRIVIALIZES, and the stress tensor of (IK4) is
     the dilaton itself.

 (3) THE ANOMALY.  The field-independent part of the variation of a composite
     comes from its LINEAR part alone (a quadratic term varies into terms at
     least linear in the field).  The linear part of Psi^{-1} O Psi with
     Psi = exp(-u) is -O u, and O = 1 on the algebra that acts, so

        bold T = -u + (covariant terms),     delta_f bold T = K_f(bold T) + d_t f,

     by item 7) of Proposition 6.2.4 with mu = 0.  Verified here in the
     two-half model of n4dyn on explicit constrained superfields q:
     delta_f(log det q) - K_f(log det q) = tr B, whence

        delta_f u = K_f(u) - d_t f - mu(f)/2   for the B of (shiftq), and
        delta_f u = K_f(u) - d_t f + mu(f)/2   for the B of (deltaq).

     (The text's item 7) prints the second sign for the first B; corrected.)

 (4) THE COCYCLE.  With Res = the coefficient of t^{-1} theta_1..theta_4,

        c(f, g) := (-1)^{p(f)} Res( g d_t f ),   a(f) = d_t f,

     which is super-skew-symmetric and satisfies the cocycle identity (both
     verified), and which -- in a basis of the acting algebra normalized so
     that ALL the relations (svectrel) hold at lambda = 0 -- equals exactly
     the cocycle (svectcoc), which is the c = 6 point:

        c(L_m, L_-m) = 1/2 (m^3 - m),       c(J^+_m, J^-_-m) = m,
        c(J^0_m, J^0_-m) = 2m,              c(G^r_a, bG^{-r b}) = -delta (r^2 - 1/4).

     All four sectors return one and the same central charge, |c| = 6.

Usage:  python3 n4stress.py           (writes n4stress-out.txt)
"""

import itertools
import sys

from fractions import Fraction as Q

import n4emb
from n4emb import C, I, cx, mono, add, smul, mul, d_t, d_odd, kbr, show, eq, \
    parity, euler
import n4shift
from n4shift import KAgens, D, cD, bD, Xi, XiBar, kerXi, recip, cnullspace

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


VOL = (1, 2, 3, 4)


def Res(f):
    """The Berezin residue: the coefficient of t^{-1} theta_1 theta_2 theta_3 theta_4."""
    return f.get((-1, VOL), C())


# ===========================================================================
# (1) the weight: div K_f = (2 - N) d_t f, in a general-N contact model
# ===========================================================================
#
# Keys (a, S), a = deg t, S a sorted tuple of odd labels 1..N.  The formulas
# are those of n4emb, with the number of odd indeterminates a parameter.


def gmono(a, S=(), c=1, N=4):
    S = tuple(S)
    if len(set(S)) != len(S):
        return {}
    # sort with sign
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    c = cx(c) * sgn
    return {(a, tuple(S)): c} if c else {}


def gmul(f, g):
    out = {}
    for (a1, S1), c1 in f.items():
        for (a2, S2), c2 in g.items():
            m = gmono(a1 + a2, S1 + S2, c1 * c2)
            for k, v in m.items():
                out[k] = out.get(k, C()) + v
    return {k: v for k, v in out.items() if v}


def gd_t(f):
    out = {}
    for (a, S), c in f.items():
        if a:
            out[(a - 1, S)] = out.get((a - 1, S), C()) + c * a
    return {k: v for k, v in out.items() if v}


def gd_odd(j, f):
    out = {}
    for (a, S), c in f.items():
        if j in S:
            p = S.index(j)
            T = S[:p] + S[p + 1:]
            out[(a, T)] = out.get((a, T), C()) + c * ((-1) ** p)
    return {k: v for k, v in out.items() if v}


def geuler(f):
    return {k: v * len(k[1]) for k, v in f.items() if v and len(k[1])}


def gparity(f):
    ps = {len(S) % 2 for (a, S) in f}
    return ps.pop() if len(ps) == 1 else 0


def gkbr(f, g, N):
    """The contact bracket on C^{1|N}, the formula of n4emb.kbr."""
    if not f or not g:
        return {}
    twoE_f = add(smul(2, f), smul(-1, geuler(f)))
    twoE_g = add(smul(2, g), smul(-1, geuler(g)))
    out = add(gmul(twoE_f, gd_t(g)), smul(-1, gmul(gd_t(f), twoE_g)))
    s = C(-1) if gparity(f) == 0 else C(1)
    for i in range(1, N + 1):
        out = add(out, smul(s, gmul(gd_odd(i, f), gd_odd(i, g))))
    return out


def gKf(f, g, N):
    """K_f(g) = {f, g} + 2 (d_t f) g, the action on functions."""
    return add(gkbr(f, g, N), smul(2, gmul(gd_t(f), g)))


def gdivK(f, N):
    """div K_f, read off the coefficients K_f(t), K_f(theta_i)."""
    A = gKf(f, gmono(1, (), 1), N)
    out = gd_t(A)
    sgn = -1 if gparity(f) == 0 else 1
    for i in range(1, N + 1):
        B = gKf(f, gmono(0, (i,), 1), N)
        out = add(out, smul(sgn, gd_odd(i, B)))
    return out


def sec1():
    log("")
    log("=" * 74)
    log("(1)  THE WEIGHT of the coadjoint module of k^L(1|N)")
    log("=" * 74)
    log("    div K_f = (2 - N) d_t f, hence <g, rho> = Res(g rho) is invariant")
    log("    for densities of weight 2 - N/2:")
    log("")
    log("       N        div K_f        weight of bold T     where")
    for N in (0, 1, 2, 3, 4):
        tests = []
        for a in range(-3, 4):
            for r in range(0, N + 1):
                for S in itertools.combinations(range(1, N + 1), r):
                    m = gmono(a, S, 1)
                    if m:
                        tests.append(m)
        bad = 0
        for f in tests:
            if not eq(gdivK(f, N), smul(2 - N, gd_t(f))):
                bad += 1
        check(bad == 0, "div K_f = (2-N) d_t f for N = %d" % N)
        w = Q(4 - N, 2)
        where = {0: "Proposition 2.3.4", 1: "Theorem 4.2.1",
                 2: "Theorem 5.3.1", 3: "(no scalar equation)",
                 4: "HERE"}[N]
        log("       %d    (2-%d) d_t f  [%3d monomials, %d bad]      %-4s   %s"
            % (N, N, len(tests), bad, w, where))
    log("")
    log("    So at N = 4 the coadjoint module is the module of FUNCTIONS, and")
    log("    the dilaton u -- of weight 0 -- already lies in it.  This is the")
    log("    structural reason the recipe of item 1) of Theorem 5.3.1 has nothing")
    log("    left to do at N = 4.")
    log("")

    # the invariance of the pairing at N = 4, directly
    tests = []
    for a in range(-3, 4):
        for r in range(0, 5):
            for S in itertools.combinations(VOL, r):
                m = mono(a, S)
                if m:
                    tests.append(m)
    bad = 0
    n = 0
    for f in tests:
        for h in tests:
            n += 1
            if Res(n4shift.Kf(f, h)) != 2 * Res(mul(d_t(f), h)):
                bad += 1
    check(bad == 0, "Res(K_f h) = 2 Res((d_t f) h)")
    log("    the invariance of the residue pairing at N = 4, directly:")
    log("       Res(K_f(h)) = 2 Res((d_t f) h)   on %d pairs, %d bad" % (n, bad))


# ===========================================================================
# (2) D_1 D_2 D_3 D_4 = d_t^2 on the algebra that acts, and there only
# ===========================================================================


def D4(f):
    g = f
    for i in (4, 3, 2, 1):
        g = D(i, g)
    return g


def dt2(f):
    return d_t(d_t(f))


def kernel_D4(lo, hi):
    """A basis of {f | D_1D_2D_3D_4 f = d_t^2 f}, degree by degree."""
    out = []
    for n in range(lo, hi + 1):
        keys, idx = n4shift.basis_window(n, n)
        okeys, oidx = n4shift.basis_window(n - 4, n + 2)
        M = [[C() for _ in keys] for _ in range(len(okeys))]
        for j, k in enumerate(keys):
            f = mono(k[0], k[1])
            for kk, cc in add(D4(f), smul(-1, dt2(f))).items():
                M[oidx[kk]][j] += cc
        for vec in cnullspace(M):
            f = {}
            for j, k in enumerate(keys):
                if vec[j]:
                    f = add(f, mono(k[0], k[1], vec[j]))
            if f:
                out.append((n, f))
    return out


def sec2():
    log("")
    log("=" * 74)
    log("(2)  D_1 D_2 D_3 D_4 = d_t^2 -- on the acting algebra, and there only")
    log("=" * 74)
    log("    For f = g(t) + Theta h(t), Theta := theta_1 theta_2 theta_3 theta_4,")
    log("       D_1D_2D_3D_4 f = Theta g'''' + h,      d_t^2 f = g'' + Theta h'',")
    log("    so the two agree exactly when h = g''.  The twist of (KAemb) at")
    log("    kappa = 0 is precisely Theta d_t^2 of the untwisted tower:")
    log("       L_n = -1/2 (1 + Theta d_t^2)(t^{n+1}).")
    log("")
    for kappa in (Q(0), Q(1), Q(1, 2), Q(-1)):
        gens = KAgens(kappa, -4, 4)
        bad = [nm for nm, f in gens if not eq(D4(f), dt2(f))]
        log("      kappa = %-4s :  %2d of %d generators of (KAemb) fail"
            % (kappa, len(bad), len(gens)))
        if kappa == 0:
            check(not bad, "D4 = d_t^2 on (KAemb) at kappa = 0")
        else:
            check(bool(bad), "D4 != d_t^2 on (KAemb) at kappa = %s" % kappa)
    log("")
    st = kerXi(-3, 3, [Xi, XiBar])
    bad = [n for n, f in st if not eq(D4(f), dt2(f))]
    check(not bad, "D4 = d_t^2 on the whole stabilizer (KAplus)")
    log("      on the stabilizer k^Xi cap k^bXi of (KAplus): %d of %d fail"
        % (len(bad), len(st)))

    # is the solution space LARGER than the stabilizer?
    ker = kernel_D4(-3, 3)
    log("      dim {f | D4 f = d_t^2 f} on degrees -3..3 : %d" % len(ker))
    log("      dim (KAplus) = k^A(1|2) + C h  on the same : %d" % len(st))
    # per degree
    from collections import Counter
    ck = Counter(n for n, f in ker)
    cs = Counter(n for n, f in st)
    log("      per degree of t:")
    for n in range(-3, 4):
        log("         deg %2d :  kernel %d,  stabilizer %d" % (n, ck[n], cs[n]))
    check(all(ck[n] >= cs[n] for n in range(-3, 4)),
          "the stabilizer lies in the kernel")
    log("")
    log("    So the operator D_1D_2D_3D_4 d_t^{-2} of the recipe is the identity")
    log("    on the algebra that acts -- and the recipe therefore returns the")
    log("    dilaton itself.  It is NOT the identity on k^L(1|4):")
    tests = []
    for a in range(-2, 3):
        for r in range(0, 5):
            for S in itertools.combinations(VOL, r):
                m = mono(a, S)
                if m:
                    tests.append(m)
    bad = [f for f in tests if not eq(D4(f), dt2(f))]
    check(len(bad) > 0, "D4 != d_t^2 on k^L(1|4)")
    # the space {g + Theta g''} is stable under d_t, so no antiderivative is
    # ever formed: writing the operator of the recipe as d_t^{-2} D_1..D_4,
    # it sends d_t f to d_t^{-2} d_t^2 (d_t f) = d_t f for f in the algebra.
    gens0 = KAgens(Q(0), -4, 4)
    bad2 = [nm for nm, f in gens0 if not eq(D4(d_t(f)), dt2(d_t(f)))]
    check(not bad2, "D4 = d_t^2 on d_t f as well, for f in (KAemb)")
    log("")
    log("    The space is stable under d_t, so, the derivatives commuting with")
    log("    d_t, the operator of the recipe may be written d_t^{-2} D_1..D_4:")
    log("    it sends d_t f to d_t^{-2} d_t^2 (d_t f) = d_t f, and no anti-")
    log("    derivative is ever formed.  Checked on d_t f for all %d generators:"
        % len(gens0))
    log("       %d fail" % len(bad2))
    log("       %d of %d monomials of k^L(1|4) fail it, e.g. f = %s:"
        % (len(bad), len(tests), show(bad[0])))
    log("          D4 f - d_t^2 f = %s"
        % show(add(D4(bad[0]), smul(-1, dt2(bad[0])))))


# ===========================================================================
# (3) the anomaly of the dilaton, in the two-half model
# ===========================================================================


def sec3():
    log("")
    log("=" * 74)
    log("(3)  THE ANOMALY: the law of the dilaton u = -1/2 log det q")
    log("=" * 74)
    import n4dyn as ND
    mins = ND.half_solutions(-1, 2, "minus")
    plus = ND.half_solutions(-1, 2, "plus")
    qs = ND.sample_qs(mins, plus, 3)
    for i, q in enumerate(qs):
        check(ND.check_constraints(q) == 0, "q%d satisfies both constraints" % i)
    fs = [(nm, f) for nm, f in ND.acting_algebra(-1, 2) if f]
    hs = [(nm, f) for nm, f in ND.stabilizer(-1, 2) if f]
    log("      %d test superfields q (both constraints), %d generators of the"
        % (len(qs), len(fs)))
    log("      acting algebra (KAemb) at kappa = 0 and %d of the stabilizer"
        % len(hs))
    log("      (KAplus).  det q = exp(-2u) by (IK4q), tr(phi.sigma) = 0.")
    log("")

    def dlogdet(q, f, withmu):
        co = ND.Kcoeffs(f)
        pf = n4emb.parity(f) if not ND.is_two(f) else ND.parity(f)
        B = ND.Bmat(f, ND.mu_of(f) if withmu else None)
        dq = ND.WM(pf, ND.madd(ND.mapply(lambda g: ND.Kf(f, g, co), q),
                               ND.mmul(q, B)))
        Dt = ND.mdet(q)
        qi = ND.minv(q)
        dD = ND.mul(Dt, ND.mtr(ND.mmul(qi, dq)))
        return ND.mul(ND.inv_scalar(Dt), ND.sub(dD, ND.W(pf, ND.Kf(f, Dt, co)))), pf

    # tr Lam = mu - 2 d_t f
    bad = 0
    for nm, f in fs + hs:
        if not ND.eq(ND.trLam(f),
                     ND.sub(ND.mu_of(f), ND.smul(2, ND.d_xm(ND.EMB(f))))):
            bad += 1
    check(bad == 0, "tr Lam = mu - 2 d_t f")
    log("      tr Lam(f) = mu(f) - 2 d_t f :   %d of %d generators fail"
        % (bad, len(fs + hs)))

    for withmu, name, sign in ((True, "(shiftq): B = -Lam^T + mu.1", -1),
                               (False, "(deltaq): B = -Lam^T", +1)):
        bad = 0
        n = 0
        for nm, f in fs + hs:
            for q in qs:
                n += 1
                val, pf = dlogdet(q, f, withmu)
                du = ND.smul(Q(-1, 2), val)                   # u = -1/2 log det q
                tgt = ND.W(pf, ND.add(ND.smul(-1, ND.d_xm(ND.EMB(f))),
                                      ND.smul(Q(sign, 2), ND.mu_of(f))))
                if not ND.eq(du, tgt):
                    bad += 1
        check(bad == 0, "the dilaton law for %s" % name)
        log("      %-28s :  delta_f u = K_f(u) - d_t f %s mu/2"
            % (name, "-" if sign < 0 else "+"))
        log("         %d of %d pairs (f, q) fail" % (bad, n))

    nz = [nm for nm, f in fs if ND.mu_of(f)]
    check(not nz, "mu = 0 on the acting algebra")
    log("      generators of the acting algebra with mu != 0:  %s"
        % (", ".join(nz) if nz else "none"))
    nzh = [nm for nm, f in hs if ND.mu_of(f)]
    log("      generators of the stabilizer with mu != 0    :  %s"
        % (", ".join(nzh) if nzh else "none"))
    log("")
    log("    So on the algebra that acts,  delta_f u = K_f(u) - d_t f, and the")
    log("    stress tensor bold T = -u + (covariant) has anomaly a(f) = d_t f.")
    log("    A quadratic composite varies into terms at least linear in the")
    log("    field, so the field-INDEPENDENT part of the variation of bold T is")
    log("    that of its linear part alone -- which is what is computed here.")


# ===========================================================================
# (4) the cocycle and the value of the central charge
# ===========================================================================


def coc(f, g):
    """c(f, g) = (-1)^{p(f)} Res(g a(f)),  a(f) = d_t f the anomaly of bold T."""
    s = C(1) if parity(f) == 0 else C(-1)
    return s * Res(mul(g, d_t(f)))


def normalized_basis(lo, hi):
    """The image of (KAemb) at kappa = 0, normalized so that ALL the relations
    (svectrel) hold at lambda = 0.  Returns (L, Gh, Gb, J0, Jp, Jm)."""
    raw = dict(KAgens(Q(0), lo, hi))

    def L(n):
        return raw["L_%d" % n]

    def Gh(r, a):
        return smul(Q(1, 2), raw["G^%d_%d" % (int(r + Q(1, 2)), a)])

    def Gb(r, a):
        return smul(Q(1, 2), raw["bG^%d%d" % (int(r + Q(1, 2)), a)])

    def sJ(m, a, b, s=None):
        """(sigma.J)_a^b at mode m, from the mixed bracket with r + s = m."""
        s = Q(-1, 2) if s is None else s
        if Q(m) - s - s == 0:
            s = Q(-3, 2)
        r = Q(m) - s
        br = kbr(Gh(r, a), Gb(s, b))
        if a == b:
            br = add(br, L(m))
        return smul(recip(C(r - s)), br)

    J0 = lambda m: smul(2, sJ(m, 1, 1))
    Jp = lambda m: sJ(m, 2, 1)
    Jm = lambda m: sJ(m, 1, 2)
    return L, Gh, Gb, J0, Jp, Jm, sJ


def sec4():
    log("")
    log("=" * 74)
    log("(4)  THE COCYCLE and the central charge")
    log("=" * 74)
    L, Gh, Gb, J0, Jp, Jm, sJ = normalized_basis(-9, 9)

    # --- the basis really satisfies (svectrel) at lambda = 0 -----------------
    log("    (a) the basis satisfies ALL the relations (svectrel), lambda = 0:")
    bad = 0
    n = 0
    for m in range(-3, 4):
        for k in range(-3, 4):
            n += 1
            if not eq(kbr(L(m), L(k)), smul(m - k, L(m + k))):
                bad += 1
    check(bad == 0, "[L_m, L_n] = (m-n) L_{m+n}")
    log("        [L_m,L_n] = (m-n)L_{m+n}                 %d of %d fail" % (bad, n))

    bad = 0
    n = 0
    for m in range(-2, 3):
        for k in range(-2, 3):
            for Jf in (J0, Jp, Jm):
                n += 1
                if not eq(kbr(L(m), Jf(k)), smul(-k, Jf(m + k))):
                    bad += 1
    check(bad == 0, "[L_m, J^a_n] = -n J^a_{m+n}")
    log("        [L_m,J^a_n] = -n J^a_{m+n}               %d of %d fail" % (bad, n))

    bad = 0
    n = 0
    for m in range(-2, 3):
        for k in range(-2, 3):
            n += 3
            if not eq(kbr(Jp(m), Jm(k)), J0(m + k)):
                bad += 1
            if not eq(kbr(J0(m), Jp(k)), smul(2, Jp(m + k))):
                bad += 1
            if not eq(kbr(J0(m), Jm(k)), smul(-2, Jm(m + k))):
                bad += 1
    check(bad == 0, "the sl(2) of the currents")
    log("        [J^+,J^-]=J^0, [J^0,J^pm]=+-2J^pm        %d of %d fail" % (bad, n))

    bad = 0
    n = 0
    for m in range(-2, 3):
        for a in (1, 2):
            for b in (1, 2):
                vals = []
                for ss in (-5, -3, -1, 1, 3):
                    s = Q(ss, 2)
                    if Q(m) - s - s == 0:
                        continue
                    vals.append(sJ(m, a, b, s))
                n += 1
                if not all(eq(vals[0], v) for v in vals[1:]):
                    bad += 1
    check(bad == 0, "the currents do not depend on the mode split")
    log("        the currents are well defined (any split) %d of %d fail" % (bad, n))

    SJ = {(1, 1): ("0", Q(1, 2)), (2, 2): ("0", Q(-1, 2)),
          (1, 2): ("-", Q(1)), (2, 1): ("+", Q(1))}
    JF = {"0": J0, "+": Jp, "-": Jm}
    bad = 0
    n = 0
    for rr in (-5, -3, -1, 1, 3, 5):
        for ss in (-5, -3, -1, 1, 3, 5):
            r = Q(rr, 2)
            s = Q(ss, 2)
            for a in (1, 2):
                for b in (1, 2):
                    n += 1
                    lhs = kbr(Gh(r, a), Gb(s, b))
                    rhs = {}
                    if a == b:
                        rhs = add(rhs, smul(-1, L(int(r + s))))
                    nm, cf = SJ[(a, b)]
                    rhs = add(rhs, smul(C(r - s) * C(cf), JF[nm](int(r + s))))
                    if not eq(lhs, rhs):
                        bad += 1
    check(bad == 0, "{G^r_a, bG^{s b}} = -delta L + (r-s) (sigma.J)")
    log("        {G^r_a,bG^{sb}} = -d L + (r-s)(s.J)      %d of %d fail" % (bad, n))

    bad = 0
    n = 0
    for rr in (-5, -3, -1, 1, 3, 5):
        for ss in (-5, -3, -1, 1, 3, 5):
            for a in (1, 2):
                for b in (1, 2):
                    n += 2
                    if kbr(Gh(Q(rr, 2), a), Gh(Q(ss, 2), b)):
                        bad += 1
                    if kbr(Gb(Q(rr, 2), a), Gb(Q(ss, 2), b)):
                        bad += 1
    check(bad == 0, "both doublets abelian")
    log("        both doublets abelian                    %d of %d fail" % (bad, n))

    # --- c is a cocycle ------------------------------------------------------
    log("")
    log("    (b) c(f,g) = -(-1)^{p(f)} Res(g d_t f) is a 2-cocycle:")
    tests = []
    for a in range(-3, 4):
        for r in range(0, 5):
            for S in itertools.combinations(VOL, r):
                m = mono(a, S)
                if m:
                    tests.append(m)
    bad = 0
    n = 0
    for f in tests:
        for g in tests:
            n += 1
            sgn = C(-1) if (parity(f) == 1 and parity(g) == 1) else C(1)
            if coc(f, g) != -sgn * coc(g, f):
                bad += 1
    check(bad == 0, "super-skew-symmetry of c")
    log("        super-skew-symmetry                      %d of %d fail" % (bad, n))

    sub = tests[:40]
    bad = 0
    n = 0
    for f in sub:
        for g in sub:
            for h in sub:
                n += 1
                t1 = C((-1) ** (parity(f) * parity(h))) * coc(kbr(f, g), h)
                t2 = C((-1) ** (parity(g) * parity(f))) * coc(kbr(g, h), f)
                t3 = C((-1) ** (parity(h) * parity(g))) * coc(kbr(h, f), g)
                if t1 + t2 + t3 != 0:
                    bad += 1
    check(bad == 0, "the cocycle identity")
    log("        the cocycle identity                     %d of %d fail" % (bad, n))
    log("        (the sign (-1)^{p(f)} is the one for which both hold, as in")
    log("         the proofs of Theorem 4.2.1 and Theorem 5.3.1.)")

    # --- the four sectors ----------------------------------------------------
    log("")
    log("    (c) the value, sector by sector, against (svectcoc):")
    log("")
    log("        m    c(L_m,L_-m)    1/2(m^3-m)   c(J^+_m,J^-_-m)   m   "
        "c(J^0_m,J^0_-m)   2m")
    bad = 0
    for m in range(-4, 5):
        v1, t1 = coc(L(m), L(-m)), Q(1, 2) * (m ** 3 - m)
        v2, t2 = coc(Jp(m), Jm(-m)), Q(m)
        v3, t3 = coc(J0(m), J0(-m)), Q(2 * m)
        for v, t in ((v1, t1), (v2, t2), (v3, t3)):
            if v != C(t):
                bad += 1
        log("      %3d   %10s   %10s   %12s %6s   %12s %6s"
            % (m, v1, t1, v2, t2, v3, t3))
    check(bad == 0, "the three even sectors equal (svectcoc)")
    log("")
    log("        r      a b    c(G^r_a, bG^{-r b})    -delta_a^b (r^2 - 1/4)")
    for rr in (-5, -3, -1, 1, 3, 5):
        r = Q(rr, 2)
        for a in (1, 2):
            for b in (1, 2):
                v = coc(Gh(r, a), Gb(-r, b))
                t = -(r * r - Q(1, 4)) if a == b else Q(0)
                if v != C(t):
                    bad += 1
                    FAIL[0] += 1
                if a <= b:
                    log("      %5s    %d %d    %14s        %14s" % (r, a, b, v, t))
    CHECKS[0] += 1
    log("")
    # off-diagonal
    off = 0
    for m in range(-3, 4):
        for k in range(-3, 4):
            if m + k != 0:
                if coc(L(m), L(k)) or coc(Jp(m), Jm(k)) or coc(J0(m), J0(k)):
                    off += 1
    check(off == 0, "c vanishes off the diagonal")
    log("        c vanishes off m + n = 0 :  %d exceptions" % off)
    log("")
    log("    (d) THE VALUE.  In the normalization of items 4) of Statements")
    log("        Theorem 4.2.1 and Theorem 5.3.1,")
    log("           [L_m,L_n] = (m-n)L_{m+n} + (c/12)(m^3-m) delta,")
    log("           [J^0_m,J^0_n] = (c/3) m delta,")
    log("           [J^+_m,J^-_n] = J^0_{m+n} + (c/6) m delta,")
    log("           {G^r_a,bG^{sb}} = -delta L + (r-s)(sigma.J) - delta (c/6)(r^2-1/4),")
    log("        the four sectors give")
    log("           c/12 = 1/2,   c/3 = 2,   c/6 = 1,   c/6 = 1,")
    log("        i.e. c = 6 four times over:   |c| = 6.")


def run():
    log("n4stress.py -- the N=4 stress tensor and the value of the anomaly")
    log("(exact arithmetic over Q(i); no floating point anywhere)")
    sec1()
    sec2()
    sec3()
    sec4()
    log("")
    log("=" * 74)
    log("%d checks, %d failures" % (CHECKS[0], FAIL[0]))
    log("=" * 74)
    with open("n4stress-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    return FAIL[0]


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
