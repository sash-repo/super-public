"""
Verification of every claim in the Appendix of the text ("Checking cocycles in [KvdL]"), from scratch, in exact rational arithmetic.

Part I  (Subsec. A.1, cocycles on k^M(1|n), n=1,2,3,4):
   1. the grading declared at the beginning of the Appendix is additive for the
      bracket {,}_Rb printed there;
   2. {,}_Rb is super skew-symmetric and satisfies the super Jacobi identity;
   3. the eight-line table of non-zero values IS a 2-cocycle for {,}_Rb
      (identity (214)), and is super skew-symmetric;
   4. every intermediate bracket and every arithmetic step of the derivation
      in Subsec. A.1 ("Deducing the expression of the cocycles"), as an identity in the integer parameters k,m,n;
   5. (bonus) the class of c is non-trivial: c is not a coboundary.

Part II (Subsec. A.2, vect^L(1|2) and svect^L_lambda(1|2)):
   6. the two computations printed there, and the closing claim about the two
      last rows of (53), recomputed from (51) with lambda kept
      as a formal parameter.

Only stringy.py's Grassmann-Laurent arithmetic (mono/add/mul/d_t/d_odd) is
reused; the bracket, the cocycle and every check below are written afresh
against the printed text.
"""

from fractions import Fraction as Q
from itertools import combinations

from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, show

HALF = Q(1, 2)

# --------------------------------------------------------------- Part I: k^M

# odd labels: 0 = zeta, 1..n-1 = theta_i


def nabla(f):
    """nabla = 2 - E,  E = zeta d_zeta + sum theta_i d_theta_i."""
    return {(a, S): (2 - len(S)) * c for (a, S), c in f.items() if 2 - len(S)}


def Dcal(f):
    """D = d_t - zeta/(2t) d_zeta."""
    return add(d_t(f), smul(-HALF, mul(mono(-1, (0,)), d_odd(0, f))))


def br(n, f, g):
    """{f,g}_Rb = nabla(f) D(g) - D(f) nabla(g) + {f,g}_MPb ,
       {f,g}_MPb = (-1)^p(f) ( (1/t) d_z f d_z g + sum_i d_i f d_i g )."""
    res = add(mul(nabla(f), Dcal(g)), smul(-1, mul(Dcal(f), nabla(g))))
    s = mul(mono(-1), mul(d_odd(0, f), d_odd(0, g)))
    for i in range(1, n):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, smul((-1) ** parity(f), s))


def monkeys(n, arange):
    """all monomial keys (a, S) of R^M(n) with a in arange."""
    labels = tuple(range(n))            # 0 = zeta, 1..n-1 = theta
    out = []
    for a in arange:
        for r in range(n + 1):
            for S in combinations(labels, r):
                out.append((a, S))
    return out


def gr(n, key):
    """the grading declared at the beginning of the Appendix."""
    a, S = key
    if n == 1:
        # deg t = 1, deg zeta = 0;  gr = deg - 1
        return a - 1
    # deg t = 2, deg zeta = 0, deg theta = 1;  gr = deg - 2
    return 2 * a + len([j for j in S if j != 0]) - 2


# ---------------------------------------------------- the table of the cocycle

def cval(n, k1, k2):
    """The value of the cocycle of the Appendix on two monomials of R^M(n),
    given by their canonical keys.  Zero unless the odd parts coincide.

    Canonical order puts zeta (label 0) first; the printed table writes zeta
    last.  Both arguments carry the same odd monomial, so the two reordering
    signs cancel -- asserted below rather than assumed."""
    (a1, S1), (a2, S2) = k1, k2
    if S1 != S2:
        return Q(0)
    z = 0 in S1
    th = tuple(j for j in S1 if j != 0)
    r = len(th)
    # reordering sign, canonical (zeta first) -> printed (zeta last)
    sgn = (-1) ** r if z else 1
    assert sgn * sgn == 1
    if not z and r == 0:                       # c(t^{m+1}, t^{n+1})
        m, nn = a1 - 1, a2 - 1
        return Q(m * (m * m - 1)) if m + nn == 0 else Q(0)
    if z and r == 0:                           # c(t^{m+1} zeta, t^{n+1} zeta)
        m, nn = a1 - 1, a2 - 1
        return Q(m * m) - Q(1, 4) if m + nn == 0 else Q(0)
    if not z and r == 1:                       # c(t^{m+1} th_i, t^{n+1} th_i)
        m, nn = a1 - 1, a2 - 1
        return Q(m * (m + 1)) if m + nn + 1 == 0 else Q(0)
    if z and r == 1:                           # c(t^m th_i zeta, t^n th_i zeta)
        m, nn = a1, a2
        return -(Q(m) - HALF) if m + nn - 1 == 0 else Q(0)
    if not z and r == 2:                       # c(t^m th_i th_j, ...)
        m, nn = a1, a2
        return Q(-m) if m + nn == 0 else Q(0)
    if z and r == 2:                           # c(t^m th_i th_j zeta, ...)
        m, nn = a1, a2
        return Q(-1) if m + nn == 0 else Q(0)
    if not z and r == 3:                       # c(t^m th1 th2 th3, ...)
        m, nn = a1, a2
        return Q(-1) if m + nn + 1 == 0 else Q(0)
    if z and r == 3:                           # c(t^{m-1} th1 th2 th3 zeta,...)
        m, nn = a1 + 1, a2 + 1
        return Q(1) / (Q(m) - HALF) if m + nn - 1 == 0 else Q(0)
    return Q(0)


def cbil(n, f, g):
    """the cocycle, extended bilinearly to arbitrary elements."""
    tot = Q(0)
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            tot += c1 * c2 * cval(n, k1, k2)
    return tot


# ---------------------------------------------------------------- Part I runs

def check_grading(n, arange=range(-3, 4)):
    bad = []
    for k1 in monkeys(n, arange):
        for k2 in monkeys(n, arange):
            b = br(n, mono(*k1), mono(*k2))
            for k3 in b:
                if gr(n, k3) != gr(n, k1) + gr(n, k2):
                    bad.append((k1, k2, k3))
    return bad


def check_skew_bracket(n, arange=range(-3, 4)):
    bad = []
    for k1 in monkeys(n, arange):
        for k2 in monkeys(n, arange):
            f, g = mono(*k1), mono(*k2)
            lhs = br(n, f, g)
            rhs = smul(-(-1) ** (parity(f) * parity(g)), br(n, g, f))
            if not eq(lhs, rhs):
                bad.append((k1, k2))
    return bad


def check_jacobi(n, arange=range(-2, 3)):
    bad = []
    for k1 in monkeys(n, arange):
        f = mono(*k1)
        for k2 in monkeys(n, arange):
            g = mono(*k2)
            s = (-1) ** (parity(f) * parity(g))
            for k3 in monkeys(n, arange):
                h = mono(*k3)
                lhs = br(n, f, br(n, g, h))
                rhs = add(br(n, br(n, f, g), h), smul(s, br(n, g, br(n, f, h))))
                if not eq(lhs, rhs):
                    bad.append((k1, k2, k3))
    return bad


def check_cocycle(n, arange=range(-3, 4)):
    """identity (214): c(f,{g,h}) = c({f,g},h) + (-1)^{p(f)p(g)} c(g,{f,h})"""
    bad = []
    for k1 in monkeys(n, arange):
        f = mono(*k1)
        for k2 in monkeys(n, arange):
            g = mono(*k2)
            s = (-1) ** (parity(f) * parity(g))
            for k3 in monkeys(n, arange):
                h = mono(*k3)
                lhs = cbil(n, f, br(n, g, h))
                rhs = cbil(n, br(n, f, g), h) + s * cbil(n, g, br(n, f, h))
                if lhs != rhs:
                    bad.append((k1, k2, k3, lhs, rhs))
    return bad


def check_skew_cocycle(n, arange=range(-4, 5)):
    """c(f,g) = -(-1)^{p(f)p(g)} c(g,f)"""
    bad = []
    for k1 in monkeys(n, arange):
        f = mono(*k1)
        for k2 in monkeys(n, arange):
            g = mono(*k2)
            s = -(-1) ** (parity(f) * parity(g))
            if cval(n, k1, k2) != s * cval(n, k2, k1):
                bad.append((k1, k2))
    return bad


def check_nontrivial(n, arange=range(-3, 4)):
    """Is c a coboundary, i.e. c(f,g) = phi({f,g}) for a linear phi?
    c has weight 0, so only the weight-0 part of phi matters.  Solve the
    linear system over the window; report whether it is solvable."""
    import sys
    import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
    from exactlin import solve
    keys = monkeys(n, arange)
    zero = [k for k in keys if gr(n, k) == 0]          # unknowns phi(k)
    idx = {k: i for i, k in enumerate(zero)}
    rows, rhs = [], []
    for k1 in keys:
        for k2 in keys:
            v = cval(n, k1, k2)
            b = br(n, mono(*k1), mono(*k2))
            row = [Q(0)] * len(zero)
            ok = True
            for k, cf in b.items():
                if gr(n, k) != 0:
                    continue
                if k not in idx:            # outside the window: skip this pair
                    ok = False
                    break
                row[idx[k]] += cf
            if not ok:
                continue
            if any(row) or v:
                rows.append(row)
                rhs.append(v)
    return solve(rows, rhs) is not None   # solvable => coboundary on the window


# ------------------------------------------------- Part I: derivation, step by step

def T(a, S=(), c=1):
    return mono(a, S, c)


def derivation_steps(rng=range(-4, 5)):
    """Every explicit bracket printed in Subsec. A.1 ("Deducing the expression of the cocycles"), as an identity in the
    integer parameters.  Returns a list (label, #checked, #failed, examples)."""
    out = []

    def rec(label, n, lhs, rhs, params):
        bad = [p for p in params if not eq(lhs(*p), rhs(*p))]
        out.append((label, len(params), len(bad), bad[:3]))

    P1 = [(m,) for m in rng]
    P2 = [(m, nn) for m in rng for nn in rng]
    P3 = [(k, m, nn) for k in rng for m in rng for nn in rng]

    # --- block 1: c(t^{m+1} zeta, t^{n+1} zeta)
    rec("{t^{m+1}z, t^{n+1}z} = -t^{m+n+1}", 1,
        lambda m, nn: br(1, T(m + 1, (0,)), T(nn + 1, (0,))),
        lambda m, nn: T(m + nn + 1, (), -1), P2)
    rec("{t^{k+1}, t^{m+1}z} = (2m-k) t^{k+m+1} z", 1,
        lambda k, m: br(1, T(k + 1), T(m + 1, (0,))),
        lambda k, m: T(k + m + 1, (0,), 2 * m - k), P2)

    # --- block 2: c(t^{m+1} theta, t^{n+1} theta)   (n=2: theta = theta_1)
    rec("{t^{m+1}th, t^{n+1}th} = -t^{m+n+2}", 2,
        lambda m, nn: br(2, T(m + 1, (1,)), T(nn + 1, (1,))),
        lambda m, nn: T(m + nn + 2, (), -1), P2)
    rec("{t^{k+1}, t^{m+1}th} = (2m-k+1) t^{k+m+1} th", 2,
        lambda k, m: br(2, T(k + 1), T(m + 1, (1,))),
        lambda k, m: T(k + m + 1, (1,), 2 * m - k + 1), P2)

    # --- block 3: c(t^m theta zeta, t^n theta zeta)
    # printed monomials are  t^m theta zeta = - t^m zeta theta  (canonical)
    rec("{th, t^{n+1}z} = (n+1/2) t^n th z", 2,
        lambda nn: br(2, T(0, (1,)), T(nn + 1, (0,))),
        lambda nn: T(nn, (1, 0), Q(2 * nn + 1, 2)), P1)
    rec("{t^m th z, th} = t^m z", 2,
        lambda m: br(2, T(m, (1, 0)), T(0, (1,))),
        lambda m: T(m, (0,)), P1)
    rec("{t^m th z, t^{n+1} z} = -t^{m+n} th", 2,
        lambda m, nn: br(2, T(m, (1, 0)), T(nn + 1, (0,))),
        lambda m, nn: T(m + nn, (1,), -1), P2)

    # --- block 4: c(t^m th_i th_j, t^n th_i th_j)     (i=1, j=2; n=3)
    rec("{th_i, t^{n+1} th_j} = (n+1) t^n th_i th_j", 3,
        lambda nn: br(3, T(0, (1,)), T(nn + 1, (2,))),
        lambda nn: T(nn, (1, 2), nn + 1), P1)
    rec("{t^m th_i th_j, th_i} = t^m th_j", 3,
        lambda m: br(3, T(m, (1, 2)), T(0, (1,))),
        lambda m: T(m, (2,)), P1)
    rec("{t^m th_i th_j, t^{n+1} th_j} = -t^{m+n+1} th_i", 3,
        lambda m, nn: br(3, T(m, (1, 2)), T(nn + 1, (2,))),
        lambda m, nn: T(m + nn + 1, (1,), -1), P2)

    # --- block 5: c(t^m th_i th_j zeta, t^n th_i th_j zeta)
    # printed t^m th_i th_j zeta = t^m zeta th_i th_j (canonical, sign +1)
    rec("{t^{n+1} th_i th_j, z} = -(n+1) t^n th_i th_j z", 3,
        lambda nn: br(3, T(nn + 1, (1, 2)), T(0, (0,))),
        lambda nn: T(nn, (0, 1, 2), -(nn + 1)), P1)
    rec("{t^m th_i th_j z, t^{n+1} th_i th_j} = 0", 3,
        lambda m, nn: br(3, T(m, (0, 1, 2)), T(nn + 1, (1, 2))),
        lambda m, nn: {}, P2)
    rec("{t^m th_i th_j z, z} = -t^{m-1} th_i th_j", 3,
        lambda m: br(3, T(m, (0, 1, 2)), T(0, (0,))),
        lambda m: T(m - 1, (1, 2), -1), P1)

    # --- block 6: c(t^m th1 th2 th3, t^n th1 th2 th3)   (n=4)
    rec("{t^{n+1} th1 th2, th3} = -(n+1) t^n th1 th2 th3", 4,
        lambda nn: br(4, T(nn + 1, (1, 2)), T(0, (3,))),
        lambda nn: T(nn, (1, 2, 3), -(nn + 1)), P1)
    rec("{t^m th1 th2 th3, t^{n+1} th1 th2} = 0", 4,
        lambda m, nn: br(4, T(m, (1, 2, 3)), T(nn + 1, (1, 2))),
        lambda m, nn: {}, P2)
    rec("{t^m th1 th2 th3, th3} = -t^m th1 th2", 4,
        lambda m: br(4, T(m, (1, 2, 3)), T(0, (3,))),
        lambda m: T(m, (1, 2), -1), P1)

    # --- block 7: c(t^{m-1} th1 th2 th3 zeta, ...)
    # printed t^{m-1} th1 th2 th3 zeta = - t^{m-1} zeta th1 th2 th3 (canonical)
    rec("{t^n th1 th2 th3, z} = -(n-1/2) t^{n-1} th1 th2 th3 z", 4,
        lambda nn: br(4, T(nn, (1, 2, 3)), T(0, (0,))),
        lambda nn: T(nn - 1, (0, 1, 2, 3), -Q(2 * nn - 1, 2) * (-1)), P1)
    rec("{t^{m-1} th1 th2 th3 z, t^n th1 th2 th3} = 0", 4,
        lambda m, nn: br(4, T(m - 1, (0, 1, 2, 3), -1), T(nn, (1, 2, 3))),
        lambda m, nn: {}, P2)
    rec("{t^{m-1} th1 th2 th3 z, z} = -t^{m-2} th1 th2 th3", 4,
        lambda m: br(4, T(m - 1, (0, 1, 2, 3), -1), T(0, (0,))),
        lambda m: T(m - 2, (1, 2, 3), -1), P1)

    return out


def derivation_arithmetic(rng=range(-5, 6)):
    """The '=>' steps of Subsec. dedCoc: each derivation solves an equation
    for the unknown value.  Check the equation and its solution."""
    msgs = []

    # (1) -k(k^2-1) d_{k+2m,0} = 2(2m-k) c(t^{m+1}z, t^{k+m+1}z)
    bad = [(k, m) for k in rng for m in rng
           if -k * (k * k - 1) * (1 if k + 2 * m == 0 else 0)
           != 2 * (2 * m - k) * cval(1, (m + 1, (0,)), (k + m + 1, (0,)))]
    msgs.append(("equation for c(t^{m+1}z,.) holds", not bad, bad[:3]))
    # solution step: k = -2m, 2m(4m^2-1) = 8m c  =>  c = m^2 - 1/4  (needs m != 0)
    bad = [m for m in rng if m and Q(2 * m * (4 * m * m - 1), 8 * m) != Q(m * m) - Q(1, 4)]
    msgs.append(("=> c = m^2-1/4 (m!=0)", not bad, bad[:3]))
    msgs.append(("NB: m=0 not determined by this step; value -1/4 checked by (214)",
                 True, []))

    # (2) -k(k^2-1) d_{k+2m+1,0} = 2(2m-k+1) c(t^{m+1}th, t^{k+m+1}th)
    bad = [(k, m) for k in rng for m in rng
           if -k * (k * k - 1) * (1 if k + 2 * m + 1 == 0 else 0)
           != 2 * (2 * m - k + 1) * cval(2, (m + 1, (1,)), (k + m + 1, (1,)))]
    msgs.append(("equation for c(t^{m+1}th,.) holds", not bad, bad[:3]))
    # k = -2m-1: (2m+1)(4m^2+4m) = 2(4m+2) c  => c = m(m+1)
    bad = [m for m in rng
           if (2 * m + 1) * (4 * m * m + 4 * m) != 2 * (4 * m + 2) * m * (m + 1)]
    msgs.append(("=> c = m(m+1), all m (2m+1 never 0)", not bad, bad[:3]))
    # and the printed LHS -k(k^2-1) at k=-2m-1 really is (2m+1)(4m^2+4m)
    bad = [m for m in rng
           if -(-2 * m - 1) * ((2 * m + 1) ** 2 - 1) != (2 * m + 1) * (4 * m * m + 4 * m)]
    msgs.append(("-k(k^2-1)|_{k=-2m-1} = (2m+1)(4m^2+4m)", not bad, bad[:3]))

    # (3) (n+1/2) c(t^m th z, t^n th z) = (m-3/2)(m-1/2) d_{m+n-1,0}
    bad = []
    for m in rng:
        for nn in rng:
            lhs = (Q(nn) + HALF) * cval(2, (m, (1, 0)), (nn, (1, 0)))
            rhs = (Q(m) - Q(3, 2)) * (Q(m) - HALF) * (1 if m + nn - 1 == 0 else 0)
            if lhs != rhs:
                bad.append((m, nn))
    msgs.append(("equation for c(t^m th z,.) holds", not bad, bad[:3]))
    # printed intermediate: c(t^m z, t^{n+1} z) = ((m-1)^2-1/4) d = (m-3/2)(m-1/2) d
    bad = [m for m in rng if (Q(m) - 1) ** 2 - Q(1, 4) != (Q(m) - Q(3, 2)) * (Q(m) - HALF)]
    msgs.append(("((m-1)^2-1/4) = (m-3/2)(m-1/2)", not bad, bad[:3]))
    # solution: on m+n-1=0, n+1/2 = -m+3/2 and c = -(m-1/2)
    bad = [m for m in rng if Q(-m) + Q(3, 2) != Q(1 - m) + HALF]
    msgs.append(("n+1/2 = -m+3/2 on m+n=1", not bad, bad[:3]))
    bad = [m for m in rng
           if (Q(-m) + Q(3, 2)) * (-(Q(m) - HALF)) != (Q(m) - Q(3, 2)) * (Q(m) - HALF)]
    msgs.append(("=> c = -(m-1/2)", not bad, bad[:3]))

    # (4) (n+1) c(t^m th_i th_j, t^n th_i th_j) = m(m-1) d_{m+n,0}
    bad = []
    for m in rng:
        for nn in rng:
            lhs = (nn + 1) * cval(3, (m, (1, 2)), (nn, (1, 2)))
            rhs = m * (m - 1) * (1 if m + nn == 0 else 0)
            if lhs != rhs:
                bad.append((m, nn))
    msgs.append(("equation for c(t^m th_i th_j,.) holds", not bad, bad[:3]))
    bad = [m for m in rng if m != 1 and Q(m * (m - 1), 1 - m) != -m]
    msgs.append(("=> c = -m  (m != 1)", not bad, bad[:3]))
    msgs.append(("NB: m=1 not determined by this step; value -1 checked by (214)",
                 True, []))

    # (5) -(n+1) c(t^m th_i th_j z, ...) = (n+1) d_{m+n,0}
    bad = []
    for m in rng:
        for nn in rng:
            lhs = -(nn + 1) * cval(3, (m, (0, 1, 2)), (nn, (0, 1, 2)))
            rhs = (nn + 1) * (1 if m + nn == 0 else 0)
            if lhs != rhs:
                bad.append((m, nn))
    msgs.append(("equation for c(t^m th_i th_j z,.) holds", not bad, bad[:3]))
    msgs.append(("NB: n=-1 not determined by this step; value -1 checked by (214)",
                 True, []))

    # (6) -(n+1) c(t^m th123, t^n th123) = (n+1) d_{m+n+1,0}
    bad = []
    for m in rng:
        for nn in rng:
            lhs = -(nn + 1) * cval(4, (m, (1, 2, 3)), (nn, (1, 2, 3)))
            rhs = (nn + 1) * (1 if m + nn + 1 == 0 else 0)
            if lhs != rhs:
                bad.append((m, nn))
    msgs.append(("equation for c(t^m th1th2th3,.) holds", not bad, bad[:3]))

    # (7) -(n-1/2) C = d_{m+n-1,0}
    bad = []
    for m in rng:
        for nn in rng:
            C = cval(4, (m - 1, (0, 1, 2, 3)), (nn - 1, (0, 1, 2, 3)))
            lhs = -(Q(nn) - HALF) * C
            rhs = Q(1 if m + nn - 1 == 0 else 0)
            if lhs != rhs:
                bad.append((m, nn))
    msgs.append(("equation for C holds", not bad, bad[:3]))
    bad = [m for m in rng if -(Q(1 - m) - HALF) != Q(m) - HALF]
    msgs.append(("-n+1/2 = m-1/2 on m+n=1", not bad, bad[:3]))
    return msgs


# ------------------------------------------------- Part II: vect^L(1|2) etc.

# Basis elements are written as dicts { name : coefficient }, where name runs
# over the monomial vector fields  t^a * (odd monomial) * d_(t|theta_1|theta_2).
# The cocycle (51) is given on those, extended bilinearly.  lambda is
# kept as a formal parameter: coefficients are polynomials in lam, encoded as
# tuples (c0, c1, c2) meaning c0 + c1 lam + c2 lam^2.

def pol(*cs):
    c = list(cs) + [Q(0)] * (3 - len(cs))
    return tuple(Q(x) for x in c)


def padd(p, q):
    return tuple(a + b for a, b in zip(p, q))


def pmul(p, q):
    out = [Q(0)] * 3
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            if a and b:
                assert i + j < 3
                out[i + j] += a * b
    return tuple(out)


def pscal(c, p):
    return tuple(Q(c) * x for x in p)


PZERO = pol(0)


def c130(x, y):
    """(51), on monomial vector fields
       x = (a, S, d) meaning t^a * (odd monomial S) * d_d, d in {1,2} (theta),
    extended by super skew-symmetry; everything else vanishes."""
    tab = {
        (((1,), 1), ((2,), 2)): +1,     # c(t^k th1 d1, t^l th2 d2) = k
        (((1,), 2), ((2,), 1)): -1,     # c(t^k th1 d2, t^l th2 d1) = -k
        (((1, 2), 1), ((), 2)): -1,     # c(t^k th1th2 d1, t^l d2)  = -k
        (((1, 2), 2), ((), 1)): +1,     # c(t^k th1th2 d2, t^l d1)  = k
    }
    (a1, S1, d1), (a2, S2, d2) = x, y
    if a1 + a2 != 0:
        return Q(0)
    key = ((S1, d1), (S2, d2))
    if key in tab:
        return Q(tab[key] * a1)
    rkey = ((S2, d2), (S1, d1))
    if rkey in tab:
        # super skew-symmetry; parity of t^a S d_theta is len(S)+1 mod 2
        p1 = (len(S1) + 1) % 2
        p2 = (len(S2) + 1) % 2
        return -Q((-1) ** (p1 * p2)) * Q(tab[rkey] * a2)
    return Q(0)


def cvec(X, Y):
    """cocycle on linear combinations; X, Y are dicts {(a,S,d): poly-in-lam}"""
    tot = PZERO
    for x, cx in X.items():
        for y, cy in Y.items():
            v = c130(x, y)
            if v:
                tot = padd(tot, pscal(v, pmul(cx, cy)))
    return tot


def LL(m):
    """LL_m = t^m (t d_t + (lam+m+1)/2 (th1 d1 + th2 d2));  t-part is in ker c."""
    h = pscal(Q(1, 2), pol(m + 1, 1))          # (lam+m+1)/2
    return {(m, (1,), 1): h, (m, (2,), 2): h}


def S(m, j):
    """S_m^j = t^m (th_j (t d_t + (lam+m+1)(th1 d1 + th2 d2)));
    th_j * (th1 d1 + th2 d2) = th_j th_{3-j} d_{3-j}; t-part is in ker c."""
    h = pol(m + 1, 1)                           # lam+m+1
    i = 3 - j
    # th_j th_i d_i, brought to the canonical order th1 th2
    sgn = 1 if j == 1 else -1
    return {(m, (1, 2), i): pscal(sgn, h)}


def Dth(m, i):
    return {(m, (), i): pol(1)}


def Rot(m):
    return {(m, (1,), 1): pol(1), (m, (2,), 2): pol(-1)}


def find_determining_triples(arange=range(-3, 4)):
    """Four values of the table are NOT determined by the derivation printed in
    Subsec. dedCoc: at one exceptional value of the parameter the coefficient
    in front of the unknown vanishes.  For each, search for a triple (f,g,h) of
    monomials for which (214) does determine it."""
    # (n, key1, key2, printed value)
    gaps = [
        (1, (1, (0,)), (1, (0,)), Q(-1, 4),
         "c(t z, t z) = -1/4      (m=0 in the zeta row)"),
        (3, (1, (1, 2)), (-1, (1, 2)), Q(-1),
         "c(t th_i th_j, t^-1 th_i th_j) = -1   (m=1 in the th_i th_j row)"),
        (3, (1, (0, 1, 2)), (-1, (0, 1, 2)), Q(-1),
         "c(t th_i th_j z, t^-1 th_i th_j z) = -1   (n=-1 in that row)"),
        (4, (0, (1, 2, 3)), (-1, (1, 2, 3)), Q(-1),
         "c(th1th2th3, t^-1 th1th2th3) = -1   (n=-1 in that row)"),
    ]
    out = []
    for n, ka, kb, val, label in gaps:
        def cu(k1, k2):
            """value as (const, coefficient of the unknown u)"""
            if (k1, k2) == (ka, kb):
                return (Q(0), Q(1))
            if (k1, k2) == (kb, ka):
                p = (len(ka[1]) % 2) * (len(kb[1]) % 2)
                return (Q(0), -Q((-1) ** p))
            return (cval(n, k1, k2), Q(0))

        def cub(f, g):
            c0, c1 = Q(0), Q(0)
            for k1, x in f.items():
                for k2, y in g.items():
                    a, b = cu(k1, k2)
                    c0 += x * y * a
                    c1 += x * y * b
            return (c0, c1)

        found = None
        keys = monkeys(n, arange)
        for kf in keys:
            f = mono(*kf)
            for kg in keys:
                g = mono(*kg)
                s = (-1) ** (parity(f) * parity(g))
                for kh in keys:
                    h = mono(*kh)
                    l0, l1 = cub(f, br(n, g, h))
                    r0a, r1a = cub(br(n, f, g), h)
                    r0b, r1b = cub(g, br(n, f, h))
                    A = l1 - r1a - s * r1b
                    B = l0 - r0a - s * r0b
                    if A:
                        sol = -B / A
                        if found is None:
                            found = (kf, kg, kh, sol)
                        assert sol == val, (label, kf, kg, kh, sol)
        out.append((label, found))
    return out


# ------------- Part III: the Appendix vs the main text, (49) & (50)

BASIS = {                       # name -> key of the generating function, as (49)
    "L_n":               lambda m: (m + 1, ()),
    "T^{th_i th_j}_n":   lambda m: (m, (1, 2)),
    "T^{th_i zeta}_n":   lambda m: (m, (1, 0)),          # printed order th_i zeta
    "S_n":               lambda m: (m - 1, (1, 2, 3, 0)),
    "G^{th_i}_n":        lambda m: (m + 1, (1,)),
    "wt F_n":            lambda m: (m, (1, 2, 3)),
    "G^{zeta}_n":        lambda m: (m + 1, (0,)),
    "F^{th_i th_j}_n":   lambda m: (m, (1, 2, 0)),
}
PRINTED_DEG = {                 # the degree column of (49)
    "L_n": lambda m: 2 * m, "T^{th_i th_j}_n": lambda m: 2 * m,
    "T^{th_i zeta}_n": lambda m: 2 * m + 1, "S_n": lambda m: 2 * m + 1,
    "G^{th_i}_n": lambda m: 2 * m + 1, "wt F_n": lambda m: 2 * m + 1,
    "G^{zeta}_n": lambda m: 2 * m, "F^{th_i th_j}_n": lambda m: 2 * m,
}
COCYCR = {                      # the table (50) of the main text
    "L_n":             lambda m, nn: Q(m * (m * m - 1)) if m + nn == 0 else Q(0),
    "G^{zeta}_n":      lambda m, nn: Q(m * m) - Q(1, 4) if m + nn == 0 else Q(0),
    "G^{th_i}_n":      lambda m, nn: Q(m * (m + 1)) if m + nn + 1 == 0 else Q(0),
    "T^{th_i th_j}_n": lambda m, nn: Q(-m) if m + nn == 0 else Q(0),
    "T^{th_i zeta}_n": lambda m, nn: -(Q(m) - HALF) if m + nn - 1 == 0 else Q(0),
    "F^{th_i th_j}_n": lambda m, nn: Q(-1) if m + nn == 0 else Q(0),
    "wt F_n":          lambda m, nn: Q(-1) if m + nn + 1 == 0 else Q(0),
    "S_n":             lambda m, nn: Q(2, 2 * m - 1) if m + nn - 1 == 0 else Q(0),
}


def part3(rng=range(-4, 5)):
    out = []
    for name, f in BASIS.items():
        # (a) does the printed degree agree with the grading of the Appendix?
        bad = [m for m in rng if gr(4, f(m)) != PRINTED_DEG[name](m)]
        out.append(("(49) degree of %-16s = %s" % (
            name, "as printed" if not bad
            else "WRONG: printed %s, true %s" % (PRINTED_DEG[name](1), gr(4, f(1)))), not bad))
        # (b) does (50) agree with the Appendix's table?
        bad = []
        for m in rng:
            for nn in rng:
                k1, k2 = f(m), f(nn)
                # bring the printed monomial to canonical order, keeping the sign
                s1 = mono(k1[0], k1[1])
                s2 = mono(k2[0], k2[1])
                got = cbil(4, s1, s2)
                if got != COCYCR[name](m, nn):
                    bad.append((m, nn, got, COCYCR[name](m, nn)))
        out.append(("(50) row  %-16s = Appendix's row" % name, not bad))
    return out


def vkeys(arange):
    """basis of vect^L(1|2): t^a * (odd monomial) * d_x, x in {t, 1, 2}."""
    out = []
    for a in arange:
        for r in range(3):
            for Sq in combinations((1, 2), r):
                for d in ('t', 1, 2):
                    out.append((a, Sq, d))
    return out


def as_vf(key, coeff=1):
    a, Sq, d = key
    from stringy import VF
    return VF({d: mono(a, Sq, coeff)}, (1, 2))


def decomp(X):
    """VF -> {(a,S,d): coeff}"""
    out = {}
    for d, f in X.c.items():
        for (a, Sq), c in f.items():
            out[(a, Sq, d)] = out.get((a, Sq, d), Q(0)) + c
    return out


def vpar(key):
    a, Sq, d = key
    return (len(Sq) + (0 if d == 't' else 1)) % 2


def c130_lin(X, Y):
    """c130 extended bilinearly to elements given as {key: coeff}"""
    tot = Q(0)
    for kx, cx in X.items():
        for ky, cy in Y.items():
            tot += cx * cy * c130(kx, ky)
    return tot


def check_130_cocycle(arange=range(-2, 3)):
    """Is (51), extended by zero, a 2-cocycle on vect^L(1|2)?"""
    keys = vkeys(arange)
    bad = []
    for kf in keys:
        F = as_vf(kf)
        for kg in keys:
            G = as_vf(kg)
            s = (-1) ** (vpar(kf) * vpar(kg))
            FG = decomp(F.bracket(G))
            for kh in keys:
                H = as_vf(kh)
                lhs = c130_lin({kf: Q(1)}, decomp(G.bracket(H)))
                rhs = c130_lin(FG, {kh: Q(1)}) + s * c130_lin({kg: Q(1)}, decomp(F.bracket(H)))
                if lhs != rhs:
                    bad.append((kf, kg, kh, lhs, rhs))
    return bad


def part2(rng=range(-4, 5)):
    out = []
    # row 1 of (53), and the Appendix's own value
    bad = []
    for m in rng:
        for nn in rng:
            got = cvec(LL(m), LL(nn))
            # Appendix: 1/2 m ((lam+1)^2 - m^2) delta
            want = PZERO
            if m + nn == 0:
                want = pscal(Q(m, 2), padd(pmul(pol(1, 1), pol(1, 1)), pol(-m * m)))
            if got != want:
                bad.append((m, nn, got, want))
    out.append(("c(LL_m,LL_n) = 1/2 m((lam+1)^2-m^2) d   [Appendix]", not bad, bad[:2]))
    bad = []
    for m in rng:
        for nn in rng:
            got = cvec(LL(m), LL(nn))
            want = PZERO
            if m + nn == 0:                      # (53): 1/2 m(m^2-(lam+1)^2)
                want = pscal(Q(m, 2), padd(pol(m * m), pscal(-1, pmul(pol(1, 1), pol(1, 1)))))
            if got != want:
                bad.append((m, nn, got, want))
    out.append(("c(LL_m,LL_n) = 1/2 m(m^2-(lam+1)^2) d   [(53)]", not bad, bad[:2]))

    # row 2: the Appendix computes c(t^m d_1, S_n^2); (53) has delta_{ij}
    for (i, j) in ((1, 1), (1, 2), (2, 1), (2, 2)):
        vals = {}
        for m in rng:
            for nn in rng:
                if m + nn:
                    continue
                vals[(m, nn)] = cvec(Dth(m, i), S(nn, j))
        allzero = all(v == PZERO for v in vals.values())
        # Appendix's number: m(m-(lam+1)) delta
        okval = all(vals[(m, -m)] == pmul(pol(m), pol(m - 1, -1)) for m in rng)
        out.append(("c(t^m d_%d, S_n^%d): identically zero = %s ; equals m(m-(lam+1))d = %s"
                    % (i, j, allzero, okval), True, []))

    # rows 3, 4
    bad = []
    for m in rng:
        for nn in rng:
            got = cvec(Rot(m), Rot(nn))
            want = pol(-2 * m) if m + nn == 0 else PZERO
            if got != want:
                bad.append((m, nn, got, want))
    out.append(("c(t^m(th1d1-th2d2), t^n(...)) = -2m d  (row 3 of (53) is +2m)",
                not bad, bad[:2]))
    bad = []
    for m in rng:
        for nn in rng:
            got = cvec({(m, (1,), 2): pol(1)}, {(nn, (2,), 1): pol(1)})
            want = pol(-m) if m + nn == 0 else PZERO
            if got != want:
                bad.append((m, nn, got, want))
    out.append(("c(t^m th1 d2, t^n th2 d1) = -m d  (row 4 of (53) is +m)",
                not bad, bad[:2]))
    return out


# ------------------------------------------------------------------- main

def main():
    print("backend: exact rational arithmetic (fractions.Fraction)")
    print()
    print("=" * 72)
    print("PART I.  Subsec. A.1: the cocycles on k^M(1|n)")
    print("=" * 72)
    for n in (1, 2, 3, 4):
        g = check_grading(n)
        s = check_skew_bracket(n)
        print("k^M(1|%d): grading additive: %-4s  bracket skew-symmetric: %-4s"
              % (n, "OK" if not g else "%d BAD" % len(g),
                 "OK" if not s else "%d BAD" % len(s)))
    print()
    for n in (1, 2, 3, 4):
        j = check_jacobi(n)
        print("k^M(1|%d): super Jacobi on all triples of monomials, -2<=deg t<=2: %s"
              % (n, "OK" if not j else "%d BAD %s" % (len(j), j[:2])))
    print()
    for n in (1, 2, 3, 4):
        sk = check_skew_cocycle(n)
        cc = check_cocycle(n)
        print("k^M(1|%d): table skew-symmetric: %-4s   (214) on all triples: %s"
              % (n, "OK" if not sk else "%d BAD" % len(sk),
                 "OK" if not cc else "%d BAD %s" % (len(cc), cc[:2])))
    print()
    for n in (1, 2, 3, 4):
        cob = check_nontrivial(n)
        print("k^M(1|%d): c is a coboundary on the window? %s" % (n, cob))
    print()
    print("-" * 72)
    print("Subsec. dedCoc: every printed bracket, as an identity in k,m,n")
    print("-" * 72)
    nbad = 0
    for label, tot, bad, ex in derivation_steps():
        nbad += bad
        print("  %-52s %3d cases  %s" % (label, tot, "OK" if not bad else "FAIL %s" % (ex,)))
    print("  ---> %d failures" % nbad)
    print()
    print("-" * 72)
    print("Subsec. dedCoc: the arithmetic of each '=>' step")
    print("-" * 72)
    for label, ok, ex in derivation_arithmetic():
        print("  %-58s %s" % (label, "OK" if ok else "FAIL %s" % (ex,)))
    print()
    print("-" * 72)
    print("Values the printed derivation leaves undetermined, and a triple that")
    print("does determine each (the value found is then asserted equal to the table)")
    print("-" * 72)
    for label, found in find_determining_triples():
        kf, kg, kh, sol = found
        print("  %-52s f=%s g=%s h=%s -> %s" % (label, kf, kg, kh, sol))
    print()
    print("=" * 72)
    print("PART II.  Subsec. A.2: vect^L(1|2) and svect^L_lambda(1|2)")
    print("=" * 72)
    b130 = check_130_cocycle()
    print("  (51), extended by zero, is a 2-cocycle on vect^L(1|2): %s"
          % ("OK" if not b130 else "%d BAD %s" % (len(b130), b130[:2])))
    for label, ok, ex in part2():
        print("  %-72s %s" % (label, "OK" if ok else "FAIL %s" % (ex,)))
    print()
    print("=" * 72)
    print("PART III.  The Appendix against the main text: (49), (50)")
    print("=" * 72)
    for label, ok in part3():
        print("  %-62s %s" % (label, "OK" if ok else "MISMATCH"))


if __name__ == "__main__":
    main()
