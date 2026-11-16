"""
The xi-eta realisation of the formulas of Section 3 of the text (the distinguished
simple stringy Lie superalgebras: brackets, bases, cocycles).

Everything is exact arithmetic in the field  K = Q(sqrt(-1), sqrt 2)  (the two
irrationalities the change of variables and the o(2k+1)-basis need).

The change of variables (n = 2k or n = 2k+1):

    xi_i  := (theta_i + I theta_{k+i}) / sqrt 2,
    eta_i := (theta_i - I theta_{k+i}) / sqrt 2,   1 <= i <= k,
    theta := theta_n   (only for n = 2k+1),

equivalently

    theta_i     = (xi_i + eta_i)/sqrt 2,
    theta_{k+i} = -I (xi_i - eta_i)/sqrt 2.

What is verified, for the Neveu-Schwarz (L) case and for the Ramond (M) case:

 (a) the change of variables turns  sum theta_i d theta_i  into
     sum (xi_i d eta_i + eta_i d xi_i) (+ theta d theta), i.e. M^T M is the
     hyperbolic Gram matrix -- so (1) and (7) hold as printed;
 (b) the field K_f of (3) written in xi-eta coordinates is the push-forward of
     the field K_f of (3) written in theta coordinates, for every monomial f;
 (c) same for the contact bracket (4), the Moebius contact field (8)-(10) and
     the Ramond/Moebius-Poisson brackets (11)-(12);
 (d) independently of (b)-(c): in the xi-eta realisation f |--> K_f is a
     homomorphism, [K_f, K_g] = K_{{f,g}}, and the bracket satisfies Jacobi;
 (e) the Poisson bracket (130) of po(0|n) in xi-eta coordinates, and the
     weights: {xi_i eta_i, xi_j} = delta_ij xi_j, {xi_i eta_i, eta_j} =
     -delta_ij eta_j, {xi_i eta_i, theta} = 0 -- the maximal torus of o(n);
 (f) the constant in theta_1 ... theta_n = c * xi_1 eta_1 ... xi_k eta_k
     (* theta), needed in (6).

Run:  python3 xieta.py
"""

from fractions import Fraction as Q
from itertools import combinations, product
import sys

import stringy


# --------------------------------------------------------------- the field K
# a + b I + (c + d I) sqrt2, with a, b, c, d in Q.

class N:
    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a=0, b=0, c=0, d=0):
        self.a, self.b, self.c, self.d = Q(a), Q(b), Q(c), Q(d)

    @staticmethod
    def of(x):
        if isinstance(x, N):
            return x
        return N(Q(x))

    def __add__(self, o):
        o = N.of(o)
        return N(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    __radd__ = __add__

    def __neg__(self):
        return N(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, o):
        return self + (-N.of(o))

    def __rsub__(self, o):
        return N.of(o) + (-self)

    def __mul__(self, o):
        o = N.of(o)
        # (x1 + y1 s)(x2 + y2 s) = (x1 x2 + 2 y1 y2) + (x1 y2 + y1 x2) s,
        # x, y Gaussian:  (p+qI)(r+sI) = (pr - qs) + (ps + qr) I.
        def g(p, q, r, s):
            return (p * r - q * s, p * s + q * r)
        x1x2 = g(self.a, self.b, o.a, o.b)
        y1y2 = g(self.c, self.d, o.c, o.d)
        x1y2 = g(self.a, self.b, o.c, o.d)
        y1x2 = g(self.c, self.d, o.a, o.b)
        return N(x1x2[0] + 2 * y1y2[0], x1x2[1] + 2 * y1y2[1],
                 x1y2[0] + y1x2[0], x1y2[1] + y1x2[1])

    __rmul__ = __mul__

    def __eq__(self, o):
        o = N.of(o)
        return (self.a, self.b, self.c, self.d) == (o.a, o.b, o.c, o.d)

    def __ne__(self, o):
        return not self.__eq__(o)

    def __bool__(self):
        return any((self.a, self.b, self.c, self.d))

    def __hash__(self):
        return hash((self.a, self.b, self.c, self.d))

    def __repr__(self):
        parts = []
        for v, s in ((self.a, ""), (self.b, "I"), (self.c, "r2"), (self.d, "I r2")):
            if v:
                parts.append("%s%s" % (v, (" " + s) if s else ""))
        return " + ".join(parts) if parts else "0"


ONE = N(1)
I = N(0, 1)
R2 = N(0, 0, 1)          # sqrt 2
INV_R2 = N(0, 0, Q(1, 2))  # 1/sqrt2 = sqrt2 / 2

# teach the stringy engine to use K as the coefficient ring
stringy.sympify_S = lambda c: c if isinstance(c, N) else N.of(c)

from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, VF   # noqa: E402


def sm(c, f):
    return smul(N.of(c), f)


# ------------------------------------------------------- coordinates, labels
# L case, n odd coordinates.  theta-realisation: labels 1..n = theta_1..theta_n.
# xi-eta realisation:  1..k = xi_1..xi_k,  k+1..2k = eta_1..eta_k,  n = theta
# (the last only if n is odd).
#
# M case, n odd coordinates: label 0 = zeta, and 1..n-1 as above with k =
# (n-1)//2.

def labels(n):
    return tuple(range(1, n + 1))


def change_matrices(m):
    """m = number of *untwisted* odd coordinates (n in the L case, n-1 in the
    M case).  Returns (M, Ninv): theta_j = sum_a M[j][a] u_a  and
    u_a = sum_j Ninv[a][j] theta_j, indices 1..m."""
    k = m // 2
    M = {j: {} for j in range(1, m + 1)}
    Nv = {a: {} for a in range(1, m + 1)}
    for i in range(1, k + 1):
        xi, eta = i, k + i
        th1, th2 = i, k + i
        # theta_i = (xi + eta)/sqrt2 ; theta_{k+i} = -I (xi - eta)/sqrt2
        M[th1][xi] = INV_R2
        M[th1][eta] = INV_R2
        M[th2][xi] = -I * INV_R2
        M[th2][eta] = I * INV_R2
        # xi = (theta_i + I theta_{k+i})/sqrt2 ; eta = (theta_i - I th_{k+i})/sqrt2
        Nv[xi][th1] = INV_R2
        Nv[xi][th2] = I * INV_R2
        Nv[eta][th1] = INV_R2
        Nv[eta][th2] = -I * INV_R2
    if m % 2:
        M[m][m] = ONE
        Nv[m][m] = ONE
    return M, Nv


def gram(M, m):
    """(M^T M)_{ab} = sum_j M[j][a] M[j][b]."""
    G = {}
    for a in range(1, m + 1):
        for b in range(1, m + 1):
            s = N(0)
            for j in range(1, m + 1):
                s = s + M[j].get(a, N(0)) * M[j].get(b, N(0))
            if s:
                G[(a, b)] = s
    return G


def hyperbolic(m):
    """the Gram matrix of sum (xi_i d eta_i + eta_i d xi_i) (+ theta d theta)."""
    k = m // 2
    G = {}
    for i in range(1, k + 1):
        G[(i, k + i)] = ONE
        G[(k + i, i)] = ONE
    if m % 2:
        G[(m, m)] = ONE
    return G


def subst_fun(f, M, shift=0):
    """theta-realisation function -> xi-eta realisation function.
    shift: the odd labels of the M case start at 1 for theta_1 (0 is zeta,
    which is not changed)."""
    out = {}
    for (a, S), c in f.items():
        term = {(a, ()): c}
        for j in S:
            if j == 0:                      # zeta: untouched
                repl = mono(0, (0,))
            else:
                repl = {}
                for u, coeff in M[j].items():
                    repl = add(repl, smul(coeff, mono(0, (u,))))
            term = mul(term, repl)
        out = add(out, term)
    return out


def push_vf(X, Nv, odd_labels):
    """push a vector field forward through the same linear change."""
    comps = {}
    if 't' in X.c:
        comps['t'] = subst_fun(X.c['t'], MCUR)
    for a in odd_labels:
        s = {}
        for j, Xj in X.c.items():
            if j == 't':
                continue
            if j == 0:                       # zeta component is untouched
                if a == 0:
                    s = add(s, subst_fun(Xj, MCUR))
                continue
            if a == 0:
                continue
            coeff = Nv[a].get(j)
            if coeff is not None:
                s = add(s, smul(coeff, subst_fun(Xj, MCUR)))
        if s:
            comps[a] = s
    return VF(comps, odd_labels)


# ------------------------------------------------------ the L (NS) case, (3), (4)

def euler(f, odd):
    out = {}
    for (a, S), c in f.items():
        v = N.of(len(S)) * c
        if v:
            out[(a, S)] = v
    return out


def nabla(f):
    """(2 - E)(f)."""
    out = {}
    for (a, S), c in f.items():
        v = N.of(2 - len(S)) * c
        if v:
            out[(a, S)] = v
    return out


def K_theta(f, n):
    """eq. (3) as printed, theta-coordinates."""
    odd = labels(n)
    comps = {'t': nabla(f)}
    sgn = -((-1) ** parity(f))
    for i in odd:
        ci = add(d_odd(i, f), sm(-1, mul(mono(0, (i,)), d_t(f))))
        ci = sm(sgn, ci)
        if ci:
            comps[i] = ci
    return VF(comps, odd)


def K_xieta(f, n):
    """eq. (3) in xi-eta coordinates:
       K_f = (2-E)(f) d_t
             - (-1)^{p(f)} sum_i [ (d_{eta_i}f - xi_i d_t f) d_{xi_i}
                                 + (d_{xi_i}f - eta_i d_t f) d_{eta_i} ]
             - (-1)^{p(f)} (d_theta f - theta d_t f) d_theta   [n odd]."""
    odd = labels(n)
    k = n // 2
    comps = {'t': nabla(f)}
    sgn = -((-1) ** parity(f))
    for i in range(1, k + 1):
        xi, eta = i, k + i
        cxi = add(d_odd(eta, f), sm(-1, mul(mono(0, (xi,)), d_t(f))))
        ceta = add(d_odd(xi, f), sm(-1, mul(mono(0, (eta,)), d_t(f))))
        if sm(sgn, cxi):
            comps[xi] = sm(sgn, cxi)
        if sm(sgn, ceta):
            comps[eta] = sm(sgn, ceta)
    if n % 2:
        c = add(d_odd(n, f), sm(-1, mul(mono(0, (n,)), d_t(f))))
        if sm(sgn, c):
            comps[n] = sm(sgn, c)
    return VF(comps, odd)


def kb_theta(f, g, n):
    """eq. (4) as printed."""
    res = add(mul(nabla(f), d_t(g)), sm(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in labels(n):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, sm(-((-1) ** parity(f)), s))


def kb_xieta(f, g, n):
    """eq. (4) in xi-eta coordinates."""
    res = add(mul(nabla(f), d_t(g)), sm(-1, mul(d_t(f), nabla(g))))
    k = n // 2
    s = {}
    for i in range(1, k + 1):
        xi, eta = i, k + i
        s = add(s, mul(d_odd(xi, f), d_odd(eta, g)),
                mul(d_odd(eta, f), d_odd(xi, g)))
    if n % 2:
        s = add(s, mul(d_odd(n, f), d_odd(n, g)))
    return add(res, sm(-((-1) ** parity(f)), s))


# ------------------------------------------- the M (Ramond) case, (7)-(12)

def E_M(f, n, xieta):
    """cE = zeta d_zeta + sum theta_i d_theta_i  =  zeta d_zeta + sum (xi d_xi
    + eta d_eta) (+ theta d_theta): the Euler operator, the same in both."""
    out = {}
    for (a, S), c in f.items():
        v = N.of(len(S)) * c
        if v:
            out[(a, S)] = v
    return out


def nablaM(f):
    return nabla(f)                     # 2 - cE, same shape


def DM(f):
    """cD = d_t - (zeta / 2t) d_zeta."""
    return add(d_t(f), sm(Q(-1, 2), mul(mono(-1, (0,)), d_odd(0, f))))


def H_theta(f, n):
    """eq. (10) as printed (theta-coordinates), n-1 untwisted thetas."""
    odd = (0,) + labels(n - 1)
    comps = {}
    sgn = (-1) ** parity(f)
    cz = sm(sgn, mul(mono(-1), d_odd(0, f)))
    if cz:
        comps[0] = cz
    for i in labels(n - 1):
        c = sm(sgn, d_odd(i, f))
        if c:
            comps[i] = c
    return VF(comps, odd)


def H_xieta(f, n):
    """eq. (10) in xi-eta coordinates."""
    m = n - 1
    k = m // 2
    odd = (0,) + labels(m)
    comps = {}
    sgn = (-1) ** parity(f)
    cz = sm(sgn, mul(mono(-1), d_odd(0, f)))
    if cz:
        comps[0] = cz
    for i in range(1, k + 1):
        xi, eta = i, k + i
        cxi = sm(sgn, d_odd(eta, f))
        ceta = sm(sgn, d_odd(xi, f))
        if cxi:
            comps[xi] = cxi
        if ceta:
            comps[eta] = ceta
    if m % 2:
        c = sm(sgn, d_odd(m, f))
        if c:
            comps[m] = c
    return VF(comps, odd)


def KM(f, n, xieta):
    """eq. (8): K^M_f = nabla(f) cD + H^M_f + cD(f) cE."""
    m = n - 1
    odd = (0,) + labels(m)
    # nabla(f) cD, the coefficient standing on the left of the derivative
    comps = {'t': nablaM(f),
             0: sm(Q(-1, 2), mul(nablaM(f), mono(-1, (0,))))}
    X = VF({k_: v for k_, v in comps.items() if v}, odd)
    H = H_xieta(f, n) if xieta else H_theta(f, n)
    # cD(f) cE
    Df = DM(f)
    ce = {}
    for a in odd:
        c = mul(Df, mono(0, (a,)))
        if c:
            ce[a] = c
    Y = VF(ce, odd)
    return X.plus(H).plus(Y)


def MP_theta(f, g, n):
    """eq. (12), theta-coordinates (with n-1 thetas, see the note in the text)."""
    sgn = (-1) ** parity(f)
    s = mul(mono(-1), mul(d_odd(0, f), d_odd(0, g)))
    for i in labels(n - 1):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return sm(sgn, s)


def MP_xieta(f, g, n):
    """eq. (12) in xi-eta coordinates."""
    m = n - 1
    k = m // 2
    sgn = (-1) ** parity(f)
    s = mul(mono(-1), mul(d_odd(0, f), d_odd(0, g)))
    for i in range(1, k + 1):
        xi, eta = i, k + i
        s = add(s, mul(d_odd(xi, f), d_odd(eta, g)),
                mul(d_odd(eta, f), d_odd(xi, g)))
    if m % 2:
        s = add(s, mul(d_odd(m, f), d_odd(m, g)))
    return sm(sgn, s)


def Rb(f, g, n, xieta):
    """eq. (11): {f,g}_R.b. = nabla(f) cD(g) - cD(f) nabla(g) + {f,g}_MP."""
    res = add(mul(nablaM(f), DM(g)), sm(-1, mul(DM(f), nablaM(g))))
    mp = MP_xieta(f, g, n) if xieta else MP_theta(f, g, n)
    return add(res, mp)


# --------------------------------------------------------------- test inputs

def monomials(odd, arange):
    out = []
    for a in arange:
        for r in range(len(odd) + 1):
            for S in combinations(odd, r):
                out.append(mono(a, S))
    return out


MCUR = None       # the change matrix currently in use (set by the callers)


def report(name, bad, total):
    tag = "OK " if bad == 0 else "FAIL"
    print("  [%s] %-58s %d/%d" % (tag, name, total - bad, total))
    return bad


def hom_M():
    """(g) intrinsic check on the Ramond side, in the xi-eta realisation:
    is f |--> K^M_f a homomorphism for the bracket (11)-(12) as printed, and
    for the one with the opposite sign of the Moebius-Poisson part?"""
    print("=" * 78)
    print("(g) M case, intrinsic: [K^M_f, K^M_g] = K^M_{{f,g}_R.b.}, xi-eta")
    print("=" * 78)
    for n in (2, 3, 4):
        m = n - 1
        odd = (0,) + labels(m)
        fs = monomials(odd, range(-1, 2))
        for sign, tag in ((+1, "as printed, eq. (12)"), (-1, "opposite sign of MP")):
            bad = cnt = 0
            for f in fs:
                for g in fs:
                    cnt += 1
                    br = add(mul(nablaM(f), DM(g)), sm(-1, mul(DM(f), nablaM(g))),
                             sm(sign, MP_xieta(f, g, n)))
                    lhs = KM(f, n, True).bracket(KM(g, n, True))
                    rhs = KM(br, n, True)
                    if not (lhs == rhs):
                        bad += 1
            report("n = %d, %-22s" % (n, tag), bad, cnt)


def main():
    bad_total = 0
    ar = range(-2, 3)
    if "--homM" in sys.argv:
        hom_M()
        return 0

    print("=" * 78)
    print("(a) the change of variables:  M^T M  =  hyperbolic Gram matrix")
    print("=" * 78)
    for m in range(1, 7):
        M, Nv = change_matrices(m)
        G, H = gram(M, m), hyperbolic(m)
        bad_total += report("m = %d  (theta_1..theta_%d -> xi,eta%s)"
                            % (m, m, ", theta" if m % 2 else ""),
                            0 if G == H else 1, 1)

    print()
    print("=" * 78)
    print("(b), (c) L case: the xi-eta formulas (3), (4) are the push-forwards")
    print("=" * 78)
    global MCUR
    for n in (1, 2, 3, 4):
        M, Nv = change_matrices(n)
        MCUR = M
        odd = labels(n)
        fs = monomials(odd, ar)
        # (3)
        bad = 0
        for f in fs:
            lhs = push_vf(K_theta(f, n), Nv, odd)
            rhs = K_xieta(subst_fun(f, M), n)
            if not (lhs == rhs):
                bad += 1
        bad_total += report("n = %d: phi_*(K_f^theta) = K_{phi(f)}^{xi-eta}, eq. (3)"
                            % n, bad, len(fs))
        # (4)
        bad = 0
        cnt = 0
        for f in fs:
            for g in fs:
                cnt += 1
                lhs = subst_fun(kb_theta(f, g, n), M)
                rhs = kb_xieta(subst_fun(f, M), subst_fun(g, M), n)
                if not eq(lhs, rhs):
                    bad += 1
        bad_total += report("n = %d: phi({f,g}^theta) = {phi f, phi g}^{xi-eta}, eq. (4)"
                            % n, bad, cnt)

    print()
    print("=" * 78)
    print("(d) L case, intrinsic: [K_f, K_g] = K_{{f,g}} in xi-eta coordinates")
    print("=" * 78)
    for n in (1, 2, 3, 4):
        odd = labels(n)
        fs = monomials(odd, range(-1, 2))
        bad = cnt = 0
        for f in fs:
            for g in fs:
                cnt += 1
                lhs = K_xieta(f, n).bracket(K_xieta(g, n))
                rhs = K_xieta(kb_xieta(f, g, n), n)
                if not (lhs == rhs):
                    bad += 1
        bad_total += report("n = %d: homomorphism, xi-eta realisation" % n, bad, cnt)

    print()
    print("=" * 78)
    print("(c) M case: (7), (9), (10), (11), (12) in xi-eta coordinates")
    print("=" * 78)
    for n in (2, 3, 4, 5):
        m = n - 1
        M, Nv = change_matrices(m)
        MCUR = M
        odd = (0,) + labels(m)
        fs = monomials(odd, ar)
        # (7): the Gram matrix of the theta-part; zeta is untouched
        G, H = gram(M, m), hyperbolic(m)
        bad_total += report("n = %d: eq. (7), the untwisted part of tilde alpha" % n,
                            0 if G == H else 1, 1)
        # (8)+(10)
        bad = 0
        for f in fs:
            lhs = push_vf(KM(f, n, False), Nv, odd)
            rhs = KM(subst_fun(f, M), n, True)
            if not (lhs == rhs):
                bad += 1
        bad_total += report("n = %d: phi_*(K^M_f) = K^M_{phi f}, eqs. (8)-(10)" % n,
                            bad, len(fs))
        # (11)+(12)
        bad = cnt = 0
        for f in fs:
            for g in fs:
                cnt += 1
                lhs = subst_fun(Rb(f, g, n, False), M)
                rhs = Rb(subst_fun(f, M), subst_fun(g, M), n, True)
                if not eq(lhs, rhs):
                    bad += 1
        bad_total += report("n = %d: phi({f,g}_R.b.) = {phi f, phi g}_R.b., (11)-(12)"
                            % n, bad, cnt)

    print()
    print("=" * 78)
    print("(e) po(0|n): the bracket (130) in xi-eta coordinates, weights")
    print("=" * 78)
    for n in (2, 3, 4, 5, 6):
        M, Nv = change_matrices(n)
        odd = labels(n)
        fs = [mono(0, S) for r in range(n + 1) for S in combinations(odd, r)]
        bad = cnt = 0
        for f in fs:
            for g in fs:
                cnt += 1
                lhs = subst_fun(kb_theta(f, g, n), M)
                rhs = kb_xieta(subst_fun(f, M), subst_fun(g, M), n)
                if not eq(lhs, rhs):
                    bad += 1
        bad_total += report("n = %d: (130), t-free restriction of (4)" % n,
                            bad, cnt)
    # the weights
    for n in (4, 5):
        k = n // 2
        bad = cnt = 0
        for i in range(1, k + 1):
            h = mono(0, (i, k + i))                 # xi_i eta_i
            for j in range(1, k + 1):
                xj, ej = mono(0, (j,)), mono(0, (k + j,))
                want_x = sm(1 if i == j else 0, xj)
                want_e = sm(-1 if i == j else 0, ej)
                cnt += 2
                if not eq(kb_xieta(h, xj, n), want_x):
                    bad += 1
                if not eq(kb_xieta(h, ej, n), want_e):
                    bad += 1
            if n % 2:
                cnt += 1
                if kb_xieta(h, mono(0, (n,)), n):
                    bad += 1
            cnt += 1
            if kb_xieta(h, mono(3, ()), n):          # weight of t^m is 0
                bad += 1
        bad_total += report("n = %d: {xi_i eta_i, -} = weights +-eps_i, 0" % n,
                            bad, cnt)

    print()
    print("=" * 78)
    print("(f) the top monomial: theta_1...theta_n = c * xi_1 eta_1 ... (theta)")
    print("=" * 78)
    for n in (2, 3, 4, 5, 6, 7):
        k = n // 2
        M, Nv = change_matrices(n)
        lhs = subst_fun(mono(0, labels(n)), M)
        S = tuple(x for i in range(1, k + 1) for x in (i, k + i))
        if n % 2:
            S = S + (n,)
        base = mono(0, S)
        # lhs = c * base ?
        key = list(base)[0]
        c = lhs.get(key)
        if c is not None:
            c = c * base[key]            # base[key] = +-1, so this divides
        ok = c is not None and eq(lhs, smul(c, base))
        # predicted: I^k * (-1)^{k(k-1)/2}
        pv = ONE
        for _ in range(k):
            pv = pv * I
        pv = pv * N((-1) ** (k * (k - 1) // 2))
        print("     n = %d: c = %-20s predicted I^k (-1)^{k(k-1)/2} = %-14s %s"
              % (n, c, pv, "OK" if (ok and c == pv) else "MISMATCH"))
        bad_total += 0 if (ok and c == pv) else 1

    print()
    print("=" * 78)
    print("TOTAL FAILURES: %d" % bad_total)
    print("=" * 78)
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
