"""Golod's elliptic su(2)-algebra and its relation to the loop algebra.

Exact arithmetic over Q[A1,A2,A3].

The algebra g_A (Golod, TMF 70:1 (1987) 18-29, formulas (5) and (6)) has basis
X_i^m, i in {1,2,3}, m in Z, and the bracket

    [X_i^m, X_j^n] = eps_ijk (X_k^{m+n} - alpha X_k^{m+n-2}),

    alpha = 0     if m, n both odd,
          = A_i   if m odd,  n even,
          = A_j   if m even, n odd,
          = A_k   if m, n both even.

At A = 0 this is the loop algebra sl(2) tensor C[lam, lam^-1] with
X_i^m = X_i lam^m.

The realization behind it (Golod's (5)): on the curve w_i^2 = u - A_i, u = w^2,

    X_i^{2l+1} = u^l w_i X_i,      X_i^{2l+2} = u^l w_j w_k X_i.
"""

from fractions import Fraction
from itertools import product

# ---------------------------------------------------------------- polynomials
# A polynomial in A1, A2, A3 over Q is a dict {(e1,e2,e3): Fraction}, zero = {}.

ZERO = {}
ONE = {(0, 0, 0): Fraction(1)}


def pconst(c):
    c = Fraction(c)
    return {} if c == 0 else {(0, 0, 0): c}


def pvar(i):
    """A_i, i = 1, 2, 3."""
    e = [0, 0, 0]
    e[i - 1] = 1
    return {tuple(e): Fraction(1)}


def padd(p, q):
    r = dict(p)
    for e, c in q.items():
        c2 = r.get(e, Fraction(0)) + c
        if c2 == 0:
            r.pop(e, None)
        else:
            r[e] = c2
    return r


def pneg(p):
    return {e: -c for e, c in p.items()}


def psub(p, q):
    return padd(p, pneg(q))


def pmul(p, q):
    r = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = (e1[0] + e2[0], e1[1] + e2[1], e1[2] + e2[2])
            c = r.get(e, Fraction(0)) + c1 * c2
            if c == 0:
                r.pop(e, None)
            else:
                r[e] = c
    return r


def pscal(c, p):
    c = Fraction(c)
    return {} if c == 0 else {e: c * v for e, v in p.items()}


def pdeg(p):
    """Total degree in the A_i (-1 for the zero polynomial)."""
    return max((sum(e) for e in p), default=-1)


def ptrunc(p, N):
    """Drop the monomials of total degree > N."""
    return {e: c for e, c in p.items() if sum(e) <= N}


def psubs(p, vals):
    """Evaluate at A_i = vals[i-1]; returns a Fraction."""
    s = Fraction(0)
    for e, c in p.items():
        t = c
        for i in range(3):
            t *= Fraction(vals[i]) ** e[i]
        s += t
    return s


# ------------------------------------------------------------------- elements
# An element is a dict {(i, m): poly}, i in {1,2,3}, m in Z.


def eadd(x, y):
    r = dict(x)
    for k, p in y.items():
        s = padd(r.get(k, ZERO), p)
        if s:
            r[k] = s
        else:
            r.pop(k, None)
    return r


def eneg(x):
    return {k: pneg(p) for k, p in x.items()}


def esub(x, y):
    return eadd(x, eneg(y))


def escal(p, x):
    r = {}
    for k, q in x.items():
        s = pmul(p, q)
        if s:
            r[k] = s
    return r


def etrunc(x, N):
    r = {}
    for k, p in x.items():
        q = ptrunc(p, N)
        if q:
            r[k] = q
    return r


def ebasis(i, m):
    return {(i, m): ONE}


EPS = {}
for _p in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
    EPS[_p] = 1
    EPS[(_p[1], _p[0], _p[2])] = -1


def eps(i, j):
    """(eps_ijk, k) for i != j; (0, None) for i == j."""
    if i == j:
        return 0, None
    k = 6 - i - j
    return EPS[(i, j, k)], k


# ------------------------------------------------------------------- brackets


def alpha_index(i, j, k, m, n):
    """The index of the A that corrects the bracket, or None."""
    mo, no = m % 2, n % 2
    if mo and no:
        return None
    if mo and not no:
        return i
    if no and not mo:
        return j
    return k


def br_basis(i, m, j, n, A=True):
    """[X_i^m, X_j^n] in g_A (A=True) or in the loop algebra (A=False)."""
    s, k = eps(i, j)
    if s == 0:
        return {}
    out = {(k, m + n): pconst(s)}
    if A:
        a = alpha_index(i, j, k, m, n)
        if a is not None:
            out[(k, m + n - 2)] = pscal(-s, pvar(a))
    return {kk: v for kk, v in out.items() if v}


def bracket(x, y, A=True):
    out = {}
    for (i, m), p in x.items():
        for (j, n), q in y.items():
            b = br_basis(i, m, j, n, A)
            if not b:
                continue
            c = pmul(p, q)
            if not c:
                continue
            out = eadd(out, escal(c, b))
    return out


# ------------------------------------- independent model: functions on a curve
# B_A = C[u, u^-1] tensor span{w_S : S subset of {1,2,3}}, w_S w_T =
# prod_{i in S cap T} (u - A_i) * w_{S xor T}.  An element of B_A is a dict
# {(S, l): poly} standing for sum poly * u^l * w_S, S a frozenset.


def fmul(f, g):
    out = {}
    for (S, l), p in f.items():
        for (T, n), q in g.items():
            inter = S & T
            sym = S ^ T
            c = pmul(p, q)
            # prod_{i in inter} (u - A_i)
            terms = {(sym, l + n): c}
            for i in sorted(inter):
                new = {}
                for (U, e), cc in terms.items():
                    for key, val in (((U, e + 1), cc), ((U, e), pneg(pmul(pvar(i), cc)))):
                        s = padd(new.get(key, ZERO), val)
                        if s:
                            new[key] = s
                        else:
                            new.pop(key, None)
                terms = new
            for key, val in terms.items():
                s = padd(out.get(key, ZERO), val)
                if s:
                    out[key] = s
                else:
                    out.pop(key, None)
    return out


def coeff_of(i, m):
    """The function attached to X_i in Golod's basis vector X_i^m, as in (5)."""
    if m % 2 == 1:                       # m = 2l+1
        l = (m - 1) // 2
        return {(frozenset([i]), l): ONE}
    l = (m - 2) // 2                      # m = 2l+2
    j, k = [t for t in (1, 2, 3) if t != i]
    return {(frozenset([j, k]), l): ONE}


def from_function(k, f):
    """Rewrite f * X_k in Golod's basis; f must be supported on w_{k} and
    w_{complement of k}."""
    i, j = [t for t in (1, 2, 3) if t != k]
    out = {}
    for (S, l), p in f.items():
        if S == frozenset([k]):
            m = 2 * l + 1
        elif S == frozenset([i, j]):
            m = 2 * l + 2
        else:
            raise ValueError("function not in the k-th component: %s" % (S,))
        s = padd(out.get((k, m), ZERO), p)
        if s:
            out[(k, m)] = s
        else:
            out.pop((k, m), None)
    return out


def br_basis_via_functions(i, m, j, n):
    """[X_i^m, X_j^n] computed in the function realization."""
    s, k = eps(i, j)
    if s == 0:
        return {}
    f = fmul(coeff_of(i, m), coeff_of(j, n))
    return escal(pconst(s), from_function(k, f))


# ------------------------------------------------ the trivializing isomorphism
# Phi sends X_i^m to the expansion of the same function with w_i replaced by
# lam sqrt(1 - A_i lam^-2), i.e. the loop-algebra element it becomes after
# the substitution.  deg_A of the n-th term is n, its lam-degree is m - 2n.


def _binom_half(n):
    """binomial(1/2, n)."""
    r = Fraction(1)
    for t in range(n):
        r *= (Fraction(1, 2) - t) / (t + 1)
    return r


def sqrt_series(i, N):
    """sqrt(1 - A_i/u) up to A-degree N: list of (n, coefficient poly)."""
    out = []
    for n in range(N + 1):
        c = _binom_half(n) * Fraction((-1) ** n)
        out.append((n, pscal(c, pmul_pow(pvar(i), n))))
    return out


def pmul_pow(p, n):
    r = ONE
    for _ in range(n):
        r = pmul(r, p)
    return r


def Phi_basis(i, m, N):
    """Phi(X_i^m) modulo A-degree > N."""
    out = {}
    if m % 2 == 1:
        for n, c in sqrt_series(i, N):
            key = (i, m - 2 * n)
            s = padd(out.get(key, ZERO), c)
            if s:
                out[key] = s
            else:
                out.pop(key, None)
    else:
        j, k = [t for t in (1, 2, 3) if t != i]
        sj, sk = sqrt_series(j, N), sqrt_series(k, N)
        for nj, cj in sj:
            for nk, ck in sk:
                if nj + nk > N:
                    continue
                key = (i, m - 2 * (nj + nk))
                s = padd(out.get(key, ZERO), pmul(cj, ck))
                if s:
                    out[key] = s
                else:
                    out.pop(key, None)
    return out


def Phi(x, N):
    out = {}
    for (i, m), p in x.items():
        out = eadd(out, escal(p, Phi_basis(i, m, N)))
    return etrunc(out, N)


# --------------------------------------------- the infinitesimal (co)boundary
# phi_1(X_i^{2l+1}) = -(A_i/2) X_i^{2l-1},
# phi_1(X_i^{2l+2}) = -((A_j+A_k)/2) X_i^{2l}.


def phi1_basis(i, m):
    if m % 2 == 1:
        return {(i, m - 2): pscal(Fraction(-1, 2), pvar(i))}
    j, k = [t for t in (1, 2, 3) if t != i]
    return {(i, m - 2): pscal(Fraction(-1, 2), padd(pvar(j), pvar(k)))}


def phi1(x):
    out = {}
    for (i, m), p in x.items():
        out = eadd(out, escal(p, phi1_basis(i, m)))
    return out


def dphi1(i, m, j, n):
    """(d phi_1)(X_i^m, X_j^n) = [phi_1 x, y]_0 + [x, phi_1 y]_0 - phi_1 [x,y]_0."""
    x, y = ebasis(i, m), ebasis(j, n)
    t1 = bracket(phi1(x), y, A=False)
    t2 = bracket(x, phi1(y), A=False)
    t3 = phi1(bracket(x, y, A=False))
    return esub(eadd(t1, t2), t3)


def mu1(i, m, j, n):
    """The first-order term of the deformed bracket (here: the whole
    correction, the family being linear in A)."""
    return esub(br_basis(i, m, j, n, A=True), br_basis(i, m, j, n, A=False))


# ------------------------------------------------------------- centroid pieces
# T = multiplication by u (shift by 2), Z = multiplication by w_1 w_2 w_3.


def T_basis(i, m):
    return {(i, m + 2): ONE}


def Z_basis(i, m):
    """Z(X_i^m), read off from v * (function of X_i^m)."""
    f = fmul({(frozenset([1, 2, 3]), 0): ONE}, coeff_of(i, m))
    return from_function(i, f)


def apply_op(op, x):
    out = {}
    for (i, m), p in x.items():
        out = eadd(out, escal(p, op(i, m)))
    return out
