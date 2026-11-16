"""
The Schwarz derivative of a superdiffeomorphism of S^{1|2} in the VECTORIAL
realisation: Radul's Sect. 6 (Stockholm Reports 1986-21, pp. 53-54 = his pp.
(14)-(15)).

Everything is exact and fully general: a superdiffeomorphism of S^{1|2} is

    F:  x = f(y, eta_1, eta_2),  zeta_1 = phi(...),  zeta_2 = psi(...),

with f even and phi, psi odd, so that over Lambda(eta_1, eta_2)

    f   = f0(y) + f12(y) eta_1 eta_2,
    phi = a1(y) eta_1 + a2(y) eta_2,     psi = b1(y) eta_1 + b2(y) eta_2,

the six coefficients being ARBITRARY functions of y (sympy Functions).  Nothing
is expanded in a parameter: what is verified below holds for every F.

The ambient algebra is the free supercommutative algebra

    Phi~ = O(S^{1|2})[dx, dx^{-1}, d zeta_1, d zeta_2],

with Radul's own parities  p(dx) = 0,  p(d zeta_i) = 1  (p. 53: "whose parities
are dx~ = 0, d zeta_i~ = 1"), which is what makes dx invertible and d zeta_i
nilpotent.  A monomial is stored as

    (k, S) -> sympy coefficient in y,

k in Z the power of dy, S a tuple of odd generators in one GLOBAL order

    eta_1 < eta_2 < d eta_1 < d eta_2,

so that every sign is produced by one sorting routine and never by hand.
"""
import sympy as sp

y = sp.Symbol('y')

# odd generators, in the one global order
E1, E2, D1, D2 = 'e1', 'e2', 'd1', 'd2'
ORDER = {E1: 0, E2: 1, D1: 2, D2: 3}

ETA = (E1, E2)
DETA = (D1, D2)


# ------------------------------------------------------------------ algebra

def _sort(S):
    """Sort a tuple of odd generators; return (sorted tuple, sign), 0 if repeated."""
    S = list(S)
    sign = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if ORDER[S[j]] > ORDER[S[j + 1]]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sign = -sign
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sign


def clean(A):
    out = {}
    for key, c in A.items():
        c = sp.expand(sp.simplify(c)) if c.free_symbols else sp.expand(c)
        if c != 0:
            out[key] = c
    return out


def mono(c, k=0, S=()):
    S, sgn = _sort(S)
    if sgn == 0:
        return {}
    c = sp.expand(sgn * sp.sympify(c))
    return {} if c == 0 else {(k, S): c}


def add(*As):
    out = {}
    for A in As:
        for key, c in A.items():
            out[key] = out.get(key, 0) + c
    return clean(out)


def neg(A):
    return {key: -c for key, c in A.items()}


def sub(A, B):
    return add(A, neg(B))


def mul(A, B):
    """Product in the free supercommutative algebra; dy is even and central."""
    out = {}
    for (ka, Sa), ca in A.items():
        for (kb, Sb), cb in B.items():
            S, sgn = _sort(tuple(Sa) + tuple(Sb))
            if sgn == 0:
                continue
            out[(ka + kb, S)] = out.get((ka + kb, S), 0) + sgn * ca * cb
    return clean(out)


def scal(c, A):
    return clean({key: sp.expand(sp.sympify(c) * v) for key, v in A.items()})


def is_zero(A):
    return not clean(A)


def parity(S):
    return len(S) % 2


ONE = mono(1)


def inv(A):
    """Inverse of an element whose odd-generator-free part is invertible.

    A = u dy^k (1 + N) with N nilpotent: only four odd generators exist, so
    N^5 = 0 and the Neumann series terminates.
    """
    head = {key: c for key, c in A.items() if not key[1]}
    assert len(head) == 1, f'no unique invertible part: {head}'
    (k, S), u = next(iter(head.items()))
    hinv = mono(1 / u, -k)
    N = mul(hinv, sub(A, head))          # nilpotent
    out = ONE
    term = ONE
    for _ in range(5):
        term = mul(term, neg(N))
        if is_zero(term):
            break
        out = add(out, term)
    res = mul(out, hinv)
    assert is_zero(sub(mul(res, A), ONE)), 'inverse failed'
    return res


# --------------------------------------------------------------- functions
# A superfunction is an element with k = 0 and S subset of (eta_1, eta_2).

def d_y(A):
    return clean({key: sp.diff(c, y) for key, c in A.items()})


def d_eta(i, A):
    """Left derivative with respect to eta_i (i = 1, 2)."""
    g = ETA[i - 1]
    out = {}
    for (k, S), c in A.items():
        if g not in S:
            continue
        pos = S.index(g)
        rest = tuple(x for x in S if x != g)
        out[(k, rest)] = out.get((k, rest), 0) + (-1) ** pos * c
    return clean(out)


# ------------------------------------------------------- the superdiffeo F

def generic_F(names=('f0', 'f12', 'a1', 'a2', 'b1', 'b2')):
    f0, f12, a1, a2, b1, b2 = (sp.Function(n)(y) for n in names)
    f = add(mono(f0), mono(f12, 0, (E1, E2)))
    phi = add(mono(a1, 0, (E1,)), mono(a2, 0, (E2,)))
    psi = add(mono(b1, 0, (E1,)), mono(b2, 0, (E2,)))
    return f, phi, psi


def pullback_dx(f):
    """F^*(dx) = d_y(f) dy + d_{eta_1}(f) d eta_1 + d_{eta_2}(f) d eta_2."""
    out = mul(d_y(f), mono(1, 1))
    for i in (1, 2):
        out = add(out, mul(d_eta(i, f), mono(1, 0, (DETA[i - 1],))))
    return out


def berezinian(f, phi, psi):
    """Ber of the Jacobian of F, in the convention Ber = det(A - B D^{-1} C)/det D.

    A = (d_y f), B = (d_{eta_j} f), C = (d_y phi, d_y psi)^t,
    D = (d_{eta_j} phi ; d_{eta_j} psi).
    """
    A = d_y(f)
    B = [d_eta(j, f) for j in (1, 2)]
    C = [d_y(phi), d_y(psi)]
    D = [[d_eta(j, phi) for j in (1, 2)], [d_eta(j, psi) for j in (1, 2)]]
    # D is an even 2x2 matrix over the (supercommutative) even part: ordinary algebra
    det_D = sub(mul(D[0][0], D[1][1]), mul(D[0][1], D[1][0]))
    Dinv = [[mul(inv(det_D), D[1][1]), neg(mul(inv(det_D), D[0][1]))],
            [neg(mul(inv(det_D), D[1][0])), mul(inv(det_D), D[0][0])]]
    # A - B D^{-1} C
    corr = {}
    for j in (1, 2):
        for k in (1, 2):
            corr = add(corr, mul(mul(B[j - 1], Dinv[j - 1][k - 1]), C[k - 1]))
    schur = sub(A, corr)
    return mul(schur, inv(det_D))


# ------------------------------------------------- the Schwarzian, twice over

def theta_element():
    """The element  dy^{-1} d eta_1 d eta_2  of Phi~."""
    return mono(1, -1, (D1, D2))


def pullback_theta(f, phi, psi):
    """Ber(J F) * F^*(dx)^{-1} F^*(d zeta_1) F^*(d zeta_2)."""
    Fdx = pullback_dx(f)
    Fd1 = pullback_dx(phi)
    Fd2 = pullback_dx(psi)
    body = mul(mul(inv(Fdx), Fd1), Fd2)
    return mul(body, berezinian(f, phi, psi))


def radul_SW(f, phi, psi):
    """Radul's printed formula, p. 54:

    S_W(F) = v (x) [ dy (fd^{-1} phid psid)
                   + d eta_1 (fd^{-1} phi_{eta_1} psid - fd^{-1} phid psi_{eta_1}
                              - fd^{-2} f_{eta_1} phid psid)
                   + d eta_2 (same with eta_2) ] Ber(J(F)),
    where a dot is d/dy.  The coefficients are written to the RIGHT of the
    differential, as printed.
    """
    fd, phid, psid = d_y(f), d_y(phi), d_y(psi)
    fdi = inv(fd)
    fdi2 = mul(fdi, fdi)
    out = mul(mono(1, 1), mul(fdi, mul(phid, psid)))
    for j in (1, 2):
        c = sub(mul(fdi, mul(d_eta(j, phi), psid)),
                mul(fdi, mul(phid, d_eta(j, psi))))
        c = sub(c, mul(fdi2, mul(d_eta(j, f), mul(phid, psid))))
        out = add(out, mul(mono(1, 0, (DETA[j - 1],)), c))
    return mul(out, berezinian(f, phi, psi))
