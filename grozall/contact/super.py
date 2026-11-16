"""The contact Lie superalgebra k(1|n) and its weighted densities, exactly.

Coordinates: t even, xi_1..xi_n odd;  contact form  alpha = dt + sum xi_i d xi_i,
generating function normalized by  f = alpha(K_f), so that  L_{K_f} alpha = f_t alpha.
With  E = sum xi_i d_{xi_i}  (Euler) this makes

    K_f = (1/2) [ (2-E)(f) d_t  -  (-1)^{p(f)} sum_i ( d_{xi_i}f - xi_i f_t ) d_{xi_i} ],

i.e. half the field of the house convention (whose alpha_1(K_f) = 2f); the halving is
what makes  {f,g} = K_f(g) - f_t g  a bracket with  [K_f,K_g] = K_{{f,g}}  and the
weights normalized as in the even case (Vol = F_{(2-n)/2}, adjoint = F_{-1}).

Grading:  deg t = 2, deg xi = 1;  k_{-2} = <K_1>, k_{-1} = <K_{xi_i}>,
k_0 = <K_t, K_{xi_i xi_j}> = Kee z + o(n),  z = K_{2t}.

Frame (right-invariant, i.e. commuting with k_{-2}+k_{-1}):

    D_i = d_{xi_i} - xi_i d_t     (odd, weighted order 1),
    R   = d_t = K_1               (even, weighted order 2).

Weighted densities: F_lam = functions with  L_{K_f}F = K_f(F) + lam f_t F.  For n = 2
the zero part k_0 is 2-dimensional abelian, so there is a SECOND weight; the module is
then  L_{K_f}F = K_f(F) + lam f_t F + nu D_1D_2(f) F,  the second multiplier being the
frame operator D_1D_2 applied to the generating function (weights.py derives the space
of multipliers: it is 1-dimensional for every n except n = 2, where it is 2-dimensional).
Everything asserted here is checked in test_super.py.
"""

from fractions import Fraction as Fr
from itertools import combinations

# ------------------------------------------------------------------ functions
# A function is a dict {(a, S): Fraction}: the monomial t^a * xi_S, S a sorted tuple.


def zero():
    return {}


def mono(a=0, S=(), c=1):
    c = Fr(c)
    S, sgn = _sort(tuple(S))
    if sgn == 0 or c == 0:
        return {}
    return {(a, S): c * sgn}


def _sort(S):
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


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            n = out.get(k, Fr(0)) + v
            if n:
                out[k] = n
            elif k in out:
                del out[k]
    return out


def scal(c, f):
    c = Fr(c)
    if not c:
        return {}
    return {k: c * v for k, v in f.items()}


def mul(f, g):
    out = {}
    for (a1, S1), c1 in f.items():
        for (a2, S2), c2 in g.items():
            S, sgn = _sort(S1 + S2)
            if sgn == 0:
                continue
            key = (a1 + a2, S)
            n = out.get(key, Fr(0)) + sgn * c1 * c2
            if n:
                out[key] = n
            elif key in out:
                del out[key]
    return out


def parity(f):
    ps = {len(S) % 2 for (a, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity")
    return ps.pop() if ps else 0


def d_t(f):
    out = {}
    for (a, S), c in f.items():
        if a == 0:
            continue
        key = (a - 1, S)
        n = out.get(key, Fr(0)) + a * c
        if n:
            out[key] = n
        elif key in out:
            del out[key]
    return out


def d_xi(i, f):
    """Left odd derivative d/d xi_i."""
    out = {}
    for (a, S), c in f.items():
        if i not in S:
            continue
        pos = S.index(i)
        sgn = (-1) ** pos
        key = (a, S[:pos] + S[pos + 1:])
        n = out.get(key, Fr(0)) + sgn * c
        if n:
            out[key] = n
        elif key in out:
            del out[key]
    return out


def euler(f):
    """E = sum xi_i d_{xi_i}: multiplication by the xi-degree."""
    return {k: len(k[1]) * v for k, v in f.items() if len(k[1])}


XI = lambda i: mono(0, (i,))


# ------------------------------------------------------- the contact fields
HALF = Fr(1, 2)


def Kf_coeffs(n, f):
    """(A, (B_1..B_n)) with K_f = A d_t + sum B_i d_{xi_i}, coefficients on the left."""
    A = scal(HALF, add(scal(2, f), scal(-1, euler(f))))
    p = parity(f)
    B = []
    for i in range(1, n + 1):
        Bi = scal(-HALF * (-1) ** p, add(d_xi(i, f), scal(-1, mul(XI(i), d_t(f)))))
        B.append(Bi)
    return A, B


def Kf(n, f, h):
    """K_f applied to h."""
    A, B = Kf_coeffs(n, f)
    out = mul(A, d_t(h))
    for i in range(1, n + 1):
        out = add(out, mul(B[i - 1], d_xi(i, h)))
    return out


def kbracket(n, f, g):
    """{f,g} = K_f(g) - f_t g."""
    return add(Kf(n, f, g), scal(-1, mul(d_t(f), g)))


# -------------------------------------------------------------- the frame
def D(i, h):
    """D_i = d_{xi_i} - xi_i d_t."""
    return add(d_xi(i, h), scal(-1, mul(XI(i), d_t(h))))


def R(h):
    return d_t(h)


def apply_frame(S, s, h):
    """D_S R^s applied to h, the D's in the order given by S (increasing)."""
    out = h
    for _ in range(s):
        out = R(out)
        if not out:
            return {}
    for i in reversed(S):          # D_{i1} D_{i2} ... applied right to left
        out = D(i, out)
        if not out:
            return {}
    return out


# ------------------------------------------------------------- the modules
class Dens:
    """F_lam (n != 2), or F_{lam,nu} (n = 2, the second weight of the o(2) in k_0)."""

    def __init__(self, n, lam, nu=0):
        self.n = n
        self.lam = Fr(lam)
        self.nu = Fr(nu)

    def __repr__(self):
        return "F_%s" % self.lam if self.n != 2 else "F_{%s,%s}" % (self.lam, self.nu)

    def act(self, f, F):
        out = add(Kf(self.n, f, F), scal(self.lam, mul(d_t(f), F)))
        if self.nu:
            # the second multiplier, DERIVED in weights.py: it is D_1D_2 of the
            # generating function -- not d_{xi_1}d_{xi_2}, which is not a cocycle.
            c = D(1, D(2, f))
            out = add(out, scal(self.nu, mul(c, F)))
        return out


# ---------------------------------------------------------- graded pieces
def monomials_of_degree(n, d):
    """t^a xi_S with 2a + |S| = d."""
    out = []
    for a in range(d // 2 + 1):
        k = d - 2 * a
        if k > n:
            continue
        for S in combinations(range(1, n + 1), k):
            out.append(mono(a, S))
    return out


def k_piece(n, i):
    return monomials_of_degree(n, i + 2)
