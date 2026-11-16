"""The contact Lie SUPERalgebra k(2n+1|m) and its weighted densities, exactly.

Coordinates: t and p_1..p_n, q_1..q_n even, xi_1..xi_m odd;

    alpha = dt + (1/2) sum_i (p_i dq_i - q_i dp_i) + sum_{a,b} G_{ab} xi^a d xi^b,

G the (symmetric, invertible) Gram matrix of the form on the ODD part of the
distribution.  Two shapes of it are used here:

    G = 1     the orthonormal shape,  sum_j xi_j d xi_j;
    G = Pi    the SYMMETRIC (hyperbolic) shape,  sum_j (xi_j d eta_j + eta_j d xi_j)
              + theta d theta if m is odd,

odd coordinates in the second numbered xi_1,eta_1,...,xi_k,eta_k[,theta].  The two are
related by the change theta_1 = (xi + eta)/2 . ... over Kee = Cee, so the answers must
agree; that agreement is the calibration of this file (test_kmix.py, khyp.py).  The
symmetric shape is the one in which the maximal torus of k_0 is diagonal, i.e. in which
the frame operators are weight vectors -- see torus.py.

Generating function normalized by f = alpha(K_f), so that L_{K_f} alpha = f_t alpha:

    K_f = A d_t + sum_i B^i d_{p_i} + sum_i C^i d_{q_i} + sum_a Th^a d_{xi_a},
    A    = f - (1/2)(E_{pq} + E_xi)(f),
    B^i  = (1/2) p_i f_t - f_{q_i},        C^i = f_{p_i} + (1/2) q_i f_t,
    Th^a = -(1/2)(-1)^{p(f)} ( sum_b (G^{-1})^{ab} d_{xi_b} f - xi_a f_t ),

coefficients on the left, d_{xi} the LEFT odd derivative.  This is contactn.py and
super.py at once: m = 0 gives the first, n = 0 the second, and (n,m) = (1,1) is the
algebra k(3|1) whose zero part is Kee z + osp(1|2).

Contact (Lagrange) bracket {f,g} = K_f(g) - f_t g, and [K_f,K_g] = K_{{f,g}} with the
SUPER commutator.  Grading: deg t = 2, deg p = deg q = deg xi = 1;
k_0 = Kee z + osp(m|2n), z = K_{2t}.

Frame (right-invariant, commuting with k_{-2}+k_{-1}):

    e_{2i-1} = d_{p_i} + (q_i/2) d_t,   e_{2i} = (p_i/2) d_t - d_{q_i}   (even, W = 1),
    D_a      = d_{xi_a} - sum_c G_{ac} xi^c d_t                          (odd,  W = 1),
    R        = d_t = K_1                                                 (even, W = 2),

with [e_a,e_b] = varpi_{ab} R,  {D_a,D_b} = -2 G_{ab} R,  [e_a,D_b] = 0.  In the
symmetric shape D_xi^2 = D_eta^2 = 0 and {D_xi,D_eta} = -2R: the odd frame operators
are then weight vectors of the maximal torus, which is what they are not for G = 1.

Weighted densities F_lam: L_{K_f} F = K_f(F) + lam f_t F.  (weightsmix.py computes the
space of admissible multipliers: it is 1-dimensional whenever n >= 1, so a density over
k(2n+1|m) with n >= 1 carries ONE weight, unlike k(1|2).)

Everything is exact rational arithmetic; test_kmix.py checks every claim above and
reproduces contact.py (m=0, n=1) and super.py (n=0) term by term.
"""

from fractions import Fraction as Fr
from itertools import combinations, combinations_with_replacement

HALF = Fr(1, 2)


# ------------------------------------------------------------------ functions
# A function is a dict {(even exponent tuple of length 2n+1, sorted odd index tuple):
# Fraction}; index 0 = t, 1..n = p, n+1..2n = q; odd indices run 1..m.

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
            nv = out.get(k, Fr(0)) + v
            if nv:
                out[k] = nv
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
    for (e1, S1), c1 in f.items():
        for (e2, S2), c2 in g.items():
            S, sgn = _sort(S1 + S2)
            if sgn == 0:
                continue
            key = (tuple(a + b for a, b in zip(e1, e2)), S)
            nv = out.get(key, Fr(0)) + sgn * c1 * c2
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
    return out


def parity(f):
    ps = {len(S) % 2 for (e, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity")
    return ps.pop() if ps else 0


def identity_gram(m):
    return [[Fr(1) if a == b else Fr(0) for b in range(m)] for a in range(m)]


def hyperbolic_gram(m):
    """The SYMMETRIC shape: xi_1,eta_1,...,xi_k,eta_k[,theta], i.e. the odd part of
    alpha is sum_j (xi_j d eta_j + eta_j d xi_j) [+ theta d theta]."""
    G = [[Fr(0)] * m for _ in range(m)]
    for j in range(m // 2):
        G[2 * j][2 * j + 1] = G[2 * j + 1][2 * j] = Fr(1)
    if m % 2:
        G[m - 1][m - 1] = Fr(1)
    return G


def _inverse(G):
    """Exact inverse of a small symmetric rational matrix."""
    m = len(G)
    A = [list(map(Fr, row)) + [Fr(1) if i == j else Fr(0) for j in range(m)]
         for i, row in enumerate(G)]
    for c in range(m):
        piv = next((r for r in range(c, m) if A[r][c]), None)
        if piv is None:
            raise ValueError("the Gram matrix is degenerate")
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(m):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[m:] for row in A]


class KMix:
    """k(2n+1|m).

    `gram` is the Gram matrix of the form on the odd part of the distribution:
    None (= the orthonormal shape, the default, which is what super.py and the first
    version of this file use), or hyperbolic_gram(m) for the symmetric shape.
    """

    def __init__(self, n, m, gram=None):
        self.n, self.m = n, m
        self.nev = 2 * n + 1
        self.dim_ev = 2 * n            # even directions of the distribution
        self.dim_od = m
        self.G = identity_gram(m) if gram is None else [list(map(Fr, r)) for r in gram]
        if len(self.G) != m or any(len(r) != m for r in self.G):
            raise ValueError("the Gram matrix must be m x m")
        if any(self.G[a][b] != self.G[b][a] for a in range(m) for b in range(m)):
            raise ValueError("the Gram matrix must be symmetric")
        self.Ginv = _inverse(self.G) if m else []
        self.shape = "orthonormal" if gram is None else "symmetric"

    def gram(self, a, b):
        """G_{ab}, indices 1-based."""
        return self.G[a - 1][b - 1]

    def gram_up(self, a, b):
        """(G^{-1})^{ab}, indices 1-based."""
        return self.Ginv[a - 1][b - 1]

    # ---------------------------------------------------------- constructors
    def zero(self):
        return {}

    def const(self, c=1):
        c = Fr(c)
        return {((0,) * self.nev, ()): c} if c else {}

    def evar(self, i, c=1):
        """The even coordinate of index i (0 = t, 1..n = p, n+1..2n = q)."""
        e = [0] * self.nev
        e[i] = 1
        return {(tuple(e), ()): Fr(c)}

    def ovar(self, j, c=1):
        return {((0,) * self.nev, (j,)): Fr(c)}

    def ip(self, i):
        return i

    def iq(self, i):
        return self.n + i

    # ------------------------------------------------------------ derivatives
    def de(self, i, f):
        out = {}
        for (e, S), c in f.items():
            if e[i] == 0:
                continue
            ee = list(e)
            ee[i] -= 1
            key = (tuple(ee), S)
            nv = out.get(key, Fr(0)) + c * e[i]
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
        return out

    def do(self, j, f):
        """Left odd derivative d/d xi_j."""
        out = {}
        for (e, S), c in f.items():
            if j not in S:
                continue
            pos = S.index(j)
            key = (e, S[:pos] + S[pos + 1:])
            nv = out.get(key, Fr(0)) + (-1) ** pos * c
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
        return out

    def dt(self, f):
        return self.de(0, f)

    def euler(self, f):
        """E = sum p d_p + q d_q + xi d_xi: multiplication by the degree in p,q,xi."""
        out = {}
        for (e, S), c in f.items():
            d = sum(e[1:]) + len(S)
            if d:
                out[(e, S)] = d * c
        return out

    # ------------------------------------------------------- contact fields
    def Kf_coeffs(self, f):
        n, m = self.n, self.m
        ft = self.dt(f)
        A = add(f, scal(-HALF, self.euler(f)))
        B, C = [], []
        for i in range(1, n + 1):
            B.append(add(scal(HALF, mul(self.evar(self.ip(i)), ft)),
                         scal(-1, self.de(self.iq(i), f))))
            C.append(add(self.de(self.ip(i), f),
                         scal(HALF, mul(self.evar(self.iq(i)), ft))))
        p = parity(f) if f else 0
        Th = []
        for a in range(1, m + 1):
            raise_ = {}
            for b in range(1, m + 1):
                g = self.gram_up(a, b)
                if g:
                    raise_ = add(raise_, scal(g, self.do(b, f)))
            Th.append(scal(-HALF * (-1) ** p,
                           add(raise_, scal(-1, mul(self.ovar(a), ft)))))
        return A, B, C, Th

    def Kf(self, f, h):
        A, B, C, Th = self.Kf_coeffs(f)
        out = mul(A, self.dt(h))
        for i in range(1, self.n + 1):
            out = add(out, mul(B[i - 1], self.de(self.ip(i), h)),
                      mul(C[i - 1], self.de(self.iq(i), h)))
        for j in range(1, self.m + 1):
            out = add(out, mul(Th[j - 1], self.do(j, h)))
        return out

    def kbracket(self, f, g):
        return add(self.Kf(f, g), scal(-1, mul(self.dt(f), g)))

    # -------------------------------------------------------------- the frame
    def e(self, a, h):
        i = (a + 1) // 2
        if a % 2 == 1:
            return add(self.de(self.ip(i), h),
                       scal(HALF, mul(self.evar(self.iq(i)), self.dt(h))))
        return add(scal(HALF, mul(self.evar(self.ip(i)), self.dt(h))),
                   scal(-1, self.de(self.iq(i), h)))

    def D(self, a, h):
        """D_a = d_{xi_a} - sum_c G_{ac} xi^c d_t."""
        out = self.do(a, h)
        ht = self.dt(h)
        if ht:
            for c in range(1, self.m + 1):
                g = self.gram(a, c)
                if g:
                    out = add(out, scal(-g, mul(self.ovar(c), ht)))
        return out

    def R(self, h):
        return self.dt(h)

    def varpi_low(self, a, b):
        """[e_a,e_b] = varpi_{ab} R  (even slots)."""
        if a == b:
            return Fr(0)
        if (a + 1) // 2 != (b + 1) // 2:
            return Fr(0)
        return Fr(1) if a % 2 == 1 else Fr(-1)

    def varpi_up(self, a, b):
        return -self.varpi_low(a, b)

    # ---------------------------------------------------------- graded pieces
    def monomials_of_degree(self, d):
        """t^a * (monomial of degree d-2a in p,q,xi)."""
        out = []
        for a in range(d // 2 + 1):
            rest = d - 2 * a
            for k in range(min(rest, self.m) + 1):
                for S in combinations(range(1, self.m + 1), k):
                    for combo in combinations_with_replacement(
                            range(1, 2 * self.n + 1), rest - k):
                        e = [0] * self.nev
                        e[0] = a
                        for c in combo:
                            e[c] += 1
                        out.append({(tuple(e), S): Fr(1)})
        return out

    def k_piece(self, i):
        return self.monomials_of_degree(i + 2)


class Dens:
    """F_lam over k(2n+1|m), and F_{lam,nu} over k(1|2).

    k(1|2) is the one contact superalgebra whose k_0 has a two-dimensional space of
    characters, so its densities carry a pair of weights; the second multiplier is the
    frame operator D_1D_2 applied to the generating function (mixchecks.py).  In the
    symmetric shape D_1 = D_xi and D_2 = D_eta, so that multiplier is the eigenvalue of
    the toral element K_{2 xi eta} -- a weight in the literal sense (torus.py)."""

    def __init__(self, alg, lam, nu=0):
        self.alg = alg
        self.lam = Fr(lam)
        self.nu = Fr(nu)
        if self.nu and (alg.n, alg.m) != (0, 2):
            raise ValueError("a second weight exists only over k(1|2)")

    def __repr__(self):
        return "F_%s" % self.lam if not self.nu else "F_{%s,%s}" % (self.lam, self.nu)

    def act(self, f, F):
        out = add(self.alg.Kf(f, F),
                  scal(self.lam, mul(self.alg.dt(f), F)))
        if self.nu:
            out = add(out, scal(self.nu, mul(self.alg.D(1, self.alg.D(2, f)), F)))
        return out
