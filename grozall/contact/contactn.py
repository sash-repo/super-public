"""The contact Lie algebra k(2n+1) in ANY dimension, and its tensor modules T(k,lam).

This is contact.py with n free.  Coordinates

    t, p_1..p_n, q_1..q_n,          alpha = dt + (1/2) sum_i (p_i dq_i - q_i dp_i),

so d alpha = sum_i dp_i ^ dq_i and the generating function is normalized by f = alpha(K_f):

    K_f = (1 - E/2)(f) d_t + H_f + (1/2) f_t E,
    E = sum_i (p_i d_{p_i} + q_i d_{q_i}),   H_f = sum_i (f_{p_i} d_{q_i} - f_{q_i} d_{p_i}),

componentwise  K_f = A d_t + sum_i B^i d_{p_i} + sum_i C^i d_{q_i}  with

    A = f - (1/2) E(f),   B^i = (1/2) p_i f_t - f_{q_i},   C^i = f_{p_i} + (1/2) q_i f_t,

and L_{K_f} alpha = f_t alpha.  Contact (Lagrange) bracket {f,g} = K_f(g) - f_t g.

Grading: deg t = 2, deg p = deg q = 1; k_i = {generating functions of degree i+2};
k_0 = csp(2n) = sp(2n) + Kee z with z = K_{2t}.

The distribution D = ker alpha is framed by the RIGHT-invariant fields

    e_{2i-1} = q_i R - K_{q_i} = d_{p_i} + (q_i/2) d_t,
    e_{2i}   = p_i R - K_{p_i} = (p_i/2) d_t - d_{q_i},        R = K_1 = d_t,

which commute with k_{-2}+k_{-1} and satisfy [e_a,e_b] = varpi_{ab} R with
varpi_{2i-1,2i} = 1.  Its inverse varpi^{ab} (varpi^{2i-1,2i} = -1) raises indices.

Tensor module T(k,lam) = sections of S^k D^* (x) L^lam: an element is a polynomial in
(t,p,q) and in the fiber variables u^1..u^{2n}, homogeneous of degree k in u, with

    L_{K_f} F = K_f(F) - (sum_{a,b} C_a^b u^a d_{u^b}) F + lam f_t F,
    [K_f,e_a] = sum_b C_a^b e_b.

Exact rational arithmetic throughout; test_contactn.py checks every claim above and, for
n = 1, reproduces the answers of contact.py.
"""

from fractions import Fraction as Fr
from itertools import combinations_with_replacement

HALF = Fr(1, 2)


# ---------------------------------------------------------------- polynomials
# A polynomial in (t,p_1..p_n,q_1..q_n) is a dict {exponent tuple: Fraction},
# the tuple of length 2n+1 with index 0 = t, 1..n = p, n+1..2n = q.

def pzero():
    return {}


def pconst(c, n):
    c = Fr(c)
    return {(0,) * (2 * n + 1): c} if c else {}


def pvar(i, n):
    """The coordinate of index i (0 = t, 1..n = p, n+1..2n = q)."""
    e = [0] * (2 * n + 1)
    e[i] = 1
    return {tuple(e): Fr(1)}


def padd(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            nv = out.get(k, Fr(0)) + v
            if nv:
                out[k] = nv
            elif k in out:
                del out[k]
    return out


def pscal(c, p):
    c = Fr(c)
    if not c:
        return {}
    return {k: c * v for k, v in p.items()}


def pmul(p1, p2):
    out = {}
    for k1, v1 in p1.items():
        for k2, v2 in p2.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            nv = out.get(k, Fr(0)) + v1 * v2
            if nv:
                out[k] = nv
            elif k in out:
                del out[k]
    return out


def pdiff(p, i):
    out = {}
    for k, v in p.items():
        if k[i] == 0:
            continue
        kk = list(k)
        kk[i] -= 1
        key = tuple(kk)
        nv = out.get(key, Fr(0)) + v * k[i]
        if nv:
            out[key] = nv
        elif key in out:
            del out[key]
    return out


class Contact:
    """The contact algebra k(2n+1) with all the structure attached to it."""

    def __init__(self, n):
        self.n = n
        self.nv = 2 * n + 1
        self.dim = 2 * n                       # of the distribution

    # ------------------------------------------------------------- indices
    def ip(self, i):
        """Index of p_i (i = 1..n)."""
        return i

    def iq(self, i):
        return self.n + i

    def dt(self, p):
        return pdiff(p, 0)

    def dp(self, i, p):
        return pdiff(p, self.ip(i))

    def dq(self, i, p):
        return pdiff(p, self.iq(i))

    # ------------------------------------------------------ contact fields
    def Kf_coeffs(self, f):
        """(A, [B^1..B^n], [C^1..C^n]) with K_f = A d_t + B^i d_{p_i} + C^i d_{q_i}."""
        n = self.n
        ft = self.dt(f)
        E = pzero()
        for i in range(1, n + 1):
            E = padd(E, pmul(pvar(self.ip(i), n), self.dp(i, f)),
                     pmul(pvar(self.iq(i), n), self.dq(i, f)))
        A = padd(f, pscal(-HALF, E))
        B, C = [], []
        for i in range(1, n + 1):
            B.append(padd(pscal(HALF, pmul(pvar(self.ip(i), n), ft)),
                          pscal(-1, self.dq(i, f))))
            C.append(padd(self.dp(i, f), pscal(HALF, pmul(pvar(self.iq(i), n), ft))))
        return A, B, C

    def Kf(self, f, h):
        A, B, C = self.Kf_coeffs(f)
        out = pmul(A, self.dt(h))
        for i in range(1, self.n + 1):
            out = padd(out, pmul(B[i - 1], self.dp(i, h)), pmul(C[i - 1], self.dq(i, h)))
        return out

    def kbracket(self, f, g):
        return padd(self.Kf(f, g), pscal(-1, pmul(self.dt(f), g)))

    # ------------------------------------------------------------ the frame
    def e(self, a, h):
        """e_a, a = 1..2n:  e_{2i-1} = d_{p_i} + (q_i/2) d_t,  e_{2i} = (p_i/2) d_t - d_{q_i}."""
        n = self.n
        i = (a + 1) // 2
        if a % 2 == 1:
            return padd(self.dp(i, h),
                        pscal(HALF, pmul(pvar(self.iq(i), n), self.dt(h))))
        return padd(pscal(HALF, pmul(pvar(self.ip(i), n), self.dt(h))),
                    pscal(-1, self.dq(i, h)))

    def R(self, h):
        return self.dt(h)

    def varpi_low(self, a, b):
        """[e_a,e_b] = varpi_{ab} R."""
        if a == b:
            return Fr(0)
        ia, ib = (a + 1) // 2, (b + 1) // 2
        if ia != ib:
            return Fr(0)
        return Fr(1) if a % 2 == 1 else Fr(-1)

    def varpi_up(self, a, b):
        """The inverse matrix: varpi^{2i-1,2i} = -1."""
        return -self.varpi_low(a, b)

    def Cmat(self, f):
        """C_a^b with [K_f,e_a] = sum_b C_a^b e_b; returned as a dict (a,b) -> poly."""
        A, B, C = self.Kf_coeffs(f)
        out = {}
        for a in range(1, 2 * self.n + 1):
            for j in range(1, self.n + 1):
                out[(a, 2 * j - 1)] = pscal(-1, self.e(a, B[j - 1]))
                out[(a, 2 * j)] = self.e(a, C[j - 1])
        return out

    # ---------------------------------------------------------- graded pieces
    def monomials_of_weighted_degree(self, d):
        """t^a * (monomial in p,q of degree d-2a)."""
        n = self.n
        out = []
        for a in range(d // 2 + 1):
            rest = d - 2 * a
            for combo in combinations_with_replacement(range(1, 2 * n + 1), rest):
                e = [0] * (2 * n + 1)
                e[0] = a
                for c in combo:
                    e[c] += 1
                out.append({tuple(e): Fr(1)})
        return out

    def k_piece(self, i):
        return self.monomials_of_weighted_degree(i + 2)


# ------------------------------------------------------------------- modules
# An element of T(k,lam) is a dict {(exponent tuple, fiber monomial): Fraction},
# the fiber monomial being a tuple of length 2n of u-exponents summing to k.

def fiber_monomials(dim, k):
    out = []
    for combo in combinations_with_replacement(range(dim), k):
        e = [0] * dim
        for c in combo:
            e[c] += 1
        out.append(tuple(e))
    return out


class Module:
    """T(k,lam) over k(2n+1)."""

    def __init__(self, alg, k, lam):
        self.alg = alg
        self.k = k
        self.lam = Fr(lam)
        self.fmons = fiber_monomials(alg.dim, k)

    def __repr__(self):
        return "T(%d,%s)" % (self.k, self.lam)

    def component(self, F, mu):
        return {key[0]: v for key, v in F.items() if key[1] == mu}

    def from_components(self, comps):
        out = {}
        for mu, p in comps.items():
            for k, v in p.items():
                key = (k, mu)
                nv = out.get(key, Fr(0)) + v
                if nv:
                    out[key] = nv
                elif key in out:
                    del out[key]
        return out

    def act(self, f, F):
        """L_{K_f} F = K_f(F) - (sum C_a^b u^a d_{u^b}) F + lam f_t F."""
        alg = self.alg
        ft = alg.dt(f)
        Cm = alg.Cmat(f)
        out = {}

        def acc(key, val):
            if not val:
                return
            nv = out.get(key, Fr(0)) + val
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]

        comps = {}
        for (e, mu), v in F.items():
            comps.setdefault(mu, {})[e] = comps.setdefault(mu, {}).get(e, Fr(0)) + v
        for mu, comp in comps.items():
            main = padd(alg.Kf(f, comp), pscal(self.lam, pmul(ft, comp)))
            for key, v in main.items():
                acc((key, mu), v)
            # - sum_{a,b} C_a^b u^a d_{u^b}:  u^b -> u^a lowers exponent b, raises a
            for b in range(1, alg.dim + 1):
                nb = mu[b - 1]
                if not nb:
                    continue
                for a in range(1, alg.dim + 1):
                    c = Cm[(a, b)]
                    if not c:
                        continue
                    nu = list(mu)
                    nu[b - 1] -= 1
                    nu[a - 1] += 1
                    nu = tuple(nu)
                    term = pscal(-nb, pmul(c, comp))
                    for key, v in term.items():
                        acc((key, nu), v)
        return out

    def basis_monomials(self, maxdeg):
        out = []
        nv = self.alg.nv
        for e in _exps(nv, maxdeg):
            for mu in self.fmons:
                out.append({(e, mu): Fr(1)})
        return out


def _exps(nv, maxdeg):
    """All exponent tuples of length nv with total degree <= maxdeg."""
    if nv == 0:
        yield ()
        return
    for d in range(maxdeg + 1):
        for rest in _exps(nv - 1, maxdeg - d):
            yield (d,) + rest
