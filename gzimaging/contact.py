"""The contact Lie algebra k(3) and its tensor modules T(k,lambda), exactly.

Coordinates (t,p,q) on R^3, contact form in the SYMMETRIC shape (alpha_1 halved):

    alpha = dt + (1/2)(p dq - q dp),        d alpha = dp ^ dq.

The halving normalizes the generating function to  f = alpha(K_f)  (alpha_1 = 2 alpha
goes with alpha_1(K_f) = 2f); then

    K_f = (1 - E/2)(f) d_t + H_f + (1/2) f_t E,
    E = p d_p + q d_q  (Euler),   H_f = f_p d_q - f_q d_p  (Hamiltonian),

i.e. componentwise  K_f = A d_t + B d_p + C d_q  with

    A = f - (p f_p + q f_q)/2,   B = (1/2) p f_t - f_q,   C = f_p + (1/2) q f_t,

and  L_{K_f} alpha = f_t alpha.  In particular K_1 = d_t and {p,q} = 1.

Contact (Lagrange) bracket   {f,g} = K_f(g) - f_t g,   [K_f,K_g] = K_{{f,g}}.

Grading:  deg t = 2, deg p = deg q = 1;  k(3) = \oplus_{i>=-2} k_i, k_i = {deg = i+2}.
The grading element is K_{2t} (in the asymmetric shape it was K_{2t-pq}).

The distribution D = ker alpha is framed by the RIGHT-invariant fields, i.e. by the
K_f of k_{-1} corrected by a Reeb term, which is what turns a left translation into
a right one:

    e_1 = q R - K_q = d_p + (q/2) d_t,
    e_2 = p R - K_p = (p/2) d_t - d_q,      R = K_1 = d_t = [e_1,e_2] (Reeb),

symmetric under p <-> q -- one of the things the symmetric shape of alpha buys.  These
are NOT contact fields (an operator invariant under the Heisenberg algebra k_{-2}+k_{-1}
has constant coefficients in the frame commuting with it, and the K_f do not commute
with each other).  Finally,  [K_f, e_a] = sum_b C_a^b e_b  stays in D because
alpha(e_a)=0 and L_{K_f}alpha ~ alpha.

Tensor module  T(k,lam) = sections of S^k D^* (x) L^lam,  L = the line bundle trivialized
by alpha (so Vol = L^2 on a 3-fold).  An element is a polynomial in (t,p,q) and in the
auxiliary u = (u^1,u^2) (the components of a D-vector), homogeneous of degree k in u:

    L_{K_f} F = K_f(F) - (sum_{a,b} C_a^b u^a d_{u^b}) F + lam f_t F.

Everything below is exact rational arithmetic; nothing is asserted that test_contact.py
does not check.
"""

from fractions import Fraction as Fr
from itertools import product

# ---------------------------------------------------------------- polynomials
# A polynomial in t,p,q is a dict {(a,b,c): Fraction}; zero coefficients are pruned.

VARS = ("t", "p", "q")


def pzero():
    return {}


def pconst(c):
    c = Fr(c)
    return {(0, 0, 0): c} if c else {}


def pmono(a, b, c, coef=1):
    coef = Fr(coef)
    return {(a, b, c): coef} if coef else {}


def padd(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            n = out.get(k, Fr(0)) + v
            if n:
                out[k] = n
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
            k = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2])
            n = out.get(k, Fr(0)) + v1 * v2
            if n:
                out[k] = n
            elif k in out:
                del out[k]
    return out


def pdiff(p, i):
    """d/d(t,p,q)[i]."""
    out = {}
    for k, v in p.items():
        if k[i] == 0:
            continue
        kk = list(k)
        kk[i] -= 1
        key = tuple(kk)
        n = out.get(key, Fr(0)) + v * k[i]
        if n:
            out[key] = n
        elif key in out:
            del out[key]
    return out


DT = lambda p: pdiff(p, 0)
DP = lambda p: pdiff(p, 1)
DQ = lambda p: pdiff(p, 2)

P = pmono(0, 1, 0)  # the coordinate p
Q = pmono(0, 0, 1)  # the coordinate q
HALF = Fr(1, 2)


# ------------------------------------------------------- contact vector field
def Kf_coeffs(f):
    """(A,B,C) with K_f = A d_t + B d_p + C d_q,  alpha = dt + (p dq - q dp)/2."""
    A = padd(f, pscal(-HALF, padd(pmul(P, DP(f)), pmul(Q, DQ(f)))))
    B = padd(pscal(HALF, pmul(P, DT(f))), pscal(-1, DQ(f)))
    C = padd(DP(f), pscal(HALF, pmul(Q, DT(f))))
    return A, B, C


def Kf(f, h):
    """The vector field K_f applied to a polynomial h."""
    A, B, C = Kf_coeffs(f)
    return padd(pmul(A, DT(h)), pmul(B, DP(h)), pmul(C, DQ(h)))


def kbracket(f, g):
    """The contact bracket {f,g} = K_f(g) - f_t g."""
    return padd(Kf(f, g), pscal(-1, pmul(DT(f), g)))


# ------------------------------------------------------- the invariant frame
# e_1 = q R - K_q = d_p + (q/2) d_t ,   e_2 = p R - K_p = (p/2) d_t - d_q ,  R = K_1.
def e1(h):
    return padd(DP(h), pscal(HALF, pmul(Q, DT(h))))


def e2(h):
    return padd(pscal(HALF, pmul(P, DT(h))), pscal(-1, DQ(h)))


def reeb(h):
    return DT(h)


FRAME = {"e1": e1, "e2": e2, "R": reeb}


def Cmat(f):
    """C_a^b with [K_f, e_a] = sum_b C_a^b e_b.  Returns ((C11,C12),(C21,C22)).

    [K_f,e_a] has components (K_f(x_a) - e_a(A), -e_a(B), -e_a(C)) in (d_t,d_p,d_q),
    where x_a is the d_t-coefficient of e_a; a field c_1 e_1 + c_2 e_2 has d_p-component
    c_1 and d_q-component -c_2, whence the read-off below (the d_t-component is then
    automatic -- checked in test_contact.py).
    """
    A, B, C = Kf_coeffs(f)
    C11, C12 = pscal(-1, e1(B)), e1(C)
    C21, C22 = pscal(-1, e2(B)), e2(C)
    return ((C11, C12), (C21, C22))


# --------------------------------------------------------------- the modules
# An element of T(k,lam) is a dict {(a,b,c,i): Fraction}: the monomial
# t^a p^b q^c u_1^i u_2^(k-i).


def mzero():
    return {}


def madd(*ms):
    out = {}
    for m in ms:
        for k, v in m.items():
            n = out.get(k, Fr(0)) + v
            if n:
                out[k] = n
            elif k in out:
                del out[k]
    return out


def mscal(c, m):
    c = Fr(c)
    if not c:
        return {}
    return {k: c * v for k, v in m.items()}


def component(F, i):
    """The coefficient of u_1^i u_2^(k-i): a polynomial in (t,p,q)."""
    return {k[:3]: v for k, v in F.items() if k[3] == i}


def from_components(comps):
    """comps: dict i -> polynomial."""
    out = {}
    for i, p in comps.items():
        for k, v in p.items():
            key = k + (i,)
            n = out.get(key, Fr(0)) + v
            if n:
                out[key] = n
            elif key in out:
                del out[key]
    return out


def apply_poly_op(F, op):
    """Apply a (t,p,q)-operator componentwise."""
    return from_components({i: op(component(F, i)) for i in set(k[3] for k in F)})


class Module:
    """T(k,lam)."""

    def __init__(self, k, lam):
        self.k = k
        self.lam = Fr(lam)

    def __repr__(self):
        return "T(%d,%s)" % (self.k, self.lam)

    def basis(self, maxdeg):
        """Monomials with t,p,q-degree (ordinary, not weighted) at most maxdeg."""
        out = []
        for a in range(maxdeg + 1):
            for b in range(maxdeg + 1 - a):
                for c in range(maxdeg + 1 - a - b):
                    for i in range(self.k + 1):
                        out.append({(a, b, c, i): Fr(1)})
        return out

    def act(self, f, F):
        """L_{K_f} F."""
        k, lam = self.k, self.lam
        ft = DT(f)
        (C11, C12), (C21, C22) = Cmat(f)
        out = {}

        def acc(key, val):
            if not val:
                return
            n = out.get(key, Fr(0)) + val
            if n:
                out[key] = n
            elif key in out:
                del out[key]

        for i in range(k + 1):
            comp = component(F, i)
            if not comp:
                continue
            # K_f(comp) + lam f_t comp   -- stays in the same u-monomial
            main = padd(Kf(f, comp), pscal(lam, pmul(ft, comp)))
            # - (C_1^1 i + C_2^2 (k-i)) comp   -- same u-monomial
            diag = padd(pscal(i, C11), pscal(k - i, C22))
            main = padd(main, pscal(-1, pmul(diag, comp)))
            for key, v in main.items():
                acc(key + (i,), v)
            # - C_1^2 (k-i) * (u_1^{i+1} u_2^{k-i-1})
            if k - i > 0:
                term = pscal(-(k - i), pmul(C12, comp))
                for key, v in term.items():
                    acc(key + (i + 1,), v)
            # - C_2^1 i * (u_1^{i-1} u_2^{k-i+1})
            if i > 0:
                term = pscal(-i, pmul(C21, comp))
                for key, v in term.items():
                    acc(key + (i - 1,), v)
        return out


# ------------------------------------------------------------ graded pieces
def monomials_of_weighted_degree(d):
    """t^a p^b q^c with 2a+b+c = d."""
    out = []
    for a in range(d // 2 + 1):
        rest = d - 2 * a
        for b in range(rest + 1):
            out.append(pmono(a, b, rest - b))
    return out


def k_piece(i):
    """A basis of k_i = {generating functions of weighted degree i+2}."""
    return monomials_of_weighted_degree(i + 2)
