"""
Finite contact transformations of the superstring S^{1|n}, as truncated flows,
for checking Radul's Sect. 5.2 (Stockholm Reports 1986-21, pp. 52-53 = his pp.
(13)-(14)).

Everything is exact: the ground ring is Q[t] adjoined two flow parameters e, d,
truncated at total degree ORD; the odd coordinates theta_1..theta_n are honest
Grassmann generators with one global order.

A finite contact diffeomorphism is G = exp(e K_g); its pull-back on functions is
the operator exp(e K_g), so no substitution routine is needed, and
    (G o H)^* = H^* o G^*.

Conventions are those of the text:
    alpha_1 = dt + sum theta_i d theta_i,
    K_f     = (2 - E)(f) d_t - (-1)^{p(f)} sum (d_{theta_i} f - theta_i d_t f) d_{theta_i},
    D_i     = theta_i d_t + d_{theta_i}   (Y-type, D_i^2 =  d_t),
    K_i     = theta_i d_t - d_{theta_i}   (X-type, K_i^2 = -d_t),
    m_G     = d_t(G^* t) + sum_i (G^* theta_i) d_t(G^* theta_i).
Radul's frame operator obeys theta_{zeta_i}^2 = -d_x, i.e. it is our K_i; which
of the two his formulas actually want is decided here, not assumed.
"""
import sympy as sp
from itertools import combinations

t, e, d = sp.symbols('t e d')
ORD = 3                      # truncation in the total degree in (e, d)


def set_order(k):
    global ORD
    ORD = k


# ---------------------------------------------------------------- the algebra
# An element is a dict {sorted tuple of odd indices: sympy expr in t, e, d}.

def trunc(x):
    x = sp.expand(x)
    if x == 0:
        return sp.Integer(0)
    out = 0
    for mono, c in sp.Poly(x, e, d).terms():
        if mono[0] + mono[1] <= ORD:
            out += c * e ** mono[0] * d ** mono[1]
    return sp.expand(out)


def clean(A):
    return {S: c for S, c in ((S, trunc(c)) for S, c in A.items()) if c != 0}


def num(c):
    return clean({(): sp.sympify(c)})


def th(i):
    return {(i,): sp.Integer(1)}


def eadd(*As):
    out = {}
    for A in As:
        for S, c in A.items():
            out[S] = out.get(S, 0) + c
    return clean(out)


def escal(c, A):
    return clean({S: sp.expand(c * v) for S, v in A.items()})


def eneg(A):
    return escal(-1, A)


def esub(A, B):
    return eadd(A, eneg(B))


def _merge(SA, SB):
    """Concatenate two sorted index tuples, returning (sorted tuple, sign)."""
    if set(SA) & set(SB):
        return None, 0
    merged = list(SA) + list(SB)
    sign = 1
    # bubble sort, counting transpositions
    for i in range(len(merged)):
        for j in range(len(merged) - 1 - i):
            if merged[j] > merged[j + 1]:
                merged[j], merged[j + 1] = merged[j + 1], merged[j]
                sign = -sign
    return tuple(merged), sign


def emul(A, B):
    out = {}
    for SA, ca in A.items():
        for SB, cb in B.items():
            S, sign = _merge(SA, SB)
            if sign == 0:
                continue
            out[S] = out.get(S, 0) + sign * ca * cb
    return clean(out)


def eeq(A, B):
    return not esub(A, B)


def d_t(A):
    return clean({S: sp.diff(c, t) for S, c in A.items()})


def d_th(i, A):
    """Left derivative with respect to theta_i."""
    out = {}
    for S, c in A.items():
        if i not in S:
            continue
        k = S.index(i)
        out[tuple(x for x in S if x != i)] = out.get(tuple(x for x in S if x != i), 0) \
            + (-1) ** k * c
    return clean(out)


def parity_split(A):
    """(even part, odd part) by the number of odd generators."""
    ev = {S: c for S, c in A.items() if len(S) % 2 == 0}
    od = {S: c for S, c in A.items() if len(S) % 2 == 1}
    return clean(ev), clean(od)


def body(A):
    """The theta-free component."""
    return A.get((), sp.Integer(0))


# --------------------------------------------------------------- vector fields
class VF:
    """a0 d_t + sum a_i d_{theta_i}, coefficients written on the left."""

    def __init__(self, a0, a):
        self.a0 = clean(a0)
        self.a = [clean(x) for x in a]

    def __call__(self, h):
        out = emul(self.a0, d_t(h))
        for i, ai in enumerate(self.a, start=1):
            out = eadd(out, emul(ai, d_th(i, h)))
        return out


def D_op(i):
    """D_{theta_i} = theta_i d_t + d_{theta_i}."""
    n = MAXN
    return VF(th(i), [num(1) if j == i else {} for j in range(1, n + 1)])


def K_op(i):
    """K_{theta_i} = theta_i d_t - d_{theta_i}."""
    n = MAXN
    return VF(th(i), [num(-1) if j == i else {} for j in range(1, n + 1)])


MAXN = 3          # set by the caller through setup()


def setup(n):
    global MAXN
    MAXN = n


WORLD = 'B'     # see K_f below


def K_f(f, n, world=None):
    """The contact field of the generating function f.

    Eq. (6) of the text prints

      (A)  K_f = (2-E)(f) d_t - (-1)^{p(f)} sum_i (d_{theta_i}f - theta_i d_t f) d_{theta_i},

    which does generate a contact algebra -- but the distribution
    it preserves is spanned by  theta_i d_t - d_{theta_i}, and its own value at
    f = theta_i is  theta_i d_t + d_{theta_i}: exactly the opposite of the two
    names in eq. (54) of the same text.  The mirror convention

      (B)  K_f = (2-E)(f) d_t + (-1)^{p(f)} sum_i (d_{theta_i}f + theta_i d_t f) d_{theta_i}

    -- (A) read in the coordinate -t -- gives  K_{theta_i} = theta_i d_t -
    d_{theta_i}  and the preserved frame  D_{theta_i} = theta_i d_t +
    d_{theta_i}, D^2 = d_t, which is (54) and the prose of the text, and the
    convention in which its Table (194) and Table (61) are written.  We
    work in (B); the only sign that has to be flipped in (6) to reach it is
    the inner one, not just the outer one.
    """
    if world is None:
        world = WORLD
    fe, fo = parity_split(f)
    Ef = {}
    for i in range(1, n + 1):
        Ef = eadd(Ef, emul(th(i), d_th(i, f)))
    a0 = esub(escal(2, f), Ef)
    a = []
    for i in range(1, n + 1):
        if world == 'A':
            tv = esub(d_th(i, fe), emul(th(i), d_t(fe)))      # p(f) = 0
            tw = esub(d_th(i, fo), emul(th(i), d_t(fo)))      # p(f) = 1
            a.append(esub(tv, tw))
        else:
            tv = eadd(d_th(i, fe), emul(th(i), d_t(fe)))
            tw = eadd(d_th(i, fo), emul(th(i), d_t(fo)))
            a.append(esub(tv, tw))
    if world == 'A':
        a = [eneg(x) for x in a]
    return VF(a0, a)


def frame(i):
    """The frame of the contact distribution: D_{theta_i} in world B."""
    return D_op(i) if WORLD == 'B' else K_op(i)


def flow(X, par, h):
    """exp(par * X) applied to h, truncated."""
    out = dict(h)
    term = dict(h)
    fact = 1
    for k in range(1, ORD + 1):
        term = X(term)
        if not term:
            break
        fact *= k
        out = eadd(out, escal(par ** k / sp.Integer(fact), term))
    return clean(out)


class Diffeo:
    """A finite contact diffeomorphism exp(par K_g), stored by its pull-back."""

    def __init__(self, g, par, n):
        self.n = n
        self.par = par
        self.X = K_f(g, n)
        self.g = g

    def star(self, h):
        return flow(self.X, self.par, h)


class Composite:
    """G o H, with (G o H)^* = H^* o G^*."""

    def __init__(self, G, H):
        self.n = G.n
        self.G, self.H = G, H

    def star(self, h):
        return self.H.star(self.G.star(h))


# ---------------------------------------------------------------- the multiplier
def coords(G, n):
    return G.star({(): t}), [G.star(th(i)) for i in range(1, n + 1)]


def multiplier(G, n):
    """m_G = d_t(f) + sum phi_i d_t(phi_i), eq. (192) of the text."""
    f, phi = coords(G, n)
    m = d_t(f)
    for p in phi:
        m = eadd(m, emul(p, d_t(p)))
    return m


def U_matrix(G, n, op=D_op):
    """U_{ij} = A_{theta_i}(G^* theta_j)."""
    phi = [G.star(th(j)) for j in range(1, n + 1)]
    return [[op(i)(phi[j - 1]) for j in range(1, n + 1)] for i in range(1, n + 1)]


# ----------------------------------------------- inverses and fractional powers
def body0(A):
    """The numerical body: the theta-free part at e = d = 0."""
    return sp.expand(body(A).subs({e: 0, d: 0}))


def _nilpotent_part(A):
    """A - body0(A); nilpotent, since e, d are truncated and the theta's square
    to zero -- every term carries a theta or a positive power of e or d."""
    return esub(A, num(body0(A)))


def einv(A):
    """1/A for A with invertible numerical body; terminating Neumann series."""
    b = body0(A)
    assert b != 0, "einv: the body vanishes"
    N = _nilpotent_part(A)
    inv = num(1 / b)
    term = num(1 / b)
    for _ in range(MAXN + ORD + 2):
        term = escal(-1 / b, emul(term, N))
        if not term:
            return inv
        inv = eadd(inv, term)
    raise AssertionError("einv: the Neumann series did not terminate")


def epow(A, r):
    """A^r for rational r, A with numerical body 1; binomial series."""
    assert sp.simplify(body0(A) - 1) == 0, "epow expects the body 1"
    N = _nilpotent_part(A)
    out = num(1)
    term = num(1)
    for k in range(1, MAXN + ORD + 2):
        term = emul(term, N)
        if not term:
            return out
        out = eadd(out, escal(sp.binomial(sp.Rational(r), k), term))
    raise AssertionError("epow: the binomial series did not terminate")
