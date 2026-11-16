"""
The Moebius (Ramond) superstring S^{1|n}_M, as truncated flows, for the table
of the Schwarz derivatives S^M_n, Table (201) of the text.

The algebra (Grassmann coordinates over Q(t) with two flow parameters) is that
of radul.py and is imported from it; what is new here is the world:

    alpha~ = dt + t zeta dzeta + sum_{i<n} theta_i dtheta_i,      eq. (10)
    cE     = zeta d_zeta + sum theta_i d_theta_i,  nabla = 2 - cE,
    cD     = d_t - zeta/(2t) d_zeta                  = (1/2) K^M_1,
    K^M_f  = nabla(f) cD + H^M_f + cD(f) cE,         eq. (11)
    H^M_f  = (-1)^{p(f)} ( (1/t) d_zeta(f) d_zeta + sum d_theta_i(f) d_theta_i ),
    D^M_zeta    = zeta d_t + (1/t) d_zeta,           eq. (56)
    D^M_theta_i = theta_i d_t + d_theta_i - theta_i zeta/(2t) d_zeta,
    m_G = d_y(f) + f psi psi' + sum phi_i phi_i',    eq. (193).

Indices: 1 is zeta, 2, ..., n are theta_1, ..., theta_{n-1}; so the algebra
k^M(1|n) has n odd coordinates, one of them twisted, and n = 0 is k^L(1|0) =
witt, where the Moebius bundle has nothing to twist.

As in radul.py, which of the two mirror conventions the printed formula (11)
belongs to is decided here and not assumed: WORLD = 'B' is the one in which the
preserved frame is the D^M of eq. (56), and that is the convention of the
tables.
"""
import sympy as sp

import radul as R
from radul import (t, e, d, num, th, eadd, esub, emul, escal, eeq, d_t, d_th,
                   VF, flow, clean, body, einv, epow, parity_split, body0)

ZETA = 1                      # the index of the twisted odd coordinate
WORLD = 'B'


def setup(n):
    """n = the total number of odd coordinates (zeta included)."""
    R.setup(max(n, 1))


def _nodd():
    return R.MAXN


# ------------------------------------------------------------ the vector fields
def cD_field(n):
    """cD = d_t - zeta/(2t) d_zeta."""
    a = [{} for _ in range(n)]
    if n >= 1:
        a[ZETA - 1] = escal(-sp.Rational(1, 2) / t, th(ZETA))
    return VF(num(1), a)


def cE_field(n):
    """cE = sum over ALL the odd coordinates of theta_i d_theta_i."""
    return VF({}, [th(i) for i in range(1, n + 1)])


def D_M(i, n):
    """The invariant (Y-type) frame, eq. (56)."""
    a = [{} for _ in range(n)]
    if i == ZETA:
        a[ZETA - 1] = num(1 / t)
        return VF(th(ZETA), a)
    a[i - 1] = num(1)
    a[ZETA - 1] = escal(-sp.Rational(1, 2) / t, emul(th(i), th(ZETA)))
    return VF(th(i), a)


def K_M(i, n):
    """The X-type operators of eq. (56)."""
    a = [{} for _ in range(n)]
    if i == ZETA:
        a[ZETA - 1] = num(-1 / t)
        return VF(th(ZETA), a)
    a[i - 1] = num(-1)
    a[ZETA - 1] = escal(-sp.Rational(1, 2) / t, emul(th(i), th(ZETA)))
    return VF(th(i), a)


def K_f(f, n, world=None):
    """The Moebius contact field of the generating function f, eq. (11).

    World 'B' is (11) as printed; world 'A' is the same with the sign of H^M_f
    reversed.  Which of the two preserves the frame (56) is checked, not
    assumed, in test_mobius.py.
    """
    if world is None:
        world = WORLD
    cD = cD_field(n)
    fe, fo = parity_split(f)
    Ef = {}
    for i in range(1, n + 1):
        Ef = eadd(Ef, emul(th(i), d_th(i, f)))
    nabla_f = esub(escal(2, f), Ef)
    cDf = cD(f)
    sgn = 1 if world == 'B' else -1

    a0 = nabla_f                                   # the d_t component of nabla(f) cD
    a = []
    for i in range(1, n + 1):
        # nabla(f) cD  contributes  -nabla(f) zeta/(2t)  to  d_zeta
        comp = {}
        if i == ZETA:
            comp = escal(-sp.Rational(1, 2) / t, emul(nabla_f, th(ZETA)))
        # H^M_f
        w = sp.Rational(1, 1) / t if i == ZETA else sp.Integer(1)
        comp = eadd(comp, escal(sgn * w, esub(d_th(i, fe), d_th(i, fo))))
        # cD(f) cE
        comp = eadd(comp, emul(cDf, th(i)))
        a.append(comp)
    return VF(a0, a)


# ------------------------------------------------------------------ diffeos
class Diffeo:
    """exp(par K^M_g), stored by its pull-back."""

    def __init__(self, g, par, n):
        self.n = n
        self.par = par
        self.g = g
        self.X = K_f(g, n)

    def star(self, h):
        return flow(self.X, self.par, h)


class Composite:
    def __init__(self, G, H):
        self.n = G.n
        self.G, self.H = G, H

    def star(self, h):
        return self.H.star(self.G.star(h))


def coords(G, n):
    """(G^* t;  G^* zeta, G^* theta_1, ...)."""
    return G.star({(): t}), [G.star(th(i)) for i in range(1, n + 1)]


def multiplier(G, n):
    """m_G = d_t(f) + f psi psi' + sum phi_i phi_i',  eq. (193)."""
    f, odd = coords(G, n)
    m = d_t(f)
    if n >= 1:
        psi = odd[ZETA - 1]
        m = eadd(m, emul(emul(f, psi), d_t(psi)))
    for j in range(2, n + 1):
        phi = odd[j - 1]
        m = eadd(m, emul(phi, d_t(phi)))
    return m


# --------------------------------------------------------------- the commutator
def bracket(X, Y, n):
    """[X, Y] for two EVEN vector fields, as a vector field."""
    a0 = esub(X(Y.a0), Y(X.a0))
    a = [esub(X(Y.a[i]), Y(X.a[i])) for i in range(n)]
    return VF(a0, a)


def anti(X, Y, n):
    """{X, Y} = XY + YX for two ODD vector fields."""
    a0 = eadd(X(Y.a0), Y(X.a0))
    a = [eadd(X(Y.a[i]), Y(X.a[i])) for i in range(n)]
    return VF(a0, a)


def vf_eq(X, Y, n):
    return eeq(X.a0, Y.a0) and all(eeq(X.a[i], Y.a[i]) for i in range(n))


def vf_scal(c, X, n):
    return VF(emul(c, X.a0), [emul(c, X.a[i]) for i in range(n)])


def vf_add(X, Y, n):
    return VF(eadd(X.a0, Y.a0), [eadd(X.a[i], Y.a[i]) for i in range(n)])


# ------------------------------------------------------------- the Schwarzians
def S_M(G, n):
    """The entry of the row n of the table (201), as the function part.

    n = 0:  the classical Schwarzian, in which no odd coordinate occurs;
    n = 1:  (1/2)( D^M_zeta cD(m)/m - (3/2) D^M_zeta(m) cD(m)/m^2 );
    n = 2:  (1/2)( D^M_zeta D^M_theta(m)/m - (3/2) D^M_zeta(m) D^M_theta(m)/m^2 ).
    """
    m = multiplier(G, n)
    mi = einv(m)
    mi2 = emul(mi, mi)
    cD = cD_field(n)
    if n == 0:
        A = emul(d_t(d_t(m)), mi)
        B = emul(emul(d_t(m), d_t(m)), mi2)
    elif n == 1:
        Dz = D_M(ZETA, n)
        A = emul(Dz(cD(m)), mi)
        B = emul(emul(Dz(m), cD(m)), mi2)
    elif n == 2:
        Dz, Dth = D_M(ZETA, n), D_M(2, n)
        A = emul(Dz(Dth(m)), mi)
        B = emul(emul(Dz(m), Dth(m)), mi2)
    else:
        raise ValueError("the table (201) has the rows n <= 2 only")
    return escal(sp.Rational(1, 2), esub(A, escal(sp.Rational(3, 2), B)))


def cocycle(Sfun, w, n, G, H):
    """S(GH) = S(H) + m_H^w H^*(S(G))."""
    GH = Composite(G, H)
    mH = multiplier(H, n)
    return eeq(Sfun(GH), eadd(Sfun(H), emul(epow(mH, w), H.star(Sfun(G)))))
