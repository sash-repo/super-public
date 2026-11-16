"""The PERICONTACT Lie superalgebra m(n) and its weighted densities, exactly.

Coordinates:  tau odd;  q_1..q_n even;  xi_1..xi_n odd.  Pericontact form

    alpha_0 = d tau + sum_i (xi_i d q_i + q_i d xi_i),      omega_1 := d alpha_0

(the periplectic form on ker alpha_0).  The generating function is normalized as in
the even contact case of this paper -- by  f = alpha_0(M_f) -- which halves the field
of the house convention (whose alpha_0(M_f) = 2f):

    M_f = (1/2)[ (2-E)(f) d_tau - Le_f - (-1)^{p(f)} f_tau E ],
    E   = sum_i ( q_i d_{q_i} + xi_i d_{xi_i} )                  (Euler),
    Le_f= sum_i ( f_{q_i} d_{xi_i} + (-1)^{p(f)} f_{xi_i} d_{q_i} )   (periplectic).

Note  p(M_f) = p(f) + 1:  the map f |-> M_f is ODD.  In particular M_1 = d_tau is an
ODD vector field -- the Reeb field of a pericontact structure is odd, and therefore
R^2 = 0.  That single fact is what makes the pericontact census finite in a way the
contact one is not.

Grading:  deg tau = 2, deg q = deg xi = 1;  m_i = <M_f : deg f = i+2>, so

    m_{-2} = <M_1>  (odd),   m_{-1} = <M_{q_i}, M_{xi_i}>  (sdim n|n),
    m_0    = <M_tau, M_{q_iq_j}, M_{q_i xi_j}, M_{xi_i xi_j}> = c(pe(n)),

and m(n) = (ba(n), c(pe(n)))_*  with ba(n) = m_{-2} + m_{-1} of sdim (n|n+1).

Frame (right-invariant, i.e. commuting with m_{-2}+m_{-1}); the two constants are
SOLVED for in frame(), not assumed:

    X_i = d_{q_i} + gamma xi_i d_tau   (even, weighted order 1),
    Th_i= d_{xi_i}+ delta q_i  d_tau   (odd,  weighted order 1),
    R   = d_tau = M_1                  (odd,  weighted order 2).

Since tau is odd, R^2 = 0; and Th_i^2 = 0 as well, so only the n even frame directions
give operators of unbounded order.

Weighted densities:  F_{a,b} = sections of alpha_0^a Vol^b.  The zero part m_0 has a
TWO-dimensional abelianization (the grading element and the gl(n)-trace), so a rank-1
module over m(n) carries TWO weights -- unlike the contact case, where there is one
(except for k(1|2)).  The space of admissible multipliers is DERIVED in weightsm.py,
not assumed; Dens below takes the basis it returns.

Everything is exact rational arithmetic; nothing is asserted that test_mperi.py does
not check.
"""

from fractions import Fraction as Fr
from itertools import combinations

# --------------------------------------------------------------------- algebra
# A function is a dict {(qexp, mask): Fraction}.  qexp is a tuple of n exponents of
# the even q_i; mask is a bitmask of the odd generators in ONE global order:
#     bit 0 = tau,  bit i = xi_i  (1 <= i <= n).
# Every sign in this module comes from that single convention.

TAU = 0


def zero():
    return {}


def _odd_sort(idxs):
    """Sort a tuple of odd indices, returning (mask, sign); 0 if repeated."""
    idxs = list(idxs)
    sgn = 1
    for i in range(len(idxs)):
        for j in range(len(idxs) - 1 - i):
            if idxs[j] == idxs[j + 1]:
                return 0, 0
            if idxs[j] > idxs[j + 1]:
                idxs[j], idxs[j + 1] = idxs[j + 1], idxs[j]
                sgn = -sgn
    for i in range(len(idxs) - 1):
        if idxs[i] == idxs[i + 1]:
            return 0, 0
    m = 0
    for i in idxs:
        m |= 1 << i
    return m, sgn


def mono(n, qexp=None, odd=(), c=1):
    c = Fr(c)
    if c == 0:
        return {}
    if qexp is None:
        qexp = (0,) * n
    qexp = tuple(qexp)
    m, s = _odd_sort(odd)
    if s == 0:
        return {}
    return {(qexp, m): c * s}


def one(n):
    return mono(n, c=1)


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
    if c == 0:
        return {}
    return {k: c * v for k, v in f.items()}


def sub(f, g):
    return add(f, scal(-1, g))


def _merge(m1, m2):
    """Sign of  theta_{m1} . theta_{m2}  written in the global order."""
    if m1 & m2:
        return 0, 0
    sgn = 1
    mm = m2
    while mm:
        b = mm & -mm
        i = b.bit_length() - 1
        higher = m1 >> (i + 1)
        if bin(higher).count("1") % 2:
            sgn = -sgn
        mm ^= b
    return m1 | m2, sgn


def mul(f, g):
    out = {}
    for (qe1, m1), c1 in f.items():
        for (qe2, m2), c2 in g.items():
            m, s = _merge(m1, m2)
            if s == 0:
                continue
            key = (tuple(a + b for a, b in zip(qe1, qe2)), m)
            nv = out.get(key, Fr(0)) + s * c1 * c2
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
    return out


def parity(f):
    ps = {bin(m).count("1") % 2 for (_, m) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous function")
    return ps.pop() if ps else 0


def d_odd(i, f):
    """Left derivative d/d theta_i,  theta_0 = tau, theta_i = xi_i."""
    out = {}
    bit = 1 << i
    for (qe, m), c in f.items():
        if not (m & bit):
            continue
        lower = m & (bit - 1)
        s = -1 if bin(lower).count("1") % 2 else 1
        key = (qe, m ^ bit)
        nv = out.get(key, Fr(0)) + s * c
        if nv:
            out[key] = nv
        elif key in out:
            del out[key]
    return out


def d_tau(f):
    return d_odd(TAU, f)


def d_xi(i, f):
    return d_odd(i, f)


def d_q(i, f):
    """d/d q_i;  i is 1-based."""
    out = {}
    j = i - 1
    for (qe, m), c in f.items():
        if qe[j] == 0:
            continue
        ne = list(qe)
        ne[j] -= 1
        key = (tuple(ne), m)
        nv = out.get(key, Fr(0)) + c * qe[j]
        if nv:
            out[key] = nv
        elif key in out:
            del out[key]
    return out


def q(n, i, c=1):
    e = [0] * n
    e[i - 1] = 1
    return mono(n, e, (), c)


def xi(n, i, c=1):
    return mono(n, None, (i,), c)


def tau(n, c=1):
    return mono(n, None, (TAU,), c)


def wdeg(f):
    """Weighted degree: deg tau = 2, deg q = deg xi = 1.  Homogeneous input."""
    ds = set()
    for (qe, m) in f:
        d = sum(qe) + bin(m >> 1).count("1") + 2 * (m & 1)
        ds.add(d)
    if len(ds) > 1:
        raise ValueError("inhomogeneous in the weighted degree")
    return ds.pop() if ds else 0


def monomials(n, d):
    """All monomials of weighted degree d."""
    out = []
    for t in (0, 1):
        for k in range(n + 1):
            for S in combinations(range(1, n + 1), k):
                rest = d - 2 * t - k
                if rest < 0:
                    continue
                for qe in _compositions(n, rest):
                    odd = ((TAU,) if t else ()) + S
                    out.append(mono(n, qe, odd))
    return out


def _compositions(n, total):
    if n == 0:
        return [()] if total == 0 else []
    if n == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for rest in _compositions(n - 1, total - first):
            out.append((first,) + rest)
    return out


# --------------------------------------------------------------- vector fields
# A vector field is a dict {key: function}, keys  ("tau",), ("q",i), ("xi",i).

def vkeys(n):
    return [("tau",)] + [("q", i) for i in range(1, n + 1)] \
        + [("xi", i) for i in range(1, n + 1)]


def vparity(key):
    return 1 if key[0] in ("tau", "xi") else 0


def vderiv(key, f):
    if key[0] == "tau":
        return d_tau(f)
    if key[0] == "q":
        return d_q(key[1], f)
    return d_xi(key[1], f)


def vzero():
    return {}


def vadd(*Xs):
    out = {}
    for X in Xs:
        for k, v in X.items():
            nv = add(out.get(k, {}), v)
            if nv:
                out[k] = nv
            elif k in out:
                del out[k]
    return out


def vscal(c, X):
    return {k: scal(c, v) for k, v in X.items() if scal(c, v)}


def vapply(X, f):
    """X(f) = sum_v X^v d_v f."""
    out = {}
    for k, coef in X.items():
        out = add(out, mul(coef, vderiv(k, f)))
    return out


def vparity_of(X):
    ps = {(parity(v) + vparity(k)) % 2 for k, v in X.items() if v}
    if len(ps) > 1:
        raise ValueError("inhomogeneous vector field")
    return ps.pop() if ps else 0


def vbracket(X, Y):
    """[X,Y]^v = X(Y^v) - (-1)^{p(X)p(Y)} Y(X^v)."""
    pX, pY = vparity_of(X), vparity_of(Y)
    sgn = (-1) ** (pX * pY)
    out = {}
    for k in set(X) | set(Y):
        a = vapply(X, Y.get(k, {}))
        b = vapply(Y, X.get(k, {}))
        v = add(a, scal(-sgn, b))
        if v:
            out[k] = v
    return out


# ----------------------------------------------------------- pericontact fields

def Mf(n, f):
    """The pericontact field of the generating function f, normalized by f = alpha_0(M_f)."""
    if not f:
        return {}
    p = parity(f)
    sp = (-1) ** p
    ft = d_tau(f)
    # (2 - E)(f)
    Ef = {}
    for i in range(1, n + 1):
        Ef = add(Ef, mul(q(n, i), d_q(i, f)), mul(xi(n, i), d_xi(i, f)))
    A = add(scal(2, f), scal(-1, Ef))
    out = {("tau",): scal(Fr(1, 2), A)}
    for i in range(1, n + 1):
        # coefficient of d_{q_i}:  -(-1)^{p} f_{xi_i}  -  (-1)^{p} f_tau q_i
        B = add(scal(-sp, d_xi(i, f)), scal(-sp, mul(ft, q(n, i))))
        # coefficient of d_{xi_i}: -f_{q_i}  -  (-1)^{p} f_tau xi_i
        C = add(scal(-1, d_q(i, f)), scal(-sp, mul(ft, xi(n, i))))
        B, C = scal(Fr(1, 2), B), scal(Fr(1, 2), C)
        if B:
            out[("q", i)] = B
        if C:
            out[("xi", i)] = C
    return {k: v for k, v in out.items() if v}


def alpha0(n, X):
    """alpha_0(X) = X^tau + sum_i ( X^{q_i} xi_i - X^{xi_i} q_i ).  Equals f for X = M_f.

    The coefficient stands to the LEFT of the coordinate: the pairing is the right
    one, and with the coefficient on the other side the formula fails exactly on the
    monomials whose coefficient is odd (f = tau xi_i is the smallest).
    """
    out = X.get(("tau",), {})
    for i in range(1, n + 1):
        out = add(out,
                  mul(X.get(("q", i), {}), xi(n, i)),
                  scal(-1, mul(X.get(("xi", i), {}), q(n, i))))
    return out


def mbracket(n, f, g):
    """The pericontact bracket:  M_{ {f,g} } = [M_f, M_g]."""
    return alpha0(n, vbracket(Mf(n, f), Mf(n, g)))


def m_piece(n, i, maxq=None):
    """A basis of m_i = <M_f : wdeg f = i+2> as generating functions."""
    d = i + 2
    if d < 0:
        return []
    return monomials(n, d)


# ----------------------------------------------------------------------- frame

_FRAME_CACHE = {}


def frame_constants(n):
    """Solve for (gamma, delta) making X_i, Th_i commute with m_{-2}+m_{-1}."""
    if n in _FRAME_CACHE:
        return _FRAME_CACHE[n]
    from exactlin import nullspace
    # Candidate X_1 = d_{q_1} + gamma xi_1 d_tau, Th_1 = d_{xi_1} + delta q_1 d_tau.
    # Impose [cand, M_h] = 0 for h in m_{-2}+m_{-1}; the condition is affine in the
    # single unknown, so two samples determine it.
    sols = []
    for which in ("X", "Th"):
        vals = []
        for cval in (Fr(0), Fr(1)):
            if which == "X":
                cand = {("q", 1): one(n), ("tau",): scal(cval, xi(n, 1))}
            else:
                cand = {("xi", 1): one(n), ("tau",): scal(cval, q(n, 1))}
            tot = {}
            for h in [one(n)] + [q(n, i) for i in range(1, n + 1)] \
                    + [xi(n, i) for i in range(1, n + 1)]:
                tot = vadd(tot, vbracket(cand, Mf(n, h)))
            vals.append(tot)
        # tot(c) = tot(0) + c * (tot(1) - tot(0));  solve tot(c) = 0
        base, slope = vals[0], vadd(vals[1], vscal(-1, vals[0]))
        c = None
        for k in set(base) | set(slope):
            for key, bv in base.get(k, {}).items():
                sv = slope.get(k, {}).get(key, Fr(0))
                if sv:
                    cc = -bv / sv
                    if c is None:
                        c = cc
                    elif c != cc:
                        raise ValueError("no frame constant")
        if c is None:
            c = Fr(0)
        # confirm
        if which == "X":
            cand = {("q", 1): one(n), ("tau",): scal(c, xi(n, 1))}
        else:
            cand = {("xi", 1): one(n), ("tau",): scal(c, q(n, 1))}
        for h in [one(n)] + [q(n, i) for i in range(1, n + 1)] \
                + [xi(n, i) for i in range(1, n + 1)]:
            if vbracket(cand, Mf(n, h)):
                raise ValueError("frame constant fails for %s" % which)
        sols.append(c)
    _FRAME_CACHE[n] = tuple(sols)
    return _FRAME_CACHE[n]


def frame(n):
    """[X_1..X_n, Th_1..Th_n, R] -- the right-invariant frame; R = M_1 = d_tau."""
    g, d = frame_constants(n)
    out = []
    for i in range(1, n + 1):
        out.append({("q", i): one(n), ("tau",): scal(g, xi(n, i))})
    for i in range(1, n + 1):
        out.append({("xi", i): one(n), ("tau",): scal(d, q(n, i))})
    out.append({("tau",): one(n)})
    return out


def apply_frame(n, term, F):
    """term = (a, S, s):  X^a Th_S R^s applied to F (R first, then Th, then X)."""
    a, S, s = term
    fr = frame(n)
    out = F
    if s:
        out = vapply(fr[2 * n], out)
        if not out:
            return {}
    for i in reversed(S):
        out = vapply(fr[n + i - 1], out)
        if not out:
            return {}
    for i in range(n, 0, -1):
        for _ in range(a[i - 1]):
            out = vapply(fr[i - 1], out)
            if not out:
                return {}
    return out


def term_wdeg(term):
    a, S, s = term
    return sum(a) + len(S) + 2 * s


def term_parity(term):
    """X_i even, Th_i odd, R odd."""
    _, S, s = term
    return (len(S) + s) % 2


def frame_terms(n, W, maxq=None):
    """All frame monomials X^a Th_S R^s of weighted order exactly W."""
    out = []
    for s in (0, 1):
        for k in range(n + 1):
            for S in combinations(range(1, n + 1), k):
                rest = W - 2 * s - k
                if rest < 0:
                    continue
                for a in _compositions(n, rest):
                    out.append((a, S, s))
    return out


def term_name(n, term):
    a, S, s = term
    bits = []
    for i in range(1, n + 1):
        if a[i - 1] == 1:
            bits.append("X_%d" % i)
        elif a[i - 1] > 1:
            bits.append("X_%d^%d" % (i, a[i - 1]))
    for i in S:
        bits.append("T_%d" % i)
    if s:
        bits.append("R")
    return "".join(bits) if bits else "1"


# ------------------------------------------------------------------- densities

def Delta_m(n, f):
    """Delta^m(f) = (1-E) f_tau - sum_i d^2 f/(d xi_i d q_i)."""
    ft = d_tau(f)
    Eft = {}
    for i in range(1, n + 1):
        Eft = add(Eft, mul(q(n, i), d_q(i, ft)), mul(xi(n, i), d_xi(i, ft)))
    out = add(ft, scal(-1, Eft))
    for i in range(1, n + 1):
        out = add(out, scal(-1, d_xi(i, d_q(i, f))))
    return out


class Dens:
    """The module F_{a,b} of sections of alpha_0^a Vol^b.

        L_{M_f} F = M_f(F) + (-1)^{p(f)} ( a f_tau + b Delta^m f ) F

    (halved convention).  The two multipliers are the basis returned by weightsm.py.
    """

    def __init__(self, n, a, b):
        self.n, self.a, self.b = n, Fr(a), Fr(b)

    def rho(self, f):
        n = self.n
        sp = (-1) ** parity(f)
        return scal(sp, add(scal(self.a, d_tau(f)), scal(self.b, Delta_m(n, f))))

    def act(self, f, F):
        return add(vapply(Mf(self.n, f), F), mul(self.rho(f), F))


# --------------------------------------------------------------- b_{a,b}(n)

def div_lambda(n, f, a, b):
    """The operator that singles out b_{a,b}(n) in m(n):  Div_lambda f = 0.

    The formula of the text:  Div_lambda = (bn - aE) d_tau - a Delta,  Delta = sum d^2/dq dxi.
    Written on the generating function; a, b are the PARAMETERS of the family.
    """
    a, b = Fr(a), Fr(b)
    ft = d_tau(f)
    Eft = {}
    for i in range(1, n + 1):
        Eft = add(Eft, mul(q(n, i), d_q(i, ft)), mul(xi(n, i), d_xi(i, ft)))
    out = add(scal(b * n, ft), scal(-a, Eft))
    for i in range(1, n + 1):
        out = add(out, scal(-a, d_xi(i, d_q(i, f))))
    return out


def b_piece(n, i, a, b):
    """A basis of the degree-i part of b_{a,b}(n), as generating functions."""
    import os, sys
    _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
    from exactlin import nullspace
    basis = m_piece(n, i)
    if not basis:
        return []
    # assemble the matrix of Div_lambda on this graded piece
    images = [div_lambda(n, f, a, b) for f in basis]
    keys = sorted({k for im in images for k in im}, key=repr)
    if not keys:
        return basis
    rows = [[im.get(k, Fr(0)) for im in images] for k in keys]
    ns = nullspace(rows)
    out = []
    for v in ns:
        f = {}
        for c, bf in zip(v, basis):
            if c:
                f = add(f, scal(c, bf))
        if f:
            out.append(f)
    return out
