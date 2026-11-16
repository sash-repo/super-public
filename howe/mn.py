"""
mn.py -- exact engine for the Lie superalgebras M(n) = m(n) (odd contact / pericontact)
and their deformations M_{a,b}(n), following

  Yu. Yu. Kochetkov, "Irreducible induced representations of the Lie superalgebras of the
  series M and SM", Uspekhi Mat. Nauk 39:1 (1984), 235-236 (a communication to the Moscow
  Math. Society, submitted 3 Nov 1983).

Kochetkov's conventions, verbatim:

    W(n,n+1) = vector fields in x_1..x_n (even), xi_1..xi_n, tau (odd),
    deg x_i = deg xi_i = 1,  deg tau = 2,  deg D = -2   (D = d/d tau),
    alpha    = d tau + sum (x_i d xi_i + xi_i d x_i),
    M(n)     = { X : L_X alpha = g . alpha },   SM(n) = M(n) cap S(n,n+1).

M(n) is realised on the whole of C[x, xi, tau] with the bracket

    [f,g]_M = Delta f . dg/d tau - (-1)^k df/d tau . Delta g + [f,g]_BB,
    k = (p(f)+1)(p(g)+1),   Delta f = 2f - sum (x_i df/dx_i + xi_i df/dxi_i),

so Delta f = (2 - deg_{x,xi} f) f, and [f,g]_BB is the Buttin antibracket (le.abr).
The M(n)-degree of f is  deg f - 2  with deg tau = 2; p('f) = p(f) + 1.

Internally tau is the odd variable of index n, i.e. a monomial is (a, S) with a of
length n and S a sorted subset of {0,...,n}, n meaning tau.  Everything else (pmul,
dx, dxi, abr, rref, nullspace) is inherited from le.py unchanged.

Low degrees:
    L_{-2} = <1>            '1 = 2D
    L_{-1} = <x_i, -xi_i>   'x_i =: 'delta_i,  '(-xi_i) =: 'partial_i   (Kochetkov's names)
    L_0    = <deg-2 in x,xi> + <tau> = pe(n) + C z,  z = 'tau  central in L_0
             = spe(n) + C h + C z,   h = sum x_i xi_i
    L_1    = <deg-3 in x,xi> + <tau x_i, tau xi_i>
Brackets in L_{<0}:  ['partial_i, 'delta_j] = 2 delta_{ij} D,  D central, D^2 = 0.
"""

from fractions import Fraction as F
from itertools import combinations

from le import (pmul, padd, pscal, dx, dxi, is_hom_par, abr, rref, nullspace)


# ---------------------------------------------------------------- basic pieces
def mono(n, a, S, c=1):
    return {(tuple(a), tuple(sorted(S))): F(c)}


def one(n):
    return mono(n, [0] * n, [])


def tau(n):
    return mono(n, [0] * n, [n])


def xdeg(m, n):
    """deg_{x,xi} of a monomial (tau not counted)"""
    a, S = m
    return sum(a) + sum(1 for i in S if i < n)


def mdeg(f, n):
    """the M(n)-degree of a homogeneous generating function"""
    ds = {xdeg(m, n) + 2 * (1 if n in m[1] else 0) - 2 for m in f}
    assert len(ds) <= 1, "inhomogeneous M-degree"
    return ds.pop() if ds else None


def Delta(f, n):
    """Delta f = 2f - E f,  E = sum (x_i d/dx_i + xi_i d/dxi_i)"""
    return {m: (2 - xdeg(m, n)) * c for m, c in f.items() if (2 - xdeg(m, n)) != 0}


def dtau(f, n):
    return dxi(f, n)


def par_m(f):
    """parity of the element 'f of M(n) (= p(f) + 1)"""
    return (is_hom_par(f) + 1) % 2


def mbr(f, g, n):
    """[f,g]_M.

    Kochetkov prints  [f,g]_M = Delta f . d_tau g - (-1)^k d_tau f . Delta g + [f,g]_BB
    with k = (p(f)+1)(p(g)+1).  As printed this is *not* super-anticommutative for the
    parity p(f)+1: since polynomials supercommute, moving Delta g past d_tau f costs a
    sign the formula does not carry, and the identity fails precisely on the pairs with
    p(f) = 0, p(g) = 1.  The exponent must be k = p(f)+1, i.e. the second sign is
    +(-1)^{p(f)}; see msign.py, where this is the unique correction (up to tau -> -tau
    and a parity-dependent rescaling of f) making the bracket a Lie superalgebra bracket.
    """
    pf = is_hom_par(f)
    out = pmul(Delta(f, n), dtau(g, n))
    out = padd(out, pmul(dtau(f, n), Delta(g, n)), (-1) ** pf)
    out = padd(out, abr(f, g, n))
    return out


# ------------------------------------------------------------------ weights
def mwt(f, n):
    """weight in Kochetkov's basis {'(x_i xi_i), i=1..n; z}: an (n+1)-tuple.
    The e_i-weight of x^a xi^S is [i in S] - a_i; the z-weight is the M-degree."""
    ws = set()
    for m in f:
        a, S = m
        e = tuple((1 if i in S else 0) - a[i] for i in range(n))
        ws.add(e + (xdeg(m, n) + 2 * (1 if n in S else 0) - 2,))
    assert len(ws) == 1, "not a weight vector"
    return ws.pop()


# -------------------------------------------------------- vector fields in W(n,n+1)
class MVF:
    """vector field: comp[g] for g = 0..n-1 (d/dx_g), n..2n-1 (d/dxi), 2n (d/dtau)"""

    def __init__(self, n, comp, par):
        self.n = n
        self.comp = {g: v for g, v in comp.items() if v}
        self.par = par

    def apply(self, p):
        n = self.n
        out = {}
        for g, coef in self.comp.items():
            d = dx(p, g) if g < n else dxi(p, g - n)
            out = padd(out, pmul(coef, d))
        return out

    def bracket(self, other):
        n = self.n
        sg = (-1) ** (self.par * other.par)
        comp = {}
        for g in range(2 * n + 1):
            comp[g] = padd(self.apply(other.comp.get(g, {})),
                           other.apply(self.comp.get(g, {})), -sg)
        return MVF(n, comp, (self.par + other.par) % 2)

    def div(self):
        """div(sum f_g D_g) = sum (-1)^{p(f_g) p(D_g)} D_g(f_g)"""
        n = self.n
        out = {}
        for g, coef in self.comp.items():
            pg = 0 if g < n else 1
            for m, c in coef.items():
                pf = len(m[1]) % 2
                d = dx({m: c}, g) if g < n else dxi({m: c}, g - n)
                out = padd(out, d, (-1) ** (pf * pg))
        return out


def Mvf(f, n):
    """the vector field 'f of the generating function f (Kochetkov's two formulas)"""
    comp = {g: {} for g in range(2 * n + 1)}
    f0 = {m: c for m, c in f.items() if n not in m[1]}          # tau-free part
    f1 = {m: c for m, c in f.items() if n in m[1]}              # part = g.tau
    # 'f contains Delta f . D in both cases
    comp[2 * n] = Delta(f, n)
    if f0:
        q = is_hom_par(f0)
        for i in range(n):
            comp[n + i] = padd(comp[n + i], dx(f0, i))
            comp[i] = padd(comp[i], dxi(f0, i), (-1) ** q)
    if f1:
        g = {(a, tuple(t for t in S if t != n)): c for (a, S), c in f1.items()}
        q = is_hom_par(g)
        tt = tau(n)
        for i in range(n):
            # (-1)^{p(g)} tau . ''g
            comp[n + i] = padd(comp[n + i], pmul(tt, dx(g, i)), (-1) ** q)
            comp[i] = padd(comp[i], pmul(tt, dxi(g, i)), (-1) ** q * (-1) ** q)
            # g . sum (x_i d_i + xi_i delta_i)
            xi_ = mono(n, [1 if k == i else 0 for k in range(n)], [])
            comp[i] = padd(comp[i], pmul(g, xi_))
            comp[n + i] = padd(comp[n + i], pmul(g, mono(n, [0] * n, [i])))
    return MVF(n, comp, par_m(f))


def mdiv(f, n):
    return Mvf(f, n).div()


# ------------------------------------------------------------------ bases
def monos(n, d, t):
    """all monomials with deg_{x,xi} = d and tau-degree t in {0,1}"""
    out = []

    def rec(i, a, rem):
        if i == n:
            for k in range(min(rem, n) + 1):
                if k == rem:
                    for S in combinations(range(n), k):
                        out.append((tuple(a), tuple(S) + ((n,) if t else ())))
            return
        for e in range(rem + 1):
            rec(i + 1, a + [e], rem - e)

    rec(0, [], d)
    return [{m: F(1)} for m in out]


def layer(n, j):
    """a monomial basis of M(n)_j = <deg_{x,xi} = j+2, no tau> + <deg_{x,xi} = j, times tau>"""
    out = monos(n, j + 2, 0) if j + 2 >= 0 else []
    out += monos(n, j, 1) if j >= 0 else []
    return out
