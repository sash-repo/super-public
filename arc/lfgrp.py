"""
The linear fractional transformations of the distinguished superstrings,
explicitly.

Grassmann algebra over the field of rational functions in the even variable t
and the even parameters A, B, C, D, ... ; odd generators: the coordinates
theta_1..theta_n and the odd parameters al_i, be_i.

Claim tested here (the (2|n)-format generalisation of eq. (105) of the text,
i.e. the projective action of OSp(n|2) on the superstring):

    tilde t     = (A t + B + sum_j ga_j th_j) / (C t + D + sum_j de_j th_j),
    tilde th_i  = (al_i t + be_i + sum_j E_ij th_j) / (C t + D + sum_j de_j th_j),

where, solving X^{st} B X = B  (B = antidiag(1,-1) + 1_n) for the odd blocks,

    Delta := AD - BC = 1 + sum_i al_i be_i,
    ga_j  = Delta^{-1} sum_i (B al_i - A be_i) E_ij,
    de_j  = Delta^{-1} sum_i (D al_i - C be_i) E_ij,      E^t E = 1 - N,
    N_ij  = ga_i de_j - de_i ga_j   (symmetric, nilpotent).

Checked: contactness  G^*(alpha_1) = m_G alpha_1  with
    m_G = (C t + D + sum_j de_j th_j)^{-2},
the invariant (convention-free) description of the multiplier
    sum_i D_i(tilde th_j) D_i(tilde th_l) = m_G delta_jl,      D_i = d_th_i + th_i d_t,
the superconformality  D_i(tilde t) = sum_j tilde th_j D_i(tilde th_j),
and  ber(Jacobi matrix)^2 = m_G^{2-n}.

Convention for d: p(dz) = p(z) + 1 (d is an odd derivation), coefficients of a
1-form written on the LEFT:  df = dt d_t(f) + sum_j d th_j d_th_j(f).
This convention is forced: in the "even d, left coefficients" one the
supertranslation  t -> t + eps th_i,  th_i -> th_i + eps  is not contact.
"""
import sympy as sp
from itertools import combinations

t = sp.Symbol('t')


# ------------------------------------------------------- Grassmann arithmetic

class G:
    """element of Lambda(odd gens) (x) Q(t, even params); {tuple: sympy expr}."""

    order = []          # global list fixing the order of the odd generators

    def __init__(self, d=None):
        self.d = {}
        for k, v in (d or {}).items():
            if v == 0:
                continue
            v = sp.cancel(v) if getattr(v, 'is_Atom', False) is False else v
            if v != 0:
                self.d[k] = v

    @staticmethod
    def sort(S):
        S = list(S)
        s = 1
        for i in range(len(S)):
            for j in range(len(S) - 1 - i):
                if G.order.index(S[j]) > G.order.index(S[j + 1]):
                    S[j], S[j + 1] = S[j + 1], S[j]
                    s = -s
                elif S[j] == S[j + 1]:
                    return (), 0
        for i in range(len(S) - 1):
            if S[i] == S[i + 1]:
                return (), 0
        return tuple(S), s

    def __add__(self, o):
        out = dict(self.d)
        for k, v in o.d.items():
            out[k] = out.get(k, 0) + v
        return G(out)

    def __sub__(self, o):
        return self + o * (-1)

    def __mul__(self, o):
        if not isinstance(o, G):
            return G({k: v * o for k, v in self.d.items()})
        out = {}
        for k1, v1 in self.d.items():
            for k2, v2 in o.d.items():
                k, s = G.sort(k1 + k2)
                if s:
                    out[k] = out.get(k, 0) + s * v1 * v2
        return G(out)

    __rmul__ = __mul__

    def inv(self):
        """inverse of an even invertible element u0 (1 + nu)."""
        u0 = self.d.get((), 0)
        assert u0 != 0
        nu = G({k: v / u0 for k, v in self.d.items() if k})
        out = G({(): sp.Integer(1)})
        term = G({(): sp.Integer(1)})
        for _ in range(len(G.order) + 1):
            term = term * nu * (-1)
            if not term.d:
                break
            out = out + term
        return out * (1 / u0)

    def __truediv__(self, o):
        return self * (o.inv() if isinstance(o, G) else 1 / o)

    def is_zero(self):
        return all(sp.simplify(sp.cancel(sp.together(v))) == 0 for v in self.d.values())

    def dt(self):
        return G({k: sp.diff(v, t) for k, v in self.d.items()})

    def dodd(self, g):
        """LEFT derivative with respect to the odd generator g."""
        out = {}
        for k, v in self.d.items():
            if g in k:
                i = k.index(g)
                out[tuple(x for x in k if x != g)] = (-1) ** i * v
        return G(out)

    def __repr__(self):
        if not self.d:
            return "0"
        return " + ".join("(%s)%s" % (sp.cancel(sp.together(v)), "".join(k)) for k, v in self.d.items())


def const(x):
    return G({(): sp.sympify(x)})


def gen(name):
    return G({(name,): sp.Integer(1)})


# ------------------------------------------------------------ superconformal

class Map:
    """t -> f, th_i -> ph_i."""

    def __init__(self, f, ph, ths):
        self.f, self.ph, self.ths = f, list(ph), list(ths)

    def D(self, i, h):
        """D_i = d_th_i + th_i d_t  applied to h."""
        return h.dodd(self.ths[i]) + gen(self.ths[i]) * h.dt()

    def multiplier_form(self):
        """dt-coefficient of G^*(alpha_1), and the check that the d th_j
        coefficients are m * th_j (odd d, left coefficients)."""
        m = self.f.dt()
        for p in self.ph:
            m = m - p * p.dt()
        ok = True
        for j, thj in enumerate(self.ths):
            c = self.f.dodd(thj)
            for p in self.ph:
                c = c + p * p.dodd(thj)
            if not (c - m * gen(thj)).is_zero():
                ok = False
        return m, ok

    def multiplier_D(self):
        """m from the invariant description: sum_i D_i(ph_j) D_i(ph_l) = m d_jl."""
        n = len(self.ths)
        M = [[sum((self.D(i, self.ph[j]) * self.D(i, self.ph[l]) for i in range(n)),
                  const(0)) for l in range(n)] for j in range(n)]
        m = M[0][0]
        ok = all((M[j][l] - (m if j == l else const(0))).is_zero()
                 for j in range(n) for l in range(n))
        return m, ok

    def superconformal(self):
        """D_i(tilde t) - sum_j tilde th_j D_i(tilde th_j) = 0 ?"""
        n = len(self.ths)
        for i in range(n):
            r = self.D(i, self.f)
            for j in range(n):
                r = r - self.ph[j] * self.D(i, self.ph[j])
            if not r.is_zero():
                return False
        return True

    def ber(self):
        """ber of the Jacobi matrix  ((d f/d t, d ph/d t),(d f/d th, d ph/d th))
        in the (1|n) format: ber = (a - b d^{-1} c) det(d)^{-1}."""
        n = len(self.ths)
        a = self.f.dt()
        b = [self.ph[j].dt() for j in range(n)]                 # 1 x n (odd)
        c = [self.f.dodd(self.ths[i]) for i in range(n)]        # n x 1 (odd)
        dmat = [[self.ph[j].dodd(self.ths[i]) for j in range(n)] for i in range(n)]
        dinv = mat_inv(dmat, n)
        red = a
        for i in range(n):
            for j in range(n):
                red = red - b[j] * dinv[j][i] * c[i]   # b d^{-1} c
        return red * mat_det(dmat, n).inv()


def power(u, k):
    """u^k for an even invertible u and an integer k."""
    if k < 0:
        u, k = u.inv(), -k
    out = const(1)
    for _ in range(k):
        out = out * u
    return out


def mat_det(M, n):
    """determinant of an n x n matrix of EVEN Grassmann elements."""
    out = const(0)
    import itertools
    for perm in itertools.permutations(range(n)):
        s = perm_sign(perm)
        p = const(s)
        for i in range(n):
            p = p * M[i][perm[i]]
        out = out + p
    return out


def perm_sign(p):
    s = 1
    p = list(p)
    for i in range(len(p)):
        for j in range(len(p) - 1 - i):
            if p[j] > p[j + 1]:
                p[j], p[j + 1] = p[j + 1], p[j]
                s = -s
    return s


def mat_inv(M, n):
    """inverse of an n x n matrix of even Grassmann elements (adjugate)."""
    if n == 1:
        return [[M[0][0].inv()]]
    det = mat_det(M, n)
    dinv = det.inv()
    out = [[const(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            minor = [[M[a][b] for b in range(n) if b != i] for a in range(n) if a != j]
            out[i][j] = const((-1) ** (i + j)) * mat_det(minor, n - 1) * dinv
    return out


# ------------------------------------------------------------- the OSp point

def osp_point(n, rot=False):
    """the C-point of OSp(n|2) in the solved parametrisation; returns the Map."""
    A, B, C = sp.symbols('A B C')
    ths = ['th%d' % i for i in range(1, n + 1)]
    als = ['al%d' % i for i in range(1, n + 1)]
    bes = ['be%d' % i for i in range(1, n + 1)]
    G.order = als + bes + ths
    al = [gen(a) for a in als]
    be = [gen(b) for b in bes]
    Delta = const(1) + sum((al[i] * be[i] for i in range(n)), const(0))
    Dinv = Delta.inv()
    # AD - BC = Delta  ==>  D is not free:
    Dg = (Delta + const(B * C)) * const(1 / A)
    # E: orthogonal part O times the nilpotent correction; solve E^t E = 1 - N
    # by iteration, starting from E = O.
    if rot and n >= 2:
        ph = sp.Symbol('ps')
        O = [[const(0)] * n for _ in range(n)]
        O[0][0] = const(sp.cos(ph)); O[0][1] = const(-sp.sin(ph))
        O[1][0] = const(sp.sin(ph)); O[1][1] = const(sp.cos(ph))
        for i in range(2, n):
            O[i][i] = const(1)
    else:
        O = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
    E = [row[:] for row in O]
    for _ in range(4):
        ga = [sum(((B * al[i] - A * be[i]) * E[i][j] for i in range(n)), const(0)) * Dinv
              for j in range(n)]
        de = [sum(((Dg * al[i] - const(C) * be[i]) * E[i][j] for i in range(n)),
                  const(0)) * Dinv for j in range(n)]
        N = [[ga[i] * de[j] - de[i] * ga[j] for j in range(n)] for i in range(n)]
        # E = O (1 - N/2 - N^2/8 - ...) : solve E^t E = 1 - N with E = O S, S^t S = 1-N
        # S = (1 - N)^{1/2} = sum_k binom(1/2,k) (-N)^k , N nilpotent
        S = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
        Npow = [[const(1 if i == j else 0) for j in range(n)] for i in range(n)]
        for k in range(1, 2 * n + 2):
            Npow = [[sum((Npow[i][l] * N[l][j] for l in range(n)), const(0))
                     for j in range(n)] for i in range(n)]
            if all(not Npow[i][j].d for i in range(n) for j in range(n)):
                break
            cf = sp.binomial(sp.Rational(1, 2), k) * (-1) ** k
            for i in range(n):
                for j in range(n):
                    S[i][j] = S[i][j] + Npow[i][j] * cf
        E = [[sum((O[i][k] * S[k][j] for k in range(n)), const(0)) for j in range(n)]
             for i in range(n)]
    # the check that E^t E = 1 - N
    ok_E = True
    for i in range(n):
        for j in range(n):
            v = sum((E[k][i] * E[k][j] for k in range(n)), const(0))
            tgt = const(1 if i == j else 0) - N[i][j]
            if not (v - tgt).is_zero():
                ok_E = False
    den = const(C) * const(t) + Dg + sum((de[j] * gen(ths[j]) for j in range(n)),
                                               const(0))
    dinv = den.inv()
    num_t = const(A) * const(t) + const(B) + sum((ga[j] * gen(ths[j]) for j in range(n)),
                                                 const(0))
    f = num_t * dinv
    ph = [(al[i] * const(t) + be[i]
           + sum((E[i][j] * gen(ths[j]) for j in range(n)), const(0))) * dinv
          for i in range(n)]
    return Map(f, ph, ths), den, ok_E, (A, B, C, Dg)


if __name__ == "__main__":
    import sys
    # n values from the command line, e.g.  python3 lfgrp.py 1 2 3   (n = 4 takes hours)
    for n in [int(a) for a in sys.argv[1:]] or [1, 2, 3]:
        print("=" * 70)
        print("n = %d :  OSp(%d|2) acting on S_c^{1|n} by the fractions of eq. (105)" % (n, n))
        M, den, ok_E, (A, B, C, Dg) = osp_point(n)
        print("   E^t E = 1 - N                    :", "OK" if ok_E else "FAILED")
        print("   superconformality D_i(t~)=th~D_i(th~):",
              "OK" if M.superconformal() else "FAILED")
        m, ok_D = M.multiplier_D()
        print("   sum_i D_i(th~_j)D_i(th~_l)=m d_jl:", "OK" if ok_D else "FAILED")
        print("   m = (Ct+D+sum de_j th_j)^{-2}    :",
              "OK" if (m - (den * den).inv()).is_zero() else "NO")
        mp = M.f.dt()
        mm = M.f.dt()
        for p in M.ph:
            mp = mp + p * p.dt()
            mm = mm - p * p.dt()
        print("   m = f\' + sum ph_i ph_i\'  (eq. (192)) :",
              "OK" if (mp - m).is_zero() else "NO")
        print("   m = f\' - sum ph_i ph_i\'          :",
              "OK" if (mm - m).is_zero() else "NO")
        b = M.ber()
        print("   ber^2 = m^{2-n}                  :",
              "OK" if (b * b - power(m, 2 - n)).is_zero() else "NO")
        if n == 2:
            print("   ber = 1 (n=2)                    :",
                  "OK" if (b - const(1)).is_zero() else "NO")
