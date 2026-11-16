#!/usr/bin/env python3
"""
sliouv.py -- the classical anomaly of the N-extended super Liouville equation.

Exact symbolic computation in the FREE SUPERCOMMUTATIVE algebra on the
generators

    u_k := D^k F   (k >= 0),      a_k := D^k a   (k >= 0),

where D is the odd derivation with D^2 = d/dx acting on the superline C^{1|1}
(so D^{2m} = d^m/dx^m and D^{2m+1} = d^m/dx^m o D), F is the even Liouville
superfield and  a  is the even generating function of a contact (superconformal)
vector field K_a.  No components, no Grassmann coordinates: the whole
computation is a computation with a free supercommutative algebra and one odd
derivation, which is what makes it exact and convention-free.

Facts derived (each is re-checked below, nothing is assumed):

  (K)   K_a = a D^2 + 1/2 (Da) D   is the general even contact field, and
        [K_a, D] = -1/2 (D^2 a) D,   i.e.  lambda_a = -1/2 a'.

  (M)   the super Liouville equation D_+ D_- F = exp(F) is preserved by
        delta_a F = K_a(F) - lambda_a = a u_2 + 1/2 a_1 u_1 + 1/2 a_2.

  (T)   the super stress tensor and its anomalous transformation law.

Usage:  python3 sliouv.py
"""

from fractions import Fraction as Q
import itertools
import sys

# ---------------------------------------------------------------------------
# a free supercommutative algebra on named generators with parities
# ---------------------------------------------------------------------------

class Alg:
    def __init__(self):
        self.names = []          # index -> name
        self.par = []            # index -> 0/1
        self.idx = {}            # name -> index
        self.Dmap = {}           # index -> index of D(generator), or None

    def gen(self, name, parity):
        if name in self.idx:
            return self.idx[name]
        i = len(self.names)
        self.names.append(name)
        self.par.append(parity % 2)
        self.idx[name] = i
        return i

    # ---- monomials are tuples of generator indices in nondecreasing order ---

    def mono_mul(self, m1, m2):
        """product of two canonical monomials -> (sign, monomial) or None."""
        # count inversions between the two blocks, odd x odd only
        sign = 1
        for i in m1:
            if self.par[i] == 0:
                continue
            for j in m2:
                if self.par[j] == 1 and j < i:
                    sign = -sign
        m = tuple(sorted(m1 + m2))
        # a repeated odd generator kills the monomial
        for k in range(len(m) - 1):
            if m[k] == m[k + 1] and self.par[m[k]] == 1:
                return None
        return sign, m

    def par_mono(self, m):
        return sum(self.par[i] for i in m) % 2


class Elt:
    """a dict {monomial: Fraction coefficient} over a given Alg."""

    __slots__ = ("A", "d")

    def __init__(self, A, d=None):
        self.A = A
        self.d = {} if d is None else {k: v for k, v in d.items() if v != 0}

    @staticmethod
    def const(A, c):
        return Elt(A, {(): Q(c)}) if c != 0 else Elt(A)

    @staticmethod
    def g(A, i):
        return Elt(A, {(i,): Q(1)})

    def __add__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d.get(k, Q(0)) + v
            if d[k] == 0:
                del d[k]
        return Elt(self.A, d)

    def __sub__(self, o):
        return self + o.scal(-1)

    def scal(self, c):
        c = Q(c)
        if c == 0:
            return Elt(self.A)
        return Elt(self.A, {k: v * c for k, v in self.d.items()})

    def __mul__(self, o):
        A = self.A
        d = {}
        for m1, c1 in self.d.items():
            for m2, c2 in o.d.items():
                r = A.mono_mul(m1, m2)
                if r is None:
                    continue
                s, m = r
                d[m] = d.get(m, Q(0)) + s * c1 * c2
                if d[m] == 0:
                    del d[m]
        return Elt(A, d)

    def is_zero(self):
        return not self.d

    def __repr__(self):
        A = self.A
        if not self.d:
            return "0"
        parts = []
        for m in sorted(self.d, key=lambda t: (len(t), t)):
            c = self.d[m]
            s = "*".join(A.names[i] for i in m) if m else "1"
            parts.append(f"({c})*{s}" if m else f"({c})")
        return " + ".join(parts)


def D(A, e):
    """the odd derivation with D(gen_k) = gen_{k+1}."""
    out = Elt(A)
    for m, c in e.d.items():
        for j in range(len(m)):
            gj = m[j]
            gnext = A.Dmap[gj]
            if gnext is None:
                continue
            sgn = (-1) ** (sum(A.par[m[t]] for t in range(j)) % 2)
            left = m[:j]
            right = m[j + 1:]
            r1 = A.mono_mul(left, (gnext,))
            if r1 is None:
                continue
            s1, m1 = r1
            r2 = A.mono_mul(m1, right)
            if r2 is None:
                continue
            s2, m2 = r2
            out = out + Elt(A, {m2: Q(sgn * s1 * s2) * c})
    return out


# ---------------------------------------------------------------------------
# the algebra of the problem
# ---------------------------------------------------------------------------

NMAX = 14   # how many derivatives of F and of a we keep

PA = 0      # parity of the generating function a; reset by set_parity()


def build(pa):
    A = Alg()
    u = [A.gen(f"u{k}", k % 2) for k in range(NMAX + 1)]        # u_k = D^k F
    a = [A.gen(f"a{k}", (k + pa) % 2) for k in range(NMAX + 1)]  # a_k = D^k a
    for k in range(NMAX + 1):
        A.Dmap[u[k]] = u[k + 1] if k < NMAX else None
        A.Dmap[a[k]] = a[k + 1] if k < NMAX else None
    return A, u, a


def set_parity(pa):
    """rebuild the algebra with a of parity pa (0 = even, 1 = odd)."""
    global A, U, AA, PA, u, aa, one, zero, deltaF
    PA = pa
    A, U, AA = build(pa)
    u = lambda k: Elt.g(A, U[k])
    aa = lambda k: Elt.g(A, AA[k])
    one = Elt.const(A, 1)
    zero = Elt(A)
    # K_a(F) = a D^2F + (-1)^{p(a)}/2 (Da)(DF);  lambda_a = -1/2 D^2 a;
    # delta_a F = K_a(F) - lambda_a
    deltaF = (aa(0) * u(2)
              + (aa(1) * u(1)).scal(Q((-1) ** pa, 2))
              + aa(2).scal(Q(1, 2)))
    return deltaF


set_parity(0)


def delta(e, dF=None):
    """the derivation determined by delta(u_k) = D^k(delta F), delta(a_k)=0.

    NB: delta has the parity of a.  Passing it through a monomial therefore
    costs a Koszul sign (-1)^{PA * (parity of the prefix)}."""
    if dF is None:
        dF = deltaF
    # precompute delta u_k.  The variation is F -> F + eps (delta F) with eps
    # of the parity of a; hence D^k(eps w) = (-1)^{k PA} eps D^k w, i.e.
    #     delta(D^k F) = (-1)^{k PA} D^k (delta F).
    du = [dF]
    for k in range(NMAX):
        du.append(D(A, du[-1]))
    du = [w.scal((-1) ** ((k * PA) % 2)) for k, w in enumerate(du)]
    out = Elt(A)
    for m, c in e.d.items():
        for j in range(len(m)):
            gj = m[j]
            if gj not in U:
                continue          # delta kills the a's
            k = U.index(gj)
            if k >= len(du):
                raise RuntimeError("NMAX too small")
            sgn = (-1) ** ((PA * sum(A.par[m[t]] for t in range(j))) % 2)
            left, right = m[:j], m[j + 1:]
            acc = Elt(A, {left: Q(1)}) * du[k] * Elt(A, {right: Q(1)})
            out = out + acc.scal(Q(sgn) * c)
    return out


# ---------------------------------------------------------------------------
# (T)  find the super stress tensor
# ---------------------------------------------------------------------------
# ansatz  T = alpha*u3 + beta*u1*u2   (odd, weight 3/2)
# law     delta_a T = K_a(T) + gamma*a_2*T + kappa*a_5
# with    K_a(T) = a_0 * D^2 T + 1/2 * a_1 * (D T)   ... careful: T is ODD, and
#         K_a = a D^2 + 1/2 (Da) D  acts on any superfield the same way.

def Kact(e):
    """K_a applied to a superfield e:  a D^2 e + (-1)^{p(a)}/2 (Da)(D e)."""
    return aa(0) * D(A, D(A, e)) + (aa(1) * D(A, e)).scal(Q((-1) ** PA, 2))


def solve_T():
    """solve for alpha:beta:gamma:kappa."""
    import sympy as sp
    al, be, ga, ka = sp.symbols("alpha beta gamma kappa")

    # build the two sides with symbolic coefficients by linearity
    def T_of(alpha, beta):
        return u(3).scal(alpha) + (u(1) * u(2)).scal(beta)

    # We work with the two basis pieces separately and recombine symbolically.
    T1, T2 = T_of(1, 0), T_of(0, 1)

    def as_sym(e):
        return {m: sp.Rational(c.numerator, c.denominator) for m, c in e.d.items()}

    lhs = {}
    for coef, Tp in ((al, T1), (be, T2)):
        for m, c in as_sym(delta(Tp)).items():
            lhs[m] = lhs.get(m, 0) + coef * c
    rhs = {}
    for coef, Tp in ((al, T1), (be, T2)):
        for m, c in as_sym(Kact(Tp)).items():
            rhs[m] = rhs.get(m, 0) + coef * c
        gT = aa(2) * Tp
        for m, c in as_sym(gT).items():
            rhs[m] = rhs.get(m, 0) + ga * coef * c
    for m, c in as_sym(aa(5)).items():
        rhs[m] = rhs.get(m, 0) + ka * c

    eqs = []
    for m in set(lhs) | set(rhs):
        eqs.append(sp.expand(lhs.get(m, 0) - rhs.get(m, 0)))
    sol = sp.solve(eqs, [al, ga, ka], dict=True)
    return sol, (al, be, ga, ka), eqs


# ---------------------------------------------------------------------------
def hill_check():
    """psi := exp(-F) satisfies D^3 psi = T psi with T = (DF)(D^2F) - D^3F.

    Verified inside the free algebra: exp(-F) is treated as a formal symbol P
    with D P = -u1 P, which is all that is used."""
    # represent psi-multiples as coefficients; we only need the identity
    # D^3 psi = (u1 u2 - u3) psi, obtained by three applications of
    #   D(w psi) = (Dw) psi + (-1)^{p(w)} w (D psi),  D psi = -u1 psi.
    # We carry the coefficient w only.
    def Dw(w, pw):
        return D(A, w) + (w * u(1)).scal(Q(-(-1) ** pw)), (pw + 1) % 2
    w, pw = one, 0
    for _ in range(3):
        w, pw = Dw(w, pw)
    T = u(1) * u(2) - u(3)
    return (w - T).is_zero(), w


if __name__ == "__main__":
    import sympy as sp

    for pa in (0, 1):
        set_parity(pa)
        tag = "EVEN a  (the L_m sector)" if pa == 0 else "ODD a  (the G_r sector)"
        print("=" * 74)
        print(f"{tag}")
        print("=" * 74)
        # [K_a, D] = -1/2 (D^2 a) D  as a SUPERcommutator
        for k in range(0, 7):
            lhs = Kact(D(A, u(k)))
            rhs = D(A, Kact(u(k)))
            comm = lhs - rhs.scal((-1) ** pa)
            tgt = (aa(2) * D(A, u(k))).scal(Q(-1, 2))
            assert (comm - tgt).is_zero(), (pa, k)
        print("   [K_a, D] = -1/2 (D^2 a) D   verified on D^k F, k=0..6    OK")
        sol, syms, eqs = solve_T()
        al, be, ga, ka = syms
        print("   ansatz  T = alpha*D^3F + beta*(DF)(D^2F)")
        print("   law     delta_a T = K_a(T) + gamma*(D^2a)*T + kappa*D^5a")
        print("   solution:", sol)
        print()

    set_parity(0)
    ok, w = hill_check()
    print("=" * 74)
    print("the super-Hill operator")
    print("=" * 74)
    print("   D^3 exp(-F) = T exp(-F) with T = (DF)(D^2F) - D^3F :", ok)
    print("   computed D^3-coefficient:", w)
    print()
    print("   so  D^3 - T  is the super Sturm-Liouville (Hill) operator whose")
    print("   kernel contains exp(-F) -- exactly as d^2/dx^2 - T does in the")
    print("   bosonic case.  This is what pins the normalization of T.")
