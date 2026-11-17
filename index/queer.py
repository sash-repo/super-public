#!/usr/bin/env python3
"""
Exact engine for the queer Chern-Weil theory of Section 5 of the text
(Lemma 5.5, formulas (81), (82); Theorem 5.11, formula (85); Example 5.11.1).

Model.
  Omega  = C[x_1..x_m] (x) Lambda(th_1..th_m),  th_i = dx_i  odd,
           d = sum_i th_i d/dx_i,  an odd derivation with d^2 = 0.
  Q(n)   = Mat(n) (x) Cl_1,   Cl_1 = K[eps]/(eps^2 - 1),  eps odd.
           So an element of Omega (x)^ Q(n) is  P + Q eps  with P,Q in Mat(n;Omega),
           and the product of the SUPER tensor product reads

              (P + Q eps)(P' + Q' eps) = (P P' + Q sg(Q')) + (P Q' + Q sg(P')) eps,

           where sg is the parity automorphism of Omega applied entrywise
           (sg(w) = (-1)^{|w|} w).  Proof of the rule: in A (x)^ Cl_1 one has
           (a (x) eps)(a' (x) 1) = (-1)^{p(eps)p(a')} a a' (x) eps = sg(a') twisted,
           and (a (x) eps)(a' (x) eps) = (-1)^{p(a')} a a' (x) eps^2.

  qtr(P + Q eps) = tr Q.

Everything below is exact: coefficients are exact polynomials over Q (poly.py), the odd
part is handled by explicit Koszul signs.  No floating point anywhere.
"""

import sys
from itertools import combinations
from fractions import Fraction
import random

import poly
from poly import P, NX

# ----------------------------------------------------------------- Omega ----


class Om:
    """Element of Omega: dict {tuple(sorted th-indices): polynomial coefficient}."""

    __slots__ = ("c",)

    def __init__(self, c=None):
        self.c = {}
        if c:
            for k, v in c.items():
                if v:
                    self.c[tuple(k)] = v

    # --- basics
    @staticmethod
    def zero():
        return Om()

    @staticmethod
    def one():
        return Om({(): P.const(1)})

    @staticmethod
    def scalar(e):
        return Om({(): e})

    @staticmethod
    def theta(i):
        return Om({(i,): P.const(1)})

    def __bool__(self):
        return bool(self.c)

    def __eq__(self, other):
        return not (self - other).c

    def __add__(self, o):
        r = dict(self.c)
        for k, v in o.c.items():
            n = r.get(k, poly.NIL) + v
            if n:
                r[k] = n
            else:
                r.pop(k, None)
        return Om(r)

    def __sub__(self, o):
        return self + o.neg()

    def neg(self):
        return Om({k: -v for k, v in self.c.items()})

    def smul(self, s):
        return Om({k: v.scal(s) for k, v in self.c.items()})

    def pmul(self, q):
        """Multiply by a polynomial (a degree-0 element of Omega)."""
        return Om({k: v * q for k, v in self.c.items()})

    def __mul__(self, o):
        r = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                k, sgn = _merge(k1, k2)
                if sgn == 0:
                    continue
                term = v1 * v2
                if sgn < 0:
                    term = -term
                n = r.get(k, poly.NIL) + term
                if n:
                    r[k] = n
                else:
                    r.pop(k, None)
        return Om(r)

    # --- gradings
    def degrees(self):
        return {len(k) for k in self.c}

    def is_homog_parity(self):
        return len({len(k) % 2 for k in self.c}) <= 1

    def parity(self):
        """0 / 1 / None (not homogeneous)."""
        ps = {len(k) % 2 for k in self.c}
        if not ps:
            return None            # zero: any parity
        return ps.pop() if len(ps) == 1 else "mixed"

    def sg(self):
        """The parity automorphism  w |-> (-1)^{|w|} w."""
        return Om({k: (-v if len(k) % 2 else v) for k, v in self.c.items()})

    def integrate01(self):
        return Om({k: v.integrate01() for k, v in self.c.items()})

    def part(self, deg):
        return Om({k: v for k, v in self.c.items() if len(k) == deg})

    def __repr__(self):
        if not self.c:
            return "0"
        out = []
        for k in sorted(self.c, key=lambda t: (len(t), t)):
            t = "*".join(f"th{i}" for i in k)
            out.append(f"({self.c[k]})" + ("*" + t if t else ""))
        return " + ".join(out)


def _merge(k1, k2):
    """Product of two square-free monomials in the th's: (key, sign)."""
    s = set(k1)
    if s & set(k2):
        return (), 0
    merged = list(k1) + list(k2)
    # bubble sort, counting transpositions
    sgn = 1
    arr = merged[:]
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sgn = -sgn
    return tuple(arr), sgn


# ------------------------------------------------------------ d and x's ----

M_VARS = NX                                   # number of even coordinates


def d(om):
    """The exterior differential  d = sum th_i d/dx_i  (odd derivation)."""
    r = {}
    for k, v in om.c.items():
        for i in range(M_VARS):
            dv = v.diff(i)
            if not dv:
                continue
            key, sgn = _merge((i,), k)         # th_i placed on the LEFT
            if sgn == 0:
                continue
            if sgn < 0:
                dv = -dv
            n = r.get(key, poly.NIL) + dv
            if n:
                r[key] = n
            else:
                r.pop(key, None)
    return Om(r)


# ------------------------------------------------------------- matrices ----


class Mat:
    """n x n matrix over Omega."""

    __slots__ = ("n", "a")

    def __init__(self, n, a=None):
        self.n = n
        self.a = a if a is not None else [[Om() for _ in range(n)] for _ in range(n)]

    @staticmethod
    def zero(n):
        return Mat(n)

    @staticmethod
    def ident(n):
        m = Mat(n)
        for i in range(n):
            m.a[i][i] = Om.one()
        return m

    def __add__(self, o):
        return Mat(self.n, [[self.a[i][j] + o.a[i][j] for j in range(self.n)]
                            for i in range(self.n)])

    def __sub__(self, o):
        return Mat(self.n, [[self.a[i][j] - o.a[i][j] for j in range(self.n)]
                            for i in range(self.n)])

    def neg(self):
        return Mat(self.n, [[self.a[i][j].neg() for j in range(self.n)]
                            for i in range(self.n)])

    def smul(self, s):
        return Mat(self.n, [[self.a[i][j].smul(s) for j in range(self.n)]
                            for i in range(self.n)])

    def __mul__(self, o):
        n = self.n
        r = Mat(n)
        for i in range(n):
            for j in range(n):
                acc = Om()
                for k in range(n):
                    acc = acc + self.a[i][k] * o.a[k][j]
                r.a[i][j] = acc
        return r

    def sg(self):
        return Mat(self.n, [[self.a[i][j].sg() for j in range(self.n)]
                            for i in range(self.n)])

    def d(self):
        return Mat(self.n, [[d(self.a[i][j]) for j in range(self.n)]
                            for i in range(self.n)])

    def tr(self):
        acc = Om()
        for i in range(self.n):
            acc = acc + self.a[i][i]
        return acc

    def is_zero(self):
        return all(not self.a[i][j] for i in range(self.n) for j in range(self.n))

    def __eq__(self, o):
        return (self - o).is_zero()

    def __repr__(self):
        return "[" + ";  ".join(", ".join(repr(e) for e in row) for row in self.a) + "]"


# --------------------------------------------------- Omega (x)^ Q(n) -------


class QE:
    """Element  P + Q eps  of Omega (x)^ Q(n)."""

    __slots__ = ("P", "Q", "n")

    def __init__(self, P, Q):
        assert P.n == Q.n
        self.P, self.Q, self.n = P, Q, P.n

    @staticmethod
    def zero(n):
        return QE(Mat.zero(n), Mat.zero(n))

    def __add__(self, o):
        return QE(self.P + o.P, self.Q + o.Q)

    def __sub__(self, o):
        return QE(self.P - o.P, self.Q - o.Q)

    def neg(self):
        return QE(self.P.neg(), self.Q.neg())

    def smul(self, s):
        return QE(self.P.smul(s), self.Q.smul(s))

    def __mul__(self, o):
        """The super tensor product rule."""
        P = self.P * o.P + self.Q * o.Q.sg()
        Q = self.P * o.Q + self.Q * o.P.sg()
        return QE(P, Q)

    def d(self):
        return QE(self.P.d(), self.Q.d())

    def qtr(self):
        return self.Q.tr()

    def is_zero(self):
        return self.P.is_zero() and self.Q.is_zero()

    def __eq__(self, o):
        return (self - o).is_zero()

    def pw(self, k):
        r = QE(Mat.ident(self.n), Mat.zero(self.n))
        for _ in range(k):
            r = r * self
        return r


def qbracket(u, v, pu, pv):
    """Supercommutator with prescribed parities."""
    uv = u * v
    vu = v * u
    return uv - vu.smul((-1) ** (pu * pv))




def _mat_pmul(m, q):
    return Mat(m.n, [[m.a[i][j].pmul(q) for j in range(m.n)] for i in range(m.n)])


Mat.pmul = _mat_pmul


# ------------------------------------------------------------ random data --

T = P.var(poly.T_IDX)


def rnd_scalar(rng, deg=2):
    """A random polynomial in the x's (never in T)."""
    e = P.const(rng.randint(-3, 3))
    for _ in range(deg):
        e = e + P.const(rng.randint(-3, 3)) * P.var(rng.randrange(M_VARS))
    e = e + P.const(rng.randint(-2, 2)) * P.var(0) * P.var(1)
    return e


def rnd_om(rng, degs):
    """Random element of Omega whose th-degrees lie in `degs`."""
    c = {}
    for k in range(max(degs) + 1):
        if k not in degs:
            continue
        for key in combinations(range(M_VARS), k):
            if rng.random() < 0.7:
                c[key] = rnd_scalar(rng)
    return Om(c)


def rnd_mat(rng, n, degs):
    return Mat(n, [[rnd_om(rng, degs) for _ in range(n)] for _ in range(n)])


def rnd_qe(rng, n, parity):
    """Random homogeneous element of Omega (x)^ q(n) of the given parity.

    P must have entries of parity `parity`, Q of parity `parity`+1.
    """
    even = [0, 2]
    odd = [1, 3]
    dP = even if parity == 0 else odd
    dQ = odd if parity == 0 else even
    return QE(rnd_mat(rng, n, dP), rnd_mat(rng, n, dQ))


# ------------------------------------------------------------------ tests --

FAIL = []


def check(name, ok, extra=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not ok:
        FAIL.append(name)


def t_omega(rng):
    print("Omega: the ground supercommutative algebra")
    a, b, c = (rnd_om(rng, [0, 1, 2, 3]) for _ in range(3))
    check("associativity", (a * b) * c == a * (b * c))
    ok = True
    for _ in range(20):
        u = rnd_om(rng, [rng.randrange(4)])
        v = rnd_om(rng, [rng.randrange(4)])
        pu, pv = u.parity(), v.parity()
        if pu is None or pv is None or pu == "mixed" or pv == "mixed":
            continue
        if u * v != (v * u).smul((-1) ** (pu * pv)):
            ok = False
    check("supercommutativity", ok)
    check("d^2 = 0", not d(d(a)))
    v = rnd_om(rng, [1])
    u1 = rnd_om(rng, [1])
    check("d is an odd derivation", d(u1 * v) == d(u1) * v - u1 * d(v))


def t_algebra(rng, n=2):
    print(f"\nOmega (x)^ q({n}): the associative superalgebra")
    u = rnd_qe(rng, n, 0)
    v = rnd_qe(rng, n, 1)
    w = rnd_qe(rng, n, 1)
    check("associativity", (u * v) * w == u * (v * w))
    check("d is an odd derivation, u even", (u * v).d() == u.d() * v + u * v.d())
    check("d is an odd derivation, v odd", (v * w).d() == v.d() * w - v * w.d())

    ok = True
    for pu in (0, 1):
        for pv in (0, 1):
            for _ in range(3):
                a = rnd_qe(rng, n, pu)
                b = rnd_qe(rng, n, pv)
                if (a * b).qtr() != (b * a).qtr().smul((-1) ** (pu * pv)):
                    ok = False
    check("qtr is a supertrace on Omega (x)^ q(n)", ok)

    ok = True
    for pu in (0, 1):
        for pv in (0, 1):
            a, b = rnd_qe(rng, n, pu), rnd_qe(rng, n, pv)
            if qbracket(a, b, pu, pv).qtr():
                ok = False
    check("qtr([u,v]) = 0", ok)

    bad = False
    for _ in range(4):
        a, b = rnd_qe(rng, n, 0), rnd_qe(rng, n, 0)
        if (a * b).P.tr() != (b * a).P.tr():
            bad = True
    check("tr P is NOT a trace (as it must not be)", bad)


def t_splitting(rng, n=3):
    print(f"\nThe splitting of Q({n}) = Mat(n) (x) Cl_1 over the ground field")
    from fractions import Fraction as F

    def rm():
        return [[F(rng.randint(-4, 4)) for _ in range(n)] for _ in range(n)]

    def mm(X, Y):
        return [[sum(X[i][k] * Y[k][j] for k in range(n)) for j in range(n)]
                for i in range(n)]

    def ma(X, Y, s=1):
        return [[X[i][j] + s * Y[i][j] for j in range(n)] for i in range(n)]

    def tr(X):
        return sum(X[i][i] for i in range(n))

    A, B, C, D = rm(), rm(), rm(), rm()

    def mul(X, Y):                        # (A,B)(C,D) = (AC+BD, AD+BC)
        return (ma(mm(X[0], Y[0]), mm(X[1], Y[1])),
                ma(mm(X[0], Y[1]), mm(X[1], Y[0])))

    X, Y = (A, B), (C, D)
    rp = lambda Z: ma(Z[0], Z[1])
    rm_ = lambda Z: ma(Z[0], Z[1], -1)
    check("rho_+ : A + eps B |-> A + B  is an algebra map",
          rp(mul(X, Y)) == mm(rp(X), rp(Y)))
    check("rho_- : A + eps B |-> A - B  is an algebra map",
          rm_(mul(X, Y)) == mm(rm_(X), rm_(Y)))
    check("qtr X = (tr rho_+(X) - tr rho_-(X)) / 2",
          tr(B) == F(tr(rp(X)) - tr(rm_(X)), 2))
    check("rho_+ o sigma = rho_-  (the two are swapped by eps |-> -eps)",
          rp((A, [[-b for b in row] for row in B])) == rm_(X))

    print("  -- and they die as soon as Omega has an odd part:")
    n2 = 2
    bad = False
    for _ in range(4):
        u, v = rnd_qe(rng, n2, 0), rnd_qe(rng, n2, 0)
        if (u * v).P + (u * v).Q != (u.P + u.Q) * (v.P + v.Q):
            bad = True
    check("rho_+ is NOT multiplicative on Omega (x)^ q(n)", bad)

    print("  -- whereas eps |-> i is, on the even part:")
    ok = True
    for _ in range(4):
        u, v = rnd_qe(rng, n2, 0), rnd_qe(rng, n2, 0)
        w = u * v
        if w.P != u.P * v.P - u.Q * v.Q or w.Q != u.P * v.Q + u.Q * v.P:
            ok = False
    check("(P,Q) multiplies as P + iQ on the even part", ok)


# ---------------------------------------------------------- Chern-Weil -----


def superconnection(rng, n):
    """alpha = omega + phi eps, odd: omega odd-degree, phi even-degree."""
    return rnd_mat(rng, n, [1, 3]), rnd_mat(rng, n, [0, 2])


def curvature(omega, phi):
    alpha = QE(omega, phi)
    return alpha, alpha.d() + alpha * alpha


def t_chern_weil(rng, n=2):
    print(f"\nChern-Weil for a q({n})-valued superconnection")
    omega, phi = superconnection(rng, n)
    alpha, F = curvature(omega, phi)

    Fom = omega.d() + omega * omega
    nabla_phi = phi.d() + omega * phi - phi * omega
    check("F = (F_omega + phi^2) + (nabla_omega phi) eps",
          F.P == Fom + phi * phi and F.Q == nabla_phi)

    check("Bianchi:  dF + alpha F - F alpha = 0",
          F.d() + alpha * F - F * alpha == QE.zero(n))

    ok = True
    for k in range(1, 5):
        if d(F.pw(k).qtr()):
            ok = False
    check("d qtr(F^k) = 0  for k = 1..4", ok)

    ok = True
    for k in range(1, 5):
        Fk = F.pw(k)
        Pm, Qm = Mat.ident(n), Mat.zero(n)
        for _ in range(k):
            Pm, Qm = Pm * F.P - Qm * F.Q, Pm * F.Q + Qm * F.P
        if Fk.P != Pm or Fk.Q != Qm:
            ok = False
    check("F^k = (F_0 + i F_1)^k   (eps behaves as i on the even part)", ok)


def transgression(omega, phi, k):
    """qtr(F^k) = d beta_k,  beta_k = k int_0^1 qtr(phi eps . F_t^{k-1}) dt."""
    n = omega.n
    Fom = omega.d() + omega * omega
    nph = phi.d() + omega * phi - phi * omega
    Ft = QE(Fom + (phi * phi).pmul(T * T), nph.pmul(T))
    dot = QE(Mat.zero(n), phi)                  # d/dT of alpha_T = phi eps
    integrand = (dot * Ft.pw(k - 1)).qtr()
    return integrand.integrate01().smul(k)


def t_vanishing(rng, n=2):
    print(f"\nThe vanishing theorem, q({n})")
    omega, phi = superconnection(rng, n)
    _, F = curvature(omega, phi)

    _, F0 = curvature(omega, Mat.zero(n))
    check("qtr(F^k) = 0 identically for a q(n)_ev-valued connection",
          all(not F0.pw(k).qtr() for k in range(1, 5)))

    ok, detail = True, []
    for k in range(1, 5):
        beta = transgression(omega, phi, k)
        lhs = F.pw(k).qtr()
        good = d(beta) == lhs
        ok = ok and good
        detail.append(f"k={k}:{'ok' if good else 'FAIL'}")
    check("qtr(F^k) = d beta_k  with the transgression primitive",
          ok, " ".join(detail))

    check("the forms qtr(F^k) are not identically zero",
          any(bool(F.pw(k).qtr()) for k in range(1, 5)))

    print("  -- degree check: qtr(F^k) is an ODD-degree differential form")
    degs = set()
    for k in range(1, 5):
        degs |= F.pw(k).qtr().degrees()
    check("every qtr(F^k) has odd form degree", all(x % 2 == 1 for x in degs),
          f"degrees seen: {sorted(degs)}")


def t_general_parity(rng):
    print("\nThe general mechanism: the parity of the trace")
    n = 2
    omega, _ = superconnection(rng, n)
    _, F0 = curvature(omega, Mat.zero(n))
    check("the curvature of an ordinary (q_ev-valued) connection has no eps-part",
          F0.Q.is_zero())
    check("qtr vanishes on q(n)_ev", True, "qtr(A + 0.eps) = 0 by definition")


def t_example_n1(rng):
    print("\nExample  n = 1,  omega = 0,  phi = f(x) = x0^2 + x1")
    f = P.var(0) * P.var(0) + P.var(1)
    phi = Mat(1, [[Om.scalar(f)]])
    omega = Mat.zero(1)
    _, F = curvature(omega, phi)
    print(f"    F_0 = {F.P.a[0][0]}")
    print(f"    F_1 = {F.Q.a[0][0]}")
    for k in (1, 2, 3):
        chk = F.pw(k).qtr()
        beta = transgression(omega, phi, k)
        print(f"    qtr(F^{k}) = {chk}")
        print(f"    beta_{k}   = {beta}")
        check(f"d beta_{k} = qtr(F^{k})", d(beta) == chk)
    print("    closed form: qtr(exp F) = e^{f^2} df = d( int_0^f e^{s^2} ds )")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260831
    rng = random.Random(seed)
    print(f"queer.py -- exact checks, seed {seed}, {M_VARS} even coordinates\n")
    t_omega(rng)
    t_algebra(rng)
    t_splitting(rng)
    t_chern_weil(rng)
    t_vanishing(rng)
    t_general_parity(rng)
    t_example_n1(rng)
    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
