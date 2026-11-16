#!/usr/bin/env python3
"""
n2.py -- the classical anomaly of the N=2 super Liouville equation (IK2).

Exact symbolic computation in the FREE SUPERCOMMUTATIVE algebra on the towers
of derivatives of

    v    (chiral:      Db v = 0),        v+   (antichiral:  Dd v+ = 0),
    f    (unconstrained generating function of the contact field K_f),

with the two odd derivations of eq. (Dcx) of the paper,

    Dd := D_{th1} - i D_{th2},   Db := D_{th1} + i D_{th2},
    Dd^2 = Db^2 = 0,             Dd Db + Db Dd = -4 d/dt.

No coordinates and no components: everything below is an identity in that free
algebra, which is what makes it convention-free.  The generators are

    v_k := d^k v,        w_k  := d^k (Dd v)          (v is chiral)
    V_k := d^k v+,       W_k  := d^k (Db v+)         (v+ is antichiral)
    f_k := d^k f,  g_k := d^k(Dd f),  G_k := d^k(Db f),  h_k := d^k(Dd Db f)

and the action of Dd, Db on them is forced by the two relations above
(e.g. Db w_k = Db Dd d^k v = (-4 d - Dd Db) d^k v = -4 v_{k+1}).

What is computed:

 (K)  K_f = 2 f d - (-1)^{p(f)}/2 [ (Dd f) Db + (Db f) Dd ]  is the contact
      field whose multiplier is the one of eq. (multC) of the paper:
      [K_f, Db] = lambda(f) Db and [K_f, Dd] = lambdabar(f) Dd with
      lambda = 1/2 Db Dd f, lambdabar = 1/2 Dd Db f, lambda + lambdabar = -2 df.

 (T)  the super stress tensor of eq. (IK2) and its anomalous transformation law:
      the ansatz
          T = alpha (Dd v)(Db v+) + beta d v + gamma d v+
          delta_f T = K_f(T) + P lambda T + Q lambdabar T
                      + kappa1 d lambda + kappa2 d lambdabar
      is solved for all seven unknowns, for f even and for f odd separately.

 (H)  the normalization: the operator identity that plays the role of the
      (super) Hill operator of Proposition 2.3.4 and Theorem 4.2.1.

Usage:  python3 n2.py
"""

from fractions import Fraction as Q
import sys

# ---------------------------------------------------------------------------
# a free supercommutative algebra on named generators with parities
# ---------------------------------------------------------------------------


class Alg:
    def __init__(self):
        self.names = []
        self.par = []
        self.idx = {}

    def gen(self, name, parity):
        if name in self.idx:
            return self.idx[name]
        i = len(self.names)
        self.names.append(name)
        self.par.append(parity % 2)
        self.idx[name] = i
        return i

    def mono_mul(self, m1, m2):
        sign = 1
        for i in m1:
            if self.par[i] == 0:
                continue
            for j in m2:
                if self.par[j] == 1 and j < i:
                    sign = -sign
        m = tuple(sorted(m1 + m2))
        for k in range(len(m) - 1):
            if m[k] == m[k + 1] and self.par[m[k]] == 1:
                return None
        return sign, m

    def par_mono(self, m):
        return sum(self.par[i] for i in m) % 2


class Elt:
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


def deriv(A, rule, parity, e):
    """apply the derivation given by  rule: gen index -> Elt  to e.

    parity 0/1 is the parity of the derivation (Koszul sign on the prefix)."""
    out = Elt(A)
    for m, c in e.d.items():
        for j in range(len(m)):
            img = rule.get(m[j])
            if img is None or img.is_zero():
                continue
            sgn = 1
            if parity:
                sgn = (-1) ** (sum(A.par[m[t]] for t in range(j)) % 2)
            left = Elt(A, {m[:j]: Q(1)})
            right = Elt(A, {m[j + 1:]: Q(1)})
            out = out + (left * img * right).scal(Q(sgn) * c)
    return out


# ---------------------------------------------------------------------------
# the algebra of the problem
# ---------------------------------------------------------------------------

NMAX = 7          # how many d-derivatives of each tower we keep


def build(pf):
    """pf = parity of the generating function f."""
    A = Alg()
    v = [A.gen(f"v{k}", 0) for k in range(NMAX + 1)]        # d^k v
    w = [A.gen(f"w{k}", 1) for k in range(NMAX + 1)]        # d^k Dd v
    V = [A.gen(f"V{k}", 0) for k in range(NMAX + 1)]        # d^k v+
    W = [A.gen(f"W{k}", 1) for k in range(NMAX + 1)]        # d^k Db v+
    f = [A.gen(f"f{k}", pf) for k in range(NMAX + 1)]       # d^k f
    g = [A.gen(f"g{k}", 1 - pf) for k in range(NMAX + 1)]   # d^k Dd f
    G = [A.gen(f"G{k}", 1 - pf) for k in range(NMAX + 1)]   # d^k Db f
    h = [A.gen(f"h{k}", pf) for k in range(NMAX + 1)]       # d^k Dd Db f

    E = lambda i: Elt.g(A, i)
    Z = Elt(A)

    dt, Dd, Db = {}, {}, {}
    for k in range(NMAX + 1):
        nxt = k + 1 if k < NMAX else None
        for tower in (v, w, V, W, f, g, G, h):
            dt[tower[k]] = E(tower[k + 1]) if nxt is not None else Z
        # Dd
        Dd[v[k]] = E(w[k])
        Dd[w[k]] = Z
        Dd[V[k]] = Z
        Dd[W[k]] = E(V[k + 1]).scal(-4) if nxt is not None else None
        Dd[f[k]] = E(g[k])
        Dd[g[k]] = Z
        Dd[G[k]] = E(h[k])
        Dd[h[k]] = Z
        # Db
        Db[v[k]] = Z
        Db[w[k]] = E(v[k + 1]).scal(-4) if nxt is not None else None
        Db[V[k]] = E(W[k])
        Db[W[k]] = Z
        Db[f[k]] = E(G[k])
        Db[g[k]] = (E(f[k + 1]).scal(-4) - E(h[k])) if nxt is not None else None
        Db[G[k]] = Z
        Db[h[k]] = E(G[k + 1]).scal(-4) if nxt is not None else None
    # None marks "would need index NMAX+1": treat as truncation, flagged on use
    return A, dict(v=v, w=w, V=V, W=W, f=f, g=g, G=G, h=h), dt, Dd, Db


class Ctx:
    def __init__(self, pf):
        self.pf = pf
        self.A, self.T, self.dtr, self.Ddr, self.Dbr = build(pf)
        A = self.A
        self.E = lambda i: Elt.g(A, i)
        self.zero = Elt(A)
        self.one = Elt.const(A, 1)

    # -- the three derivations, with truncation guard ------------------------
    def _rule(self, rule):
        out = {}
        for k, val in rule.items():
            out[k] = self.zero if val is None else val
        return out

    def d(self, e):
        self._guard(e, self.dtr)
        return deriv(self.A, self._rule(self.dtr), 0, e)

    def Dd(self, e):
        self._guard(e, self.Ddr)
        return deriv(self.A, self._rule(self.Ddr), 1, e)

    def Db(self, e):
        self._guard(e, self.Dbr)
        return deriv(self.A, self._rule(self.Dbr), 1, e)

    def _guard(self, e, rule):
        for m in e.d:
            for i in m:
                if rule.get(i, "missing") is None:
                    raise RuntimeError("NMAX too small: " + self.A.names[i])

    # -- named generators ---------------------------------------------------
    def gen(self, tower, k):
        return self.E(self.T[tower][k])

    # -- the contact field --------------------------------------------------
    def K(self, e):
        """K_f(e) = 2 f d(e) - (-1)^{p(f)}/2 [ (Dd f) Db(e) + (Db f) Dd(e) ]."""
        f0 = self.gen("f", 0)
        g0 = self.gen("g", 0)
        G0 = self.gen("G", 0)
        s = Q((-1) ** self.pf, 2)
        return (f0 * self.d(e)).scal(2) - (g0 * self.Db(e) + G0 * self.Dd(e)).scal(s)

    def lam(self):
        """lambda(f) = 1/2 Db Dd f = -2 f1 - 1/2 h0."""
        return self.Db(self.Dd(self.gen("f", 0))).scal(Q(1, 2))

    def lamb(self):
        """lambdabar(f) = 1/2 Dd Db f = 1/2 h0."""
        return self.Dd(self.Db(self.gen("f", 0))).scal(Q(1, 2))

    # -- the variation ------------------------------------------------------
    def delta_rule(self):
        """delta(v_k) = d^k(dv), delta(w_k) = (-1)^{p(f)} d^k Dd(dv), etc."""
        dv = self.K(self.gen("v", 0)) + self.lam()
        dV = self.K(self.gen("V", 0)) + self.lamb()
        sg = (-1) ** self.pf
        rule = {}
        cur_v, cur_V = dv, dV
        cur_w, cur_W = self.Dd(dv).scal(sg), self.Db(dV).scal(sg)
        for k in range(NMAX + 1):
            rule[self.T["v"][k]] = cur_v
            rule[self.T["V"][k]] = cur_V
            rule[self.T["w"][k]] = cur_w
            rule[self.T["W"][k]] = cur_W
            if k < NMAX:
                cur_v, cur_V = self.d(cur_v), self.d(cur_V)
                cur_w, cur_W = self.d(cur_w), self.d(cur_W)
        for t in ("f", "g", "G", "h"):
            for k in range(NMAX + 1):
                rule[self.T[t][k]] = self.zero
        return rule

    def delta(self, e):
        if not hasattr(self, "_drule"):
            self._drule = self.delta_rule()
        return deriv(self.A, self._drule, self.pf, e)


# ---------------------------------------------------------------------------
# checks of the ambient algebra
# ---------------------------------------------------------------------------

def check_algebra(C):
    """Dd^2 = Db^2 = 0 and Dd Db + Db Dd = -4 d, on generators and products."""
    n = 0
    gens = [C.E(i) for i in range(len(C.A.names))
            if not C.A.names[i].endswith(str(NMAX))]
    tests = list(gens)
    # a few products, to test the derivation property as well
    for a in gens[:12]:
        for b in gens[:12]:
            tests.append(a * b)
    for e in tests:
        assert C.Dd(C.Dd(e)).is_zero()
        assert C.Db(C.Db(e)).is_zero()
        n += 2
    gens2 = [C.E(i) for i in range(len(C.A.names))
             if not (C.A.names[i].endswith(str(NMAX))
                     or C.A.names[i].endswith(str(NMAX - 1)))]
    for e in gens2 + [a * b for a in gens2[:10] for b in gens2[:10]]:
        lhs = C.Dd(C.Db(e)) + C.Db(C.Dd(e))
        rhs = C.d(e).scal(-4)
        assert (lhs - rhs).is_zero(), C.A.names
        n += 1
    return n


def check_derivation(C, op, parity):
    """op is a derivation of the given parity."""
    n = 0
    gens = [C.E(i) for i in range(len(C.A.names))
            if not C.A.names[i].endswith(str(NMAX))]
    for a in gens[:16]:
        for b in gens[:16]:
            pa = C.A.par_mono(next(iter(a.d)))
            lhs = op(a * b)
            rhs = op(a) * b + (a * op(b)).scal((-1) ** ((parity * pa) % 2))
            assert (lhs - rhs).is_zero(), (a, b)
            n += 1
    return n


def check_multiplier(C):
    """[K_f, Db] = lambda(f) Db  and  [K_f, Dd] = lambdabar(f) Dd."""
    n = 0
    lam, lamb = C.lam(), C.lamb()
    gens = [C.E(i) for i in range(len(C.A.names))
            if not (C.A.names[i].endswith(str(NMAX))
                    or C.A.names[i].endswith(str(NMAX - 1)))]
    sg = (-1) ** C.pf                      # K_f has parity p(f), Db is odd
    for e in gens:
        comm = C.K(C.Db(e)) - C.Db(C.K(e)).scal(sg)
        assert (comm - lam * C.Db(e)).is_zero(), ("Db", e, comm)
        comm = C.K(C.Dd(e)) - C.Dd(C.K(e)).scal(sg)
        assert (comm - lamb * C.Dd(e)).is_zero(), ("Dd", e, comm)
        n += 2
    return n


def check_constraints(C):
    """the variation preserves chirality of v and antichirality of v+."""
    dv = C.K(C.gen("v", 0)) + C.lam()
    dV = C.K(C.gen("V", 0)) + C.lamb()
    assert C.Db(dv).is_zero(), C.Db(dv)
    assert C.Dd(dV).is_zero(), C.Dd(dV)
    # and it is compatible with the tower relations Db w_k = -4 v_{k+1} etc.
    n = 2
    for k in range(NMAX - 1):
        lhs = C.Db(C.delta(C.gen("w", k))).scal((-1) ** C.pf)
        rhs = C.delta(C.Db(C.gen("w", k)))
        assert (lhs - rhs).is_zero(), ("w", k, lhs - rhs)
        lhs = C.Dd(C.delta(C.gen("W", k))).scal((-1) ** C.pf)
        rhs = C.delta(C.Dd(C.gen("W", k)))
        assert (lhs - rhs).is_zero(), ("W", k, lhs - rhs)
        n += 2
    return n


# ---------------------------------------------------------------------------
# (T) the stress tensor and its law
# ---------------------------------------------------------------------------

def solve_T(C):
    import sympy as sp
    al, be, ga, P, Qs, k1, k2 = sp.symbols("alpha beta gamma P Q kappa1 kappa2")

    T_parts = [C.gen("w", 0) * C.gen("W", 0), C.gen("v", 1), C.gen("V", 1)]
    coeffs = [al, be, ga]

    def sym(e):
        return {m: sp.Rational(c.numerator, c.denominator) for m, c in e.d.items()}

    def acc(dst, e, factor):
        for m, c in sym(e).items():
            dst[m] = dst.get(m, 0) + factor * c

    lam, lamb = C.lam(), C.lamb()

    lhs, rhs = {}, {}
    for co, Tp in zip(coeffs, T_parts):
        acc(lhs, C.delta(Tp), co)
        acc(rhs, C.K(Tp), co)
        acc(rhs, lam * Tp, P * co)
        acc(rhs, lamb * Tp, Qs * co)
    acc(rhs, C.d(lam), k1)
    acc(rhs, C.d(lamb), k2)

    eqs = [sp.expand(lhs.get(m, 0) - rhs.get(m, 0)) for m in set(lhs) | set(rhs)]
    eqs = [e for e in eqs if e != 0]
    sol = sp.solve(eqs, [al, be, ga, P, Qs, k1, k2], dict=True)
    return sol, (al, be, ga, P, Qs, k1, k2), eqs


# ---------------------------------------------------------------------------
# (H) the normalization: a Hill-type operator identity
# ---------------------------------------------------------------------------

def hill(C, mu, nu, x, y):
    """Psi := exp(mu v + nu v+);  return Psi^{-1} (x Db Dd + y Dd Db) Psi.

    Computed inside the free algebra: Psi is a formal symbol with
    Dd Psi = mu (Dd v) Psi, Db Psi = nu (Db v+) Psi; we carry the coefficient."""
    mu, nu = Q(mu), Q(nu)
    w0, W0 = C.gen("w", 0), C.gen("W", 0)

    # a "coefficient" c stands for the element c*Psi; the two odd derivations act
    # by  Dd(c Psi) = (Dd c) Psi + (-1)^{p(c)} c mu w0 Psi, and likewise for Db.
    def Dd_(c, pc):
        return C.Dd(c) + (c * w0).scal(Q((-1) ** pc) * mu), (pc + 1) % 2

    def Db_(c, pc):
        return C.Db(c) + (c * W0).scal(Q((-1) ** pc) * nu), (pc + 1) % 2

    c1, p1 = Dd_(C.one, 0)
    c1, p1 = Db_(c1, p1)          # Db Dd Psi / Psi
    c2, p2 = Db_(C.one, 0)
    c2, p2 = Dd_(c2, p2)          # Dd Db Psi / Psi
    return c1.scal(Q(x)) + c2.scal(Q(y))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sympy as sp

    print("=" * 74)
    print("N = 2 super Liouville: the stress tensor and its anomaly")
    print("=" * 74)

    sols = {}
    for pf in (0, 1):
        C = Ctx(pf)
        tag = "EVEN f (the L_m, J_m sectors)" if pf == 0 else "ODD f (the G_r sectors)"
        print()
        print("-" * 74)
        print(tag)
        print("-" * 74)
        n = check_algebra(C)
        print(f"   Dd^2 = Db^2 = 0, {{Dd,Db}} = -4 d          {n} checks   OK")
        n = check_derivation(C, C.Dd, 1) + check_derivation(C, C.Db, 1)
        n += check_derivation(C, C.d, 0) + check_derivation(C, C.K, pf)
        print(f"   d, Dd, Db, K_f are derivations            {n} checks   OK")
        n = check_multiplier(C)
        print(f"   [K_f,Db] = lambda Db, [K_f,Dd] = lambdabar Dd   {n} checks   OK")
        lam, lamb = C.lam(), C.lamb()
        assert (lam + lamb - C.d(C.gen("f", 0)).scal(-2)).is_zero()
        print("   lambda + lambdabar = -2 df                            OK")
        n = check_constraints(C)
        print(f"   the variation preserves (anti)chirality   {n} checks   OK")

        sol, syms, eqs = solve_T(C)
        print()
        print("   ansatz   T = alpha (Dd v)(Db v+) + beta dv + gamma dv+")
        print("   law      delta_f T = K_f(T) + P lambda T + Q lambdabar T")
        print("                        + kappa1 d lambda + kappa2 d lambdabar")
        print("   solution:", sol)
        sols[pf] = sol

    print()
    print("=" * 74)
    print("the Hill-type normalization")
    print("=" * 74)
    C = Ctx(0)
    for (mu, nu, x, y) in ((-1, -1, 1, -1), (-1, -1, 1, 0), (-1, -1, 0, 1),
                           (-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1)):
        e = hill(C, mu, nu, x, y)
        print(f"   Psi = exp({mu} v + {nu} v+),  ({x}) Db Dd + ({y}) Dd Db :  {e}")
