#!/usr/bin/env python3
"""Machine checks for the uniqueness of the central extension of gl_{p,c}.

Everything is exact (Fraction / int).  The checks are:

  (1) the cocycle (17) of the paper, written as
          c(X,Y) = str(X_{+-} Y_{-+}) - (-1)^{p(X)p(Y)} str(Y_{+-} X_{-+}),
      satisfies the super Chevalley-Eilenberg 2-cocycle identity on random
      homogeneous band matrices, for random parity functions p;

  (2) on the diagonal decomposition x_k(xi) = sum_i xi_i E_{i,i+k} one has
          x_k(xi) x_l(eta) = x_{k+l}(xi * T^k eta),
      and (17) restricted to d_k x d_{-k} is
          phi_k(xi,eta) = sum_i G_k(i) xi_i eta_{i+k},
          G_k(i) = (-1)^{p(i)} (theta(i) - theta(i+k)),  theta = 1_{i>=0};

  (3) the classification of degree-0 local 2-cocycles by their values on the
      matrix units, g(a,b) := c(E_{ab},E_{ba}):  the solution space of the
      cocycle identities is exactly
          g(a,b) = -( u_a - (-1)^{p_a+p_b} u_b ),   u in C^Z,
      i.e. it is parametrized by u modulo constants;  and the cocycle (17)
      is the member u_a = -(-1)^{p_a} theta(a);

  (4) the locality lemma:  phi_k(xi,eta) = 0 whenever (Supp xi + k) and
      Supp eta are disjoint -- checked on the family of (3).

Run:  python3 h2c.py
"""

import itertools
import random
import sys
from fractions import Fraction

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import rank, nullspace
except Exception:                                    # pragma: no cover
    rank = nullspace = None


# ---------------------------------------------------------------- matrices

class M:
    """A finitely supported Z x Z matrix: dict (i,j) -> coefficient."""

    __slots__ = ("d",)

    def __init__(self, d=None):
        self.d = {k: v for k, v in (d or {}).items() if v}

    def __add__(self, o):
        r = dict(self.d)
        for k, v in o.d.items():
            r[k] = r.get(k, 0) + v
        return M(r)

    def __sub__(self, o):
        return self + o.scale(-1)

    def scale(self, c):
        return M({k: c * v for k, v in self.d.items()})

    def __mul__(self, o):
        r = {}
        by_row = {}
        for (i, j), v in o.d.items():
            by_row.setdefault(i, []).append((j, v))
        for (i, k), v in self.d.items():
            for (j, w) in by_row.get(k, ()):
                r[(i, j)] = r.get((i, j), 0) + v * w
        return M(r)

    def __repr__(self):
        return "M(%r)" % (self.d,)


def E(i, j):
    return M({(i, j): 1})


def parity_of(m, p):
    """Parity of a homogeneous matrix, or None if it is empty."""
    par = None
    for (i, j) in m.d:
        q = (p(i) + p(j)) % 2
        if par is None:
            par = q
        elif par != q:
            raise ValueError("not homogeneous")
    return par


def bracket(x, y, p):
    px, py = parity_of(x, p), parity_of(y, p)
    if px is None or py is None:
        return M()
    s = -1 if (px and py) else 1
    return x * y - (y * x).scale(s)


def corner(m, sign):
    """sign = (+,-) means rows i >= 0, columns j < 0;  (-,+) the other one."""
    a, b = sign
    out = {}
    for (i, j), v in m.d.items():
        ok_i = (i >= 0) if a == "+" else (i < 0)
        ok_j = (j >= 0) if b == "+" else (j < 0)
        if ok_i and ok_j:
            out[(i, j)] = v
    return M(out)


def str_of(m, p):
    return sum(v * (-1) ** p(i) for (i, j), v in m.d.items() if i == j)


def coc(x, y, p):
    """The cocycle (17)."""
    px, py = parity_of(x, p), parity_of(y, p)
    if px is None or py is None:
        return 0
    s = -1 if (px and py) else 1
    return (str_of(corner(x, "+-") * corner(y, "-+"), p)
            - s * str_of(corner(y, "+-") * corner(x, "-+"), p))


# ------------------------------------------------------- (1) the identity

def cocycle_defect(x, y, z, p, c):
    """The super CE 2-cocycle identity, all three terms."""
    px, py, pz = (parity_of(t, p) for t in (x, y, z))
    if None in (px, py, pz):
        return 0
    s1 = (-1) ** (px * pz)
    s2 = (-1) ** (py * px)
    s3 = (-1) ** (pz * py)
    return (s1 * c(bracket(x, y, p), z, p)
            + s2 * c(bracket(y, z, p), x, p)
            + s3 * c(bracket(z, x, p), y, p))


def random_parity(rng, N):
    tab = {i: rng.randint(0, 1) for i in range(-N - 6, N + 7)}
    return lambda i: tab.get(i, 0)


def random_homogeneous(rng, N, p, want_parity=None, nterms=4):
    """A random band matrix, homogeneous for p, supported in [-N,N]^2."""
    cells = [(i, j) for i in range(-N, N + 1) for j in range(-N, N + 1)
             if abs(i - j) <= 3]
    rng.shuffle(cells)
    par = want_parity
    d = {}
    for (i, j) in cells:
        q = (p(i) + p(j)) % 2
        if par is None:
            par = q
        if q == par:
            d[(i, j)] = rng.randint(-3, 3)
            if len(d) >= nterms:
                break
    return M(d)


def check_identity(trials=300, seed=1):
    rng = random.Random(seed)
    bad = 0
    for t in range(trials):
        N = rng.randint(2, 5)
        p = random_parity(rng, N)
        xs = [random_homogeneous(rng, N, p, rng.randint(0, 1)) for _ in range(3)]
        if any(not x.d for x in xs):
            continue
        if cocycle_defect(xs[0], xs[1], xs[2], p, coc) != 0:
            bad += 1
    return bad


def check_skew(trials=300, seed=2):
    """c(x,y) = -(-1)^{p(x)p(y)} c(y,x)."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        N = rng.randint(2, 5)
        p = random_parity(rng, N)
        x = random_homogeneous(rng, N, p, rng.randint(0, 1))
        y = random_homogeneous(rng, N, p, rng.randint(0, 1))
        if not x.d or not y.d:
            continue
        px, py = parity_of(x, p), parity_of(y, p)
        if coc(x, y, p) != -((-1) ** (px * py)) * coc(y, x, p):
            bad += 1
    return bad


# --------------------------------- (2) the diagonal form of the cocycle

def x_k(k, xi):
    """sum_i xi_i E_{i,i+k};  xi is a dict i -> coefficient."""
    return M({(i, i + k): v for i, v in xi.items() if v})


def check_diagonal_product(trials=200, seed=3):
    """x_k(xi) x_l(eta) = x_{k+l}(xi . T^k eta)."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        k, l = rng.randint(-3, 3), rng.randint(-3, 3)
        xi = {i: rng.randint(-3, 3) for i in range(-6, 7)}
        eta = {i: rng.randint(-3, 3) for i in range(-6, 7)}
        lhs = x_k(k, xi) * x_k(l, eta)
        prod = {i: xi.get(i, 0) * eta.get(i + k, 0) for i in range(-6, 7)}
        rhs = x_k(k + l, prod)
        # compare inside the window where both are defined
        w = range(-4, 5)
        for i in w:
            if lhs.d.get((i, i + k + l), 0) != rhs.d.get((i, i + k + l), 0):
                bad += 1
                break
    return bad


def theta(i):
    return 1 if i >= 0 else 0


def check_G(trials=200, seed=4):
    """phi_k(xi,eta) = sum_i G_k(i) xi_i eta_{i+k},  G_k(i)=(-1)^{p(i)}(th(i)-th(i+k))."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        N = 6
        p = random_parity(rng, N)
        k = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        # xi, eta supported where the parity of the diagonal is constant,
        # so that x and y are homogeneous
        par = rng.randint(0, 1)
        supp_x = [i for i in range(-N, N + 1) if (p(i) + p(i + k)) % 2 == par]
        supp_y = [j for j in range(-N, N + 1) if (p(j) + p(j - k)) % 2 == par]
        if not supp_x or not supp_y:
            continue
        xi = {i: rng.randint(-3, 3) for i in supp_x}
        eta = {j: rng.randint(-3, 3) for j in supp_y}
        x, y = x_k(k, xi), x_k(-k, eta)
        lhs = coc(x, y, p)
        rhs = sum(((-1) ** p(i)) * (theta(i) - theta(i + k))
                  * xi.get(i, 0) * eta.get(i + k, 0) for i in range(-2 * N, 2 * N + 1))
        if lhs != rhs:
            bad += 1
    return bad


# ------------------------------------- (3) classification on matrix units

def classify(N, p, verbose=False):
    """Solve the cocycle identities for the unknowns

           g(a,b) = c(E_{ab}, E_{ba})   (a != b),   psi(a,b) = c(E_{aa},E_{bb}),

    on the index window [-N, N], under the degree-0 + locality ansatz
    (every other value of c on a pair of matrix units vanishes).  Returns
    (dim of the solution space, list of the u-family solutions).
    """
    idx = list(range(-N, N + 1))
    unk = {}                                     # name -> column
    for a in idx:
        for b in idx:
            if a != b:
                unk[("g", a, b)] = len(unk)
    for a in idx:
        for b in idx:
            if a < b:
                unk[("psi", a, b)] = len(unk)
    ncol = len(unk)

    def var(name):
        """Coefficient vector of one unknown, with the skew-symmetry built in."""
        kind = name[0]
        v = [0] * ncol
        if kind == "g":
            _, a, b = name
            v[unk[("g", a, b)]] = 1
        else:
            _, a, b = name
            if a == b:
                return v
            if a < b:
                v[unk[("psi", a, b)]] = 1
            else:
                # psi is skew and E_{aa}, E_{bb} are even
                v[unk[("psi", b, a)]] = -1
        return v

    def cval(x, y):
        """The unknown value c(x,y) for x,y matrix units, as a coefficient vector."""
        (i, j), = x.d
        (k, l), = y.d
        if i == j and k == l:
            return var(("psi", i, k))
        if i != j and k != l and (k, l) == (j, i):
            return var(("g", i, j))
        return [0] * ncol                        # degree 0 + locality

    def combo(m):
        """Write a finitely supported matrix as a list of (coeff, unit)."""
        return [(v, E(i, j)) for (i, j), v in m.d.items()]

    rows = []
    units = [E(a, b) for a in idx for b in idx]
    for x, y, z in itertools.product(units, repeat=3):
        px, py, pz = (parity_of(t, p) for t in (x, y, z))
        s1 = (-1) ** (px * pz)
        s2 = (-1) ** (py * px)
        s3 = (-1) ** (pz * py)
        row = [0] * ncol
        for s, (a, b, c_) in ((s1, (x, y, z)), (s2, (y, z, x)), (s3, (z, x, y))):
            br = bracket(a, b, p)
            for coefficient, unit in combo(br):
                if abs(max(unit.d)[0]) > N or abs(max(unit.d)[1]) > N:
                    continue
                v = cval(unit, c_)
                for t in range(ncol):
                    if v[t]:
                        row[t] += s * coefficient * v[t]
        if any(row):
            rows.append(row)

    if not rows:
        return ncol, unk
    ns = nullspace(rows) if nullspace else None
    return (len(ns) if ns is not None else None), unk


def u_family(N, p, u):
    """g(a,b) = -( u_a - (-1)^{p_a+p_b} u_b )."""
    out = {}
    for a in range(-N, N + 1):
        for b in range(-N, N + 1):
            if a != b:
                out[(a, b)] = -(u[a] - ((-1) ** ((p(a) + p(b)) % 2)) * u[b])
    return out


def check_u_family(N=4, seed=5, trials=30):
    """Every u gives a solution, and the cocycle (17) is u_a = -(-1)^{p_a} th(a)."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        p = random_parity(rng, N)
        u = {a: Fraction(rng.randint(-5, 5)) for a in range(-N, N + 1)}
        g = u_family(N, p, u)
        # the cocycle identity on all triples of matrix units
        for a, b, c_ in itertools.product(range(-N, N + 1), repeat=3):
            x, y, z = E(a, b), E(b, c_), E(c_, a)
            if a == b or b == c_ or c_ == a:
                continue
            val = 0
            px, py, pz = (parity_of(t, p) for t in (x, y, z))
            for s, (t1, t2, t3) in (((-1) ** (px * pz), (x, y, z)),
                                    ((-1) ** (py * px), (y, z, x)),
                                    ((-1) ** (pz * py), (z, x, y))):
                br = bracket(t1, t2, p)
                (i3, j3), = t3.d
                for (i, j), coefficient in br.d.items():
                    if i != j and (i3, j3) == (j, i):
                        val += s * coefficient * g[(i, j)]
                    elif i == j and i3 == j3:
                        pass                      # psi = 0 for this family
            if val != 0:
                bad += 1
        # the Tate member
        uT = {a: -((-1) ** p(a)) * theta(a) for a in range(-N, N + 1)}
        gT = u_family(N, p, uT)
        for a in range(-N, N + 1):
            for b in range(-N, N + 1):
                if a != b and coc(E(a, b), E(b, a), p) != gT[(a, b)]:
                    bad += 1
    return bad


# -------------------------------------------------- (4) the locality lemma

def check_locality(N=5, seed=6, trials=200):
    """phi_k(xi,eta) = 0 when (Supp xi + k) misses Supp eta -- for (17)."""
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        p = random_parity(rng, N)
        k = rng.choice([-3, -2, -1, 1, 2, 3])
        par = rng.randint(0, 1)
        sx = [i for i in range(-N, N + 1) if (p(i) + p(i + k)) % 2 == par]
        if not sx:
            continue
        A = set(rng.sample(sx, max(1, len(sx) // 2)))
        sy = [j for j in range(-N, N + 1)
              if (p(j) + p(j - k)) % 2 == par and j not in {i + k for i in A}]
        if not sy:
            continue
        xi = {i: rng.randint(-3, 3) for i in A}
        eta = {j: rng.randint(-3, 3) for j in sy}
        if coc(x_k(k, xi), x_k(-k, eta), p) != 0:
            bad += 1
    return bad


# ------------------------------------------------- (5) the commuting pair

def check_commuting_pair(N=8, seed=7, trials=60):
    """x = sum (-1)^{p(i)} E_{i,i-1} and y = sum E_{i,i+1} supercommute,
    while c(x,y) = 1.  Both lie in gl_{p,c}, the smallest of the four.

    The matrices are truncated to [-N,N]; the bracket is then correct in the
    window |j| < N, and c sees only the entries next to the origin.
    """
    rng = random.Random(seed)
    bad = 0
    for _ in range(trials):
        p = random_parity(rng, N + 3)
        xi = {i: (-1) ** p(i) for i in range(-N, N + 1)}
        eta = {j: 1 for j in range(-N, N + 1)}
        x, y = x_k(-1, xi), x_k(1, eta)
        # the bracket, computed piece by piece since x and y need not be
        # homogeneous: the (i)-piece of x pairs with the (i-1)-piece of y,
        # and both have parity p(i)+p(i-1)
        br = {}
        for i in range(-N + 1, N + 1):
            s = (p(i) + p(i - 1)) % 2
            t = xi[i] * eta.get(i - 1, 0)
            br[i] = br.get(i, 0) + t
            br[i - 1] = br.get(i - 1, 0) - ((-1) ** s) * t
        if any(v for j, v in br.items() if abs(j) < N - 1):
            bad += 1
        # the cocycle, again piece by piece
        val = 0
        for i in range(-N, N + 1):
            s = (p(i) + p(i - 1)) % 2
            xs = M({(i, i - 1): xi[i]})
            ys = M({(i - 1, i): eta.get(i - 1, 0)})
            sg = -1 if s else 1
            val += (str_of(corner(xs, "+-") * corner(ys, "-+"), p)
                    - sg * str_of(corner(ys, "+-") * corner(xs, "-+"), p))
        if val != 1:
            bad += 1
    return bad


# ---------------------------------------------------------------- driver

def main():
    print("backend:", end=" ")
    try:
        from exactlin import backend
        print(backend())
    except Exception:
        print("pure python")

    print("(1) cocycle identity for (17), 300 random triples ...", end=" ")
    print("%d failures" % check_identity())

    print("(1') superskewness of (17), 300 random pairs .......", end=" ")
    print("%d failures" % check_skew())

    print("(2) x_k(xi) x_l(eta) = x_{k+l}(xi . T^k eta) .......", end=" ")
    print("%d failures" % check_diagonal_product())

    print("(2') G_k(i) = (-1)^{p(i)}(theta(i)-theta(i+k)) .....", end=" ")
    print("%d failures" % check_G())

    print("(3) the u-family solves the identities, Tate = theta", end=" ")
    print(" %d failures" % check_u_family())

    print("(4) locality of (17) ...............................", end=" ")
    print("%d failures" % check_locality())

    print("(5) the supercommuting pair with c(x,y) = 1 ........", end=" ")
    print("%d failures" % check_commuting_pair())

    for N in (2, 3, 4):
        rng = random.Random(100 + N)
        p = random_parity(rng, N)
        dim, unk = classify(N, p)
        width = 2 * N + 1
        print("(3') window [-%d,%d]: solution space dim = %s, "
              "expected %d (= u mod constants)" % (N, N, dim, width - 1))

    print("(6) hat g(a,b)+hat g(b,t)+hat g(t,a)=0 for all 8 parity patterns:", end=" ")
    print("%d failures" % check_hat_reformulation())



# ------------------------- (6) the reformulation used in the written proof

def check_hat_reformulation():
    """With  hat g(a,b) := (-1)^{p(a)} g(a,b),  the three-term relation
    coming from the triple (E_ab, E_bc, E_ca) is exactly

        hat g(a,b) + hat g(b,c) + hat g(c,a) = 0,

    for every one of the eight parity patterns.  Checked symbolically: the
    relation is linear in the three unknowns, so it is enough to compare the
    coefficient vectors.
    """
    from fractions import Fraction as F
    bad = 0
    for pa, pb, pc in itertools.product((0, 1), repeat=3):
        p = {0: pa, 1: pb, 2: pc}
        a, b, c_ = 0, 1, 2
        pi = lambda i, j: (p[i] + p[j]) % 2
        # the identity as it comes out of the CE condition, in the unknowns
        # g(a,c), g(b,a), g(c,b)
        s_ac = (-1) ** (pi(a, b) * pi(c_, a))
        s_ba = (-1) ** (pi(b, c_) * pi(a, b))
        s_cb = (-1) ** (pi(c_, a) * pi(b, c_))
        # rewrite in the unknowns hat g(x,y) = (-1)^{p(x)} g(x,y), using
        # hat g(y,x) = -hat g(x,y)  (which is the superskewness of c)
        # g(a,c) = (-1)^{p(a)} hat g(a,c) = -(-1)^{p(a)} hat g(c,a)
        v = {}
        v[("c", "a")] = -s_ac * ((-1) ** p[a])
        v[("b", "a")] = s_ba * ((-1) ** p[b])
        v[("c", "b")] = s_cb * ((-1) ** p[c_])
        # target: hat g(a,b) + hat g(b,c) + hat g(c,a) = 0, i.e.
        #        -hat g(b,a) - hat g(c,b) + hat g(c,a) = 0
        w = {("c", "a"): 1, ("b", "a"): -1, ("c", "b"): -1}
        ratios = {F(v[k], w[k]) for k in w}
        if len(ratios) != 1 or 0 in ratios:
            bad += 1
    return bad


if __name__ == "__main__":
    main()
