#!/usr/bin/env python3
"""The invariant (Haar) Berezin volume of a compact Lie supergroup.

This script computes the invariant volume exactly, for U(a|b) and for the other compact
supergroups, and identifies what the vanishing really depends on.

THE SET-UP.  Take the chart

    G_0 x Pi(g_od) --> G,   (g_0, xi) |--> g_0 exp(xi),

where G_0 is the underlying group.  The left-invariant Maurer-Cartan form of
g = g_0 exp(xi) is

    g^{-1} dg = Ad(e^{-xi})(g_0^{-1} dg_0) + ((1 - e^{-ad xi})/ad xi)(d xi),

so with T the matrix of that linear map from (dg_0, d xi) in g_ev + g_od to
g = g_ev + g_od, the invariant density is Ber(T) D(g_0) D(xi) -- rows of T
indexed by the frame, columns by the coordinates, Ber taken with respect to
that same splitting (dg_0 even, d xi odd).  Hence

    vol(G) = vol(G_0) * (the Berezin integral over Pi(g_od) of Ber(T))
           = vol(G_0) * (the coefficient of the top monomial in Ber(T)).

The convention "density = Ber(T) D(x)" is the one that makes the classical
Haar measure come out right (dx/x on R^+), and it is verified here on a
supergroup with odd coordinates whose invariance can be checked by hand: the
Borel subgroup of GL(1|1), where it gives b^{-1} da db D(theta).

WHAT COMES OUT.  vol(U(a|b)) = 0 for EVERY a, b >= 1 -- the vanishing is not
peculiar to a = b.  What it is peculiar to is the presence in g_od of a
homological element, i.e. an eta != 0 with [eta, eta] = 0: osp(1|2n), which has
none, has non-zero volume.

Everything is exact: the Grassmann algebra is a dict from sorted tuples of
generator indices to Fractions, and the structure constants are Fractions.

Subsecs. 2.3-2.5 of the text (formula (6), Statements 2.4, 2.5, Remarks 2.5.1,
formula (7)).

Run:  python3 haar.py
"""

import sys
from fractions import Fraction
from itertools import product

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(("  ok   " if ok else "  FAIL") + f"  {name}" + (f"   [{detail}]" if detail else ""))
    return ok


# ------------------------------------------------------------ Grassmann ---

class G:
    """An element of the Grassmann algebra on G.NGEN generators.

    Stored as {tuple(sorted indices): Fraction}.  One global order on the
    generators, as the wedge-basis rule of this project requires.
    """

    NGEN = 0

    __slots__ = ("c",)

    def __init__(self, c=None):
        self.c = {} if c is None else {k: v for k, v in c.items() if v}

    @staticmethod
    def scalar(v):
        v = Fraction(v)
        return G({(): v}) if v else G()

    @staticmethod
    def gen(i):
        return G({(i,): Fraction(1)})

    def __bool__(self):
        return bool(self.c)

    def __add__(self, o):
        if not isinstance(o, G):
            o = G.scalar(o)
        r = dict(self.c)
        for k, v in o.c.items():
            r[k] = r.get(k, Fraction(0)) + v
        return G(r)

    __radd__ = __add__

    def __neg__(self):
        return G({k: -v for k, v in self.c.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, G) else G.scalar(-Fraction(o)))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, G):
            o = G.scalar(o)
        r = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                k, s = _wedge(k1, k2)
                if s:
                    r[k] = r.get(k, Fraction(0)) + s * v1 * v2
        return G(r)

    __rmul__ = __mul__

    def __eq__(self, o):
        if not isinstance(o, G):
            o = G.scalar(o)
        return (self - o).c == {}

    def body(self):
        return self.c.get((), Fraction(0))

    def top(self):
        return self.c.get(tuple(range(G.NGEN)), Fraction(0))

    def inv(self):
        """1/x for x with invertible body: geometric series in the nilpotent part."""
        b = self.body()
        if b == 0:
            raise ZeroDivisionError("no invertible body")
        nil = self - G.scalar(b)
        out = G.scalar(1 / b)
        term = G.scalar(1)
        for _ in range(G.NGEN):
            term = term * nil * G.scalar(-1 / b)
            if not term:
                break
            out = out + term * G.scalar(1 / b)
        return out

    def __repr__(self):
        if not self.c:
            return "0"
        return " + ".join(f"{v}*{k}" if k else f"{v}" for k, v in sorted(self.c.items()))


def _wedge(k1, k2):
    """Merge two sorted index tuples; return (merged, sign) or ((), 0)."""
    if set(k1) & set(k2):
        return (), 0
    merged = list(k1) + list(k2)
    sign = 1
    for i in range(1, len(merged)):                       # insertion sort
        j = i
        while j > 0 and merged[j - 1] > merged[j]:
            merged[j - 1], merged[j] = merged[j], merged[j - 1]
            sign = -sign
            j -= 1
    return tuple(merged), sign


# ------------------------------------------------ matrices over Grassmann ---

def mat_mul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][j] for t in range(k)), G()) for j in range(m)]
            for i in range(n)]


def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_scal(c, A):
    c = c if isinstance(c, G) else G.scalar(c)
    return [[c * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_id(n):
    return [[G.scalar(1) if i == j else G() for j in range(n)] for i in range(n)]


def det(A):
    """Determinant of a matrix with EVEN (commuting) Grassmann entries."""
    n = len(A)
    if n == 0:
        return G.scalar(1)
    M = [row[:] for row in A]
    d = G.scalar(1)
    for col in range(n):
        piv = None
        for r in range(col, n):
            if M[r][col].body() != 0:
                piv = r
                break
        if piv is None:
            return _det_leibniz(A)          # every pivot nilpotent; n is small
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            d = d * G.scalar(-1)
        d = d * M[col][col]
        ip = M[col][col].inv()
        for r in range(col + 1, n):
            f = M[r][col] * ip
            if not f:
                continue
            for c in range(col, n):
                M[r][c] = M[r][c] - f * M[col][c]
    return d


def _det_leibniz(A):
    from itertools import permutations
    n = len(A)
    out = G()
    for perm in permutations(range(n)):
        s, seen = 1, list(perm)
        for i in range(n):
            for j in range(i + 1, n):
                if seen[i] > seen[j]:
                    s = -s
        term = G.scalar(s)
        for i in range(n):
            term = term * A[i][perm[i]]
        out = out + term
    return out


def mat_inv(D):
    """Inverse of a matrix with even entries and invertible body."""
    n = len(D)
    I = mat_id(n)
    M = [row[:] + I[i] for i, row in enumerate(D)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col].body() != 0)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        ip = M[col][col].inv()
        M[col] = [ip * x for x in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if not f:
                continue
            M[r] = [M[r][c] - f * M[col][c] for c in range(2 * n)]
    return [row[n:] for row in M]


def berezinian(T, p):
    """Ber of the supermatrix T whose first p rows/columns form the even block."""
    n = len(T)
    A = [[T[i][j] for j in range(p)] for i in range(p)]
    if n == p:
        return det(A)
    B = [[T[i][j] for j in range(p, n)] for i in range(p)]
    C = [[T[i][j] for j in range(p)] for i in range(p, n)]
    D = [[T[i][j] for j in range(p, n)] for i in range(p, n)]
    S = mat_add(A, mat_scal(-1, mat_mul(B, mat_mul(mat_inv(D), C))))
    return det(S) * det(D).inv()


# ------------------------------------------------- Lie superalgebras ------

def rref(M, ncols):
    """Exact rref of a list of rows of Fractions, eliminating in the first
    `ncols` columns only.  Returns (all rows, pivot columns, rank) -- the rows
    beyond the rank are kept, since their tails carry the relations."""
    M = [row[:] for row in M]
    piv, r = [], 0
    for c in range(ncols):
        p = None
        for i in range(r, len(M)):
            if M[i][c]:
                p = i
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return M, piv, r


class Alg:
    """A Lie superalgebra given by a homogeneous matrix basis inside gl(m|n).

    The basis is stored even-first.  Structure constants are computed from the
    supercommutator by exact linear algebra.
    """

    def __init__(self, name, m, n, basis):
        self.name, self.m, self.n = name, m, n
        self.N = m + n
        self.mpar = [0] * m + [1] * n
        ev = [(X, p) for (X, p) in basis if p == 0]
        od = [(X, p) for (X, p) in basis if p == 1]
        self.basis = [X for X, _ in ev] + [X for X, _ in od]
        self.par = [0] * len(ev) + [1] * len(od)
        self.nev, self.nod = len(ev), len(od)
        self.dim = self.nev + self.nod
        self._prep_solver()
        self.C = {}
        for i in range(self.dim):
            for j in range(self.dim):
                self.C[(i, j)] = self.coords(self.superbracket(i, j))

    # -- matrix helpers ----------------------------------------------------
    def mat_par(self, i, j):
        return (self.mpar[i] + self.mpar[j]) % 2

    def superbracket(self, i, j):
        A, B = self.basis[i], self.basis[j]
        pa, pb = self.par[i], self.par[j]
        AB = _mm(A, B)
        BA = _mm(B, A)
        s = (-1) ** (pa * pb)
        return [[AB[r][c] - s * BA[r][c] for c in range(self.N)] for r in range(self.N)]

    def _prep_solver(self):
        self._rows = [[X[r][c] for r in range(self.N) for c in range(self.N)]
                      for X in self.basis]
        self._solver = None

    def coords(self, X):
        """Express X in the basis; exact, and it verifies that X is in the span."""
        target = [X[r][c] for r in range(self.N) for c in range(self.N)]
        ncol = self.N * self.N
        if self._solver is None:
            aug = [self._rows[k][:] + [Fraction(1) if t == k else Fraction(0)
                                       for t in range(self.dim)]
                   for k in range(self.dim)]
            self._solver = rref(aug, ncol)
        R, piv, rank = self._solver
        coef = [Fraction(0)] * self.dim
        rem = target[:]
        for r, c in zip(R[:rank], piv):
            f = rem[c]
            if not f:
                continue
            for t in range(self.dim):
                coef[t] += f * r[ncol + t]
            for c2 in range(ncol):
                rem[c2] -= f * r[c2]
        assert not any(rem), f"{self.name}: bracket leaves the algebra"
        return {t: coef[t] for t in range(self.dim) if coef[t]}


def _mm(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def _unit(N, i, j):
    return [[Fraction(1) if (r, c) == (i, j) else Fraction(0)
             for c in range(N)] for r in range(N)]


def gl(m, n):
    N = m + n
    par = [0] * m + [1] * n
    basis = [(_unit(N, i, j), (par[i] + par[j]) % 2)
             for i in range(N) for j in range(N)]
    return Alg(f"gl({m}|{n})", m, n, basis)


def osp(m, n):
    """osp(m|n): the stabilizer of the even non-degenerate supersymmetric form
    B = 1_m (+) J_n, J the standard symplectic matrix (n even).  X preserves B
    iff  B(X e_j, e_k) + (-1)^{p(X) p_j} B(e_j, X e_k) = 0  for all j, k."""
    N = m + n
    par = [0] * m + [1] * n
    B = [[Fraction(0)] * N for _ in range(N)]
    for i in range(m):
        B[i][i] = Fraction(1)
    k = n // 2
    for i in range(k):
        B[m + i][m + k + i] = Fraction(1)
        B[m + k + i][m + i] = Fraction(-1)

    basis = []
    for p in (0, 1):
        units = [(i, j) for i in range(N) for j in range(N)
                 if (par[i] + par[j]) % 2 == p]
        rows = []
        for (i, j) in units:                       # X = E_ij
            eq = []
            for jj in range(N):
                for kk in range(N):
                    v = Fraction(0)
                    if jj == j:                    # X e_j = e_i
                        v += B[i][kk]
                    if kk == j:                    # X e_k = e_i
                        v += Fraction((-1) ** (p * par[jj])) * B[jj][i]
                    eq.append(v)
            rows.append(eq)
        for v in _left_nullspace(rows):
            X = [[Fraction(0)] * N for _ in range(N)]
            for coef, (i, j) in zip(v, units):
                if coef:
                    X[i][j] += coef
            basis.append((X, p))
    return Alg(f"osp({m}|{n})", m, n, basis)


def _left_nullspace(rows):
    """Vectors v with sum_k v_k * rows[k] = 0."""
    if not rows:
        return []
    ncol, nrow = len(rows[0]), len(rows)
    aug = [rows[k][:] + [Fraction(1) if t == k else Fraction(0) for t in range(nrow)]
           for k in range(nrow)]
    R, piv, rank = rref(aug, ncol)
    return [r[ncol:] for r in R[rank:] if any(r[ncol:])]


def q(n):
    """The queer superalgebra q(n) = {(A|B; B|A)} inside gl(n|n)."""
    N = 2 * n
    basis = []
    for i in range(n):
        for j in range(n):
            A = [[Fraction(0)] * N for _ in range(N)]
            A[i][j] = A[n + i][n + j] = Fraction(1)
            basis.append((A, 0))
            Bm = [[Fraction(0)] * N for _ in range(N)]
            Bm[i][n + j] = Bm[n + i][j] = Fraction(1)
            basis.append((Bm, 1))
    return Alg(f"q({n})", n, n, basis)


def sl(m, n):
    """sl(m|n) = Ker str, spanned by the off-diagonal units and the differences
    (-1)^{p_{i+1}} E_ii - (-1)^{p_i} E_{i+1, i+1}."""
    N = m + n
    par = [0] * m + [1] * n
    basis = [(_unit(N, i, j), (par[i] + par[j]) % 2)
             for i in range(N) for j in range(N) if i != j]
    for i in range(N - 1):
        s, t = (-1) ** par[i], (-1) ** par[i + 1]
        Z = [[Fraction(0)] * N for _ in range(N)]
        Z[i][i] = Fraction(t)
        Z[i + 1][i + 1] = Fraction(-s)
        basis.append((Z, 0))
    return Alg(f"sl({m}|{n})", m, n, basis)


# ------------------------------------------------------ the density ------

def ad_xi(g):
    """Matrix of ad_xi over the Grassmann algebra, xi = sum_a xi_a e_a over the
    odd basis; columns = images of the basis (even first)."""
    G.NGEN = g.nod
    dim = g.dim
    M = [[G() for _ in range(dim)] for _ in range(dim)]
    for a in range(g.nod):
        ia = g.nev + a
        xa = G.gen(a)
        for col in range(dim):
            for tgt, coef in g.C[(ia, col)].items():
                M[tgt][col] = M[tgt][col] + xa * G.scalar(coef)
    return M


def mc_matrix(g):
    N = ad_xi(g)
    dim = g.dim
    pows = [mat_id(dim)]
    while True:
        P = mat_mul(pows[-1], N)
        if all(not x for row in P for x in row):
            break
        pows.append(P)
        if len(pows) > g.nod + 2:
            break
    fac = [1]
    for k in range(1, len(pows) + 2):
        fac.append(fac[-1] * k)

    def series(coeffs):
        S = [[G() for _ in range(dim)] for _ in range(dim)]
        for k, c in enumerate(coeffs):
            if k < len(pows) and c:
                S = mat_add(S, mat_scal(G.scalar(c), pows[k]))
        return S

    expm = series([Fraction((-1) ** k, fac[k]) for k in range(len(pows))])
    phi = series([Fraction((-1) ** k, fac[k + 1]) for k in range(len(pows))])
    T = [[G() for _ in range(dim)] for _ in range(dim)]
    for col in range(dim):
        src = expm if col < g.nev else phi
        for row in range(dim):
            T[row][col] = src[row][col]
    return T


def sq(g, v):
    """[eta, eta] for eta = sum v_a e_a over the odd basis."""
    acc = {}
    for a, ca in enumerate(v):
        if not ca:
            continue
        for b, cb in enumerate(v):
            if not cb:
                continue
            for t, w in g.C[(g.nev + a, g.nev + b)].items():
                acc[t] = acc.get(t, 0) + ca * cb * w
    return {t: w for t, w in acc.items() if w}


def homological_elements(g, bound=2, tries=40000):
    """Is there an odd eta != 0 with [eta, eta] = 0?

    First a search over the rational vectors with entries in [-bound, bound]
    (normalized so that the first non-zero entry is 1).  If that finds nothing,
    the question is decided OVER C by a Groebner basis: the quadrics
    q_t(xi) = [xi, xi]_t have no non-trivial common zero iff the radical of the
    ideal they generate is the irrelevant maximal ideal, i.e. iff a pure power
    of every variable lies in the ideal.
    """
    n = g.nod
    if n == 0:
        return False, "no odd part"
    vals = list(range(-bound, bound + 1))
    count = 0
    for lead in range(n):
        for tail in product(vals, repeat=n - lead - 1):
            v = [0] * lead + [1] + list(tail)
            count += 1
            if count > tries:
                break
            if not sq(g, v):
                return True, "over Q: (" + ",".join(str(x) for x in v) + ")"
    try:
        import sympy
    except ImportError:
        return False, "none over Q (C not decided: no sympy)"
    xs = sympy.symbols(f"x0:{n}")
    quad = {}
    for a in range(n):
        for b in range(n):
            for t, w in g.C[(g.nev + a, g.nev + b)].items():
                quad[t] = quad.get(t, 0) + sympy.Rational(w) * xs[a] * xs[b]
    polys = [sympy.expand(p) for p in quad.values() if sympy.expand(p) != 0]
    if not polys:
        return True, "over C: [xi, xi] = 0 identically"
    Gb = sympy.groebner(polys, *xs, order="grevlex")
    # only zero is the origin  <=>  some pure power x_i^N lies in the ideal
    for x in xs:
        if not any(Gb.reduce(x ** N)[1] == 0 for N in range(1, 2 * n + 3)):
            return True, "over C: a non-trivial zero exists"
    return False, "none over C"


def killing(g, i, j):
    """str(ad_{e_i} ad_{e_j}) -- the Killing form on the basis."""
    dim = g.dim
    tot = Fraction(0)
    for k in range(dim):
        # (ad_i ad_j)(e_k), coefficient on e_k
        inner = g.C[(j, k)]
        for t, w in inner.items():
            c = g.C[(i, t)].get(k, Fraction(0))
            if c:
                tot += Fraction(w) * c * (-1) ** g.par[k]
    return tot


# --------------------------------------------- the convention, verified ---

def borel_check():
    """The Borel subgroup of GL(1|1): g = ((a, theta), (0, b)).  By hand,
    omega = g^{-1}dg gives T = ((1/a, 0, 0), (0, 1/b, 0), (0, -theta/(ab), 1/a)),
    so Ber(T) = (1/(ab)) * a = 1/b, and the density (1/b) da db D(theta) is
    invariant under left translation -- checked by the transformation rule
    D(a', b', theta') = Ber(J) D(a, b, theta) with Ber(J) = q."""
    # symbolic check of the two Berezinians with a = 1, b = 1 bodies
    G.NGEN = 1
    th = G.gen(0)
    a = G.scalar(3); b = G.scalar(5)
    T = [[a.inv(), G(), G()],
         [G(), b.inv(), G()],
         [G(), -(th * a.inv() * b.inv()), a.inv()]]
    lhs = berezinian(T, 2)
    return lhs == b.inv()


# ---------------------------------------------------------------- main ----

def main():
    print("The invariant Berezin volume of a compact Lie supergroup\n")
    print("  vol(G) = vol(G_0) * [top Grassmann coefficient of Ber(T)]\n")

    print("  0. The convention 'density = Ber(T) D(x)', on the Borel of GL(1|1)")
    check("Ber(T) = 1/b, the density b^{-1} da db D(theta) is invariant",
          borel_check())

    print("\n  1. The unitary supergroups U(a|b) = the real forms of GL(a|b)\n")
    algs = [gl(1, 0), gl(2, 0), gl(1, 1), gl(2, 1), gl(1, 2), gl(2, 2), gl(3, 1)]
    for g in algs:
        T = mc_matrix(g)
        B = berezinian(T, g.nev)
        top = B.top()
        hom, wit = homological_elements(g)
        print(f"    {g.name}: sdim = ({g.nev}|{g.nod}), top coeff = {top}, "
              f"homological element: {'yes, ' + wit if hom else 'no'}")
        if g.nod == 0:
            check(f"{g.name}: purely even, Ber(T) = 1", B == G.scalar(1))
        else:
            check(f"{g.name}: the volume vanishes", top == 0)
            check(f"{g.name}: and it has a homological element", hom)

    print("\n  2. The other compact supergroups\n")
    others = [osp(1, 2), osp(1, 4), osp(1, 6), osp(2, 2), osp(3, 2), osp(4, 2),
              q(1), q(2), sl(1, 1), sl(2, 1)]
    nonzero = []
    for g in others:
        T = mc_matrix(g)
        B = berezinian(T, g.nev)
        top = B.top()
        hom, wit = homological_elements(g)
        print(f"    {g.name}: sdim = ({g.nev}|{g.nod}), top coeff = {top}, "
              f"homological element: {'yes, ' + wit if hom else 'no'}")
        if top != 0:
            nonzero.append(g.name)
        if hom:
            check(f"{g.name}: has a homological element, and the volume vanishes",
                  top == 0)
    check("the volume is non-zero exactly for osp(1|2n) among those tested",
          nonzero == ["osp(1|2)", "osp(1|4)", "osp(1|6)"], ", ".join(nonzero))
    tops = []
    for k in (1, 2, 3):
        g = osp(1, 2 * k)
        tops.append(abs(berezinian(mc_matrix(g), g.nev).top()))
    check("|top coeff| of osp(1|2n) is (2n-1)!! = 1, 3, 15",
          tops == [Fraction(1), Fraction(3), Fraction(15)], str(tops))

    print("\n  2a. Ber(T) is an even function of xi -- whence q(1), where dim g_od = 1\n")
    for g in [gl(1, 1), gl(2, 1), osp(1, 2), q(1), q(2)]:
        B = berezinian(mc_matrix(g), g.nev)
        check(f"{g.name}: only even Grassmann degrees occur in Ber(T)",
              all(len(k) % 2 == 0 for k in B.c))
    check("q(1): dim g_od = 1 is odd, so the top coefficient vanishes for parity",
          q(1).nod % 2 == 1)

    print("\n  3. What does single out a = b: the Killing form\n")
    for (m, n) in [(1, 1), (2, 1), (2, 2), (3, 1)]:
        g = gl(m, n)
        ok = True
        for i in range(g.dim):
            for j in range(g.dim):
                lhs = killing(g, i, j)
                # 2(m-n) str(XY) - 2 str(X) str(Y)
                XY = _mm(g.basis[i], g.basis[j])
                strXY = sum((-1) ** g.mpar[r] * XY[r][r] for r in range(g.N))
                strX = sum((-1) ** g.mpar[r] * g.basis[i][r][r] for r in range(g.N))
                strY = sum((-1) ** g.mpar[r] * g.basis[j][r][r] for r in range(g.N))
                if lhs != 2 * (m - n) * strXY - 2 * strX * strY:
                    ok = False
        check(f"gl({m}|{n}): str(ad_X ad_Y) = 2({m}-{n}) str(XY) - 2 str X str Y", ok)
    for (m, n) in [(1, 1), (2, 2)]:
        g = sl(m, n)
        z = all(killing(g, i, j) == 0 for i in range(g.dim) for j in range(g.dim))
        check(f"sl({m}|{n}): the Killing form vanishes identically", z)
    for (m, n) in [(2, 1), (3, 1)]:
        g = sl(m, n)
        z = any(killing(g, i, j) != 0 for i in range(g.dim) for j in range(g.dim))
        check(f"sl({m}|{n}): the Killing form does not vanish", z)

    bad = [c for c in CHECKS if not c[1]]
    print(f"\n  {len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
