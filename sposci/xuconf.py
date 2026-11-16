"""Xu's conformal representations of osp(2n+2|2m) and osp(2n+3|2m), checked.

X. Xu, "Conformal oscillator representations of orthosymplectic Lie
superalgebras", J. Pure Appl. Algebra 225 (2021) 106530, eqs. (1.14)-(1.30)
(the even case, osp(2n+2|2m) on C[x_1..x_2n] (x) Lambda[th_1..th_2m]) and
(1.36)-(1.40) (the odd case, osp(2n+3|2m), one extra even indeterminate x_0).

What this engine settles, exactly (over Q, with c a free parameter, i.e., the
coefficients are polynomials in c and the identities are identities of
polynomials):

  1) pi_c really is a homomorphism osp -> End(A): for every pair of basis
     elements X, Y of osp, [pi_c(X), pi_c(Y)] = pi_c([X,Y]) as operators on
     every monomial of A of degree <= DEG.  ([X,Y] is the matrix bracket,
     re-expanded in the basis by exact linear algebra.)
  2) the matrices Xu writes down do span osp: they lie in the algebra
     preserving the form of his (1.5)-(1.8), and there are dim osp of them.
  3) at c=0 every pi_0(X) is a *derivation* of A, i.e., a vector field: so
     pi_c(X) = pi_0(X) + (multiplication by a function) and the c-family is a
     twist of a subalgebra of vect(2n|2m) by a 1-cocycle with values in O.
  4) that cocycle is the divergence: pi_c(X) - pi_0(X) = lambda * Div(pi_0(X))
     with lambda = c/(2n-2m) -- so A is the module of weighted densities
     F_lambda -- EXCEPT when n=m, where Div vanishes identically on the whole
     of osp(2n+2|2n) yet the c-family is still non-trivial.
  5) the finite-dimensional submodules of Xu's Theorem 1 at c=-l.

Run:  python3 xuconf.py        (plain python3 works too)
"""
import sys, os, itertools
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import solve, rank, backend                      # noqa: E402

# ------------------------------------------------------------------ poly in c
# a coefficient is a tuple of Fractions: (a0, a1, ...) meaning a0 + a1 c + ...

ZERO = ()
ONE = (Fraction(1),)
CVAR = (Fraction(0), Fraction(1))


def ptrim(p):
    while p and p[-1] == 0:
        p = p[:-1]
    return p


def padd(p, q):
    n = max(len(p), len(q))
    return ptrim(tuple(Fraction(p[i] if i < len(p) else 0) +
                       (q[i] if i < len(q) else 0) for i in range(n)))


def pmul(p, q):
    if not p or not q:
        return ZERO
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return ptrim(tuple(out))


def pnum(x):
    x = Fraction(x)
    return (x,) if x else ZERO


def psubs(p, val):
    """evaluate the polynomial at c = val"""
    val = Fraction(val)
    s, pw = Fraction(0), Fraction(1)
    for a in p:
        s += a * pw
        pw *= val
    return s


# ------------------------------------------------- the superalgebra A of A_bar
# a monomial is (alpha, J): alpha = tuple of exponents of the even
# indeterminates, J = increasing tuple of indices of the odd ones.
# an element is a dict {monomial: poly}, normalized (no zero coefficients).

def norm(d):
    return {k: v for k, v in d.items() if v}


def eadd(a, b):
    out = dict(a)
    for k, v in b.items():
        s = padd(out.get(k, ZERO), v)
        if s:
            out[k] = s
        elif k in out:
            del out[k]
    return out


def escale(a, p):
    return norm({k: pmul(v, p) for k, v in a.items()})


class Sup:
    """the polynomial superalgebra in nx even and nth odd indeterminates"""

    def __init__(self, nx, nth):
        self.nx, self.nth = nx, nth

    def monomials(self, deg):
        """all monomials of total degree exactly deg (deg th counts as 1)"""
        out = []
        for r in range(min(deg, self.nth) + 1):
            for J in itertools.combinations(range(self.nth), r):
                for alpha in self._compositions(deg - r, self.nx):
                    out.append((alpha, J))
        return out

    def _compositions(self, total, parts):
        if parts == 0:
            return [()] if total == 0 else []
        out = []
        for first in range(total + 1):
            for rest in self._compositions(total - first, parts - 1):
                out.append((first,) + rest)
        return out

    def parity(self, mono):
        return len(mono[1]) % 2

    # --- the primitive operators, each acting on one monomial -> element

    def mul_x(self, r):
        def op(mono):
            alpha, J = mono
            a = list(alpha)
            a[r] += 1
            return {(tuple(a), J): ONE}
        return Op(op, 0, self)

    def d_x(self, r):
        def op(mono):
            alpha, J = mono
            if alpha[r] == 0:
                return {}
            a = list(alpha)
            k = a[r]
            a[r] -= 1
            return {(tuple(a), J): pnum(k)}
        return Op(op, 0, self)

    def mul_th(self, p):
        def op(mono):
            alpha, J = mono
            if p in J:
                return {}
            sgn = sum(1 for j in J if j < p)          # move th_p past them
            Jn = tuple(sorted(J + (p,)))
            return {(alpha, Jn): pnum((-1) ** sgn)}
        return Op(op, 1, self)

    def d_th(self, p):
        def op(mono):
            alpha, J = mono
            if p not in J:
                return {}
            k = J.index(p)
            Jn = J[:k] + J[k + 1:]
            return {(alpha, Jn): pnum((-1) ** k)}
        return Op(op, 1, self)

    def ident(self):
        return Op(lambda mono: {mono: ONE}, 0, self)

    def zero(self):
        return Op(lambda mono: {}, 0, self)


class Op:
    """a linear operator on Sup, given by its action on monomials"""

    def __init__(self, fn, parity, alg):
        self.fn, self.parity, self.alg = fn, parity, alg

    def __call__(self, elt):
        """apply to an element (dict) or to a single monomial (tuple)"""
        if isinstance(elt, tuple):
            return self.fn(elt)
        out = {}
        for mono, coef in elt.items():
            out = eadd(out, escale(self.fn(mono), coef))
        return out

    def __add__(self, other):
        assert self.parity == other.parity
        return Op(lambda m: eadd(self.fn(m), other.fn(m)), self.parity, self.alg)

    def __sub__(self, other):
        return self + other.scale(pnum(-1))

    def scale(self, p):
        return Op(lambda m: escale(self.fn(m), p), self.parity, self.alg)

    def comp(self, other):
        """self o other"""
        return Op(lambda m: self(other.fn(m)),
                  (self.parity + other.parity) % 2, self.alg)

    def bracket(self, other):
        s = -1 if (self.parity and other.parity) else 1
        return self.comp(other) + other.comp(self).scale(pnum(-s))

    def is_zero(self, degs):
        for d in degs:
            for mono in self.alg.monomials(d):
                if norm(self.fn(mono)):
                    return False
        return True

    def equals(self, other, degs):
        return (self - other).is_zero(degs)


def lc(*terms):
    """linear combination: lc((coef, Op), ...)"""
    ops = [op.scale(coef) for coef, op in terms]
    out = ops[0]
    for o in ops[1:]:
        out = out + o
    return out


# ------------------------------------------------------------- gl(N0|N1) part

class Mat:
    """matrices of gl(N0|N1); rows/cols 0..N0-1 even, N0..N0+N1-1 odd"""

    def __init__(self, N0, N1):
        self.N0, self.N1, self.N = N0, N1, N0 + N1

    def par(self, i):
        return 0 if i < self.N0 else 1

    def E(self, i, j):
        return {(i, j): Fraction(1)}

    def parity(self, M):
        ps = {(self.par(i) + self.par(j)) % 2 for (i, j), v in M.items() if v}
        assert len(ps) <= 1, "inhomogeneous matrix"
        return ps.pop() if ps else 0

    def add(self, A, B):
        out = dict(A)
        for k, v in B.items():
            s = out.get(k, Fraction(0)) + v
            if s:
                out[k] = s
            elif k in out:
                del out[k]
        return out

    def scale(self, A, s):
        return {k: v * s for k, v in A.items() if v * s}

    def mul(self, A, B):
        out = {}
        for (i, j), a in A.items():
            for (k, l), b in B.items():
                if j == k:
                    key = (i, l)
                    s = out.get(key, Fraction(0)) + a * b
                    if s:
                        out[key] = s
                    elif key in out:
                        del out[key]
        return out

    def bracket(self, A, B):
        s = -1 if (self.parity(A) and self.parity(B)) else 1
        return self.add(self.mul(A, B), self.scale(self.mul(B, A), -s))

    def vec(self, A):
        return [A.get((i, j), Fraction(0))
                for i in range(self.N) for j in range(self.N)]


def in_osp(mat, M, F):
    """B(Xu,v) + (-1)^{p(X)p(u)} B(u,Xv) = 0 for all basis u,v"""
    pX = mat.parity(M)
    for u in range(mat.N):
        for v in range(mat.N):
            tot = Fraction(0)
            for i in range(mat.N):                     # X e_u = sum_i M[i,u] e_i
                tot += M.get((i, u), Fraction(0)) * F[i][v]
            sgn = -1 if (pX and mat.par(u)) else 1
            for i in range(mat.N):
                tot += sgn * F[u][i] * M.get((i, v), Fraction(0))
            if tot:
                return False
    return True


# ---------------------------------------------------------------- the model

class Xu:
    """Xu's setting for a given (n, m); odd=True adds his x_0, i.e. osp(2n+3|2m)

    Index conventions (0-based in code, Xu's are 1-based):
      even matrix indices  0 .. 2n+1   <->  Xu's 1 .. 2n+2   (plus his 0 -> 2n+2)
      odd  matrix indices  2n+2 .. 2n+1+2m  <->  Xu's 2n+2+r
      even indeterminates  x[0..2n-1]   <->  Xu's x_1..x_2n  (plus x0 if odd)
      odd  indeterminates  th[0..2m-1]  <->  Xu's theta_1..theta_2m
    """

    def __init__(self, n, m, odd=False):
        self.n, self.m, self.odd = n, m, odd
        self.nx = 2 * n + (1 if odd else 0)
        self.alg = Sup(self.nx, 2 * m)
        self.N0 = 2 * n + 2 + (1 if odd else 0)
        self.mat = Mat(self.N0, 2 * m)
        self.x0 = 2 * n if odd else None               # slot of Xu's x_0
        self.i0 = 2 * n + 2 if odd else None           # matrix row of Xu's 0
        self._form()
        self._ops()
        self._basis()

    # Xu's 1-based even index k -> our 0-based; his 0 -> self.i0
    def e(self, k):
        return self.i0 if k == 0 else k - 1

    def o(self, r):                                    # his 2n+2+r, r>=1
        return self.N0 + (r - 1)

    def X(self, r):                                    # his x_r, r>=0
        return self.x0 if r == 0 else r - 1

    def T(self, r):                                    # his theta_r, r>=1
        return r - 1

    # ------------------------------------------------------------- the form
    def _form(self):
        N = self.mat.N
        F = [[Fraction(0)] * N for _ in range(N)]
        n, m = self.n, self.m
        for i in range(1, n + 2):                      # B(e_i, e_{n+1+i}) = 1
            a, b = self.e(i), self.e(n + 1 + i)
            F[a][b] = F[b][a] = Fraction(1)
        if self.odd:
            F[self.i0][self.i0] = Fraction(1)          # his x_0^2/2 in eta
        for r in range(1, m + 1):                      # B(f_r, f_{m+r}) = -1
            a, b = self.o(r), self.o(m + r)
            F[a][b] = Fraction(-1)
            F[b][a] = Fraction(1)
        self.F = F

    # -------------------------------------------------------- the operators
    def _ops(self):
        A, n, m = self.alg, self.n, self.m
        self.mx = [A.mul_x(i) for i in range(self.nx)]
        self.dx = [A.d_x(i) for i in range(self.nx)]
        self.mt = [A.mul_th(i) for i in range(2 * m)]
        self.dt = [A.d_th(i) for i in range(2 * m)]
        self.id = A.ident()
        # D = sum x_r d_{x_r} + sum th_s d_{th_s}   (his (1.13)/(1.36))
        D = A.zero()
        for r in range(self.nx):
            D = D + self.mx[r].comp(self.dx[r])
        for s in range(2 * m):
            D = D + self.mt[s].comp(self.dt[s])
        self.D = D
        # eta = sum_{i<=n} x_i x_{n+i} + sum_{r<=m} th_r th_{m+r} (+ x_0^2/2)
        eta = A.zero()
        for i in range(1, n + 1):
            eta = eta + self.mx[self.X(i)].comp(self.mx[self.X(n + i)])
        for r in range(1, m + 1):
            eta = eta + self.mt[self.T(r)].comp(self.mt[self.T(m + r)])
        if self.odd:
            eta = eta + self.mx[self.x0].comp(self.mx[self.x0]).scale(
                pnum(Fraction(1, 2)))
        self.eta = eta
        self.Dc = D + self.id.scale(CVAR)              # D + c

    # ------------------------------------------------- the spanning elements
    def _basis(self):
        """(matrix, operator) pairs from Xu's (1.14)-(1.30) [+ (1.37)-(1.40)]"""
        n, m, mat = self.n, self.m, self.mat
        E, e, o, X, T = mat.E, self.e, self.o, self.X, self.T
        mx, dx, mt, dt = self.mx, self.dx, self.mt, self.dt
        add, sc = mat.add, mat.scale
        pairs = []

        def P(M, op):
            pairs.append((M, op))

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                # (1.14)
                P(add(E(e(i), e(j)), sc(E(e(n + 1 + j), e(n + 1 + i)), -1)),
                  mx[X(i)].comp(dx[X(j)]) - mx[X(n + j)].comp(dx[X(n + i)]))
                # (1.18), (1.20)
                P(add(E(e(i), e(n + 1 + j)), sc(E(e(j), e(n + 1 + i)), -1)),
                  mx[X(i)].comp(dx[X(n + j)]) - mx[X(j)].comp(dx[X(n + i)]))
                P(add(E(e(n + 1 + i), e(j)), sc(E(e(n + 1 + j), e(i)), -1)),
                  mx[X(n + i)].comp(dx[X(j)]) - mx[X(n + j)].comp(dx[X(i)]))
        for r in range(1, m + 1):
            for s in range(1, m + 1):
                # (1.15), (1.19), (1.21)
                P(add(E(o(r), o(s)), sc(E(o(m + s), o(m + r)), -1)),
                  mt[T(r)].comp(dt[T(s)]) - mt[T(m + s)].comp(dt[T(m + r)]))
                P(add(E(o(m + r), o(s)), E(o(m + s), o(r))),
                  mt[T(m + r)].comp(dt[T(s)]) + mt[T(m + s)].comp(dt[T(r)]))
                P(add(E(o(r), o(m + s)), E(o(s), o(m + r))),
                  mt[T(r)].comp(dt[T(m + s)]) + mt[T(s)].comp(dt[T(m + r)]))
        for i in range(1, n + 1):
            for r in range(1, m + 1):
                # (1.16), (1.17), (1.22), (1.23)
                P(add(E(e(i), o(r)), sc(E(o(m + r), e(n + 1 + i)), -1)),
                  mx[X(i)].comp(dt[T(r)]) - mt[T(m + r)].comp(dx[X(n + i)]))
                P(add(E(o(r), e(i)), E(e(n + 1 + i), o(m + r))),
                  mt[T(r)].comp(dx[X(i)]) + mx[X(n + i)].comp(dt[T(m + r)]))
                P(add(E(e(i), o(m + r)), E(o(r), e(n + 1 + i))),
                  mx[X(i)].comp(dt[T(m + r)]) + mt[T(r)].comp(dx[X(n + i)]))
                P(add(E(e(n + 1 + i), o(r)), sc(E(o(m + r), e(i)), -1)),
                  mx[X(n + i)].comp(dt[T(r)]) - mt[T(m + r)].comp(dx[X(i)]))
        # (1.24)
        P(add(E(e(n + 1), e(n + 1)), sc(E(e(2 * n + 2), e(2 * n + 2)), -1)),
          self.Dc.scale(pnum(-1)))
        for i in range(1, n + 1):
            # (1.25), (1.26)
            P(add(E(e(n + 1), e(n + 1 + i)), sc(E(e(i), e(2 * n + 2)), -1)),
              dx[X(n + i)])
            P(add(E(e(n + 1), e(i)), sc(E(e(n + 1 + i), e(2 * n + 2)), -1)),
              dx[X(i)])
            # (1.27), (1.28)
            P(add(E(e(i), e(n + 1)), sc(E(e(2 * n + 2), e(n + 1 + i)), -1)),
              self.eta.comp(dx[X(n + i)]) - mx[X(i)].comp(self.Dc))
            P(add(E(e(2 * n + 2), e(i)), sc(E(e(n + 1 + i), e(n + 1)), -1)),
              mx[X(n + i)].comp(self.Dc) - self.eta.comp(dx[X(i)]))
        for r in range(1, m + 1):
            # (1.25), (1.26) odd half
            P(add(E(e(n + 1), o(r)), sc(E(o(m + r), e(2 * n + 2)), -1)),
              dt[T(r)])
            P(add(E(e(n + 1), o(m + r)), E(o(r), e(2 * n + 2))), dt[T(m + r)])
            # (1.29), (1.30)
            P(add(E(o(r), e(n + 1)), E(e(2 * n + 2), o(m + r))),
              mt[T(r)].comp(self.Dc).scale(pnum(-1)) -
              self.eta.comp(dt[T(m + r)]))
            P(add(E(e(2 * n + 2), o(r)), sc(E(o(m + r), e(n + 1)), -1)),
              mt[T(m + r)].comp(self.Dc) - self.eta.comp(dt[T(r)]))
        if self.odd:
            for i in range(1, n + 1):
                # (1.37)
                P(add(E(e(0), e(i)), sc(E(e(n + 1 + i), e(0)), -1)),
                  mx[self.x0].comp(dx[X(i)]) - mx[X(n + i)].comp(dx[self.x0]))
                P(add(E(e(0), e(n + 1 + i)), sc(E(e(i), e(0)), -1)),
                  mx[self.x0].comp(dx[X(n + i)]) - mx[X(i)].comp(dx[self.x0]))
            # (1.38): Xu prints  x_0(D+c) - eta d_{x_0}; the two terms have to
            # be interchanged, as in his own (1.27), for the family to close.
            s38 = -1 if os.environ.get("XU138_AS_PRINTED") else 1
            P(add(E(e(0), e(n + 1)), sc(E(e(2 * n + 2), e(0)), -1)),
              (self.eta.comp(dx[self.x0]) -
               mx[self.x0].comp(self.Dc)).scale(pnum(s38)))
            P(add(E(e(n + 1), e(0)), sc(E(e(0), e(2 * n + 2)), -1)),
              dx[self.x0])
            for r in range(1, m + 1):
                # (1.39), (1.40)
                P(add(E(e(0), o(r)), sc(E(o(m + r), e(0)), -1)),
                  mx[self.x0].comp(dt[T(r)]) - mt[T(m + r)].comp(dx[self.x0]))
                # (1.40): Xu prints  x_0 d_{th_{m+r}} - th_r d_{x_0}; the
                # minus makes the family fail to close (see runxu.py) -- with
                # the plus, as in his own (1.22), everything closes.  Set
                # XU140_AS_PRINTED=1 in the environment to see the failure.
                sgn = -1 if os.environ.get("XU140_AS_PRINTED") else 1
                P(add(E(e(0), o(m + r)), E(o(r), e(0))),
                  mx[self.x0].comp(dt[T(m + r)]) +
                  mt[T(r)].comp(dx[self.x0]).scale(pnum(sgn)))

        # drop zero matrices and reduce to a basis
        keep = []
        rows = []
        for M, op in pairs:
            if not M:
                continue
            v = mat.vec(M)
            if rank(rows + [v]) > len(rows):
                rows.append(v)
                keep.append((M, op))
        self.pairs = keep
        self.rows = rows

    # ------------------------------------------------------------- utilities
    def dim_osp(self):
        n, m, N0 = self.n, self.m, self.N0
        return (N0 * (N0 - 1) // 2 + m * (2 * m + 1), N0 * 2 * m)

    def expand(self, M):
        """coordinates of M in the basis self.pairs"""
        cols = [list(col) for col in zip(*self.rows)]      # transpose
        return solve(cols, self.mat.vec(M))

    def pi(self, coords):
        op = None
        for a, (M, o) in zip(coords, self.pairs):
            if a:
                t = o.scale(pnum(a))
                op = t if op is None else op + t
        return op if op is not None else self.alg.zero()
