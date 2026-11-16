#!/usr/bin/env python3
r"""
abfock.py -- the Fock spaces of the antibracket, exactly.

The objects (notation of [LSh] = Leites-Shchepochkina, TMF 126:3 (2001) 339-369,
"How to quantize the antibracket", Subsecs. P.5.2 and 4.3):

  ab(n) = W + C.z,  W of sdim (n|n) spanned by the EVEN q_1..q_n and the ODD
  theta_1..theta_n, B the non-degenerate anti-symmetric ODD form with
  B(q_i, theta_j) = delta_ij,  and  [v, w] = B(v, w) z  with z ODD and central.

Because B is odd, z is odd; hence the "Planck constant" hbar = the scalar by
which z acts is an ODD parameter, hbar^2 = 0, and the base ring is the
supercommutative C = Lambda(hbar), not a field.

The n+1 Fock modules of Theorem 6.2 of [LSh]:

  L_i  = Span(q_1..q_i, theta_{i+1}..theta_n)     (the annihilators)
  M_i  = Span(theta_1..theta_i, q_{i+1}..q_n)     (the creators)
  F_i  = C[lambda_1..lambda_i ; x_{i+1}..x_n],  the free C-module on them, with

     q_j     |->  hbar d/d lambda_j     (j <= i),      q_j |-> x_j        (j > i)
     theta_j |->  lambda_j              (j <= i),  theta_j |-> -hbar d/dx_j (j > i)

so F_i is the space of functions on a superspace of sdim (n-i | i): each of the
n+1 has exactly n indeterminates, half the 2n of b(n).

EVERYTHING here is exact: rational coefficients, one global order on the odd
generators (hbar first, then lambda_1 < ... < lambda_i), no floating point.

A module element is a dict
      {(odd_tuple, x_exponents): Fraction}
where odd_tuple is a strictly increasing tuple of odd-generator indices, the
generator 0 being hbar and the generator j (1<=j<=i) being lambda_j, and
x_exponents is a tuple of length n-i.
"""

from fractions import Fraction
from itertools import combinations


# ---------------------------------------------------------------- the module

class Fock:
    """The free C-module F_i with its operators.  n >= 1, 0 <= i <= n."""

    def __init__(self, n, i):
        assert 0 <= i <= n
        self.n, self.i = n, i
        self.nx = n - i                      # number of even indeterminates
        # odd generators: 0 = hbar, 1..i = lambda_1..lambda_i
        self.nodd = i + 1

    # -- basis ------------------------------------------------------------
    def basis(self, maxdeg=2, with_hbar=True):
        """Basis monomials with total x-degree <= maxdeg."""
        out = []
        odd_gens = list(range(0 if with_hbar else 1, self.nodd))
        for k in range(len(odd_gens) + 1):
            for S in combinations(odd_gens, k):
                for a in self._exps(maxdeg):
                    out.append((S, a))
        return out

    def _exps(self, maxdeg):
        if self.nx == 0:
            yield ()
            return
        def rec(k, left):
            if k == self.nx - 1:
                for e in range(left + 1):
                    yield (e,)
                return
            for e in range(left + 1):
                for rest in rec(k + 1, left - e):
                    yield (e,) + rest
        yield from rec(0, maxdeg)

    def mono(self, S, a, c=1):
        return {(tuple(S), tuple(a)): Fraction(c)}

    def one(self):
        return self.mono((), (0,) * self.nx)

    # -- linear algebra over Q --------------------------------------------
    @staticmethod
    def add(f, g, c=1):
        out = dict(f)
        for k, v in g.items():
            out[k] = out.get(k, Fraction(0)) + Fraction(c) * v
            if out[k] == 0:
                del out[k]
        return out

    @staticmethod
    def scal(f, c):
        if c == 0:
            return {}
        return {k: Fraction(c) * v for k, v in f.items()}

    @staticmethod
    def iszero(f):
        return all(v == 0 for v in f.values())

    # -- the elementary operators -----------------------------------------
    # Every operator is a python function  dict -> dict,  and carries a parity.

    def mul_odd(self, g):
        """multiplication by the odd generator g (0 = hbar, j = lambda_j)."""
        def op(f):
            out = {}
            for (S, a), c in f.items():
                if g in S:
                    continue                      # square of an odd generator
                # sign: move g from the left past the generators smaller than g
                sgn = (-1) ** sum(1 for s in S if s < g)
                T = tuple(sorted(S + (g,)))
                out[(T, a)] = out.get((T, a), Fraction(0)) + sgn * c
            return {k: v for k, v in out.items() if v != 0}
        return op

    def d_odd(self, g):
        """d/d(odd generator g), an odd derivation (left derivative)."""
        def op(f):
            out = {}
            for (S, a), c in f.items():
                if g not in S:
                    continue
                pos = S.index(g)
                sgn = (-1) ** pos
                T = tuple(s for s in S if s != g)
                out[(T, a)] = out.get((T, a), Fraction(0)) + sgn * c
            return {k: v for k, v in out.items() if v != 0}
        return op

    def mul_x(self, j):
        """multiplication by x_j  (j an index into 0..nx-1)."""
        def op(f):
            out = {}
            for (S, a), c in f.items():
                b = list(a)
                b[j] += 1
                out[(S, tuple(b))] = out.get((S, tuple(b)), Fraction(0)) + c
            return out
        return op

    def d_x(self, j):
        """d/dx_j, an even derivation."""
        def op(f):
            out = {}
            for (S, a), c in f.items():
                if a[j] == 0:
                    continue
                b = list(a)
                e = b[j]
                b[j] -= 1
                k = (S, tuple(b))
                out[k] = out.get(k, Fraction(0)) + e * c
            return {k: v for k, v in out.items() if v != 0}
        return op

    # -- composition of operators -----------------------------------------
    @staticmethod
    def comp(*ops):
        def op(f):
            for o in reversed(ops):
                f = o(f)
            return f
        return op

    @staticmethod
    def opadd(*terms):
        """terms = (coefficient, operator) pairs."""
        def op(f):
            out = {}
            for c, o in terms:
                out = Fock.add(out, o(f), c)
            return out
        return op

    def bracket(self, A, pA, Bop, pB):
        """super-commutator [A,B] = AB - (-1)^{pA pB} BA."""
        s = -((-1) ** (pA * pB))
        return self.opadd((1, self.comp(A, Bop)), (s, self.comp(Bop, A)))

    # -- the representation of ab(n) --------------------------------------
    def rho(self, kind, j):
        """kind = 'q' or 'th', j = 1..n.  Returns (operator, parity)."""
        i, n = self.i, self.n
        h = self.mul_odd(0)
        if kind == 'q':
            if j <= i:
                return self.comp(h, self.d_odd(j)), 0
            return self.mul_x(j - i - 1), 0
        else:
            if j <= i:
                return self.mul_odd(j), 1
            return self.opadd((-1, self.comp(h, self.d_x(j - i - 1)))), 1

    def in_L(self, kind, j):
        """is the generator in the annihilator subspace L_i?"""
        return (kind == 'q' and j <= self.i) or (kind == 'th' and j > self.i)

    def gens(self):
        for j in range(1, self.n + 1):
            yield ('q', j)
        for j in range(1, self.n + 1):
            yield ('th', j)


def parity(kind):
    return 0 if kind == 'q' else 1


def Bform(v, w):
    """B(v,w) for v,w generators;  B(q_i,theta_j)=delta, B(theta_j,q_i)=-delta."""
    (kv, jv), (kw, jw) = v, w
    if jv != jw:
        return 0
    if kv == 'q' and kw == 'th':
        return 1
    if kv == 'th' and kw == 'q':
        return -1
    return 0


# ------------------------------------------------------- pe(n) = aut(B)

def pe_basis(n):
    """
    A basis of pe(n) = aut(B), SOLVED for from the invariance equations
    rather than quoted.  An element is stored as a dict
        X[(kind, j)] = {(kind', k): coefficient}
    together with its parity.
    """
    gens = [('q', j) for j in range(1, n + 1)] + [('th', j) for j in range(1, n + 1)]
    out = []
    for p in (0, 1):
        # unknowns: X(g) = sum_h c[g,h] h  with p(h) = p(g) + p
        unk = [(g, h) for g in gens for h in gens
               if (parity(h[0]) - parity(g[0])) % 2 == p % 2]
        idx = {u: t for t, u in enumerate(unk)}
        rows = []
        for a in range(len(gens)):
            for b in range(a, len(gens)):
                v, w = gens[a], gens[b]
                # B(Xv, w) + (-1)^{p * p(v)} B(v, Xw) = 0
                row = [Fraction(0)] * len(unk)
                for h in gens:
                    if (v, h) in idx:
                        row[idx[(v, h)]] += Bform(h, w)
                sgn = (-1) ** (p * parity(v[0]))
                for h in gens:
                    if (w, h) in idx:
                        row[idx[(w, h)]] += sgn * Bform(v, h)
                if any(x != 0 for x in row):
                    rows.append(row)
        for vec in nullspace(rows, len(unk)):
            X = {}
            for u, t in idx.items():
                if vec[t] != 0:
                    X.setdefault(u[0], {})[u[1]] = vec[t]
            out.append((X, p))
    return out


def apply_X(X, g):
    """X(g) as a dict of generators -> coefficients."""
    return X.get(g, {})


def pe_bracket(X, pX, Y, pY, n):
    """[X,Y] = XY - (-1)^{pX pY} YX as operators on W."""
    gens = [('q', j) for j in range(1, n + 1)] + [('th', j) for j in range(1, n + 1)]
    Z = {}
    s = (-1) ** (pX * pY)
    for g in gens:
        acc = {}
        for h, c in apply_X(Y, g).items():
            for k, d in apply_X(X, h).items():
                acc[k] = acc.get(k, Fraction(0)) + c * d
        for h, c in apply_X(X, g).items():
            for k, d in apply_X(Y, h).items():
                acc[k] = acc.get(k, Fraction(0)) - s * c * d
        acc = {k: v for k, v in acc.items() if v != 0}
        if acc:
            Z[g] = acc
    return Z, (pX + pY) % 2


def stabilizes_L(X, n, i):
    """does X preserve L_i = Span(q_1..q_i, theta_{i+1}..theta_n)?"""
    L = [('q', j) for j in range(1, i + 1)] + [('th', j) for j in range(i + 1, n + 1)]
    Lset = set(L)
    for g in L:
        for h, c in apply_X(X, g).items():
            if c != 0 and h not in Lset:
                return False
    return True


# ----------------------------------------------------------- linear algebra

def nullspace(rows, ncols):
    """Exact nullspace over Q; returns a list of basis vectors (as rows)."""
    M = [list(map(Fraction, r)) for r in rows]
    piv = []
    r = 0
    for c in range(ncols):
        p = None
        for rr in range(r, len(M)):
            if M[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for rr in range(len(M)):
            if rr != r and M[rr][c] != 0:
                f = M[rr][c]
                M[rr] = [a - f * b for a, b in zip(M[rr], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for fc in free:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for rr, pc in enumerate(piv):
            v[pc] = -M[rr][fc]
        out.append(v)
    return out


def solve(rows, rhs, ncols):
    """One solution of rows * v = rhs, or None."""
    M = [list(map(Fraction, r)) + [Fraction(b)] for r, b in zip(rows, rhs)]
    piv = []
    r = 0
    for c in range(ncols):
        p = None
        for rr in range(r, len(M)):
            if M[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for rr in range(len(M)):
            if rr != r and M[rr][c] != 0:
                f = M[rr][c]
                M[rr] = [a - f * b for a, b in zip(M[rr], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    for rr in range(len(M)):
        if all(M[rr][c] == 0 for c in range(ncols)) and M[rr][ncols] != 0:
            return None
    v = [Fraction(0)] * ncols
    for rr, pc in enumerate(piv):
        v[pc] = M[rr][ncols]
    return v
