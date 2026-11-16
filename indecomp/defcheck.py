#!/usr/bin/env python3
r"""
defcheck.py -- exact check of the H^1 Claims of Section 4 of
"Supervarieties of indecomposable modules over Lie superalgebras and
Grassmann superalgebras".

What it computes
----------------
H^1(g; M), split by parity, for

    g = q(n), sq(n), pe(n), spe(n), sl(1|1), gl(1|1)
    M = gl(n|n) (tautological V (x) V^*),  End(g),  1,  Pi(1)

Method
------
* Everything sits inside gl(n|n); the bracket is the supercommutator
      [x,y] = xy - (-1)^{p(x)p(y)} yx .
* A g-module is given by a representation rho on a graded space; the
  modules we need are End(V) = V \otimes V^* with the action
      x . f = rho(x) f - (-1)^{p(x)p(f)} f rho(x).
* Odd cochains c: g -> M are handled as even cochains c: g -> Pi(M),
  where Pi(M) has parity shifted by 1 and the action twisted by
  (-1)^{p(x)}.  So only the even-cochain differential is ever used:
      (dc)(x,y) = x.c(y) - (-1)^{p(x)p(y)} y.c(x) - c([x,y]).
  (Pi(M) is a module: both sides of the module axiom pick up the same
  sign (-1)^{p(x)+p(y)}.)
* Reduction to weight 0: a torus h inside g_ev acts semisimply on the
  cochain complex and, by the Cartan homotopy formula
  L_h = d i_h + i_h d, acts as 0 on cohomology.  Hence
      H^*(g;M) = H^*(g;M)_0 ,
  and only weight-zero cochains have to be considered.  This is what
  makes n = 4,5 feasible.  The full (unrestricted) computation is run
  as a cross-check whenever it is small enough.
* Ranks are computed over GF(p) for two large primes; the structure
  constants are small integers, so agreement of the two is a reliable
  substitute for the (much slower) exact rank over Q.  Small cases are
  additionally redone over Q with Fractions.

Usage:  python3 defcheck.py          (runs the whole check suite)
"""

import itertools
import sys
from fractions import Fraction

import numpy as np

PRIMES = (2147483647, 2147483629)


# --------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------
def rank_mod(A, p):
    """Rank of the integer matrix A over GF(p)."""
    if A.size == 0:
        return 0
    M = np.mod(A.astype(np.int64), p)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, c]), p - 2, p)
        M[r] = np.mod(M[r] * inv, p)
        col = M[r + 1:, c].copy()
        nz = np.nonzero(col)[0]
        if nz.size:
            M[r + 1 + nz] = np.mod(M[r + 1 + nz] - np.outer(col[nz], M[r]), p)
        r += 1
        if r == rows:
            break
    return r


def rank_exact(A):
    """Rank of an integer matrix over Q (Fractions); for small matrices."""
    M = [[Fraction(int(v)) for v in row] for row in A]
    rows = len(M)
    cols = len(M[0]) if rows else 0
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(r + 1, rows):
            if M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
    return r


def rank_checked(A, exact=False):
    """Rank over two primes (and over Q if exact=True); they must agree."""
    if A.size == 0:
        return 0
    ranks = [rank_mod(A, p) for p in PRIMES]
    assert ranks[0] == ranks[1], f"prime disagreement {ranks}"
    if exact:
        re = rank_exact(A)
        assert re == ranks[0], f"GF(p) rank {ranks[0]} != exact rank {re}"
    return ranks[0]


# --------------------------------------------------------------------------
# gl(n|n) and its subalgebras
# --------------------------------------------------------------------------
def par_idx(i, n):
    return 0 if i < n else 1


def unit(N, i, j):
    m = np.zeros((N, N), dtype=np.int64)
    m[i, j] = 1
    return m


def sbr(x, px, y, py):
    """supercommutator"""
    return x @ y - (-1) ** (px * py) * y @ x


class Alg:
    """A Lie superalgebra given by a homogeneous basis of matrices in gl(N)."""

    def __init__(self, name, N, basis, torus_idx, n=None):
        self.name = name
        self.N = N
        self.basis = [(np.array(m, dtype=np.int64), p) for m, p in basis]
        self.dim = len(self.basis)
        self.par = [p for _, p in self.basis]
        self.n = n
        self._coord_setup()
        self.torus = [self.basis[i][0] for i in torus_idx]
        self.weights = self._weights()
        self._check_closed()

    # -- coordinates of an arbitrary matrix in the basis ------------------
    def _coord_setup(self):
        N = self.N
        G = [[Fraction(int(m[a, b])) for m, _ in self.basis]
             for a in range(N) for b in range(N)]
        # row-reduce [G | I] to get a left inverse on the column span
        rows = len(G)
        cols = self.dim
        aug = [G[i] + [Fraction(1) if j == i else Fraction(0)
                       for j in range(rows)] for i in range(rows)]
        r = 0
        pivots = []
        for c in range(cols):
            piv = next((i for i in range(r, rows) if aug[i][c] != 0), None)
            if piv is None:
                raise ValueError(f"{self.name}: basis is not independent")
            aug[r], aug[piv] = aug[piv], aug[r]
            pv = aug[r][c]
            aug[r] = [v / pv for v in aug[r]]
            for i in range(rows):
                if i != r and aug[i][c] != 0:
                    f = aug[i][c]
                    aug[i] = [a - f * b for a, b in zip(aug[i], aug[r])]
            pivots.append(r)
            r += 1
        self._solver = [aug[i][cols:] for i in pivots]  # dim x rows

    def coords(self, m):
        """Express matrix m in the basis; raises if m is not in the span."""
        flat = [Fraction(int(m[a, b])) for a in range(self.N)
                for b in range(self.N)]
        out = [sum(s[k] * flat[k] for k in range(len(flat)) if s[k] != 0)
               for s in self._solver]
        # verify
        rec = np.zeros((self.N, self.N), dtype=object)
        for c, (bm, _) in zip(out, self.basis):
            if c != 0:
                rec = rec + c * bm
        for a in range(self.N):
            for b in range(self.N):
                if rec[a, b] != flat[a * self.N + b]:
                    raise ValueError(f"{self.name}: element outside the algebra")
        return out

    def _check_closed(self):
        for i, (x, px) in enumerate(self.basis):
            for j, (y, py) in enumerate(self.basis):
                self.coords(sbr(x, px, y, py))  # raises if not closed

    def _weights(self):
        ws = []
        for x, px in self.basis:
            w = []
            for h in self.torus:
                c = sbr(h, 0, x, px)
                nz = np.nonzero(x)
                a, b = nz[0][0], nz[1][0]
                lam = Fraction(int(c[a, b]), int(x[a, b]))
                assert np.array_equal(c, np.array(lam * x, dtype=np.int64)), \
                    f"{self.name}: basis element is not a weight vector"
                w.append(lam)
            ws.append(tuple(w))
        return ws

    def bracket_coords(self, i, j):
        x, px = self.basis[i]
        y, py = self.basis[j]
        return self.coords(sbr(x, px, y, py))


def gl(n, m=0):
    """gl(n|m) with the matrix-unit basis; torus = diagonal."""
    N = n + m
    basis, torus = [], []
    for a in range(N):
        for b in range(N):
            p = (par_idx(a, n) + par_idx(b, n)) % 2
            basis.append((unit(N, a, b), p))
            if a == b:
                torus.append(len(basis) - 1)
    return Alg(f"gl({n}|{m})", N, basis, torus, n=n)


def q(n, special=False):
    N = 2 * n
    basis, torus = [], []
    for i in range(n):
        for j in range(n):
            basis.append((unit(N, i, j) + unit(N, n + i, n + j), 0))
            if i == j:
                torus.append(len(basis) - 1)
    for i in range(n):
        for j in range(n):
            if special and i == j:
                continue
            basis.append((unit(N, i, n + j) + unit(N, n + i, j), 1))
    if special:
        for i in range(n - 1):
            basis.append((unit(N, i, n + i) + unit(N, n + i, i)
                          - unit(N, i + 1, n + i + 1) - unit(N, n + i + 1, i + 1), 1))
    return Alg(f"{'s' if special else ''}q({n})", N, basis, torus, n=n)


def pe(n, special=False):
    N = 2 * n
    basis, torus = [], []
    for i in range(n):
        for j in range(n):
            if special and i == j:
                continue
            basis.append((unit(N, i, j) - unit(N, n + j, n + i), 0))
            if i == j:
                torus.append(len(basis) - 1)
    if special:
        for i in range(n - 1):
            basis.append((unit(N, i, i) - unit(N, n + i, n + i)
                          - unit(N, i + 1, i + 1) + unit(N, n + i + 1, n + i + 1), 0))
            torus.append(len(basis) - 1)
    for i in range(n):                       # B = B^t
        for j in range(i, n):
            m = unit(N, i, n + j) + (unit(N, j, n + i) if i != j else 0)
            basis.append((m, 1))
    for i in range(n):                       # C = -C^t
        for j in range(i + 1, n):
            basis.append((unit(N, n + i, j) - unit(N, n + j, i), 1))
    return Alg(f"{'s' if special else ''}pe({n})", N, basis, torus, n=n)


def sl11():
    N = 2
    basis = [(unit(N, 0, 0) + unit(N, 1, 1), 0),   # E
             (unit(N, 1, 0), 1),                   # X_-
             (unit(N, 0, 1), 1)]                   # X_+
    return Alg("sl(1|1)", N, basis, [0], n=1)


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------
class Module:
    """basis vectors carry (parity, weight); act[k] = matrix of x_k."""

    def __init__(self, name, par, weights, act):
        self.name = name
        self.par = par
        self.weights = weights
        self.act = act
        self.dim = len(par)


def endo_module(g, rho, vpar, vwt, name):
    """End(V) = V (x) V^*, action x.f = rho(x) f - (-1)^{p(x)p(f)} f rho(x)."""
    d = len(vpar)
    par, wt, idx = [], [], {}
    for a in range(d):
        for b in range(d):
            idx[(a, b)] = len(par)
            par.append((vpar[a] + vpar[b]) % 2)
            wt.append(tuple(x - y for x, y in zip(vwt[a], vwt[b])))
    act = []
    for k in range(g.dim):
        R = rho[k]
        pk = g.par[k]
        A = np.zeros((len(par), len(par)), dtype=np.int64)
        for a in range(d):
            for b in range(d):
                col = idx[(a, b)]
                pf = (vpar[a] + vpar[b]) % 2
                # rho(x) E_{ab}: sum_c R[c,a] E_{cb}
                for c in range(d):
                    if R[c, a]:
                        A[idx[(c, b)], col] += R[c, a]
                # - (-1)^{p(x)p(f)} E_{ab} rho(x) = ... sum_c R[b,c] E_{ac}
                s = (-1) ** (pk * pf)
                for c in range(d):
                    if R[b, c]:
                        A[idx[(a, c)], col] -= s * R[b, c]
        act.append(A)
    return Module(name, par, wt, act)


def taut_endo(g):
    """M = gl(n|n) = End(V), V the tautological gl(n|n)-module."""
    n = g.n
    N = g.N
    vpar = [par_idx(a, n) for a in range(N)]
    # weight of the basis vector e_a of V under the torus of g
    vwt = []
    for a in range(N):
        vwt.append(tuple(Fraction(int(h[a, a])) for h in g.torus))
    rho = [g.basis[k][0] for k in range(g.dim)]
    return endo_module(g, rho, vpar, vwt, f"gl({n}|{n})")


def adjoint_endo(g):
    """M = End(g) = g (x) g^*, g in the adjoint representation."""
    rho = []
    for k in range(g.dim):
        A = np.zeros((g.dim, g.dim), dtype=np.int64)
        for j in range(g.dim):
            for i, c in enumerate(g.bracket_coords(k, j)):
                if c != 0:
                    assert c.denominator == 1
                    A[i, j] = int(c)
        rho.append(A)
    return endo_module(g, rho, list(g.par), list(g.weights), f"End({g.name})")


def trivial_module(g, odd=False):
    z = tuple(Fraction(0) for _ in g.torus)
    return Module("Pi(1)" if odd else "1", [1 if odd else 0], [z],
                  [np.zeros((1, 1), dtype=np.int64) for _ in range(g.dim)])


def parity_shift(g, M):
    """Pi(M): parity +1, action twisted by (-1)^{p(x)}."""
    act = [((-1) ** g.par[k]) * M.act[k] for k in range(g.dim)]
    return Module("Pi(" + M.name + ")", [(p + 1) % 2 for p in M.par],
                  list(M.weights), act)


# --------------------------------------------------------------------------
# H^1
# --------------------------------------------------------------------------
def h1_even(g, M, restrict_weights=True, exact=False):
    """dim H^1(g;M)_even  (even cochains g -> M)."""
    # ---- weight-zero cochain basis --------------------------------------
    coch = []
    for k in range(g.dim):
        for j in range(M.dim):
            if M.par[j] != g.par[k]:
                continue
            if restrict_weights and M.weights[j] != g.weights[k]:
                continue
            coch.append((k, j))
    pos = {kj: i for i, kj in enumerate(coch)}
    ncol = len(coch)
    if ncol == 0:
        return 0

    # ---- cocycle equations ----------------------------------------------
    rows = []
    for k in range(g.dim):
        for l in range(k, g.dim):
            pk, pl = g.par[k], g.par[l]
            blk = np.zeros((M.dim, ncol), dtype=np.int64)
            for j in range(M.dim):                      # x_k . c(x_l)
                if (l, j) in pos:
                    blk[:, pos[(l, j)]] += M.act[k][:, j]
            s = (-1) ** (pk * pl)
            for j in range(M.dim):                      # -(-1)^{..} x_l . c(x_k)
                if (k, j) in pos:
                    blk[:, pos[(k, j)]] -= s * M.act[l][:, j]
            for i, c in enumerate(g.bracket_coords(k, l)):   # - c([x_k,x_l])
                if c != 0:
                    assert c.denominator == 1
                    for j in range(M.dim):
                        if (i, j) in pos:
                            blk[j, pos[(i, j)]] -= int(c)
            nz = np.any(blk != 0, axis=1)
            if nz.any():
                rows.append(blk[nz])
    A = np.vstack(rows) if rows else np.zeros((0, ncol), dtype=np.int64)
    dimZ = ncol - rank_checked(A, exact=exact)

    # ---- coboundaries ----------------------------------------------------
    zero = tuple(Fraction(0) for _ in g.torus)
    brows = []
    for j in range(M.dim):
        if M.par[j] != 0:
            continue
        if restrict_weights and M.weights[j] != zero:
            continue
        v = np.zeros(ncol, dtype=np.int64)
        ok = True
        for k in range(g.dim):
            col = M.act[k][:, j]
            for i in np.nonzero(col)[0]:
                if (k, i) in pos:
                    v[pos[(k, i)]] += col[i]
                else:
                    ok = False
        assert ok or not restrict_weights, "coboundary left the weight-0 space"
        brows.append(v)
    B = np.array(brows, dtype=np.int64) if brows else np.zeros((0, ncol), dtype=np.int64)
    dimB = rank_checked(B, exact=exact)
    return dimZ - dimB


def H1(g, M, restrict_weights=True, exact=False):
    """(even, odd) dimensions of H^1(g;M)."""
    ev = h1_even(g, M, restrict_weights, exact)
    od = h1_even(g, parity_shift(g, M), restrict_weights, exact)
    return ev, od


def nullspace_mod(A, ncol, p):
    """Basis of the nullspace of A over GF(p), as rows."""
    if A.size == 0:
        return np.eye(ncol, dtype=np.int64)
    M = np.mod(A.astype(np.int64), p)
    rows = M.shape[0]
    r = 0
    piv_cols = []
    for c in range(ncol):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = np.mod(M[r] * pow(int(M[r, c]), p - 2, p), p)
        col = M[:, c].copy()
        col[r] = 0
        nz = np.nonzero(col)[0]
        if nz.size:
            M[nz] = np.mod(M[nz] - np.outer(col[nz], M[r]), p)
        piv_cols.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(ncol) if c not in piv_cols]
    basis = []
    for f in free:
        v = np.zeros(ncol, dtype=np.int64)
        v[f] = 1
        for i, c in enumerate(piv_cols):
            v[c] = (-int(M[i, f])) % p
        basis.append(v)
    return np.array(basis, dtype=np.int64) if basis else np.zeros((0, ncol), dtype=np.int64)


def h1_reps(g, M, gnames, mnames, restrict_weights=True, p=PRIMES[0]):
    """Explicit representatives of H^1(g;M)_even, as printable strings."""
    coch = []
    for k in range(g.dim):
        for j in range(M.dim):
            if M.par[j] != g.par[k]:
                continue
            if restrict_weights and M.weights[j] != g.weights[k]:
                continue
            coch.append((k, j))
    pos = {kj: i for i, kj in enumerate(coch)}
    ncol = len(coch)
    if ncol == 0:
        return []
    rows = []
    for k in range(g.dim):
        for l in range(k, g.dim):
            pk, pl = g.par[k], g.par[l]
            blk = np.zeros((M.dim, ncol), dtype=np.int64)
            for j in range(M.dim):
                if (l, j) in pos:
                    blk[:, pos[(l, j)]] += M.act[k][:, j]
            s = (-1) ** (pk * pl)
            for j in range(M.dim):
                if (k, j) in pos:
                    blk[:, pos[(k, j)]] -= s * M.act[l][:, j]
            for i, c in enumerate(g.bracket_coords(k, l)):
                if c != 0:
                    for j in range(M.dim):
                        if (i, j) in pos:
                            blk[j, pos[(i, j)]] -= int(c)
            nz = np.any(blk != 0, axis=1)
            if nz.any():
                rows.append(blk[nz])
    A = np.vstack(rows) if rows else np.zeros((0, ncol), dtype=np.int64)
    Z = nullspace_mod(A, ncol, p)
    zero = tuple(Fraction(0) for _ in g.torus)
    brows = []
    for j in range(M.dim):
        if M.par[j] != 0 or (restrict_weights and M.weights[j] != zero):
            continue
        v = np.zeros(ncol, dtype=np.int64)
        for k in range(g.dim):
            col = M.act[k][:, j]
            for i in np.nonzero(col)[0]:
                if (k, i) in pos:
                    v[pos[(k, i)]] += col[i]
        brows.append(v)
    B = np.array(brows, dtype=np.int64) if brows else np.zeros((0, ncol), dtype=np.int64)
    rB = rank_mod(B, p)
    out, cur = [], B
    for z in Z:
        trial = np.vstack([cur, z.reshape(1, -1)])
        if rank_mod(trial, p) > rank_mod(cur, p):
            cur = trial
            # clear denominators: find the smallest multiplier making every
            # entry a small centred integer
            for mult in range(1, 25):
                zz = np.mod(z * mult, p)
                cent = np.where(zz > p // 2, zz - p, zz)
                if np.all(np.abs(cent) < 10 ** 4):
                    z = np.mod(zz, p)
                    break
            terms = []
            for (k, j), v in zip(coch, z):
                v = int(v) % p
                if v == 0:
                    continue
                v = v - p if v > p // 2 else v
                c = "" if v == 1 else ("-" if v == -1 else f"{v} ")
                terms.append(f"{c}{mnames[j]} (x) {gnames[k]}^")
            out.append(" + ".join(terms).replace("+ -", "- "))
    assert len(out) == Z.shape[0] - rB
    return out


def test_cochain(g, M, c, p=PRIMES[0]):
    """c: dict {index of g basis -> vector of length M.dim}.
    Returns (is_cocycle, is_coboundary) for the EVEN cochain c: g -> M."""
    def cv(k):
        return c.get(k, np.zeros(M.dim, dtype=np.int64))
    ok = True
    for k in range(g.dim):
        for l in range(k, g.dim):
            s = (-1) ** (g.par[k] * g.par[l])
            val = M.act[k] @ cv(l) - s * (M.act[l] @ cv(k))
            for i, co in enumerate(g.bracket_coords(k, l)):
                if co != 0:
                    val = val - int(co) * cv(i)
            if np.any(val % p):
                ok = False
    # coboundary test: is c in the span of {x -> x.m}?
    cols, target = [], []
    for k in range(g.dim):
        target.append(cv(k))
    target = np.concatenate(target)
    for j in range(M.dim):
        if M.par[j] != 0:
            continue
        v = np.concatenate([M.act[k][:, j] for k in range(g.dim)])
        cols.append(v)
    B = np.array(cols, dtype=np.int64)
    r0 = rank_mod(B, p)
    r1 = rank_mod(np.vstack([B, target.reshape(1, -1)]), p)
    return ok, (r1 == r0)


def check_c2(n, signs=(1, -1)):
    """The paper's cocycle c_2 for q(n), as printed:

        c_2 = sum_i (m_{i,n+i} - m_{n+i,i})  (x)  hat(sum_i Pi(e_{i,i})) ,

    i.e. the value is the sign-twisted odd unit and the functional is the
    queertrace.  signs=(1,1) gives the value bar1 = Pi(1_n) that the
    shorthand `bar1 (x) hat1' at the end of the Claim names instead.
    """
    g = q(n)
    M = taut_endo(g)                      # parity-preserving cochain
    N = 2 * n
    val = np.zeros(M.dim, dtype=np.int64)
    for i in range(n):
        val[i * N + (n + i)] = signs[0]
        val[(n + i) * N + i] = signs[1]
    # odd basis elements of q(n) start at index n*n; (0,e_{ij}) at n*n + i*n + j
    c = {n * n + i * n + i: val for i in range(n)}   # functional = queertrace
    return g, M, c


def check_sq2_claim():
    """The two cochains printed in the Claim for sq(2), decoded.

    Basis of sq(2) in this file:
        0:e_{11} 1:e_{12} 2:e_{21} 3:e_{22} 4:Pi(e_{12}) 5:Pi(e_{21})
        6:Pi(e_{11}-e_{22})
    The Claim prints `Pi(e_{1,3}-e_{2,4})', which is out of range for q(2)
    (whose matrix units carry indices 1,2); we read it as Pi(e_{11}-e_{22}),
    the remaining basis vector of sq(2)_odd.  The two evident TeX typos
    (a stray \\otimes and a missing parenthesis) are repaired.
    """
    g = q(2, special=True)
    M = parity_shift(g, taut_endo(g))      # both cochains reverse parity
    N = 4

    def m(a, b, c=1):
        v = np.zeros(M.dim, dtype=np.int64)
        v[(a - 1) * N + (b - 1)] = c
        return v

    def add(*vs):
        out = np.zeros(M.dim, dtype=np.int64)
        for v in vs:
            out = out + v
        return out

    diag = add(m(1, 3, -1), m(2, 4, -1), m(3, 1), m(4, 2))
    c1 = {6: add(m(1, 1, 2), m(4, 4, 2)),
          5: add(m(2, 1, 2), m(4, 3, -2)),
          0: diag, 3: diag,
          1: m(3, 2, -2), 2: m(4, 1, 2)}
    diag2 = add(m(1, 3), m(2, 4), m(3, 1, -1), m(4, 2, -1))
    c2 = {6: add(m(3, 3, 2), m(4, 4, -2)),
          4: m(3, 4, 2), 5: m(4, 3, 2),
          0: diag2, 3: diag2}
    for name, c in (("c_1", c1), ("c_2", c2)):
        cocy, cobo = test_cochain(g, M, c)
        verdict = ("generates H^1" if cocy and not cobo
                   else "a coboundary" if cocy else "NOT a cocycle")
        print(f"     sq(2) {name} as printed: cocycle {cocy!s:5s} -> {verdict}")
    csum = dict(c1)
    for k, v in c2.items():
        csum[k] = csum.get(k, np.zeros(M.dim, dtype=np.int64)) + v
    cocy, cobo = test_cochain(g, M, csum)
    print(f"     sq(2) c_1+c_2            : cocycle {cocy!s:5s}")


def q_names(n, special=False):
    g = []
    for i in range(n):
        for j in range(n):
            g.append(f"e_{{{i+1},{j+1}}}")
    for i in range(n):
        for j in range(n):
            if special and i == j:
                continue
            g.append(f"Pi(e_{{{i+1},{j+1}}})")
    if special:
        for i in range(n - 1):
            g.append(f"Pi(e_{{{i+1},{i+1}}}-e_{{{i+2},{i+2}}})")
    return g


def m_names(N):
    return [f"m_{{{a+1},{b+1}}}" for a in range(N) for b in range(N)]


def show(g, M, expect=None, restrict_weights=True, exact=False, note=""):
    ev, od = H1(g, M, restrict_weights, exact)
    got = f"{ev}|{od}"
    flag = ""
    if expect is not None:
        flag = "  OK" if got == expect else f"  <-- MISMATCH, the text says {expect}"
    print(f"  H^1({g.name}; {M.name}) = {got}{flag}   {note}")
    return ev, od


# --------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("sanity checks (known answers)")
    print("=" * 74)
    s = sl11()
    show(s, trivial_module(s), expect="0|2", restrict_weights=False, exact=True,
         note="Lemma 3.1: dim H^1(sl(1|1)) = 0|2")
    show(s, trivial_module(s, odd=True), expect="2|0", restrict_weights=False,
         exact=True, note="Lemma 3.2: H^1(sl(1|1);Pi(1))")
    g1 = gl(1, 1)
    show(g1, trivial_module(g1), expect="1|0", restrict_weights=False, exact=True,
         note="H^1(gl(1|1)) = <str>")
    for n in (2, 3):
        a = pe(n, special=True)
        show(a, trivial_module(a), restrict_weights=False if n == 2 else True,
             note="H^1(spe(n)) -- (5.1) predicts 1|0 for n=2")
    for n in (2, 3):
        a = q(n)
        show(a, trivial_module(a), note="H^1(q(n)) = <sum qtr> : 0|1 expected")

    print()
    print("=" * 74)
    print("cross-check: weight-restricted vs. full computation")
    print("=" * 74)
    for a in (q(2), q(3), pe(2, special=True), pe(2), q(2, special=True)):
        M = taut_endo(a)
        full = H1(a, M, restrict_weights=False)
        red = H1(a, M, restrict_weights=True)
        ok = "OK" if full == red else "MISMATCH"
        print(f"  {a.name:9s} full {full[0]}|{full[1]}   weight-0 {red[0]}|{red[1]}   {ok}")
        assert full == red

    print()
    print("=" * 74)
    print("Claim (q/sq): H^1(g; gl(n|n))")
    print("=" * 74)
    for n in (2, 3, 4, 5):
        a = q(n, special=True)
        show(a, taut_endo(a), expect="1|1" if n == 2 else "0|0",
             note="text: (1|1) for n=2, 0 for n=3,4,5")
    for n in (2, 3, 4, 5):
        a = q(n)
        show(a, taut_endo(a), expect="1|1",
             note="text: c_1 even, c_2 = bar1 (x) hat1 odd")

    print()
    print("=" * 74)
    print("Claim (pe/spe), tautological module: H^1(g; gl(n|n))")
    print("=" * 74)
    for n in (2, 3, 4, 5):
        a = pe(n, special=True)
        show(a, taut_endo(a), expect="0|0", note="text: (i) = 0")
    for n in (2, 3, 4, 5):
        a = pe(n)
        show(a, taut_endo(a), expect="1|0", note="text: (ii) spanned by D (x) hat D")

    print()
    print("=" * 74)
    print("Claim (adjoint spe): H^1(spe(n); spe(n) (x) spe(n)^*)")
    print("=" * 74)
    for n in (2, 3, 4):
        a = pe(n, special=True)
        show(a, adjoint_endo(a), expect="0|1" if n == 2 else "0|0", note="text: (0|1) for n=2, 0 for n>=3")

    print()
    print("=" * 74)
    print("formula (5.1) in degree 1:  H^1(g) with trivial coefficients")
    print("=" * 74)
    for n in (2, 3, 4, 5, 6):
        for a in (pe(n), pe(n, special=True), q(n), q(n, special=True)):
            ev, od = H1(a, trivial_module(a))
            print(f"  H^1({a.name}) = {ev}|{od}")

    print()
    print("=" * 74)
    print("explicit representatives")
    print("=" * 74)
    a = q(2, special=True)
    M = taut_endo(a)
    print("  H^1(sq(2); gl(2|2)), even part:")
    for s in h1_reps(a, M, q_names(2, True), m_names(4)):
        print("     ", s)
    print("  H^1(sq(2); gl(2|2)), odd part:")
    for s in h1_reps(a, parity_shift(a, M), q_names(2, True), m_names(4)):
        print("     ", s)
    a = q(2)
    M = taut_endo(a)
    print("  H^1(q(2); gl(2|2)), odd part:")
    for s in h1_reps(a, parity_shift(a, M), q_names(2), m_names(4)):
        print("     ", s)
    print()
    print("  the two cochains printed in the Claim for sq(2):")
    check_sq2_claim()
    print()
    print("  the paper's cocycle c_2 for q(n), as printed vs. as abbreviated:")
    for n in (2, 3, 4):
        for signs, what in (((1, -1), "as printed  (m_{i,n+i} - m_{n+i,i})"),
                            ((1, 1), "as abbreviated  bar1 = Pi(1_n)   ")):
            g, M, c = check_c2(n, signs)
            cocy, cobo = test_cochain(g, M, c)
            verdict = ("generates H^1" if cocy and not cobo
                       else "coboundary" if cocy else "NOT a cocycle")
            print(f"     q({n}) {what}: cocycle {cocy!s:5s} -> {verdict}")

    a = pe(2, special=True)
    print("  H^1(spe(2); End(spe(2))), odd part:")
    Mad = adjoint_endo(a)
    names = ["A_{1,2}", "A_{2,1}", "A_{1,1}-A_{2,2}",
             "B_{1,1}", "B_{1,2}", "B_{2,2}", "C_{1,2}"]
    for s in h1_reps(a, parity_shift(a, Mad), names,
                     [f"[{names[i]}(x){names[j]}^*]" for i in range(7) for j in range(7)]):
        print("     ", s)


if __name__ == "__main__":
    main()
