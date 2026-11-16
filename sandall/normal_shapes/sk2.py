#!/usr/bin/env python3
"""Hamiltonian Lie algebras at p = 2: Poisson brackets and 2-structures.

The text gives the Poisson
brackets and the p-structures of the deformed Hamiltonian Lie algebras for
p > 2; here is the same for p = 2, together with the question of the
(2,4)-structures of [BLLS, Subsec. 2.6.1].

Everything is exact over F_2.  The divided power algebra, the divided power
de Rham complex, the Lie derivative and the non-degeneracy test are taken from
`nalt2.py`.

Conventions
-----------
* indeterminates are indexed 0..m-1; for m = 2k we write ~i = i + k;
* the standard form is omega_0 = sum_{i<k} du_i du_{~i};
* the 1st-type form is
      omega = omega_0 + sum_{i,j<k} alpha_ij ubar_i ubar_{~j} du_i du_{~j},
  where ubar_i = u_i^(2^{u_i} - 1), so ubar_i du_i = d(u_i^(2^{u_i})) and each
  summand is a product of two closed 1-forms, whence omega is closed;
* generating functions live in the AMBIENT algebra O(m; u+1), of which
  O(m; u) + Span{u_j^(2^{u_j})} is the relevant subspace (Lemma "ssSpanH"):
  the extra generating functions are exactly the u_j^(2^{u_j});
* H_f is defined by iota_{H_f} omega = df, i.e. sum_i (H_f)_i G_ij = d_j f.

Usage:  python3 sk2.py            (plain python3 works too)
"""
import sys, os, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nalt2                                                    # noqa: E402
from nalt2 import Alg, d2, lie, act, bracket, gram, det, alternating  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import nullspace, rank                            # noqa: E402


# ------------------------------------------------------------------ the ring

class Ring:
    """O(m; u) sitting inside the ambient O(m; u+1)."""

    def __init__(self, m, u=None):
        self.m = m
        self.u = tuple(u) if u is not None else (1,) * m
        self.big = Alg(m, [k + 1 for k in self.u])
        self.N = tuple(2 ** k for k in self.u)          # heights p^{u_i}
        # index sets inside the ambient
        self.small = [k for k, a in enumerate(self.big.mons)
                      if all(a[i] < self.N[i] for i in range(m))]
        self.smallset = set(self.small)
        self.smallmask = 0
        for k in self.small:
            self.smallmask |= 1 << k

    # -- elements ----------------------------------------------------------
    def mono(self, a):
        return self.big.mono(a)

    def gen(self, i):
        return self.big.gen(i)

    def divpow(self, i, e):
        a = [0] * self.m
        a[i] = e
        return self.big.mono(a)

    def ubar(self, i):
        """u_i^(2^{u_i} - 1)."""
        return self.divpow(i, self.N[i] - 1)

    def utop(self, i):
        """u_i^(2^{u_i}) -- an EXTRA generating function, outside O(m;u)."""
        return self.divpow(i, self.N[i])

    def in_small(self, f):
        return f & ~self.smallmask == 0

    def show(self, f):
        return self.big.show(f)

    # -- ring inverse (constant term 1) ------------------------------------
    def inv(self, f):
        assert self.big.const_term(f) == 1, "not invertible"
        n = f ^ self.big.one                       # nilpotent part
        out, t = self.big.one, self.big.one
        while True:
            t = self.big.mul(t, n)
            if t == 0:
                return out
            out ^= t


# --------------------------------------------------------- matrices over O

def mat_minor(G, i, j):
    return [[G[r][c] for c in range(len(G)) if c != j]
            for r in range(len(G)) if r != i]


def mat_det(alg, G):
    return det(alg, G)


def mat_inv(ring, G):
    """inverse of a matrix over O with invertible determinant (adjugate)."""
    alg = ring.big
    m = len(G)
    D = mat_det(alg, G)
    Dinv = ring.inv(D)
    out = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            # (G^{-1})_{ij} = cofactor_{ji} / det ; char 2: no signs
            out[i][j] = alg.mul(Dinv, mat_det(alg, mat_minor(G, j, i)))
    return out


def mat_mul(alg, A, B):
    n = len(A)
    return [[_dot(alg, A[i], [B[r][j] for r in range(n)]) for j in range(n)]
            for i in range(n)]


def _dot(alg, xs, ys):
    out = 0
    for x, y in zip(xs, ys):
        out ^= alg.mul(x, y)
    return out


# ------------------------------------------------------------------- forms

def omega0(ring, k):
    om = {}
    for i in range(k):
        om[('a', i, i + k)] = ring.big.one
    return om


def omega1(ring, k, alpha):
    """omega_0 + sum alpha_ij ubar_i ubar_{~j} du_i du_{~j}."""
    om = dict(omega0(ring, k))
    alg = ring.big
    for i in range(k):
        for j in range(k):
            if not alpha[i][j]:
                continue
            c = alg.mul(ring.ubar(i), ring.ubar(j + k))
            key = ('a', min(i, j + k), max(i, j + k))
            om[key] = om.get(key, 0) ^ c
    return {k_: v for k_, v in om.items() if v}


def omega2(ring, k, j):
    """the 2nd-type form d( exp(u_j^(2^{u_j})) sum_i u_i du_{~i} ).

    exp(v) = sum_a v^(a); here v = u_j^(2^{u_j}) and v^(a) = u_j^(a 2^{u_j})
    inside the ambient O(m; u+1), where only a = 0, 1 survive.
    """
    alg = ring.big
    ex = alg.one ^ ring.utop(j)                   # 1 + v  (v^(2) is outside)
    # the 1-form  exp(v) * sum_i u_i du_{~i}
    one_form = [0] * ring.m
    for i in range(k):
        one_form[i + k] = alg.mul(ex, ring.gen(i))
    return nalt2.d1(alg, one_form)


def omega_nonalt(ring, k):
    """the non-alternating shape  (du_0)^(2) + sum_{i} du_{2i-1} du_{2i}.

    Lives in m = 2k+1 indeterminates:  u_0 = x, then the k pairs.
    """
    om = {('b', 0): ring.big.one}
    for i in range(k):
        om[('a', 1 + 2 * i, 2 + 2 * i)] = ring.big.one
    return om


# -------------------------------------------------- Hamiltonian vector fields

class Ham:
    def __init__(self, ring, om):
        self.ring = ring
        self.alg = ring.big
        self.om = om
        self.G = gram(self.alg, om)
        assert self.alg.const_term(mat_det(self.alg, self.G)) == 1, "degenerate"
        self.Ginv = mat_inv(ring, self.G)
        assert not d2(self.alg, om), "form is not closed"

    def field(self, f):
        """H_f with iota_{H_f} omega = df."""
        alg = self.alg
        df = [alg.partial(j, f) for j in range(ring_m(self.ring))]
        return [_dot(alg, df, [self.Ginv[j][i] for j in range(len(df))])
                for i in range(len(df))]

    def preserves_small(self, D):
        """does D = sum f_i d_i map O(m;u) into itself and lie in W(m;u)?"""
        return all(self.ring.in_small(c) for c in D)

    def poisson(self, f, g):
        return act(self.alg, self.field(f), g)

    def square(self, D):
        """D^2, a vector field in characteristic 2."""
        return [act(self.alg, D, D[i]) for i in range(len(D))]


def ring_m(ring):
    return ring.m


# ----------------------------------------- operators on O(m;u), at the honest
# level: a vector field is compared with a SQUARE of a vector field through the
# action on functions, because at p = 2 the square of D = sum f_i d_i is
#     D^2 = sum D(f_i) d_i + sum (const term of f_i)^2 d_i^2,
# and d_i^2 is not a vector field when u_i > 1.

def op_of_field(ring, D):
    """the F_2-matrix (as a list of column bitmasks) of D acting on O(m;u)."""
    alg = ring.big
    cols = []
    for k in ring.small:
        img = act(alg, D, 1 << k)
        assert ring.in_small(img), "field does not preserve O(m;u)"
        cols.append(_pack(ring, img))
    return cols


def op_compose(ring, A, B):
    """A o B, both given as column lists over the small basis."""
    out = []
    pos = {k: t for t, k in enumerate(ring.small)}
    for col in B:
        v = 0
        c = col
        while c:
            lo = c & -c
            t = lo.bit_length() - 1
            c ^= lo
            v ^= A[t]
        out.append(v)
    return out


def op_of_operator_square(ring, D):
    """the operator D o D on O(m;u) -- always maps O(m;u) to itself."""
    A = op_of_field(ring, D)
    return op_compose(ring, A, A)


def _pack(ring, f):
    v = 0
    for t, k in enumerate(ring.small):
        if (f >> k) & 1:
            v |= 1 << t
    return v


def op_flat(cols):
    """flatten a list of column bitmasks into a single big integer."""
    w = 0
    n = 0
    for c in cols:
        w |= c << n
        n += max(c.bit_length(), 1)
    return w


def op_int(ring, cols):
    """pack an operator (list of column bitmasks) into one big integer."""
    n = len(ring.small)
    w = 0
    for t, c in enumerate(cols):
        w |= c << (t * n)
    return w


def prepare_solver(ring, ham, basis):
    """echelon form of the span of the op(H_f): leading bit -> (vec, combo)."""
    piv = {}
    for c, f in enumerate(basis):
        v = op_int(ring, op_of_field(ring, ham.field(f)))
        combo = 1 << c
        while v:
            hb = v.bit_length() - 1
            if hb in piv:
                w, cm = piv[hb]
                v ^= w
                combo ^= cm
            else:
                piv[hb] = (v, combo)
                break
    return piv


def solve_prepared(ring, piv, basis, target_cols):
    """g with op(H_g) = target, or None."""
    v = op_int(ring, target_cols)
    combo = 0
    while v:
        hb = v.bit_length() - 1
        if hb not in piv:
            return None
        w, cm = piv[hb]
        v ^= w
        combo ^= cm
    g = 0
    for c in range(len(basis)):
        if (combo >> c) & 1:
            g ^= basis[c]
    return g


def solve_operator(ring, ham, basis, target_cols):
    """find g in Span(basis) with op(H_g) = target (as operators on O(m;u))."""
    ncell = len(ring.small)
    gencols = []
    for f in basis:
        gencols.append(op_of_field(ring, ham.field(f)))
    nrow = ncell * ncell
    rows = [0] * nrow
    ncols = len(basis)
    for c, cols in enumerate(gencols):
        for t, colmask in enumerate(cols):
            cm = colmask
            while cm:
                lo = cm & -cm
                r = lo.bit_length() - 1
                cm ^= lo
                rows[t * ncell + r] |= 1 << c
    for t, colmask in enumerate(target_cols):
        cm = colmask
        while cm:
            lo = cm & -cm
            r = lo.bit_length() - 1
            cm ^= lo
            rows[t * ncell + r] |= 1 << ncols
    tbit = 1 << ncols
    piv = {}
    for w in rows:
        while w & ~tbit:
            hb = (w & ~tbit).bit_length() - 1
            if hb in piv:
                w ^= piv[hb]
            else:
                piv[hb] = w
                w = 0
                break
        if w == tbit:
            return None
    x = [0] * ncols
    for c in sorted(piv):
        w = piv[c]
        val = (w >> ncols) & 1
        for cc in range(c):
            if (w >> cc) & 1:
                val ^= x[cc]
        x[c] = val
    g = 0
    for c, v in enumerate(x):
        if v:
            g ^= basis[c]
    return g


# ------------------------------------------- the space of generating functions

def genfun_basis(ring):
    """basis of O(m;u) + Span{u_j^(2^{u_j})}, as ambient elements."""
    out = [1 << k for k in ring.small]
    out += [ring.utop(j) for j in range(ring.m)]
    return out


def field_matrix(ham, basis):
    """matrix of f -> H_f in coordinates (rows = ambient coords of the field)."""
    ring = ham.ring
    cols = []
    for f in basis:
        D = ham.field(f)
        v = []
        for i in range(ring.m):
            v.extend((D[i] >> t) & 1 for t in range(ring.big.dim))
        cols.append(v)
    return cols


def solve_field(cols, basis, D, ring):
    """express the vector field D as H_g, g a combination of `basis`."""
    target = []
    for i in range(ring.m):
        target.extend((D[i] >> t) & 1 for t in range(ring.big.dim))
    # solve  sum_c x_c cols[c] = target   over F_2
    nrows = len(target)
    M = [[cols[c][r] for c in range(len(cols))] + [target[r]]
         for r in range(nrows)]
    # gaussian elimination mod 2 with bitmasks; the augmented (target) entry
    # is kept in bit `ncols` and is NEVER used as a pivot.
    ncols = len(cols)
    tbit = 1 << ncols
    rows = []
    for r in M:
        w = 0
        for c, v in enumerate(r):
            if v:
                w |= 1 << c
        rows.append(w)
    piv = {}
    for w in rows:
        while w & ~tbit:                          # while some unknown occurs
            hb = (w & ~tbit).bit_length() - 1
            if hb in piv:
                w ^= piv[hb]
            else:
                piv[hb] = w
                w = 0
                break
        if w == tbit:                             # 0 = 1: inconsistent
            return None
    # back substitution: each pivot row has its pivot as the HIGHEST bit,
    # so its other unknowns sit in columns < c and are already known.
    x = [0] * ncols
    for c in sorted(piv):
        w = piv[c]
        val = (w >> ncols) & 1
        for cc in range(c):
            if (w >> cc) & 1:
                val ^= x[cc]
        x[c] = val
    g = 0
    for c, v in enumerate(x):
        if v:
            g ^= basis[c]
    return g


# ------------------------------------------------------- the normal shapes

def J0(k):
    """J_k(0): 1 on the superdiagonal."""
    A = [[0] * k for _ in range(k)]
    for i in range(k - 1):
        A[i][i + 1] = 1
    return A


def Ccyc(k):
    A = J0(k)
    A[k - 1][0] = 1
    return A


def Jkr(k, r, lam):
    """J_{k,r}(lambda), k = r n, as in eq. (Jkr) of the paper."""
    assert k % r == 0
    n = k // r
    A = [[0] * k for _ in range(k)]
    for c in range(n - 1):
        for a in range(r):
            A[c * r + a][(c + 1) * r + a] = 1
    # last block row: J_r(lambda) in the first block column
    for a in range(r):
        A[(n - 1) * r + a][a] = lam
        if a < r - 1:
            A[(n - 1) * r + a][a + 1] = 1
    return A


def shapes(k):
    """the shapes of table (3.3) available over F_2 (lambda = 1 only)."""
    out = []
    if k > 1:
        out.append(("J_%d(0)" % k, J0(k)))
        out.append(("C_%d" % k, Ccyc(k)))
    for r in range(1, k + 1):
        if k % r == 0:
            out.append(("J_{%d,%d}(1)" % (k, r), Jkr(k, r, 1)))
    return out


# ---------------------------------------------------------- the path formula

def admissible_paths(alpha, k, i, j):
    """paths i -> j whose interior vertices are distinct and avoid i and j."""
    out = []

    def walk(cur, path, interior):
        for nxt in range(k):
            if not alpha[cur][nxt]:
                continue
            if nxt == j:
                out.append(list(path) + [nxt])
            if nxt in (i, j) or nxt in interior:
                continue
            walk(nxt, path + [nxt], interior | {nxt})

    walk(i, [i], set())
    return out


def G_path(ring, k, alpha):
    """the matrix G of Theorem "Gpath", at p = 2 (no signs)."""
    alg = ring.big
    G = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            v = alg.one if i == j else 0
            for P in admissible_paths(alpha, k, i, j):
                w = 1
                for s in range(len(P) - 1):
                    w &= alpha[P[s]][P[s + 1]]
                if not w:
                    continue
                t = alg.mul(ring.ubar(i), ring.ubar(j + k))
                for s in P[1:-1]:
                    t = alg.mul(t, alg.mul(ring.ubar(s), ring.ubar(s + k)))
                v ^= t
            G[i][j] = v
    return G
