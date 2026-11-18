#!/usr/bin/env python3
r"""Derivations of modular vectorial Lie superalgebras -- exact engine over F_p.

The whole point of Skryabin's note (UMN 42:6 (1987) 201--202) is the identity

    Der L = N_{Der O}(L)          (the normalizer of L in Der O),

where O = O(m; N) is the divided power algebra and L is of Cartan type.  Here we
build the SUPER version of both sides from scratch and compare them:

    O(m|n; N) = O(m; N) tensor Lambda(n),
    Der O     = free O-module on  d_{i,k} = \partial_i^{p^k} (0 <= k < N_i)
                                  and delta_j = \partial_{xi_j},
    vect(m|n;N) = the O-span of the k=0 generators (the SPECIAL derivations).

Everything is exact arithmetic over F_p.

Conventions
-----------
A monomial of O is a pair (a, b) with a an m-tuple of exponents of the divided
powers x_i^{(a_i)}, a_i < p^{N_i}, and b an n-tuple in {0,1}^n of exponents of
the odd xi_j.  Parity of (a,b) is |b| mod 2.

An element of Der O is a dict {(mon, gen): coefficient}, where gen is
('x', i, k) for \partial_i^{p^k} and ('xi', j) for \partial_{xi_j}.

The bracket of two O-linear-combinations of supercommuting derivations D, E is

    [uD, vE] = u D(v) E - (-1)^{p(uD) p(vE)} v E(u) D,

which is what mul() and act() implement.
"""
from itertools import product
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import nullspace, rank                                # noqa: E402


# ---------------------------------------------------------------- binomials
_BINOM = {}


def binom(n, k):
    if k < 0 or k > n:
        return 0
    key = (n, k)
    if key not in _BINOM:
        r = 1
        for i in range(k):
            r = r * (n - i) // (i + 1)
        _BINOM[key] = r
    return _BINOM[key]


class Super:
    """The divided power superalgebra O(m|n; N) over F_p and its Der."""

    def __init__(self, p, m, n, N=None, wt=None):
        self.p, self.m, self.n = p, m, n
        self.N = tuple(N if N is not None else [1] * m)
        assert len(self.N) == m
        self.hi = tuple(p ** Ni for Ni in self.N)          # a_i < hi[i]
        # weights of the indeterminates (for the Z-grading of a subalgebra)
        self.wt = tuple(wt) if wt is not None else (1,) * (m + n)

        self.mons = [(a, b) for a in product(*[range(h) for h in self.hi])
                     for b in product(*[range(2)] * n)]
        self.mind = {u: i for i, u in enumerate(self.mons)}
        self.dimO = len(self.mons)

        self.gens = ([('x', i, k) for i in range(m) for k in range(self.N[i])]
                     + [('xi', j) for j in range(n)])
        self.gind = {g: i for i, g in enumerate(self.gens)}

        self.dbasis = [(u, g) for u in self.mons for g in self.gens]
        self.dind = {b: i for i, b in enumerate(self.dbasis)}
        self.dimD = len(self.dbasis)

    # ------------------------------------------------------------ O algebra
    def par_mon(self, u):
        return sum(u[1]) % 2

    def omul(self, u, v):
        """product of two monomials of O; returns (mon, coeff) or (None, 0)."""
        (a, b), (c, d) = u, v
        e = tuple(x + y for x, y in zip(a, c))
        if any(e[i] >= self.hi[i] for i in range(self.m)):
            return None, 0
        co = 1
        for i in range(self.m):
            co = co * binom(a[i] + c[i], a[i]) % self.p
        if co == 0:
            return None, 0
        if any(b[j] and d[j] for j in range(self.n)):
            return None, 0
        # Koszul sign of shuffling xi^b past xi^d
        s = 0
        for j in range(self.n):
            if d[j]:
                s += sum(b[l] for l in range(j + 1, self.n))
        f = tuple(b[j] + d[j] for j in range(self.n))
        return (e, f), (co * (-1) ** s) % self.p

    # -------------------------------------------------------- the generators
    def par_gen(self, g):
        return 0 if g[0] == 'x' else 1

    def gen_act(self, g, u):
        """g(u) for a generator g of Der O and a monomial u; (mon, coeff)."""
        a, b = u
        if g[0] == 'x':
            _, i, k = g
            s = self.p ** k
            if a[i] < s:
                return None, 0
            e = list(a)
            e[i] -= s
            # \partial_i^{p^k} x^{(a)} = binom(a_i, p^k)... NO:
            # on divided powers \partial_i^{p^k} x_i^{(a_i)} = x_i^{(a_i - p^k)}
            return ((tuple(e), b), 1)
        _, j = g
        if not b[j]:
            return None, 0
        s = sum(b[l] for l in range(j))          # sign: move past xi_0..xi_{j-1}
        d = list(b)
        d[j] = 0
        return ((a, tuple(d)), (-1) ** s % self.p)

    def par(self, mon, gen):
        return (self.par_mon(mon) + self.par_gen(gen)) % 2

    # ------------------------------------------------------------ Der O ring
    def zero(self):
        return {}

    def add(self, X, Y, c=1):
        Z = dict(X)
        for k, v in Y.items():
            w = (Z.get(k, 0) + c * v) % self.p
            if w:
                Z[k] = w
            else:
                Z.pop(k, None)
        return Z

    def scal(self, X, c):
        c %= self.p
        return {k: (v * c) % self.p for k, v in X.items() if (v * c) % self.p}

    def act(self, X, u):
        """X(u) for X in Der O and u a monomial of O; returns a dict on O."""
        out = {}
        for (v, g), co in X.items():
            w, c1 = self.gen_act(g, u)
            if not c1:
                continue
            # v * g(u)   -- g is a derivation, no extra sign here
            z, c2 = self.omul(v, w)
            if not c2:
                continue
            out[z] = (out.get(z, 0) + co * c1 * c2) % self.p
        return {k: v for k, v in out.items() if v}

    def bracket(self, X, Y):
        """[X, Y] for X, Y in Der O (both homogeneous or not -- we use the
        parity of each basis term, which is correct term by term)."""
        out = {}
        for (u, D), cx in X.items():
            pX = self.par(u, D)
            for (v, E), cy in Y.items():
                pY = self.par(v, E)
                # + u D(v) E
                w, c1 = self.gen_act(D, v)
                if c1:
                    z, c2 = self.omul(u, w)
                    if c2:
                        key = (z, E)
                        out[key] = (out.get(key, 0) + cx * cy * c1 * c2) % self.p
                # - (-1)^{pX pY} v E(u) D
                w, c1 = self.gen_act(E, u)
                if c1:
                    z, c2 = self.omul(v, w)
                    if c2:
                        key = (z, D)
                        sg = -((-1) ** (pX * pY))
                        out[key] = (out.get(key, 0) + sg * cx * cy * c1 * c2) % self.p
        return {k: v for k, v in out.items() if v % self.p}

    # ------------------------------------------------------------- gradings
    def deg_mon(self, u):
        a, b = u
        return (sum(self.wt[i] * a[i] for i in range(self.m))
                + sum(self.wt[self.m + j] * b[j] for j in range(self.n)))

    def deg_gen(self, g):
        if g[0] == 'x':
            _, i, k = g
            return -(self.p ** k) * self.wt[i]
        return -self.wt[self.m + g[1]]

    def deg(self, X):
        ds = {self.deg_mon(u) + self.deg_gen(g) for (u, g) in X}
        assert len(ds) <= 1, "inhomogeneous element"
        return ds.pop() if ds else None

    def parity(self, X):
        ps = {self.par(u, g) for (u, g) in X}
        assert len(ps) <= 1, "parity-inhomogeneous element"
        return ps.pop() if ps else None

    # ------------------------------------------------------------- vectors
    def vec(self, X):
        v = [0] * self.dimD
        for k, c in X.items():
            v[self.dind[k]] = c % self.p
        return v

    def unvec(self, v):
        return {self.dbasis[i]: c % self.p for i, c in enumerate(v) if c % self.p}

    # ------------------------------------------------ the standard elements
    def mono_field(self, u, g):
        return {(u, g): 1}

    def vect(self):
        """vect(m|n; N) = the O-span of the special derivations."""
        out = []
        for u in self.mons:
            for g in self.gens:
                if g[0] == 'x' and g[2] != 0:
                    continue
                out.append({(u, g): 1})
        return out

    def divergence(self, X):
        """div(sum f_i d_i + sum g_j delta_j) in O."""
        out = {}
        for (u, g), c in X.items():
            if g[0] == 'x':
                if g[2] != 0:
                    raise ValueError("divergence only for special derivations")
                w, c1 = self.gen_act(('x', g[1], 0), u)
            else:
                w, c1 = self.gen_act(g, u)
                # (-1)^{p(u)} for the odd derivative
                c1 = c1 * (-1) ** self.par_mon(u)
            if c1:
                out[w] = (out.get(w, 0) + c * c1) % self.p
        return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------- subalgebras
def span_matrix(S, L):
    return [S.vec(X) for X in L]


def echelon(rows, p):
    """row echelon form with pivot bookkeeping; returns (rows, pivots)."""
    rows = [r[:] for r in rows]
    piv, r = [], 0
    ncols = len(rows[0]) if rows else 0
    for c in range(ncols):
        pr = None
        for i in range(r, len(rows)):
            if rows[i][c] % p:
                pr = i
                break
        if pr is None:
            continue
        rows[r], rows[pr] = rows[pr], rows[r]
        inv = pow(rows[r][c], p - 2, p) if p > 2 else 1
        rows[r] = [(x * inv) % p for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] % p:
                f = rows[i][c] % p
                rows[i] = [(a - f * b) % p for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
        if r == len(rows):
            break
    return rows[:r], piv


class Echelon:
    """membership tests against a fixed subspace."""

    def __init__(self, rows, p):
        self.p = p
        self.rows, self.piv = echelon(rows, p) if rows else ([], [])

    def reduce(self, v):
        v = [x % self.p for x in v]
        for r, c in zip(self.rows, self.piv):
            if v[c]:
                f = v[c]
                v = [(a - f * b) % self.p for a, b in zip(v, r)]
        return v

    def contains(self, v):
        return not any(self.reduce(v))

    @property
    def dim(self):
        return len(self.rows)


def svect(S):
    """svect(m|n;N) = divergence-free special derivations."""
    rows = []
    basis = S.vect()
    for X in basis:
        d = S.divergence(X)
        rows.append([d.get(u, 0) for u in S.mons])
    ns = nullspace([[rows[i][j] for i in range(len(basis))]
                    for j in range(S.dimO)], p=S.p)
    out = []
    for v in ns:
        X = {}
        for i, c in enumerate(v):
            if c % S.p:
                X = S.add(X, basis[i], int(c))
        out.append(X)
    return out


def derived(S, L):
    """the derived subalgebra L^(1) of a list of spanning elements."""
    rows = []
    for i in range(len(L)):
        for j in range(i, len(L)):
            rows.append(S.vec(S.bracket(L[i], L[j])))
    E = Echelon(rows, S.p)
    return [S.unvec(r) for r in E.rows]


def poisson(S, kind, r):
    """Hamiltonian-type subalgebras given as f |-> H_f.

    kind = 'h'  : even Poisson bracket, x = (p_1..p_r, q_1..q_r) even,
                  all n odd variables paired by the odd form sum d xi_j^2.
    kind = 'le' : odd (Buttin) bracket on m|n with m = n = r,
                  x_i even, xi_i odd, form sum d x_i d xi_i.
    Returns (list of H_f, dict f -> H_f).
    """
    fields = {}
    for u in S.mons:
        X = {}
        if kind == 'h':
            for i in range(r):
                # H_f = sum (d_{p_i} f) d_{q_i} - (d_{q_i} f) d_{p_i}
                w, c = S.gen_act(('x', i, 0), u)
                if c:
                    X = S.add(X, {(w, ('x', r + i, 0)): c % S.p})
                w, c = S.gen_act(('x', r + i, 0), u)
                if c:
                    X = S.add(X, {(w, ('x', i, 0)): (-c) % S.p})
            for j in range(S.n):
                w, c = S.gen_act(('xi', j), u)
                if c:
                    sg = (-1) ** S.par_mon(u)
                    X = S.add(X, {(w, ('xi', j)): (sg * c) % S.p})
        elif kind == 'le':
            for i in range(r):
                w, c = S.gen_act(('x', i, 0), u)
                if c:
                    X = S.add(X, {(w, ('xi', i)): c % S.p})
                w, c = S.gen_act(('xi', i), u)
                if c:
                    sg = (-1) ** S.par_mon(u)
                    X = S.add(X, {(w, ('x', i, 0)): (sg * c) % S.p})
        else:
            raise ValueError(kind)
        if X:
            fields[u] = X
    E = Echelon([S.vec(X) for X in fields.values()], S.p)
    return [S.unvec(row) for row in E.rows], fields


def contact(S, r):
    """k(1+2r|n): contact fields K_f, with x_0 = t, then p_i, q_i, then xi_j.

    K_f = (2f - E(f)) d_t + H_f - (-1)^{p(f)} sum ... ;  we use the standard
    formula (see e.g. [BGLLS]):
        K_f = (2f - E(f)) d_t + X_f + (d_t f) E,
    where E = sum_{i>0} x_i d_i + sum_j xi_j d_{xi_j} is the Euler operator in
    the non-contact variables and X_f is the Hamiltonian field of f in them
    with t frozen.  Verified by the closure test in check_closed().
    """
    fields = {}
    # Euler operator in the variables other than t
    for u in S.mons:
        a, b = u
        X = {}
        # (2f - E(f)) d_t
        e = sum(a[1:]) + sum(b)
        c0 = (2 - e) % S.p
        if c0:
            X = S.add(X, {(u, ('x', 0, 0)): c0})
        # Hamiltonian part in p,q and xi (t frozen)
        for i in range(r):
            w, c = S.gen_act(('x', 1 + i, 0), u)
            if c:
                X = S.add(X, {(w, ('x', 1 + r + i, 0)): c % S.p})
            w, c = S.gen_act(('x', 1 + r + i, 0), u)
            if c:
                X = S.add(X, {(w, ('x', 1 + i, 0)): (-c) % S.p})
        for j in range(S.n):
            w, c = S.gen_act(('xi', j), u)
            if c:
                sg = (-1) ** S.par_mon(u)
                X = S.add(X, {(w, ('xi', j)): (sg * c) % S.p})
        # (d_t f) * Euler
        w, c = S.gen_act(('x', 0, 0), u)
        if c:
            for i in range(1, S.m):
                for uu in [w]:
                    z, cc = S.omul(uu, ((0,) * i + (1,) + (0,) * (S.m - i - 1),
                                        (0,) * S.n))
                    if cc:
                        X = S.add(X, {(z, ('x', i, 0)): (c * cc) % S.p})
            for j in range(S.n):
                bb = [0] * S.n
                bb[j] = 1
                z, cc = S.omul(w, ((0,) * S.m, tuple(bb)))
                if cc:
                    X = S.add(X, {(z, ('xi', j)): (c * cc) % S.p})
        if X:
            fields[u] = X
    E = Echelon([S.vec(X) for X in fields.values()], S.p)
    return [S.unvec(row) for row in E.rows], fields


# ------------------------------------------------------------------ closure
def check_closed(S, L):
    """is the span of L a subalgebra?  returns (bool, first offending pair)."""
    E = Echelon([S.vec(X) for X in L], S.p)
    for i in range(len(L)):
        for j in range(i, len(L)):
            if not E.contains(S.vec(S.bracket(L[i], L[j]))):
                return False, (i, j)
    return True, None


# --------------------------------------------------------------- normalizer
def normalizer(S, L):
    """N_{Der O}(L) = {D in Der O : [D, L] subset L}.

    The unknown D ranges over Der O (dimension S.dimD); for every X in a basis
    of L the class of [D, X] in Der O / L must vanish.
    """
    E = Echelon([S.vec(X) for X in L], S.p)
    free = [c for c in range(S.dimD) if c not in set(E.piv)]

    def coords_mod_L(v):
        w = E.reduce(v)
        return [w[c] for c in free]

    cols = S.dimD
    # brackets[k][xi] = [e_k, X_xi] reduced mod L, as a vector over `free`
    red = [[coords_mod_L(S.vec(S.bracket({b: 1}, X))) for X in L]
           for b in S.dbasis]
    eqs = []
    for xi in range(len(L)):
        for t in range(len(free)):
            row = [red[k][xi][t] % S.p for k in range(cols)]
            if any(row):
                eqs.append(row)
    if not eqs:
        return [S.unvec([1 if i == k else 0 for i in range(cols)])
                for k in range(cols)]
    ns = nullspace(eqs, p=S.p)
    return [S.unvec([int(x) for x in v]) for v in ns]


def ad_dim(S, L):
    """dim ad(L) = dim L - dim (center of L)."""
    E = Echelon([S.vec(X) for X in L], S.p)
    B = [S.unvec(r) for r in E.rows]
    rows = []
    for j, Y in enumerate(B):
        for t in range(S.dimD):
            row = [S.vec(S.bracket(X, Y))[t] for X in B]
            if any(x % S.p for x in row):
                rows.append([x % S.p for x in row])
    if not rows:
        return 0, len(B)
    z = len(nullspace(rows, p=S.p))
    return len(B) - z, z




# -------------------------------------------------------- full Der L, graded
def homogeneous_basis(S, L):
    """A basis of span(L) split into the layers of the (Z-degree, parity)
    bigrading.  Returns [(deg, par, X), ...].  Asserts that span(L) is graded."""
    layers = {}
    for k, b in enumerate(S.dbasis):
        layers.setdefault((S.deg({b: 1}), S.par(*b)), []).append(k)
    vecs = [S.vec(X) for X in L]
    d0 = Echelon(vecs, S.p).dim
    out, tot = [], 0
    for key in sorted(layers):
        cols = layers[key]
        rows = [[v[c] for c in cols] for v in vecs]
        rows = [r for r in rows if any(x % S.p for x in r)]
        if not rows:
            continue
        R, _ = echelon(rows, S.p)
        tot += len(R)
        for r in R:
            X = {S.dbasis[c]: v % S.p for c, v in zip(cols, r) if v % S.p}
            out.append((key[0], key[1], X))
    assert tot == d0, ("span(L) is not graded", tot, d0)
    return out


class Struct:
    """structure constants of a graded Lie superalgebra given by a graded basis."""

    def __init__(self, S, hom):
        self.S, self.p = S, S.p
        self.deg = [h[0] for h in hom]
        self.par = [h[1] for h in hom]
        self.B = [h[2] for h in hom]
        self.N = len(hom)
        M = [S.vec(X) for X in self.B]
        aug = [M[i][:] + [1 if j == i else 0 for j in range(self.N)]
               for i in range(self.N)]
        self.A, self.piv = echelon(aug, S.p)
        self.C = {}
        for i in range(self.N):
            for j in range(i, self.N):
                c = self.coords(S.vec(S.bracket(self.B[i], self.B[j])))
                assert c is not None, "not a subalgebra"
                self.C[(i, j)] = c

    def coords(self, v):
        S, p, N = self.S, self.p, self.N
        w = [x % p for x in v] + [0] * N
        for row, c in zip(self.A, self.piv):
            if c < S.dimD and w[c]:
                f = w[c]
                w = [(a - f * b) % p for a, b in zip(w, row)]
        if any(w[:S.dimD]):
            return None
        return [(-x) % p for x in w[S.dimD:]]

    def cc(self, i, j):
        if i <= j:
            return self.C[(i, j)]
        sg = -((-1) ** (self.par[i] * self.par[j]))
        return [(sg * x) % self.p for x in self.C[(j, i)]]


def der_dim(S, L, want_blocks=False):
    """dim Der L, computed block by block in the (degree shift, parity shift).

    Returns (dim Der L, {(d, q): dim}).
    """
    hom = homogeneous_basis(S, L)
    st = Struct(S, hom)
    N, p = st.N, S.p
    lay = {}
    for i in range(N):
        lay.setdefault((st.deg[i], st.par[i]), []).append(i)
    degs = sorted({st.deg[i] for i in range(N)})
    total, blocks = 0, {}
    for d in range(min(degs) - max(degs), max(degs) - min(degs) + 1):
        for q in (0, 1):
            slots = []
            for i in range(N):
                for j in lay.get((st.deg[i] + d, (st.par[i] + q) % 2), []):
                    slots.append((i, j))
            if not slots:
                continue
            sind = {s: k for k, s in enumerate(slots)}
            nvar = len(slots)
            eqs = []
            for i in range(N):
                for j in range(i, N):
                    cij = st.cc(i, j)
                    tgt = lay.get((st.deg[i] + st.deg[j] + d,
                                   (st.par[i] + st.par[j] + q) % 2), [])
                    if not tgt:
                        continue
                    sgn = (-1) ** (q * st.par[i])
                    for t in tgt:
                        row = [0] * nvar
                        # D([e_i,e_j]) : sum_k c^{ij}_k D_{k,t}
                        for k in range(N):
                            if cij[k] % p and (k, t) in sind:
                                row[sind[(k, t)]] = (row[sind[(k, t)]]
                                                     + cij[k]) % p
                        # - [D e_i, e_j]
                        for (a, b) in slots:
                            if a == i:
                                v = st.cc(b, j)[t]
                                if v % p:
                                    row[sind[(a, b)]] = (row[sind[(a, b)]] - v) % p
                        # - (-1)^{q p_i} [e_i, D e_j]
                        for (a, b) in slots:
                            if a == j:
                                v = st.cc(i, b)[t]
                                if v % p:
                                    row[sind[(a, b)]] = (row[sind[(a, b)]]
                                                         - sgn * v) % p
                        if any(row):
                            eqs.append(row)
            k = nvar - (rank(eqs, p=p) if eqs else 0)
            if k:
                blocks[(d, q)] = k
                total += k
    return (total, blocks) if want_blocks else total
