"""Tensor fields with a general irreducible sp(2m)-fiber.

The fiber V_lambda of an irreducible representation of sp(2m) with highest weight
a partition lambda (at most m rows) is realized inside W^{tensor k}, k = |lambda|,
as

    V_lambda = im(c_lambda)  intersect  ker(all omega-traces),

c_lambda being the Young symmetrizer of the row-first tableau of shape lambda.
All indices are DOWN (covariant), matching `symp.lie_sym2`; the trace of slots
a < b uses the bivector omega^{uv} = Symp.Om.

A tensor field of type lambda is then an array t[I], I in {0..n-1}^k, of
polynomials, whose value at every point lies in V_lambda; the Lie derivative
along a Hamiltonian field is the classical

    (L_X t)_{i_1..i_k} = X^l d_l t_{i_1..i_k} + sum_a (d_{i_a} X^l) t_{i_1..l..i_k},

which preserves V_lambda because the Jacobian of a Hamiltonian field lies in
sp(2m).  That is checked, not assumed: `check_invariance` below.
"""

import os
import sys
from fractions import Fraction
from itertools import permutations, product

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rref, nullspace  # noqa: E402

from symp import Symp, pdiff, pmul, pscale, padd, pzero, _sum_polys  # noqa: E402


# ------------------------------------------------------------------ index tools


def all_indices(n, k):
    return list(product(range(n), repeat=k))


def flat(idx, n):
    r = 0
    for i in idx:
        r = r * n + i
    return r


def unflat(f, n, k):
    out = []
    for _ in range(k):
        out.append(f % n)
        f //= n
    return tuple(reversed(out))


# ------------------------------------------------------------ Young symmetrizer


def _tableau(lam):
    """Row-first numbering of the boxes: rows[r] = list of slot numbers."""
    rows, c = [], 0
    for L in lam:
        rows.append(list(range(c, c + L)))
        c += L
    cols = []
    for j in range(max(lam)):
        col = [rows[r][j] for r in range(len(lam)) if len(rows[r]) > j]
        cols.append(col)
    return rows, cols


def _group(blocks, k):
    """All permutations of {0..k-1} permuting inside each block, with signs."""
    out = [(tuple(range(k)), 1)]
    for blk in blocks:
        new = []
        for perm, sgn in out:
            for q in permutations(blk):
                p = list(perm)
                s = sgn
                # sign of the permutation blk -> q
                s *= _sign_of(blk, q)
                for a, b in zip(blk, q):
                    p[a] = perm[b]
                new.append((tuple(p), s))
        out = new
    return out


def _sign_of(src, dst):
    pos = {v: i for i, v in enumerate(src)}
    perm = [pos[v] for v in dst]
    seen = [False] * len(perm)
    sgn = 1
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, ln = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            ln += 1
        if ln % 2 == 0:
            sgn = -sgn
    return sgn


def young_image(lam, n):
    """Row space of c_lambda acting on W^{tensor k}; returns a list of rows."""
    k = sum(lam)
    rows, cols = _tableau(lam)
    R = _group(rows, k)          # row symmetrizer (signs all +1)
    C = _group(cols, k)          # column antisymmetrizer
    R = [(p, 1) for p, _ in R]
    dim = n ** k
    vecs = []
    for I in all_indices(n, k):
        acc = {}
        for p, _ in R:
            # (sigma_p . e_I) has index J with J[p[a]] = I[a]
            J = [0] * k
            for a in range(k):
                J[p[a]] = I[a]
            for q, sq in C:
                K = [0] * k
                for a in range(k):
                    K[q[a]] = J[a]
                f = flat(tuple(K), n)
                acc[f] = acc.get(f, 0) + sq
        if acc:
            row = [Fraction(0)] * dim
            for f, c in acc.items():
                row[f] = Fraction(c)
            vecs.append(row)
    return _rowspace(vecs)


def _rowspace(vecs):
    if not vecs:
        return []
    R, _piv = rref(vecs)
    return [r for r in R if any(c != 0 for c in r)]


# --------------------------------------------------------------------- fibers


class Fiber:
    """The irreducible sp(2m)-module V_lambda realized in W^{tensor k}."""

    def __init__(self, S, lam):
        self.S = S
        self.lam = tuple(lam)
        self.k = sum(lam)
        n = S.n
        self.n = n
        assert len(lam) <= S.m, f"partition {lam} has more than m={S.m} rows"
        Y = young_image(lam, n)
        # impose omega-tracelessness on each pair of slots
        basis = Y
        for a in range(self.k):
            for b in range(a + 1, self.k):
                basis = self._kill_trace(basis, a, b)
                if not basis:
                    break
        self.basis = basis
        self.dim = len(basis)
        self.ann = self._annihilator()

    def _trace_rows(self, a, b):
        """The trace c_{ab}: W^{ok} -> W^{o(k-2)} as a matrix (rows = source)."""
        n, k = self.n, self.k
        rest = [s for s in range(k) if s not in (a, b)]
        tgt = n ** (k - 2)
        M = {}
        for u, v, w in self.S.Omnz:
            for J in all_indices(n, k - 2):
                I = [0] * k
                I[a], I[b] = u, v
                for s, j in zip(rest, J):
                    I[s] = j
                M[(flat(tuple(I), n), flat(J, n) if k > 2 else 0)] = \
                    M.get((flat(tuple(I), n), flat(J, n) if k > 2 else 0), 0) + w
        return M, tgt

    def _kill_trace(self, basis, a, b):
        if not basis:
            return basis
        M, tgt = self._trace_rows(a, b)
        # image of each basis vector under the trace
        rows = []
        for v in basis:
            img = [Fraction(0)] * tgt
            for (src, dst), w in M.items():
                if v[src]:
                    img[dst] += w * v[src]
            rows.append(img)
        # we want the combinations of the basis vectors whose image vanishes,
        # i.e. the LEFT kernel of `rows` = the right kernel of its transpose.
        cols = [[rows[s][t] for s in range(len(basis))] for t in range(tgt)]
        ns = nullspace(cols)
        out = []
        for c in ns:
            vec = [Fraction(0)] * (self.n ** self.k)
            for coef, v in zip(c, basis):
                if coef:
                    for i, x in enumerate(v):
                        if x:
                            vec[i] += coef * x
            out.append(vec)
        return _rowspace(out)

    def _annihilator(self):
        """A basis of the linear functionals vanishing on V_lambda."""
        if not self.basis:
            dim = self.n ** self.k
            return [[Fraction(1) if i == j else Fraction(0) for i in range(dim)]
                    for j in range(dim)]
        return nullspace(self.basis)

    # ---- fields ----------------------------------------------------------
    def zero_field(self):
        return [pzero() for _ in range(self.n ** self.k)]

    def field_from(self, coeffs, polys):
        """sum_s polys[s] * basis[s]  as an array of polynomials."""
        out = [dict() for _ in range(self.n ** self.k)]
        for c, v, p in zip(coeffs, self.basis, polys):
            if not p:
                continue
            for i, x in enumerate(v):
                if x:
                    q = pscale(p, x * c)
                    out[i] = padd(out[i], q)
        return out

    def rank_one_field(self, vec, poly):
        """poly(x) * vec, vec given in the ambient W^{ok} coordinates."""
        out = [dict() for _ in range(self.n ** self.k)]
        for i, x in enumerate(vec):
            if x:
                out[i] = pscale(poly, x)
        return out

    def random_vector(self, rng, spread=3):
        vec = [Fraction(0)] * (self.n ** self.k)
        for v in self.basis:
            c = rng.randint(-spread, spread)
            if c:
                for i, x in enumerate(v):
                    if x:
                        vec[i] += c * x
        return vec

    def contains(self, vec):
        return all(sum(a * b for a, b in zip(phi, vec)) == 0 for phi in self.ann)


# ------------------------------------------------- Lie derivative on W^{ok}


def lie_tensor(S, k, f, T):
    """(L_X T) for X = X_f and T an array of polynomials over {0..n-1}^k."""
    n = S.n
    X = S.ham(f)
    dX = [[pdiff(X[l], i) for i in range(n)] for l in range(n)]
    out = [None] * (n ** k)
    for fi in range(n ** k):
        I = unflat(fi, n, k)
        terms = [pmul(X[l], pdiff(T[fi], l)) for l in range(n) if X[l] and T[fi]]
        for a in range(k):
            for l in range(n):
                if not dX[l][I[a]]:
                    continue
                J = list(I)
                J[a] = l
                t = T[flat(tuple(J), n)]
                if t:
                    terms.append(pmul(t, dX[l][I[a]]))
        out[fi] = _sum_polys(terms)
    return out


def check_invariance(S, F, ntest=6, seed=1):
    """V_lambda is sp(2m)-invariant: L_X of a constant field of type lambda
    is again of type lambda, for X a linear Hamiltonian field (quadratic H)."""
    import random
    rng = random.Random(seed)
    n, k = S.n, F.k
    quads = [q for q in _quadratics(S)]
    ok = True
    for _ in range(ntest):
        vec = F.random_vector(rng)
        p = {(0,) * n: Fraction(1)}
        T = F.rank_one_field(vec, p)
        for q in quads:
            L = lie_tensor(S, k, q, T)
            val = [next(iter(L[i].values())) if L[i] else Fraction(0)
                   for i in range(n ** k)]
            if not F.contains(val):
                ok = False
    return ok


def _quadratics(S):
    n = S.n
    out = []
    for i in range(n):
        for j in range(i, n):
            e = [0] * n
            e[i] += 1
            e[j] += 1
            out.append({tuple(e): Fraction(1)})
    return out
