"""Exact engine for Diff_omega-invariant bilinear differential operators on a
symplectic manifold (Grozman, arXiv:math/0101266).

Coordinates z_1..z_n, n = 2m, with z_i = x_i (i < m) and z_{m+i} = y_i.
omega = sum_i dx_i ^ dy_i.

Everything is exact rational arithmetic on polynomials in the n coordinates,
represented as dict{exponent tuple -> Fraction}.

The point of the module: a translation-invariant, Sp(2m)-covariant bilinear
differential operator on tensor fields is automatically invariant under
h_{-1} + h_0 of the Poisson (Hamiltonian) algebra; the only remaining condition
is invariance under the cubic Hamiltonians h_1, and h_{-1}+h_0+h_1 generate
the whole of h(2m).  So the classification of invariant operators of a given
order reduces to a finite linear system.
"""

from fractions import Fraction
from itertools import combinations_with_replacement, product
import random

# ---------------------------------------------------------------- polynomials


def pzero():
    return {}


def pconst(c, n):
    c = Fraction(c)
    return {} if c == 0 else {(0,) * n: c}


def pvar(i, n):
    e = [0] * n
    e[i] = 1
    return {tuple(e): Fraction(1)}


def padd(p, q):
    r = dict(p)
    for e, c in q.items():
        v = r.get(e, 0) + c
        if v:
            r[e] = v
        elif e in r:
            del r[e]
    return r


def pscale(p, c):
    c = Fraction(c)
    if c == 0:
        return {}
    return {e: v * c for e, v in p.items()}


def pmul(p, q):
    r = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            v = r.get(e, 0) + c1 * c2
            if v:
                r[e] = v
            elif e in r:
                del r[e]
    return r


def pdiff(p, i):
    r = {}
    for e, c in p.items():
        if e[i] == 0:
            continue
        f = list(e)
        f[i] -= 1
        f = tuple(f)
        v = r.get(f, 0) + c * e[i]
        if v:
            r[f] = v
        elif f in r:
            del r[f]
    return r


def piszero(p):
    return all(c == 0 for c in p.values())


# ------------------------------------------------------------------ symplectic


class Symp:
    """The symplectic data for n = 2m coordinates."""

    def __init__(self, m):
        self.m = m
        self.n = 2 * m
        n = self.n
        # Om[i][j] = omega^{ij}, the inverse form, chosen so that the Hamiltonian
        # field X_f^k = sum_l Om[k][l] d_l f gives {f,g} = X_f(g) = f_x g_y - f_y g_x.
        Om = [[Fraction(0)] * n for _ in range(n)]
        for i in range(m):
            Om[i][m + i] = Fraction(-1)
            Om[m + i][i] = Fraction(1)
        self.Om = Om
        # sparse list of (i, j, value) with Om[i][j] != 0
        self.Omnz = [(i, j, Om[i][j]) for i in range(n) for j in range(n) if Om[i][j]]

    # --- Hamiltonian vector fields ------------------------------------------
    def ham(self, f):
        """X_f as a list of n polynomial components."""
        n = self.n
        return [
            _sum_polys([pscale(pdiff(f, l), self.Om[k][l]) for l in range(n) if self.Om[k][l]])
            for k in range(n)
        ]

    def poisson(self, f, g):
        X = self.ham(f)
        return _sum_polys([pmul(X[k], pdiff(g, k)) for k in range(self.n)])

    # --- Lie derivative on covariant symmetric 2-tensors --------------------
    def lie_sym2(self, f, A):
        """(L_X A)_{ij} = X^k d_k A_ij + A_kj d_i X^k + A_ik d_j X^k, X = X_f."""
        n = self.n
        X = self.ham(f)
        dX = [[pdiff(X[k], i) for i in range(n)] for k in range(n)]
        out = [[pzero() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                terms = [pmul(X[k], pdiff(A[i][j], k)) for k in range(n)]
                terms += [pmul(A[k][j], dX[k][i]) for k in range(n)]
                terms += [pmul(A[i][k], dX[k][j]) for k in range(n)]
                s = _sum_polys(terms)
                out[i][j] = s
                out[j][i] = s
        return out


def _sum_polys(ps):
    r = {}
    for p in ps:
        for e, c in p.items():
            v = r.get(e, 0) + c
            if v:
                r[e] = v
            elif e in r:
                del r[e]
    return r


# ------------------------------------------------------- the ansatz of terms
#
# A term of bidegree (da, db) with da + db = N derivatives is a complete
# contraction, by copies of omega^{..}, of
#       d_{p_1}...d_{p_da} A_{k l}  *  d_{q_1}...d_{q_db} A'_{r s}
# leaving two free indices i, j (then symmetrised).
#
# Slot list: the derivative slots on A, then A's own two slots, then the
# derivative slots on A', then A''s two slots.  Two slots carry the free
# indices; the remaining ones are matched in pairs by omega's.


class Term:
    def __init__(self, da, db, free, pairs):
        self.da, self.db = da, db
        self.free = free          # (slot for i, slot for j)
        self.pairs = pairs        # tuple of (slot, slot)
        self.nslots = da + 2 + db + 2

    def __repr__(self):
        return f"T(da={self.da},db={self.db},free={self.free},pairs={self.pairs})"

    def label(self):
        """Readable label: name the slots."""
        names = ([f"d{t+1}A" for t in range(self.da)] + ["A1", "A2"]
                 + [f"d{t+1}B" for t in range(self.db)] + ["B1", "B2"])
        f = f"free={names[self.free[0]]},{names[self.free[1]]}"
        p = " ".join(f"({names[a]}-{names[b]})" for a, b in self.pairs)
        return f"[{self.da}|{self.db}] {f} contr {p}"

    def slotinfo(self):
        da, db = self.da, self.db
        info = []
        for t in range(da):
            info.append(("dA", t))
        info.append(("A", 0)); info.append(("A", 1))
        for t in range(db):
            info.append(("dB", t))
        info.append(("B", 0)); info.append(("B", 1))
        return info


def enumerate_terms(N):
    """All contraction schemes with N derivatives in total."""
    terms = []
    for da in range(N + 1):
        db = N - da
        nslots = da + 2 + db + 2
        if (nslots - 2) % 2:
            continue
        for free in combinations(range(nslots), 2):
            rest = [s for s in range(nslots) if s not in free]
            for pairs in _pairings(rest):
                terms.append(Term(da, db, free, pairs))
    return terms


def combinations(it, k):
    from itertools import combinations as _c
    return list(_c(it, k))


def _pairings(items):
    if not items:
        return [()]
    a = items[0]
    out = []
    for idx in range(1, len(items)):
        b = items[idx]
        rest = items[1:idx] + items[idx + 1:]
        for p in _pairings(rest):
            out.append(((a, b),) + p)
    return out


def eval_term(S, term, A, B):
    """Evaluate the contraction on the pair of symmetric tensor fields A, B.

    Returns a symmetric n x n array of polynomials (already symmetrised in i,j).
    """
    n = S.n
    info = term.slotinfo()
    nslots = term.nslots
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    # iterate over the values of the contracted index pairs (only the nonzero
    # entries of omega contribute)
    choices = [S.Omnz] * len(term.pairs)
    for i in range(n):
        for j in range(i, n):
            acc = {}
            for combo in product(*choices):
                idx = [None] * nslots
                idx[term.free[0]] = i
                idx[term.free[1]] = j
                coef = Fraction(1)
                for (sa, sb), (va, vb, w) in zip(term.pairs, combo):
                    idx[sa] = va
                    idx[sb] = vb
                    coef *= w
                # build the two factors
                dAs = [idx[s] for s in range(nslots) if info[s][0] == "dA"]
                aidx = [idx[s] for s in range(nslots) if info[s][0] == "A"]
                dBs = [idx[s] for s in range(nslots) if info[s][0] == "dB"]
                bidx = [idx[s] for s in range(nslots) if info[s][0] == "B"]
                fa = A[aidx[0]][aidx[1]]
                for d in dAs:
                    fa = pdiff(fa, d)
                if not fa:
                    continue
                fb = B[bidx[0]][bidx[1]]
                for d in dBs:
                    fb = pdiff(fb, d)
                if not fb:
                    continue
                prod = pmul(fa, fb)
                for e, c in prod.items():
                    v = acc.get(e, 0) + c * coef
                    if v:
                        acc[e] = v
                    elif e in acc:
                        del acc[e]
            # symmetrisation in (i,j) is automatic: we take the term as it is
            # and separately add its (i<->j) partner below when i != j.
            out[i][j] = acc
            out[j][i] = acc
    # symmetrise properly: B_{ij} := (T_{ij} + T_{ji})/2 -- but the loop above
    # already set out[j][i] = out[i][j], which is NOT the symmetrisation.  Redo.
    return out


def eval_term_sym(S, term, A, B):
    """Symmetrised evaluation: (T_{ij} + T_{ji})/2."""
    n = S.n
    info = term.slotinfo()
    nslots = term.nslots
    raw = [[None] * n for _ in range(n)]
    choices = [S.Omnz] * len(term.pairs)
    combos = list(product(*choices))
    for i in range(n):
        for j in range(n):
            acc = {}
            for combo in combos:
                idx = [None] * nslots
                idx[term.free[0]] = i
                idx[term.free[1]] = j
                coef = Fraction(1)
                for (sa, sb), (va, vb, w) in zip(term.pairs, combo):
                    idx[sa] = va
                    idx[sb] = vb
                    coef *= w
                dAs = [idx[s] for s in range(nslots) if info[s][0] == "dA"]
                aidx = [idx[s] for s in range(nslots) if info[s][0] == "A"]
                dBs = [idx[s] for s in range(nslots) if info[s][0] == "dB"]
                bidx = [idx[s] for s in range(nslots) if info[s][0] == "B"]
                fa = A[aidx[0]][aidx[1]]
                for d in dAs:
                    fa = pdiff(fa, d)
                if not fa:
                    continue
                fb = B[bidx[0]][bidx[1]]
                for d in dBs:
                    fb = pdiff(fb, d)
                if not fb:
                    continue
                prod = pmul(fa, fb)
                for e, c in prod.items():
                    v = acc.get(e, 0) + c * coef
                    if v:
                        acc[e] = v
                    elif e in acc:
                        del acc[e]
            raw[i][j] = acc
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pscale(padd(raw[i][j], raw[j][i]), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


# ------------------------------------------------------------ random fields


def random_sym_field(S, deg, rng, spread=4, nterms=None):
    """A random symmetric 2-tensor field with polynomial entries of degree <= deg.

    If nterms is given, each entry gets only that many random monomials (sparse
    fields keep the polynomial arithmetic cheap for m >= 3 while staying generic
    enough: the conditions are linear in the field, so a spanning family of
    sparse fields carries exactly the same information as dense ones).
    """
    n = S.n
    mons = [e for d in range(deg + 1) for e in _monomials(n, d)]
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            p = {}
            if nterms is None:
                for e in mons:
                    c = rng.randint(-spread, spread)
                    if c:
                        p[e] = Fraction(c)
            else:
                for e in rng.sample(mons, min(nterms, len(mons))):
                    c = rng.randint(-spread, spread)
                    if c:
                        p[e] = Fraction(c)
            out[i][j] = p
            out[j][i] = p
    return out


def _monomials(n, d):
    if n == 1:
        yield (d,)
        return
    for k in range(d + 1):
        for rest in _monomials(n - 1, d - k):
            yield (k,) + rest


def monomial_polys(S, deg):
    """All monomials of exact degree `deg` as polynomials."""
    return [{e: Fraction(1)} for e in _monomials(S.n, deg)]
