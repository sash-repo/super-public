#!/usr/bin/env python3
"""Is the exceptional periplectic algebra le_{omega_Xi}(n;u) isomorphic to le(n;u)?

For p = 3 and
n = 1 the two algebras were separated by the SPAN of the [x,x] with x odd, and
for n >= 2 that span coincides (18 for n = 2, 108 for n = 3).  Here the question
is settled.

This engine builds both algebras from the closed brackets

    {f,g}_0   = sum_i ( d_qi f  d_xii g + (-1)^{p(f)} d_xii f  d_qi g )
    {f,g}_Xi  = {f,g}_0 - (-1)^{p(f)} xi_1 d_q1 f d_q1 g

on L = O(n|n;u)/K with the parity shift p_L(f) = p(f) + 1 (a Theorem of the
text), and computes ISOMORPHISM INVARIANTS of the two:

  * the derived series, the center, simplicity;
  * dim Der (by parity), dim of the inner part, dim H^1(g;g);
  * the COUNT of odd square-zero elements, #{x in g_od | [x,x] = 0}, over F_p.

The last one is the invariant that separates them.  It is not enumerable
directly (3^17 points already for n = 2), but for n <= 2 the odd part splits as
A (+) C with [C,C] = 0, so that x = a + c has

    [x,x] = [a,a] + 2 B(a,c),          B = the bracket, bilinear,

which is AFFINE in c for fixed a: the fiber over a is a coset of a linear
space, computed by one rank.  Whence the exact count as a sum over the 3^{dim A}
values of a.  The affineness hypothesis [C,C] = 0 is verified, not assumed.

Usage:  python3 leiso.py            (plain python3 works too)
"""
import sys, os, itertools, random
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import rank as _rank, rref as _rref, nullspace as _nullspace  # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# --------------------------------------------------------------- O(n|n;u)

class Fun:
    """the supercommutative divided power superalgebra O(n|n;u) over F_p."""

    def __init__(self, n, p, u=None):
        self.n, self.p = n, p
        self.u = tuple(u) if u is not None else (1,) * n
        self.cap = tuple(p ** k for k in self.u)

    # ---- monomials: (qexp, xibits) ---------------------------------------
    def basis(self, par=None):
        out = []
        for qe in itertools.product(*[range(c) for c in self.cap]):
            for xb in itertools.product(*[range(2)] * self.n):
                m = (tuple(qe), tuple(xb))
                if par is None or self.parity(m) == par:
                    out.append(m)
        return out

    def parity(self, m):
        return sum(m[1]) % 2

    def one(self):
        return {((0,) * self.n, (0,) * self.n): 1}

    def gen_q(self, i, k=1):
        qe = [0] * self.n
        qe[i] = k
        return {(tuple(qe), (0,) * self.n): 1}

    def gen_xi(self, i):
        xb = [0] * self.n
        xb[i] = 1
        return {((0,) * self.n, tuple(xb)): 1}

    # ---- arithmetic -------------------------------------------------------
    def mul_mon(self, a, b):
        p = self.p
        qe = []
        c = 1
        for i in range(self.n):
            s = a[0][i] + b[0][i]
            if s >= self.cap[i]:
                return None
            c = (c * comb(s, a[0][i])) % p
            if c == 0:
                return None
            qe.append(s)
        xb = []
        sign = 0
        for j in range(self.n):
            if b[1][j]:
                if a[1][j]:
                    return None
                sign += sum(a[1][k] for k in range(j + 1, self.n))
            xb.append(a[1][j] + b[1][j])
        if sign % 2:
            c = (-c) % p
        return (c, (tuple(qe), tuple(xb)))

    def mul(self, x, y):
        p, out = self.p, {}
        for a, ca in x.items():
            for b, cb in y.items():
                r = self.mul_mon(a, b)
                if r is None:
                    continue
                c, m = r
                v = (out.get(m, 0) + c * ca * cb) % p
                if v:
                    out[m] = v
                elif m in out:
                    del out[m]
        return out

    def add(self, *xs):
        p, out = self.p, {}
        for x in xs:
            for m, c in x.items():
                v = (out.get(m, 0) + c) % p
                if v:
                    out[m] = v
                elif m in out:
                    del out[m]
        return out

    def smul(self, c, x):
        c %= self.p
        if c == 0:
            return {}
        return {m: (c * v) % self.p for m, v in x.items()}

    def sub(self, x, y):
        return self.add(x, self.smul(-1, y))

    # ---- derivatives ------------------------------------------------------
    def d_q(self, i, x):
        out = {}
        for (qe, xb), c in x.items():
            a = qe[i]
            if a:
                nq = list(qe)
                nq[i] = a - 1
                m = (tuple(nq), xb)
                v = (out.get(m, 0) + c) % self.p
                if v:
                    out[m] = v
                elif m in out:
                    del out[m]
        return out

    def d_xi(self, i, x):
        """the LEFT derivative d/dxi_i (an odd derivation)."""
        out = {}
        for (qe, xb), c in x.items():
            if not xb[i]:
                continue
            sgn = (-1) ** sum(xb[k] for k in range(i))
            nx = list(xb)
            nx[i] = 0
            m = (qe, tuple(nx))
            v = (out.get(m, 0) + sgn * c) % self.p
            if v:
                out[m] = v
            elif m in out:
                del out[m]
        return out


# --------------------------------------------------------------- the brackets

def antibracket(F, f, g, shape="0", j=0):
    """{f,g}_0, and for shape='Xi' the deformed bracket of Theorem 6.1 2)."""
    assert all(F.parity(m) == F.parity(next(iter(f))) for m in f), "f must be homogeneous"
    pf = F.parity(next(iter(f))) if f else 0
    out = {}
    for i in range(F.n):
        out = F.add(out, F.mul(F.d_q(i, f), F.d_xi(i, g)))
        out = F.add(out, F.smul((-1) ** pf, F.mul(F.d_xi(i, f), F.d_q(i, g))))
    if shape == "Xi":
        t = F.mul(F.gen_xi(j), F.mul(F.d_q(j, f), F.d_q(j, g)))
        out = F.sub(out, F.smul((-1) ** pf, t))
    return out


class LeAlg:
    """L = O(n|n;u)/K with the bracket above and the parity shift p+1."""

    def __init__(self, n, p, u=None, shape="0"):
        self.F = Fun(n, p, u)
        self.p, self.n, self.shape = p, n, shape
        F = self.F
        const = ((0,) * n, (0,) * n)
        self.mons = [m for m in F.basis() if m != const]
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self.dim = len(self.mons)
        self.par = [(F.parity(m) + 1) % 2 for m in self.mons]      # p_L
        self.ev = [i for i in range(self.dim) if self.par[i] == 0]
        self.od = [i for i in range(self.dim) if self.par[i] == 1]
        self.const = const
        self._tab = None

    def sdim(self):
        return (len(self.ev), len(self.od))

    def vec(self, x):
        """an element of O, reduced mod constants, as a coordinate vector."""
        v = [0] * self.dim
        for m, c in x.items():
            if m == self.const:
                continue
            v[self.idx[m]] = c % self.p
        return v

    def brmon(self, i, j):
        """[e_i, e_j] as a vector."""
        F = self.F
        b = antibracket(F, {self.mons[i]: 1}, {self.mons[j]: 1}, self.shape)
        return self.vec(b)

    def table(self):
        """the structure constants, as a dim x dim array of vectors."""
        if self._tab is None:
            self._tab = [[self.brmon(i, j) for j in range(self.dim)]
                         for i in range(self.dim)]
        return self._tab

    def br(self, x, y):
        """bracket of two coordinate vectors."""
        T = self.table()
        p, out = self.p, [0] * self.dim
        for i, a in enumerate(x):
            if not a % p:
                continue
            for j, b in enumerate(y):
                if not b % p:
                    continue
                c = (a * b) % p
                row = T[i][j]
                for k, v in enumerate(row):
                    if v:
                        out[k] = (out[k] + c * v) % p
        return out


# --------------------------------------------------------------- linear algebra

def span_dim(vs, p):
    vs = [v for v in vs if any(x % p for x in v)]
    if not vs:
        return 0
    return _rank(vs, p)


def echelon_add(E, rows, p, ncols):
    """keep a row echelon basis of the span of E + rows (E is a list of rows)."""
    if not rows:
        return E
    M = E + rows
    R, piv = _rref(M, p)
    return [r for r in R if any(x % p for x in r)]


# --------------------------------------------------------------- invariants

def check_axioms(L, ntriples=4000, seed=1):
    """super anti-symmetry, Jacobi, on all pairs and on random triples."""
    p, d = L.p, L.dim
    T = L.table()
    bad = 0
    for i in range(d):
        for j in range(d):
            s = (-1) ** (L.par[i] * L.par[j])
            for k in range(d):
                if (T[i][j][k] + s * T[j][i][k]) % p:
                    bad += 1
    rnd = random.Random(seed)
    basis = [[1 if k == i else 0 for k in range(d)] for i in range(d)]
    badj = 0
    trips = []
    if d <= 12:
        trips = list(itertools.product(range(d), repeat=3))
    else:
        trips = [(rnd.randrange(d), rnd.randrange(d), rnd.randrange(d))
                 for _ in range(ntriples)]
    for (i, j, k) in trips:
        x, y, z = basis[i], basis[j], basis[k]
        lhs = L.br(x, L.br(y, z))
        r1 = L.br(L.br(x, y), z)
        s = (-1) ** (L.par[i] * L.par[j])
        r2 = L.br(y, L.br(x, z))
        for t in range(d):
            if (lhs[t] - r1[t] - s * r2[t]) % p:
                badj += 1
                break
    return bad, badj, len(trips)


def derived_series(L):
    p, d = L.p, L.dim
    basis = [[1 if k == i else 0 for k in range(d)] for i in range(d)]
    cur = basis
    dims = []
    for _ in range(4):
        vs = []
        for x in cur:
            for y in cur:
                vs.append(L.br(x, y))
        R, piv = _rref([v for v in vs if any(v)], p) if any(any(v) for v in vs) else ([], [])
        cur = [r for r in R if any(x % p for x in r)]
        dims.append(len(cur))
        if len(dims) >= 2 and dims[-1] == dims[-2]:
            break
    return dims


def center(L):
    """the center, as a subspace; returns (dim, dim_ev, dim_od)."""
    p, d = L.p, L.dim
    T = L.table()
    # x in the center <=> sum_i x_i [e_i, e_j] = 0 for all j: the LEFT nullspace
    # of the matrix M with rows = i, columns = (j,k)
    M = [[T[i][j][k] % p for j in range(d) for k in range(d)] for i in range(d)]
    MT = [[M[i][c] for i in range(d)] for c in range(len(M[0]))]
    ns = _nullspace(MT, p)
    return len(ns), ns


def ideal_generated(L, v):
    """the ideal generated by one element."""
    p, d = L.p, L.dim
    basis = [[1 if k == i else 0 for k in range(d)] for i in range(d)]
    E = echelon_add([], [[x % p for x in v]], p, d)
    while True:
        new = []
        for r in E:
            for b in basis:
                new.append(L.br(r, b))
        E2 = echelon_add(E, [v for v in new if any(x % p for x in v)], p, d)
        if len(E2) == len(E):
            return len(E)
        E = E2


def is_simple(L):
    p, d = L.p, L.dim
    for i in range(d):
        e = [1 if k == i else 0 for k in range(d)]
        if ideal_generated(L, e) != d:
            return False, i
    return True, None


def derivations(L, par):
    """dim of the space of derivations of parity par."""
    p, d = L.p, L.dim
    T = L.table()
    # unknowns: entries D[j][i] = coefficient of e_i in D(e_j), with
    # p_L(i) = p_L(j) + par
    unk = {}
    for j in range(d):
        for i in range(d):
            if (L.par[i] + L.par[j]) % 2 == par % 2:
                unk[(j, i)] = len(unk)
    N = len(unk)
    E = []
    rows = []
    for j in range(d):
        for k in range(j, d):
            # D[e_j,e_k] = [D e_j, e_k] + (-1)^{par p_L(j)} [e_j, D e_k]
            s = (-1) ** (par * L.par[j])
            for t in range(d):
                row = [0] * N
                # LHS: D applied to the vector T[j][k]
                for i in range(d):
                    c = T[j][k][i] % p
                    if not c:
                        continue
                    key = (i, t)
                    if key in unk:
                        row[unk[key]] = (row[unk[key]] + c) % p
                # RHS 1: [D e_j, e_k] = sum_i D[j][i] [e_i, e_k]
                for i in range(d):
                    key = (j, i)
                    if key not in unk:
                        continue
                    c = T[i][k][t] % p
                    if c:
                        row[unk[key]] = (row[unk[key]] - c) % p
                # RHS 2: s * [e_j, D e_k]
                for i in range(d):
                    key = (k, i)
                    if key not in unk:
                        continue
                    c = (s * T[j][i][t]) % p
                    if c:
                        row[unk[key]] = (row[unk[key]] - c) % p
                if any(row):
                    rows.append(row)
        if len(rows) >= 3000:
            E = echelon_add(E, rows, p, N)
            rows = []
    E = echelon_add(E, rows, p, N)
    return N - len(E), N


def inner_dim(L):
    """dim of the image of ad: g -> Der g."""
    p, d = L.p, L.dim
    T = L.table()
    rows = [[T[i][j][k] % p for j in range(d) for k in range(d)] for i in range(d)]
    return span_dim(rows, p)


# --------------------------------------------------------------- square-zero count

def odd_split(L):
    """split the odd part into A = xi-free monomials and C = the rest."""
    A = [i for i in L.od if sum(L.mons[i][1]) == 0]
    C = [i for i in L.od if sum(L.mons[i][1]) > 0]
    return A, C


def count_sqzero_bruteforce(L, idxs):
    """#{x in span(e_i, i in idxs) | [x,x] = 0}, by enumeration."""
    p, d = L.p, L.dim
    T = L.table()
    # precompute Q(e_i) and 2B(e_i,e_j)
    Q = {i: T[i][i] for i in idxs}
    B = {}
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            i, j = idxs[a], idxs[b]
            B[(i, j)] = [(2 * T[i][j][k]) % p for k in range(d)]
    cnt = 0
    for co in itertools.product(range(p), repeat=len(idxs)):
        val = [0] * d
        for a in range(len(idxs)):
            ca = co[a]
            if not ca:
                continue
            i = idxs[a]
            c2 = (ca * ca) % p
            for k, v in enumerate(Q[i]):
                if v:
                    val[k] = (val[k] + c2 * v) % p
            for b in range(a + 1, len(idxs)):
                cb = co[b]
                if not cb:
                    continue
                j = idxs[b]
                cc = (ca * cb) % p
                for k, v in enumerate(B[(i, j)]):
                    if v:
                        val[k] = (val[k] + cc * v) % p
        if not any(val):
            cnt += 1
    return cnt


def count_sqzero_affine(L, A, C, verbose=False):
    """#{x in span(A u C) | [x,x] = 0}, assuming [C,C] = 0 (verified here).

    x = a + c, [x,x] = [a,a] + 2 B(a,c): affine in c for fixed a.
    """
    p, d = L.p, L.dim
    T = L.table()
    # hypothesis: [C,C] = 0
    for i in C:
        for j in C:
            if any(x % p for x in T[i][j]):
                raise AssertionError("[C,C] != 0: the affine method does not apply")
    QA = {i: T[i][i] for i in A}
    BAA = {}
    for a1 in range(len(A)):
        for a2 in range(a1 + 1, len(A)):
            i, j = A[a1], A[a2]
            BAA[(i, j)] = [(2 * T[i][j][k]) % p for k in range(d)]
    BAC = {}
    for i in A:
        for j in C:
            BAC[(i, j)] = [(2 * T[i][j][k]) % p for k in range(d)]
    total = 0
    nAC = len(C)
    for co in itertools.product(range(p), repeat=len(A)):
        # rhs = -[a,a]
        val = [0] * d
        for a1 in range(len(A)):
            ca = co[a1]
            if not ca:
                continue
            i = A[a1]
            c2 = (ca * ca) % p
            for k, v in enumerate(QA[i]):
                if v:
                    val[k] = (val[k] + c2 * v) % p
            for a2 in range(a1 + 1, len(A)):
                cb = co[a2]
                if not cb:
                    continue
                j = A[a2]
                cc = (ca * cb) % p
                for k, v in enumerate(BAA[(i, j)]):
                    if v:
                        val[k] = (val[k] + cc * v) % p
        rhs = [(-x) % p for x in val]
        # columns: 2B(a, e_j) for j in C
        colsT = []
        for j in C:
            colv = [0] * d
            for a1 in range(len(A)):
                ca = co[a1]
                if not ca:
                    continue
                for k, v in enumerate(BAC[(A[a1], j)]):
                    if v:
                        colv[k] = (colv[k] + ca * v) % p
            colsT.append(colv)
        # solve sum_j c_j colsT[j] = rhs  -> matrix with rows = equations
        M = [[colsT[j][k] for j in range(nAC)] for k in range(d)]
        r = _rank(M, p) if any(any(row) for row in M) else 0
        Mb = [M[k] + [rhs[k]] for k in range(d)]
        rb = _rank(Mb, p) if any(any(row) for row in Mb) else 0
        if rb == r:
            total += p ** (nAC - r)
    return total


# --------------------------------------------------------------- report

def report(n, p, u, shapes=("0", "Xi"), do_der=True, do_count=True,
           bf_dim=None):
    algs = {}
    for sh in shapes:
        L = LeAlg(n, p, u, sh)
        algs[sh] = L
        say(f"--- n={n} p={p} u={u} shape={sh}: sdim = {L.sdim()}, dim = {L.dim}")
        bad, badj, nt = check_axioms(L)
        say(f"    axioms: antisymmetry failures {bad}, Jacobi failures {badj} of {nt} triples")
        say(f"    derived series dims: {derived_series(L)}")
        cz, ns = center(L)
        say(f"    dim center = {cz}")
        sim, wit = is_simple(L)
        say(f"    simple: {sim}" + ("" if sim else f" (proper ideal from e_{wit})"))
        sq = [L.table()[i][i] for i in L.od]
        say(f"    dim span of [x,x], x odd = {span_dim(sq, p)}")
        if do_der:
            d0, N0 = derivations(L, 0)
            d1, N1 = derivations(L, 1)
            inn = inner_dim(L)
            say(f"    dim Der = {d0 + d1}  (even {d0}, odd {d1}); "
                f"inner {inn}; dim H^1(g;g) = {d0 + d1 - inn}")
        if do_count:
            A, C = odd_split(L)
            say(f"    odd part: A (xi-free) {len(A)}, C (rest) {len(C)}")
            if len(A) + len(C) <= 14:
                nb = count_sqzero_bruteforce(L, A + C)
                say(f"    #(x odd | [x,x]=0) by enumeration = {nb} of {p**(len(A)+len(C))}")
            if C:
                try:
                    na = count_sqzero_affine(L, A, C)
                    say(f"    #(x odd | [x,x]=0) by the affine method = {na}"
                        f" of {p**(len(A)+len(C))}")
                except AssertionError as e:
                    say(f"    affine method not applicable: {e}")
            if bf_dim:
                # cross-check the two methods on a sub-cube A u C[:k]
                k = max(0, bf_dim - len(A))
                sub = A + C[:k]
                nb = count_sqzero_bruteforce(L, sub)
                na = count_sqzero_affine(L, A, C[:k])
                say(f"    cross-check on A u C[:{k}] (dim {len(sub)}): "
                    f"enumeration {nb}, affine {na}, agree = {nb == na}")
    return algs


def main():
    global LOG
    here = os.path.dirname(os.path.abspath(__file__))
    LOG = open(os.path.join(here, "leiso.log"), "w")
    from exactlin import backend
    say(f"backend: {backend()}")
    say("")
    say("=== n = 1 (the case already settled in the text: a control) ===")
    report(1, 3, (1,), do_der=True, do_count=True)
    say("")
    report(1, 3, (2,), do_der=False, do_count=True)
    say("")
    say("=== n = 2, u = One: the open case ===")
    report(2, 3, (1, 1), do_der=True, do_count=True, bf_dim=12)
    LOG.close()


if __name__ == "__main__":
    main()
