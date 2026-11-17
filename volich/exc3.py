#!/usr/bin/env python3
"""Explicit rational models of the two exceptional Lie superalgebras
       ag(2) = G(3)   (sdim 17|14)   and   ab(3) = F(4)  (sdim 24|16)
in a WEIGHT basis whose weights are literally the roots used in
the census of the Z-gradings, so that every Z-grading of
the census is simply  deg(basis vector) = <weight, h>.

Both have the same shape
        g = (s + sl(2)) + (V (x) C^2),
   G(3):  s = g(2)  acting on V = the 7-dim module,
   F(4):  s = o(7)  acting on V = spin_8,
with the odd bracket forced (up to normalisation) to be
        [v(x)u, v'(x)u'] = alpha*(v,v')_V * gamma(u,u')  +  beta*w(u,u')*pi(v^v'),
where (,)_V is the invariant symmetric form, gamma: S^2(C^2) -> sl(2),
w = the symplectic form on C^2 and pi: Lambda^2 V -> s the moment map
   pi(v^v') = sum_i <A_i v, v'>_V A_i^*   (A_i^* dual w.r.t. the trace form).
The ratio beta/alpha is determined by the (odd,odd,odd) Jacobi identity;
everything is then verified by an exact super-Jacobi check.

Ambient 7-dim space (used for both):  w_0; w_1,w_2,w_3; w_{-1},w_{-2},w_{-3}
with (w_i,w_{-i}) = 1, (w_0,w_0) = 2  (the 2 makes the Clifford action of w_0
rational).  Weights of w_0,w_i,w_{-i} are 0, e_i, -e_i.
"""
from fractions import Fraction as F
from itertools import product, combinations
from lin import span, dim, inspan, kernel, rref, zero

# --------------------------------------------------------------- V_7 and o(7)
# positions: 0 = w_0, 1,2,3 = w_1,w_2,w_3, 4,5,6 = w_{-1},w_{-2},w_{-3}
DV = 7
EWT = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
       (-1, 0, 0), (0, -1, 0), (0, 0, -1)]        # weights in e-coordinates
S7 = [[F(0)] * DV for _ in range(DV)]
S7[0][0] = F(2)
for i in (1, 2, 3):
    S7[i][i + 3] = F(1)
    S7[i + 3][i] = F(1)


def mzero(n, m=None):
    m = n if m is None else m
    return [[F(0)] * m for _ in range(n)]


def mflat(M):
    return [x for row in M for x in row]


def munflat(v, n):
    return [[v[i * n + j] for j in range(n)] for i in range(n)]


def mmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = mzero(n, p)
    for i in range(n):
        Ai = A[i]
        for k in range(m):
            a = Ai[k]
            if a:
                Bk = B[k]
                out_i = out[i]
                for j in range(p):
                    if Bk[j]:
                        out_i[j] += a * Bk[j]
    return out


def mcomm(A, B):
    return [[x - y for x, y in zip(r, s)] for r, s in zip(mmul(A, B), mmul(B, A))]


def mtrace(A):
    return sum(A[i][i] for i in range(len(A)))


def mact(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def E_(a, b, n=DV):
    M = mzero(n)
    M[a][b] = F(1)
    return M


def wtsub(a, b):
    return tuple(x - y for x, y in zip(EWT[a], EWT[b]))


def o7_basis():
    """weight-homogeneous basis of o(V_7, S7); returns list of (weight, matrix)."""
    out = []
    bywt = {}
    for a in range(DV):
        for b in range(DV):
            bywt.setdefault(wtsub(a, b), []).append((a, b))
    for wt in sorted(bywt):
        pairs = bywt[wt]
        rows = []                        # conditions A^T S + S A = 0
        for p in range(DV):
            for q in range(DV):
                row = []
                for (a, b) in pairs:
                    # A = E_ab: (A^T S)_{pq} = sum_k A_kp S_kq = [p==b] S_aq
                    # (S A)_{pq} = sum_k S_pk A_kq = S_pa [b==q]
                    val = (S7[a][q] if p == b else F(0)) + \
                          (S7[p][a] if b == q else F(0))
                    row.append(val)
                rows.append(row)
        for sol in kernel(rows, len(pairs)):
            M = mzero(DV)
            for cf, (a, b) in zip(sol, pairs):
                if cf:
                    M[a][b] += cf
            out.append((wt, M))
    return out


O7 = o7_basis()
assert len(O7) == 21, len(O7)
assert sum(1 for wt, _ in O7 if wt == (0, 0, 0)) == 3


# ------------------------------------------------------------------- g(2) < o(7)
G2WT = [(0, 0), (1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]
"""weight of w_0, w_1, w_2, w_3, w_{-1}, w_{-2}, w_{-3} in the g(2) coordinates
   (delta-free part of the coordinates of `gradings3exc.py`: e_3 = -e_1-e_2)."""


def g2_basis(phi):
    """A in gl(7) with  phi(Ax,y,z)+phi(x,Ay,z)+phi(x,y,Az) = 0,
       given as a weight-homogeneous basis (weights in the g(2) coordinates).
       For a phi in the right GL(7)-orbit this is g(2), of dim 14, and it
       automatically preserves a symmetric form (computed separately)."""
    out = []
    bywt = {}
    for a in range(DV):
        for b in range(DV):
            wt = (G2WT[a][0] - G2WT[b][0], G2WT[a][1] - G2WT[b][1])
            bywt.setdefault(wt, []).append((a, b))
    for wt in sorted(bywt):
        pairs = bywt[wt]
        rows = []
        for t in combinations(range(DV), 3):
            x, y, z = t
            row = []
            for (a, b) in pairs:
                # A = E_ab sends w_b -> w_a
                s = F(0)
                if b == x:
                    s += phi.get(tuple(sorted((a, y, z))), F(0)) * _sgn((a, y, z))
                if b == y:
                    s += phi.get(tuple(sorted((x, a, z))), F(0)) * _sgn((x, a, z))
                if b == z:
                    s += phi.get(tuple(sorted((x, y, a))), F(0)) * _sgn((x, y, a))
                row.append(s)
            rows.append(row)
        for sol in kernel(rows, len(pairs)):
            M = mzero(DV)
            for cf, (a, b) in zip(sol, pairs):
                if cf:
                    M[a][b] += cf
            out.append((wt, M))
    return out


def _sgn(t):
    """sign of the permutation sorting t (0 if repeated)."""
    a, b, c = t
    if a == b or b == c or a == c:
        return F(0)
    s = 1
    l = [a, b, c]
    for i in range(3):
        for j in range(i + 1, 3):
            if l[i] > l[j]:
                s = -s
    return F(s)


def find_g2():
    """the weight-0 3-form ansatz; every choice with nonzero constants gives a
       14-dimensional stabiliser (checked over a 4096-point grid), i.e. g(2)."""
    cand = []
    for c in (F(1), F(-1)):
        phi = {(1, 2, 3): F(1), (4, 5, 6): F(1),
               (0, 1, 4): c, (0, 2, 5): c, (0, 3, 6): c}
        g2 = g2_basis(phi)
        if len(g2) == 14 and sum(1 for wt, _ in g2 if wt == (0, 0)) == 2:
            cand.append((phi, g2))
    return cand


# ------------------------------------------------------------- spin_8 of o(7)
DS = 8
SUB = [tuple(t) for k in range(4) for t in combinations((1, 2, 3), k)]
SUB = sorted(SUB, key=lambda t: (len(t), t))
SIDX = {t: i for i, t in enumerate(SUB)}


def swt(t):
    """spin weight of the basis vector Lambda^t: sum_{i in t} e_i - (e1+e2+e3)/2"""
    return tuple(F(1, 1) * (1 if (i + 1) in t else 0) - F(1, 2) for i in range(3))


SWT = [swt(t) for t in SUB]


def clifford():
    """c(x) for x = the 7 basis vectors of V_7, as 8x8 rational matrices,
       normalised by  c(x)c(y) + c(y)c(x) = (x,y) Id."""
    C = []
    for a in range(DV):
        M = mzero(DS)
        if a == 0:
            for t in SUB:
                M[SIDX[t]][SIDX[t]] = F((-1) ** len(t))
        elif 1 <= a <= 3:
            i = a
            for t in SUB:
                if i in t:
                    continue
                nt = tuple(sorted(t + (i,)))
                sg = F((-1) ** sum(1 for j in t if j < i))
                M[SIDX[nt]][SIDX[t]] = sg
        else:
            i = a - 3
            for t in SUB:
                if i not in t:
                    continue
                nt = tuple(x for x in t if x != i)
                sg = F((-1) ** sum(1 for j in t if j < i))
                M[SIDX[nt]][SIDX[t]] = sg
        C.append(M)
    return C


CL = clifford()
for a in range(DV):
    for b in range(DV):
        lhs = [[x + y for x, y in zip(r, s)]
               for r, s in zip(mmul(CL[a], CL[b]), mmul(CL[b], CL[a]))]
        rhs = mzero(DS)
        for i in range(DS):
            rhs[i][i] = S7[a][b]
        assert lhs == rhs, ("clifford", a, b)


def spin_rep():
    """rho: o(7) -> gl(8) with [rho(A), c(z)] = c(Az)."""
    # candidate space: T_ab = c(a)c(b) - (a,b)/2
    Ts = []
    for a in range(DV):
        for b in range(DV):
            if a == b:
                continue
            M = mmul(CL[a], CL[b])
            for i in range(DS):
                M[i][i] -= S7[a][b] / 2
            Ts.append(M)
    basis = span([mflat(M) for M in Ts])          # subspace of gl(8)
    # Phi(T) = the o(7) element A with [T, c(z)] = c(Az)
    clflat = [mflat(M) for M in CL]
    rows_c, _ = rref(clflat)
    assert len(rows_c) == DV

    def phi_of(T):
        A = mzero(DV)
        for z in range(DV):
            M = mcomm(T, CL[z])
            # express M = sum_a coef_a c(a)
            sol = _solve_in_basis(clflat, mflat(M))
            assert sol is not None, "not in the Clifford span"
            for a in range(DV):
                A[a][z] = sol[a]
        return A
    imgs = [phi_of(munflat(b, DS)) for b in basis]
    # invert Phi on the 21-dim image
    mat = [mflat(A) for A in imgs]
    rho = []
    for wt, A in O7:
        sol = _solve_in_basis(mat, mflat(A))
        assert sol is not None, "o(7) element not in the image of Phi"
        M = mzero(DS)
        for cf, b in zip(sol, basis):
            if cf:
                Mb = munflat(b, DS)
                for i in range(DS):
                    for j in range(DS):
                        if Mb[i][j]:
                            M[i][j] += cf * Mb[i][j]
        rho.append(M)
    return rho


def _solve_in_basis(basis, v):
    """coefficients of v in terms of the (independent) list `basis`."""
    n = len(v)
    rows = []
    for k in range(n):
        rows.append([basis[i][k] for i in range(len(basis))] + [v[k]])
    m, piv = rref(rows)
    if any(p == len(basis) for p in piv):
        return None
    sol = [F(0)] * len(basis)
    for r, p in zip(m, piv):
        sol[p] = r[len(basis)]
    # verify
    chk = [sum(sol[i] * basis[i][k] for i in range(len(basis))) for k in range(n)]
    if chk != list(v):
        return None
    return sol


def invariant_form(ops, d):
    """symmetric G with op^T G + G op = 0 for all ops; returns the (unique) one."""
    rows = []
    unk = [(i, j) for i in range(d) for j in range(i, d)]
    pos = {}
    for k, (i, j) in enumerate(unk):
        pos[(i, j)] = k
        pos[(j, i)] = k
    for M in ops:
        for p in range(d):
            for q in range(d):
                row = [F(0)] * len(unk)
                for k in range(d):
                    if M[k][p]:
                        row[pos[(k, q)]] += M[k][p]
                    if M[k][q]:
                        row[pos[(p, k)]] += M[k][q]
                rows.append(row)
    ker = kernel(rows, len(unk))
    assert len(ker) == 1, "invariant form not unique: %d" % len(ker)
    G = mzero(d)
    for (i, j), k in [((i, j), pos[(i, j)]) for (i, j) in unk]:
        G[i][j] = ker[0][k]
        G[j][i] = ker[0][k]
    return G


# ------------------------------------------------------------------- assembly
class Model:
    """g = (s + sl2) + V (x) C^2 with sparse structure constants."""

    def __init__(self, name, s_basis, Vdim, Vform, Vweights, rho, sl2root,
                 alpha=F(1)):
        self.name = name
        self.s = s_basis                      # list of (weight, matrix on V)
        self.ds = len(s_basis)
        self.dV = Vdim
        self.G = Vform                        # invariant symmetric form on V
        self.Vwt = Vweights                   # weight of each basis vector of V
        self.rho = rho                        # action of s_basis[i] on V
        self.sl2root = sl2root                # the root of sl(2) (delta-coord)
        self.N = self.ds + 3 + 2 * self.dV
        self.iE, self.iH, self.iF = self.ds, self.ds + 1, self.ds + 2
        self.PAR = [0] * (self.ds + 3) + [1] * (2 * self.dV)
        self.WT = self._weights()
        self.gram, self.dualc = self._trace_form()
        self.B = None
        self.beta = None

    # ---- indexing
    def iodd(self, a, eps):
        return self.ds + 3 + 2 * a + eps

    def _weights(self):
        """weight of every basis vector, in the coordinates (delta, e_1, ...)
           used by gradings3exc.py."""
        r = len(self.s[0][0])
        wt = []
        h = F(self.sl2root, 2)
        for (w, M) in self.s:
            wt.append((F(0),) + tuple(F(x) for x in w))
        wt.append((F(self.sl2root),) + (F(0),) * r)
        wt.append((F(0),) * (r + 1))
        wt.append((F(-self.sl2root),) + (F(0),) * r)
        for a in range(self.dV):
            for eps in (0, 1):
                d = h if eps == 0 else -h
                wt.append((d,) + tuple(F(x) for x in self.Vwt[a]))
        return wt

    def _trace_form(self):
        """Gram matrix of the trace form on s (in the 7-dim rep) and its inverse."""
        n = self.ds
        gram = mzero(n)
        for i in range(n):
            for j in range(n):
                gram[i][j] = mtrace(mmul(self.s[i][1], self.s[j][1]))
        rows = [list(gram[i]) + [F(1) if k == i else F(0) for k in range(n)]
                for i in range(n)]
        m, piv = rref(rows)
        assert piv == list(range(n)), "trace form on s degenerate"
        inv = [[m[i][n + j] for j in range(n)] for i in range(n)]
        return gram, inv

    # ---- the moment map pi: Lambda^2 V -> s
    def pi(self, a, b):
        """pi(v_a ^ v_b) as a vector of coefficients in the s-basis."""
        vals = []
        for i in range(self.ds):
            R = self.rho[i]
            # <R v_a, v_b> = sum_k R[k][a] * G[k][b]
            s = F(0)
            for k in range(self.dV):
                if R[k][a] and self.G[k][b]:
                    s += R[k][a] * self.G[k][b]
            vals.append(s)
        return [sum(self.dualc[i][j] * vals[j] for j in range(self.ds))
                for i in range(self.ds)]

    # ---- structure constants
    def build(self, alpha, beta):
        N = self.N
        B = {}

        def put(i, j, d):
            d = {k: v for k, v in d.items() if v}
            if d:
                B[(i, j)] = d

        # s x s
        for i in range(self.ds):
            for j in range(self.ds):
                C = mcomm(self.s[i][1], self.s[j][1])
                if any(any(r) for r in C):
                    sol = _solve_in_basis([mflat(M) for (_, M) in self.s], mflat(C))
                    assert sol is not None, "s not closed"
                    put(i, j, {k: sol[k] for k in range(self.ds)})
        # sl(2)
        iE, iH, iF = self.iE, self.iH, self.iF
        put(iE, iF, {iH: F(1)}); put(iF, iE, {iH: F(-1)})
        put(iH, iE, {iE: F(2)}); put(iE, iH, {iE: F(-2)})
        put(iH, iF, {iF: F(-2)}); put(iF, iH, {iF: F(2)})
        # s on odd
        for i in range(self.ds):
            R = self.rho[i]
            for a in range(self.dV):
                for eps in (0, 1):
                    src = self.iodd(a, eps)
                    d = {}
                    for k in range(self.dV):
                        if R[k][a]:
                            d[self.iodd(k, eps)] = R[k][a]
                    put(i, src, d)
                    put(src, i, {k: -v for k, v in d.items()})
        # sl(2) on odd:  E u1 = u0, F u0 = u1, H u0 = u0, H u1 = -u1
        for a in range(self.dV):
            u0, u1 = self.iodd(a, 0), self.iodd(a, 1)
            put(iE, u1, {u0: F(1)}); put(u1, iE, {u0: F(-1)})
            put(iF, u0, {u1: F(1)}); put(u0, iF, {u1: F(-1)})
            put(iH, u0, {u0: F(1)}); put(u0, iH, {u0: F(-1)})
            put(iH, u1, {u1: F(-1)}); put(u1, iH, {u1: F(1)})
        # odd x odd
        gam = {(0, 0): (iE, F(2)), (1, 1): (iF, F(-2)),
               (0, 1): (iH, F(-1)), (1, 0): (iH, F(-1))}
        om = {(0, 1): F(1), (1, 0): F(-1), (0, 0): F(0), (1, 1): F(0)}
        for a in range(self.dV):
            for b in range(self.dV):
                p = self.pi(a, b)
                for e1 in (0, 1):
                    for e2 in (0, 1):
                        d = {}
                        if self.G[a][b]:
                            k, c = gam[(e1, e2)]
                            d[k] = d.get(k, F(0)) + alpha * self.G[a][b] * c
                        if om[(e1, e2)]:
                            for i in range(self.ds):
                                if p[i]:
                                    d[i] = d.get(i, F(0)) + \
                                        beta * om[(e1, e2)] * p[i]
                        put(self.iodd(a, e1), self.iodd(b, e2), d)
        self.B = B
        return B

    # ---- brackets
    def br(self, x, y):
        out = [F(0)] * self.N
        B = self.B
        for i, xi in enumerate(x):
            if not xi:
                continue
            for j, yj in enumerate(y):
                if not yj:
                    continue
                d = B.get((i, j))
                if d:
                    c = xi * yj
                    for k, v in d.items():
                        out[k] += c * v
        return out

    def e(self, i):
        v = [F(0)] * self.N
        v[i] = F(1)
        return v

    def jacobi_bad(self):
        """count of failing basis triples of the super-Jacobi identity."""
        N, B, PAR = self.N, self.B, self.PAR
        bad = 0
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    lhs = self.br(self.e(a), self.br(self.e(b), self.e(c)))
                    r1 = self.br(self.br(self.e(a), self.e(b)), self.e(c))
                    r2 = self.br(self.e(b), self.br(self.e(a), self.e(c)))
                    sg = F((-1) ** (PAR[a] * PAR[b]))
                    if lhs != [x + sg * y for x, y in zip(r1, r2)]:
                        bad += 1
        return bad

    def solve_beta(self):
        """odd-odd-odd Jacobi is linear in (alpha,beta); find the ratio."""
        odd = [self.iodd(a, e) for a in range(self.dV) for e in (0, 1)]
        # the Jacobiator on odd triples is linear in (alpha,beta): evaluate it
        # at (1,0) and at (0,1) and solve the resulting linear system
        Js = []
        for (al, be) in ((F(1), F(0)), (F(0), F(1))):
            self.build(al, be)
            J = {}
            for (x, y, z) in combinations(odd, 3):
                v = self.br(self.br(self.e(x), self.e(y)), self.e(z))
                v = [p + q for p, q in zip(
                    v, self.br(self.br(self.e(y), self.e(z)), self.e(x)))]
                v = [p + q for p, q in zip(
                    v, self.br(self.br(self.e(z), self.e(x)), self.e(y)))]
                J[(x, y, z)] = v
            for (x, y) in product(odd, repeat=2):
                v = self.br(self.br(self.e(x), self.e(x)), self.e(y))
                v = [p + 2 * q for p, q in zip(
                    v, self.br(self.br(self.e(x), self.e(y)), self.e(x)))]
                J[(x, x, y)] = v
            Js.append(J)
        rows = []
        for key in Js[0]:
            for k in range(self.N):
                rows.append([Js[0][key][k], Js[1][key][k]])
        ker = kernel(rows, 2)
        return ker


# ------------------------------------------------------------------ builders
def build_G3():
    cands = find_g2()
    assert cands, "no phi found"
    phi, g2 = cands[0]
    wts = [wt for wt, _ in g2]
    # sanity: the 14 weights must be the g(2) root system + (0,0) twice
    nz = sorted(w for w in wts if w != (0, 0))
    assert len(nz) == 12 and len(set(nz)) == 12, nz
    g2roots = set()
    for i in range(3):
        for s in (1, -1):
            e = [(1, 0), (0, 1), (-1, -1)][i]
            g2roots.add((s * e[0], s * e[1]))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            a = [(1, 0), (0, 1), (-1, -1)][i]
            b = [(1, 0), (0, 1), (-1, -1)][j]
            g2roots.add((a[0] - b[0], a[1] - b[1]))
    assert set(nz) == g2roots, (sorted(set(nz)), sorted(g2roots))
    rho = [M for _, M in g2]
    GV = invariant_form(rho, DV)          # the g(2)-invariant form on the 7
    m = Model("ag(2) = G(3)", g2, DV, GV, G2WT, rho, 2)
    ker = m.solve_beta()
    assert len(ker) == 1, ("beta not unique", ker)
    al, be = ker[0]
    assert al != 0
    be = be / al
    m.build(F(1), be)
    m.beta = be
    m.phi = phi
    return m


def build_F4():
    rho = spin_rep()
    for i in range(len(O7)):
        for j in range(len(O7)):
            C = mcomm(O7[i][1], O7[j][1])
            sol = _solve_in_basis([mflat(M) for (_, M) in O7], mflat(C))
            got = mcomm(rho[i], rho[j])
            exp = mzero(DS)
            for cf, M in zip(sol, rho):
                if cf:
                    for p in range(DS):
                        for q in range(DS):
                            if M[p][q]:
                                exp[p][q] += cf * M[p][q]
            assert got == exp, ("rho not a homomorphism", i, j)
    GS = invariant_form(rho, DS)
    m = Model("ab(3) = F(4)", O7, DS, GS, SWT, rho, 1)
    ker = m.solve_beta()
    assert len(ker) == 1, ("beta not unique", ker)
    al, be = ker[0]
    assert al != 0
    be = be / al
    m.build(F(1), be)
    m.beta = be
    return m


# ---------------------------------------------------------------- diagnostics
def report(m, expected_sdim):
    ev = sum(1 for p in m.PAR if p == 0)
    od = sum(1 for p in m.PAR if p == 1)
    print("%-14s  N=%d  sdim %d|%d   beta=%s" % (m.name, m.N, ev, od, m.beta))
    assert (ev, od) == expected_sdim, (ev, od)
    bad = m.jacobi_bad()
    print("   super-Jacobi failures: %d" % bad)
    assert bad == 0
    # simplicity: the ideal generated by any basis vector is everything
    for i in range(m.N):
        I = span([m.e(i)])
        changed = True
        while changed:
            changed = False
            for v in list(I):
                for k in range(m.N):
                    w = m.br(m.e(k), v)
                    if any(w) and not inspan(w, I):
                        I = span(I + [w]); changed = True
        assert len(I) == m.N, ("not simple: ideal of dim %d from basis %d"
                               % (len(I), i))
    print("   simple: yes (every basis vector generates g as an ideal)")
    return True


if __name__ == "__main__":
    g3 = build_G3()
    report(g3, (17, 14))
    f4 = build_F4()
    report(f4, (24, 16))
