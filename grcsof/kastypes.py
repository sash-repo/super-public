#!/usr/bin/env python3
"""The kas_0-modules V and the corresponding spaces of tensor fields.

kas_0 = k(1|6)_0 = c o(6) = gl(4).  For a superfunction f the matrix

      a(f):   [K_f, D_i] = sum_j a(f)_{ij} D_j                          (*)

takes values in c o(6) pointwise (verified in kastens.py), and the module of
tensor fields of type V (= coinduced from V, kas_{>0} acting trivially) is

      (L_f phi)_b = K_f(phi_b) + lam d_t(f) phi_b + sum_a rho(a(f)^T)_{ba} phi_a,

rho being the representation of c o(6) on V.  Weighted densities are V = C
with o(6) acting trivially; a general V is expanded in a fixed basis of
c o(6), so that rho enters only through 16 CONSTANT matrices.

*** For f in as one has tr a(f) = 0 and a(f) in o(6) pointwise, so BOTH the
weight lam AND the det-twist are invisible to as: an as-module of tensor
fields depends on the o(6)-module V only. ***

The types provided:
   F        1   the densities (trivial o(6)-module)
   L1       6   Lambda^1 = the vector representation of o(6) = Lambda^2(id_4)
   L2      15   Lambda^2 = the adjoint of o(6) = sl(4)
   L3p     10   the two halves of Lambda^3 = S^2(id_4), S^2(id_4^*);
   L3m     10     L3p is the one whose extreme torus weight is (1,1,1),
                  i.e. the one matching Lambda^3_xi subset as
   Sp       4   the half-spinors id_4 and id_4^* of o(6) = sl(4);
   Sm       4     Sp is the one containing the weight (1/2,1/2,1/2) vector
"""
from fractions import Fraction as Fr
from itertools import combinations

from superN import add, smul, mul, mono, parity, d_t
from kas6 import ALG, NAMES, B6
from kastens import amat, transpose, N6

# --------------------------------------------------- a basis of c o(6)

def _o6_basis():
    """basis of o(6) = {X : X^T B + B X = 0} as 6x6 rational matrices, plus
    the identity; returned with a dual basis for expanding."""
    # solve the linear conditions
    unk = [(i, j) for i in range(N6) for j in range(N6)]
    rows = []
    for i in range(N6):
        for j in range(N6):
            r = [0] * 36
            for k in range(N6):
                if B6[k][j]:
                    r[unk.index((k, i))] += B6[k][j]
                if B6[i][k]:
                    r[unk.index((k, j))] += B6[i][k]
            if any(r):
                rows.append(r)
    # nullspace
    from knclass import nullspace
    ns = nullspace(rows, 36)
    out = []
    for v in ns:
        M = [[Fr(v[unk.index((i, j))]) for j in range(N6)] for i in range(N6)]
        out.append(M)
    return out


O6 = _o6_basis()
ID6 = [[Fr(int(i == j)) for j in range(N6)] for i in range(N6)]
CO6 = O6 + [ID6]                      # 15 + 1 = 16 = dim gl(4)

# expansion of an arbitrary c o(6) matrix in CO6: pick 16 independent entries
_ENTRIES = []
_MAT = []
for M in CO6:
    _MAT.append([M[i][j] for i in range(N6) for j in range(N6)])
# choose pivot positions by echelonizing the 16 x 36 matrix
def _pivots(rows):
    R = [r[:] for r in rows]
    piv = []
    r = 0
    for c in range(len(R[0])):
        p = next((k for k in range(r, len(R)) if R[k][c]), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        pv = R[r][c]
        R[r] = [x / pv for x in R[r]]
        for k in range(len(R)):
            if k != r and R[k][c]:
                f = R[k][c]
                R[k] = [a - f * b for a, b in zip(R[k], R[r])]
        piv.append(c)
        r += 1
        if r == len(R):
            break
    return piv, R


_PIV, _RED = _pivots(_MAT)
assert len(_PIV) == 16, len(_PIV)
# _RED is the reduced form: CO6 in terms of pivot entries. To expand a matrix
# M we need the inverse of the 16x16 submatrix of _MAT at the pivot columns.
def _inv16():
    A = [[_MAT[r][c] for c in _PIV] for r in range(16)]
    n = 16
    M = [A[i][:] + [Fr(int(i == j)) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c])
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


_INV = _inv16()       # coefficients: coeff_alpha = sum_k _INV[k][alpha] * M[pivot_k]


def expand(M):
    """M a 6x6 matrix of superfunctions lying in c o(6) pointwise ->
    list of 16 superfunctions c_alpha with M = sum c_alpha CO6[alpha]."""
    ent = [M[c // N6][c % N6] for c in _PIV]
    out = []
    for al in range(16):
        s = {}
        for k in range(16):
            if _INV[k][al] and ent[k]:
                s = add(s, smul(_INV[k][al], ent[k]))
        out.append(s)
    # safety: reconstruct
    for i in range(N6):
        for j in range(N6):
            s = {}
            for al in range(16):
                if CO6[al][i][j] and out[al]:
                    s = add(s, smul(CO6[al][i][j], out[al]))
            assert not add(s, smul(-1, M[i][j])), (i, j)
    return out


# --------------------------------------------------------------- the types

class Type:
    """V with rho given by 16 constant matrices (one per basis element of
    c o(6)), and the torus weights of the basis vectors."""

    def __init__(self, name, dim, mats, labels=None):
        self.name, self.dim, self.mats = name, dim, mats
        self.labels = labels or [str(i) for i in range(dim)]
        self.wts = self._weights()

    def _weights(self):
        """weight of each basis vector under the Cartan K_{xi_k eta_k},
        normalized so that the function xi_k has weight +1 in slot k."""
        W = []
        for k in range(3):
            f = mono(0, (1 << (2 * k)) | (1 << (2 * k + 1)))     # xi_k eta_k
            R = rho_of(self, transpose(amat(f)))
            col = []
            for b in range(self.dim):
                v = R[b][b]
                assert set(v) <= {(0, 0)}, (self.name, k, b, v)
                col.append(v.get((0, 0), 0))
                for a in range(self.dim):
                    if a != b:
                        assert not R[b][a], (self.name, k, b, a)
            W.append(col)
        return [tuple(W[k][b] for k in range(3)) for b in range(self.dim)]


def rho_of(V, M):
    """rho(M) as a dim x dim matrix of superfunctions."""
    cs = expand(M)
    d = V.dim
    out = [[{} for _ in range(d)] for _ in range(d)]
    for al in range(16):
        if not cs[al]:
            continue
        A = V.mats[al]
        for b in range(d):
            for a in range(d):
                if A[b][a]:
                    out[b][a] = add(out[b][a], smul(A[b][a], cs[al]))
    return out


# ---- Lambda^k of the vector representation (gl(6) acts by derivations)

def _lam_mats(k):
    bas = [tuple(c) for c in combinations(range(N6), k)]
    idx = {A: p for p, A in enumerate(bas)}
    d = len(bas)
    mats = []
    for M in CO6:
        R = [[Fr(0)] * d for _ in range(d)]
        for p, A in enumerate(bas):
            for pos, i in enumerate(A):
                for j in range(N6):
                    if not M[j][i] or (j in A and j != i):
                        continue
                    Bn = list(A)
                    Bn[pos] = j
                    sgn = 1
                    perm = list(Bn)
                    for x in range(len(perm)):
                        for y in range(x + 1, len(perm)):
                            if perm[x] > perm[y]:
                                sgn = -sgn
                    R[idx[tuple(sorted(Bn))]][p] += sgn * M[j][i]
        mats.append(R)
    return bas, mats


def lam_type(k):
    bas, mats = _lam_mats(k)
    lab = ["".join(NAMES[i] for i in A) or "1" for A in bas]
    return Type(f"L{k}", len(bas), mats, lab)


def restrict(V, vecs, name):
    """the submodule spanned by the coordinate vectors vecs (rows)."""
    d = len(vecs)
    piv = []
    for v in vecs:
        piv.append(next(i for i, x in enumerate(v) if x))
    mats = []
    for A in V.mats:
        R = [[Fr(0)] * d for _ in range(d)]
        for j in range(d):
            img = [Fr(0)] * V.dim
            for p, c in enumerate(vecs[j]):
                if c:
                    for q in range(V.dim):
                        img[q] += c * A[q][p]
            for i in range(d):
                R[i][j] = img[piv[i]]
            for i in range(d):
                if R[i][j]:
                    for q in range(V.dim):
                        img[q] -= R[i][j] * vecs[i][q]
            assert not any(img), "not a submodule"
        mats.append(R)
    return Type(name, d, mats, [V.labels[p] for p in piv])


def submodule(V, seed):
    """the submodule of V generated by the coordinate vector seed."""
    rows = []

    def red(v):
        v = [Fr(x) for x in v]
        for r, p in rows:
            if v[p]:
                c = v[p] / r[p]
                v = [a - c * b for a, b in zip(v, r)]
        return v

    def push(v):
        w = red(v)
        p = next((i for i, x in enumerate(w) if x), None)
        if p is None:
            return None
        rows.append((w, p))
        return w

    push(seed)
    frontier = [seed]
    while frontier:
        new = []
        for v in frontier:
            for A in V.mats:
                w = [sum(A[q][p] * v[p] for p in range(V.dim) if v[p])
                     for q in range(V.dim)]
                if any(w) and push(w) is not None:
                    new.append(w)
        frontier = new
    # fully reduced echelon basis (so that restrict() can use the pivots)
    rr = sorted(rows, key=lambda rp: rp[1])
    out = []
    for r, p in rr:
        v = [x / r[p] for x in r]
        out.append(v)
    for i in range(len(out)):
        for j in range(len(out)):
            if i != j and out[i][rr[j][1]]:
                c = out[i][rr[j][1]]
                out[i] = [a - c * b for a, b in zip(out[i], out[j])]
    return out


# ---- the half-spinors: solve [rho(X), gamma_p] = (X gamma)_p on S = Lam(th)

def _spinor():
    """S = Lambda(th1,th2,th3), gamma_{2i} = 2*(th_i .), gamma_{2i+1} = d/dth_i
    (so that gamma_p gamma_q + gamma_q gamma_p = 2 B_{pq})."""
    bas = [m for m in range(8)]                       # bitmask over th1,th2,th3

    def mulm(i, m):                                   # th_i . th^m
        if m >> i & 1:
            return None, 0
        s = -1 if bin(m & ((1 << i) - 1)).count('1') % 2 else 1
        return m | (1 << i), s

    def der(i, m):
        if not (m >> i & 1):
            return None, 0
        s = -1 if bin(m & ((1 << i) - 1)).count('1') % 2 else 1
        return m ^ (1 << i), s

    G = []
    for p in range(N6):
        M = [[Fr(0)] * 8 for _ in range(8)]
        i = p // 2
        for m in bas:
            if p % 2 == 0:
                mm, s = mulm(i, m)
                if mm is not None:
                    M[mm][m] += 2 * s
            else:
                mm, s = der(i, m)
                if mm is not None:
                    M[mm][m] += s
        G.append(M)
    # check Clifford
    for p in range(N6):
        for q in range(N6):
            S = [[sum(G[p][a][c] * G[q][c][b] + G[q][a][c] * G[p][c][b]
                      for c in range(8)) for b in range(8)] for a in range(8)]
            for a in range(8):
                for b in range(8):
                    assert S[a][b] == (2 * B6[p][q] if a == b else 0), (p, q)
    return G


GAM = _spinor()


def _spin_mats():
    """for each X in CO6 find the 8x8 matrix R with

          [R, gamma_q] = sum_p X[p][q] gamma_p                          (**)

    -- the equivariance of the Clifford multiplication V (x) S --> S, which
    makes X |-> R a homomorphism c o(6) --> gl(S) restricting to X itself on
    the vector representation.  R is unique up to a scalar; the scalar is
    fixed by tracelessness on each half-spinor.  Id acts by 0 (the det-twist,
    invisible to as, see kasV.py)."""
    from knclass import nullspace
    mats = []
    for X in CO6:
        rows = []
        rhs_rows = []
        # unknown R (64 entries); equation [R, G_p] + sum_q X[q][p] G_q = 0
        eqs = []
        for p in range(N6):
            for a in range(8):
                for b in range(8):
                    r = [Fr(0)] * 65
                    for c in range(8):
                        r[a * 8 + c] += GAM[p][c][b]
                        r[c * 8 + b] -= GAM[p][a][c]
                    const = Fr(0)
                    for q in range(N6):
                        if X[q][p]:
                            const -= X[q][p] * GAM[q][a][b]
                    r[64] = const
                    if any(r):
                        eqs.append(r)
        ns = nullspace(eqs, 65)
        sol = [v for v in ns if v[64] != 0]
        if not sol:
            mats.append([[Fr(0)] * 8 for _ in range(8)])
            continue
        v = sol[0]
        v = [Fr(x) / Fr(v[64]) for x in v]
        R = [[v[a * 8 + b] for b in range(8)] for a in range(8)]
        # kill the scalar freedom: make traceless on each half
        ev = [m for m in range(8) if bin(m).count('1') % 2 == 0]
        od = [m for m in range(8) if bin(m).count('1') % 2 == 1]
        for half in (ev, od):
            tr = sum(R[m][m] for m in half) / 4
            for m in half:
                R[m][m] -= tr
        mats.append(R)
    return mats


SPIN = _spin_mats()


def spin_type(even):
    idx = [m for m in range(8) if bin(m).count('1') % 2 == (0 if even else 1)]
    mats = [[[SPIN[al][a][b] for b in idx] for a in idx] for al in range(16)]
    lab = ["".join("th%d" % (i + 1) for i in range(3) if m >> i & 1) or "1"
           for m in idx]
    return Type("Sp" if even else "Sm", 4, mats, lab)


def check_hom(V):
    """[rho(X), rho(Y)] = rho([X,Y]) for X, Y in the o(6) part of CO6?"""
    bad = []
    for i in range(15):
        for j in range(15):
            X, Y = CO6[i], CO6[j]
            C = [[sum(X[a][c] * Y[c][b] - Y[a][c] * X[c][b] for c in range(N6))
                  for b in range(N6)] for a in range(N6)]
            # expand C in CO6
            cs = expand([[{(0, 0): C[a][b]} if C[a][b] else {}
                          for b in range(N6)] for a in range(N6)])
            d = V.dim
            R = [[Fr(0)] * d for _ in range(d)]
            for al in range(16):
                c = cs[al].get((0, 0), 0)
                if c:
                    for p in range(d):
                        for q in range(d):
                            R[p][q] += c * V.mats[al][p][q]
            A, B = V.mats[i], V.mats[j]
            for p in range(d):
                for q in range(d):
                    v = sum(A[p][c] * B[c][q] - B[p][c] * A[c][q]
                            for c in range(d)) - R[p][q]
                    if v:
                        bad.append((i, j, p, q))
    return bad
