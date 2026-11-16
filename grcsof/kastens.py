#!/usr/bin/env python3
"""n = 6: tensor fields over kas of type V, for V a kas_0-module of dim > 1.

kas_0 = k(1|6)_0 = c o(6) = gl(4).  Tensor fields of type V = the module
coinduced from an irreducible kas_0-module V (kas_{>0} acting trivially),
realized as V-valued superfunctions.  Weighted densities are the 1-dimensional
V's.  This module builds the action for the o(6)-modules

      Lambda^k(W),  W = the 6-dimensional vector representation  (= the
      contact distribution <D_1,...,D_6>),

and for the two half-spinors (the 4-dimensional gl(4)-modules id, id^*).

The action is derived, not guessed: for a superfunction f one has

      [K_f, D_i] = sum_j a_{ij}(f) D_j                                   (*)

(no d_t-part: contact fields preserve the distribution), and a(f) is a
6x6 matrix of superfunctions of parity p(f), lying in c o(6) pointwise.
The module of "vector fields along the distribution" X = sum phi_i D_i with
the action ad K_f is then the tensor fields of type W (up to a shift of the
weight, which for as is invisible anyway); every other type is obtained by
letting a(f) act through the corresponding representation.
"""
from fractions import Fraction as Fr
from itertools import combinations

from superN import add, smul, mul, mono, parity, popcount, d_t, clean
from kas6 import ALG, NAMES, B6, XI, ETA, fmono, msk

N6 = 6
D = [ALG.D(i) for i in range(N6)]


def acomm(f, i):
    """[K_f, D_i] = K_f D_i - (-1)^{p(f)} D_i K_f, as an operator."""
    Kf = ALG.K(f)
    p = parity(f)
    s = -1 if p else 1

    def r(u):
        return add(Kf(D[i](u)), smul(-s, D[i](Kf(u))))
    return r


_AMAT = {}


def amat(f):
    """the matrix a(f)_{ik} of (*)."""
    key = tuple(sorted(f.items(), key=str))
    r = _AMAT.get(key)
    if r is None:
        r = _AMAT[key] = [[acomm(f, i)(mono(0, 1 << k)) for k in range(N6)]
                          for i in range(N6)]
    return r


def check_amat(f, tmax=3):
    """verify [K_f,D_i] = sum_j a_{ij}. D_j on all monomials t^a y^m, a<=tmax."""
    a = amat(f)
    bad = []
    for i in range(N6):
        C = acomm(f, i)
        for m in ALG.monos(tmax):
            u = mono(*m)
            lhs = C(u)
            rhs = {}
            for j in range(N6):
                if a[i][j]:
                    rhs = add(rhs, mul(a[i][j], D[j](u)))
            if add(lhs, smul(-1, rhs)):
                bad.append((i, m))
    return bad


if __name__ == '__main__':
    from kas6 import gens_kas, gens_as
    print("=== a(f) well defined? (contact fields preserve <D_i>) ===")
    F = gens_kas()
    F['t2'] = mono(2, 0)
    F['txi1xi2'] = mul(mono(1, 0), fmono(0, 2))
    worst = 0
    for nm, f in sorted(F.items()):
        bad = check_amat(f)
        if bad:
            print("  FAIL", nm, bad[:3])
            worst += 1
    print("  checked", len(F), "generators;", worst, "failures")

    print("\n=== the shape of a(f) ===")
    for nm in ['1', 't', 'xi1', 'xi1eta1', 'xi1xi2', 'txi1', 't2', 'c0']:
        f = F[nm]
        a = amat(f)
        print(f"  f = {nm:8s}  a(f) =")
        for i in range(N6):
            print("       ", [ALG.show(a[i][j]) for j in range(N6)])
        print()


# ------------------------------------------------------------ the types V

def transpose(a):
    return [[a[i][j] for i in range(N6)] for j in range(N6)]


def in_co6(a):
    """is a in c o(6) = o(6) + C.Id pointwise?  (o(6): X^T B + B X = 0)"""
    # scalar part: (1/6) tr
    tr = {}
    for i in range(N6):
        tr = add(tr, a[i][i])
    sc = smul(Fr(1, 6), tr)
    X = [[add(a[i][j], smul(-1, sc) if i == j else {}) for j in range(N6)]
         for i in range(N6)]
    bad = {}
    for i in range(N6):
        for j in range(N6):
            s = {}
            for k in range(N6):
                if B6[k][j]:
                    s = add(s, smul(B6[k][j], X[k][i]))     # (X^T B)_{ij}
                if B6[i][k]:
                    s = add(s, smul(B6[i][k], X[k][j]))     # (B X)_{ij}
            if s:
                bad[(i, j)] = ALG.show(s)
    return bad


class Lam:
    """V = Lambda^k(C^6), the ordinary k-th exterior power of the vector
    representation of o(6); gl(6) acts by derivations.  For k=1 this is the
    contact distribution itself, for k=2 it is sl(4) = the adjoint of o(6),
    for k=3 it is Lambda^3_xi + Lambda^3_eta."""

    def __init__(self, k, sub=None, name=None):
        self.k = k
        self.bas = [tuple(c) for c in combinations(range(N6), k)]
        self.name = name or f"Lam{k}"
        self.sub = sub          # optional list of coordinate vectors spanning
        self.dim = len(self.bas) if sub is None else len(sub)

    # rho(M) for M a 6x6 matrix of superfunctions, as a derivation of Lambda^k
    def rho(self, M):
        d = len(self.bas)
        R = [[{} for _ in range(d)] for _ in range(d)]
        idx = {A: p for p, A in enumerate(self.bas)}
        for p, A in enumerate(self.bas):
            for pos, i in enumerate(A):
                for j in range(N6):
                    if not M[j][i]:
                        continue
                    if j in A and j != i:
                        continue
                    Bn = list(A)
                    Bn[pos] = j
                    sgn = 1
                    srt = sorted(Bn)
                    # sign of the permutation sorting Bn
                    perm = list(Bn)
                    for x in range(len(perm)):
                        for y in range(x + 1, len(perm)):
                            if perm[x] > perm[y]:
                                sgn = -sgn
                    q = idx[tuple(srt)]
                    R[q][p] = add(R[q][p], smul(sgn, M[j][i]))
        return R

    def wts(self):
        """torus weight (3-vector) of each basis element, in the same
        normalization as the weights of the y's."""
        from kas6 import wt_of_subset
        return [wt_of_subset(A) for A in self.bas]


class Sub:
    """an o(6)-submodule of Lambda^k, given by a basis of coordinate vectors."""

    def __init__(self, amb, vecs, name):
        self.amb = amb
        self.vecs = [[Fr(x) for x in v] for v in vecs]
        self.dim = len(vecs)
        self.name = name
        # left inverse: express an ambient vector in the basis
        self.piv = []
        rows = [v[:] for v in self.vecs]
        for r in range(len(rows)):
            p = next(i for i, x in enumerate(rows[r]) if x)
            self.piv.append(p)
        self.bas = [amb.bas[p] for p in self.piv]

    def rho(self, M):
        R0 = self.amb.rho(M)
        d = self.dim
        # rho on the submodule: image of the j-th basis vector, re-expanded
        out = [[{} for _ in range(d)] for _ in range(d)]
        for j in range(d):
            img = [{} for _ in range(len(self.amb.bas))]
            for p, c in enumerate(self.vecs[j]):
                if not c:
                    continue
                for q in range(len(self.amb.bas)):
                    if R0[q][p]:
                        img[q] = add(img[q], smul(c, R0[q][p]))
            # solve img = sum_i coef_i * vecs[i]  (using the pivots)
            for i in range(d):
                out[i][j] = img[self.piv[i]]
            # consistency: subtract and require 0
            for i in range(d):
                c = out[i][j]
                if c:
                    for q in range(len(self.amb.bas)):
                        if self.vecs[i][q]:
                            img[q] = add(img[q], smul(-self.vecs[i][q], c))
            if any(img):
                raise ValueError("not a submodule")
        return out

    def wts(self):
        return [self.amb.wts()[p] for p in self.piv]


def cub_vectors(basis_fs):
    """turn the Lambda^3_xi basis of kas6 (superfunctions) into coordinate
    vectors in the Lambda^3 basis of Lam(3)."""
    L3 = Lam(3)
    out = []
    for f in basis_fs:
        v = [f.get((0, msk(*A)), 0) for A in L3.bas]
        out.append(v)
    return L3, out


# ---------------------------------------------------------- the L-action

_LACT = {}


def Lact(V, f, lam):
    """the matrix R(f) with  (L_f phi)_b = K_f(phi_b) + lam d_t(f) phi_b
    + sum_a R[b][a] . phi_a ."""
    key = (id(V), tuple(sorted(f.items(), key=str)), lam)
    if key in _LACT:
        return _LACT[key]
    R = V.rho(transpose(amat(f)))
    d = V.dim
    dtf = smul(lam, d_t(f)) if lam else {}
    out = [[dict(R[b][a]) for a in range(d)] for b in range(d)]
    if dtf:
        for b in range(d):
            out[b][b] = add(out[b][b], dtf)
    _LACT[key] = out
    return out


class Triv:
    dim = 1
    name = 'F'
    bas = [()]

    def rho(self, M):
        return [[{}]]

    def wts(self):
        return [(0, 0, 0)]


_KOP = {}


def Kop(f):
    key = tuple(sorted(f.items(), key=str))
    r = _KOP.get(key)
    if r is None:
        r = _KOP[key] = ALG.K(f)
    return r


def apply_L(V, f, lam, phi):
    """phi = list of superfunctions of length V.dim."""
    Kf = Kop(f)
    R = Lact(V, f, lam)
    out = []
    for b in range(V.dim):
        s = Kf(phi[b])
        for a in range(V.dim):
            if R[b][a] and phi[a]:
                s = add(s, mul(R[b][a], phi[a]))
        out.append(s)
    return out


def check_module(V, lam, gens, tmax=2, ntest=None):
    """L_{{f,g}} = L_f L_g - (-1)^{p(f)p(g)} L_g L_f ?"""
    bad = []
    mons = ALG.monos(tmax)
    for nf, f in gens.items():
        for ng, g in gens.items():
            fg = ALG.kb(f, g)
            s = -1 if (parity(f) and parity(g)) else 1
            for b0 in range(V.dim):
                for m in mons:
                    phi = [{} for _ in range(V.dim)]
                    phi[b0] = mono(*m)
                    lhs = apply_L(V, fg, lam, phi) if fg else \
                        [{} for _ in range(V.dim)]
                    r1 = apply_L(V, f, lam, apply_L(V, g, lam, phi))
                    r2 = apply_L(V, g, lam, apply_L(V, f, lam, phi))
                    rhs = [add(x, smul(-s, y)) for x, y in zip(r1, r2)]
                    if any(add(x, smul(-1, y)) for x, y in zip(lhs, rhs)):
                        bad.append((nf, ng, b0, m))
                        break
                if bad and bad[-1][0] == nf and bad[-1][1] == ng:
                    break
    return bad
