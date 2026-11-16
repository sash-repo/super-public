"""The canonical Sp(2m)-equivariant projector pi: W^{ox 3} --> V_{(2,1)}.

The multiplicity of V_{(2,1)} in W^{ox 3} is 2, so a projector onto "the" copy
is not canonical.  What IS canonical is the composite

    pi := (traceless part)  o  (Young projector of the tableau  1 2 / 3),

because the Young projector p_lambda is canonical once the tableau is fixed
(it is the GL(W)-equivariant projector onto im c_lambda), and inside
im c_lambda = S^{(2,1)}(W) the sp-module V_{(2,1)} occurs ONCE, the complement
being the trace part, isomorphic to W.  Both maps commute with sp(2m), hence so
does pi.

This module builds pi as an explicit 64 x 64 (in general n^3 x n^3) rational
matrix and checks: pi^2 = pi, pi|_V = id, im pi = V, and equivariance under the
fiberwise sp(2m)-action.
"""

import os
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank, rref  # noqa: E402

from fiber import _tableau, _group, flat, unflat  # noqa: E402


def young_matrix(lam, n):
    """The matrix of c_lambda on W^{ox k}: out[J][I] with c(e_I) = sum_J out[J][I] e_J."""
    k = sum(lam)
    rows, cols = _tableau(lam)
    R = [(p, 1) for p, _ in _group(rows, k)]
    C = _group(cols, k)
    dim = n ** k
    M = [[Fraction(0)] * dim for _ in range(dim)]
    for fi in range(dim):
        I = unflat(fi, n, k)
        for p, _ in R:
            J = [0] * k
            for a in range(k):
                J[p[a]] = I[a]
            for q, sq in C:
                K = [0] * k
                for a in range(k):
                    K[q[a]] = J[a]
                M[flat(tuple(K), n)][fi] += sq
    return M


def _matmul(A, B):
    n, m2 = len(A), len(B[0])
    out = [[Fraction(0)] * m2 for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for k, a in enumerate(Ai):
            if a:
                Bk = B[k]
                oi = out[i]
                for j, b in enumerate(Bk):
                    if b:
                        oi[j] += a * b
    return out


def _apply(M, v):
    return [sum((M[i][j] * v[j] for j in range(len(v)) if v[j] and M[i][j]),
                Fraction(0)) for i in range(len(M))]


def _inverse(M):
    n = len(M)
    aug = [list(M[i]) + [Fraction(1) if i == j else Fraction(0) for j in range(n)]
           for i in range(n)]
    R, piv = rref(aug)
    assert list(piv[:n]) == list(range(n)), "matrix is singular"
    return [R[i][n:] for i in range(n)]


class Projector:
    """pi onto V_lambda inside W^{ox k}, for lambda with at most 2 rows."""

    def __init__(self, S, F):
        self.S, self.F = S, F
        n, k = S.n, F.k
        self.n, self.k = n, k
        dim = n ** k
        self.dim = dim
        Y = young_matrix(F.lam, n)
        # normalize: c^2 = (k!/f^lambda) c
        col = None
        for j in range(dim):
            if any(Y[i][j] for i in range(dim)):
                col = j
                break
        e = [Fraction(1) if i == col else Fraction(0) for i in range(dim)]
        y1 = _apply(Y, e)
        y2 = _apply(Y, y1)
        s = None
        for i in range(dim):
            if y1[i]:
                s = y2[i] / y1[i]
                break
        assert s, "degenerate Young symmetrizer"
        self.scale = s
        P = [[Y[i][j] / s for j in range(dim)] for i in range(dim)]
        self.P = P                                   # idempotent Young projector
        # the trace part inside im P: p(omega_{ab} (x) u), all placements
        Omd = [[-S.Om[i][j] for j in range(n)] for i in range(n)]
        tr = []
        for pos in range(k):                         # slot carrying u
            for u in range(n):
                v = [Fraction(0)] * dim
                slots = [s2 for s2 in range(k) if s2 != pos]
                for a in range(n):
                    for b in range(n):
                        if not Omd[a][b]:
                            continue
                        I = [0] * k
                        I[pos] = u
                        I[slots[0]], I[slots[1]] = a, b
                        v[flat(tuple(I), n)] += Omd[a][b]
                tr.append(_apply(P, v))
        self.trace_part = [r for r in rref(tr)[0] if any(c != 0 for c in r)]
        Vb = F.basis
        assert rank(Vb + self.trace_part) == len(Vb) + len(self.trace_part), \
            "V and the trace part are not independent"
        # basis of the whole space: V, trace part, ker P
        kerP = [r for r in rref(_kernel_rows(P))[0] if any(c != 0 for c in r)]
        B = list(Vb) + list(self.trace_part) + list(kerP)
        assert len(B) == dim and rank(B) == dim, \
            f"basis has {len(B)} vectors of rank {rank(B)}, want {dim}"
        # columns = basis vectors
        Bm = [[B[j][i] for j in range(dim)] for i in range(dim)]
        Binv = _inverse(Bm)
        coord = Binv[:len(Vb)]                       # the V-coordinates of a vector
        Vm = [[Vb[j][i] for j in range(len(Vb))] for i in range(dim)]
        self.M = _matmul(Vm, coord)

    def apply_vec(self, v):
        return _apply(self.M, v)

    def apply_field(self, T):
        """pi applied to an array of polynomials over the ambient indices."""
        from symp import padd, pscale
        out = [dict() for _ in range(self.dim)]
        for j in range(self.dim):
            if not T[j]:
                continue
            for i in range(self.dim):
                c = self.M[i][j]
                if c:
                    out[i] = padd(out[i], pscale(T[j], c))
        return out


def _kernel_rows(P):
    from exactlin import nullspace
    return nullspace(P)


# ------------------------------------------------------------------- self-test


def selftest(m=2, lam=(2, 1)):
    import random
    from symp import Symp
    from fiber import Fiber, lie_tensor, _quadratics
    S = Symp(m)
    F = Fiber(S, lam)
    pi = Projector(S, F)
    rng = random.Random(11)
    ok = True
    # idempotent, identity on V, image inside V
    for _ in range(4):
        v = [Fraction(rng.randint(-5, 5)) for _ in range(pi.dim)]
        w = pi.apply_vec(v)
        if pi.apply_vec(w) != w:
            ok = False
            print("  FAIL: pi^2 != pi")
        if not F.contains(w):
            ok = False
            print("  FAIL: im pi is not inside V")
        u = F.random_vector(rng)
        if pi.apply_vec(u) != u:
            ok = False
            print("  FAIL: pi is not the identity on V")
    # equivariance: pi commutes with the fiberwise sp-action
    for _ in range(3):
        v = [Fraction(rng.randint(-5, 5)) for _ in range(pi.dim)]
        p1 = {(0,) * S.n: Fraction(1)}
        T = [dict() if not v[i] else {(0,) * S.n: v[i]} for i in range(pi.dim)]
        for q in _quadratics(S):
            L1 = lie_tensor(S, F.k, q, pi.apply_field(T))
            L2 = pi.apply_field(lie_tensor(S, F.k, q, T))
            for i in range(pi.dim):
                d = dict(L1[i])
                for e, c in L2[i].items():
                    d[e] = d.get(e, 0) - c
                if any(c for c in d.values()):
                    ok = False
                    print("  FAIL: pi is not sp-equivariant")
                    break
    print(f"projector self-test (m={m}, lambda={lam}): {'PASS' if ok else 'FAIL'}")
    print(f"  dim V = {F.dim}, dim im(Young) = {F.dim + len(pi.trace_part)}, "
          f"trace part {len(pi.trace_part)}")
    return ok


if __name__ == "__main__":
    selftest()
