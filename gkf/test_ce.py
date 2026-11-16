"""Validation of the Chevalley--Eilenberg engine against known answers.

1) sl(2) and sl(3) over Q: H^* = Lambda(x_3) and Lambda(x_3, x_5).
2) a purely odd abelian superalgebra: H^q = K for every q.
3) d^2 = 0 for h^{(1)}(0|n), n = 3..6.
4) H^0 = K and H^1 = 0 for a simple algebra (the derived algebra is everything).
"""

import sys
from po import Po
from ce import CE, cohomology


class MatrixAlg:
    """A Lie algebra given by structure constants on a basis of matrices."""

    def __init__(self, mats, name):
        self.name = name
        self.mats = mats
        self.dim = len(mats)
        self.N = len(mats[0])
        self._tab = None

    def par(self, k):
        return 0

    def deg(self, k):
        return 0

    def weight(self, k):
        return ()

    def _mul(self, A, B):
        n = self.N
        return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
                for i in range(n)]

    def _decompose(self, C):
        # solve C = sum lambda_a mats[a] by matching entries (basis is a set of
        # matrix units / diagonal elements, so this is triangular in practice)
        rem = [row[:] for row in C]
        out = {}
        for a, M in enumerate(self.mats):
            # find a position where M is nonzero and no earlier basis element is
            pos = None
            for i in range(self.N):
                for j in range(self.N):
                    if M[i][j]:
                        pos = (i, j)
                        break
                if pos:
                    break
            i, j = pos
            if rem[i][j]:
                lam = rem[i][j] / M[i][j]
                if lam == int(lam):
                    lam = int(lam)
                out[a] = out.get(a, 0) + lam
                for x in range(self.N):
                    for y in range(self.N):
                        rem[x][y] -= lam * M[x][y]
        assert all(all(v == 0 for v in r) for r in rem), (C, out)
        return {k: v for k, v in out.items() if v}

    def table(self):
        if self._tab is None:
            self._tab = []
            for a in range(self.dim):
                row = []
                for b in range(self.dim):
                    AB = self._mul(self.mats[a], self.mats[b])
                    BA = self._mul(self.mats[b], self.mats[a])
                    C = [[AB[i][j] - BA[i][j] for j in range(self.N)]
                         for i in range(self.N)]
                    row.append(self._decompose(C))
                self._tab.append(row)
        return self._tab


def sl(n):
    mats, names = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                M = [[0] * n for _ in range(n)]
                M[i][j] = 1
                mats.append(M)
    for i in range(n - 1):
        M = [[0] * n for _ in range(n)]
        M[i][i] = 1
        M[i + 1][i + 1] = -1
        mats.append(M)
    return MatrixAlg(mats, f"sl({n})")


class OddAbelian:
    def __init__(self, m):
        self.dim = m

    def par(self, k):
        return 1

    def deg(self, k):
        return 0

    def weight(self, k):
        return ()

    def table(self):
        return [[dict() for _ in range(self.dim)] for _ in range(self.dim)]


def betti(h, qmax):
    out = [0] * (qmax + 1)
    for key, v in h.items():
        if key[0] <= qmax:
            out[key[0]] += v
    return out


if __name__ == "__main__":
    print("== 1) d^2 = 0 and the sign convention ==")
    for g, name in [(sl(2), "sl(2)"), (sl(3), "sl(3)"), (OddAbelian(2), "odd abelian(2)")] + \
                   [(Po(n, "h1"), f"h1(0|{n})") for n in range(3, 7)]:
        ce = CE(g)
        bad = ce.check_d2()
        print(f"   {name:16s} dim {g.dim:3d}   d^2 = 0 on generators: "
              f"{'YES' if not bad else 'NO ' + str(bad[:4])}")

    print("== 2) known cohomology ==")
    ce = CE(sl(2))
    b = betti(cohomology(ce, 3), 3)
    print(f"   sl(2): H^0..H^3 = {b}   (expected [1, 0, 0, 1])")
    assert b == [1, 0, 0, 1], b

    ce = CE(sl(3))
    b = betti(cohomology(ce, 5), 5)
    print(f"   sl(3): H^0..H^5 = {b}   (expected [1, 0, 0, 1, 0, 1])")
    assert b == [1, 0, 0, 1, 0, 1], b

    ce = CE(OddAbelian(2))
    b = betti(cohomology(ce, 4), 4)
    print(f"   odd abelian(2): H^0..H^4 = {b}   (expected [1, 2, 3, 4, 5])")
    assert b == [1, 2, 3, 4, 5], b

    print("== 3) h^{(1)}(0|4): H^0 = K, H^1 = 0 (it is simple) ==")
    ce = CE(Po(4, "h1"))
    h = cohomology(ce, 2)
    b = betti(h, 2)
    print(f"   H^0, H^1, H^2 = {b}")
    assert b[0] == 1 and b[1] == 0, b
    print("   all validations passed")
