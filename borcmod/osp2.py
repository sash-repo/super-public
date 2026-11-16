"""osp(2|2) in k(1;N|2) over F_p acting on weighted densities F_{lam,mu} = O(1;N|2) alpha^{lam/2} (x) mu.

Coordinates t (divided powers, t^(r), r < P = p^N), xi, eta odd; contact form alpha_1 = dt + xi d eta + eta d xi.
Basis of O(1;N|2): t^(r) xi^e1 eta^e2, index 4r + 2e1 + e2, parity e1 + e2.
K_f = (2-E)(f) d_t + d_t(f) E + (-1)^{p(f)} (d_xi(f) d_eta + d_eta(f) d_xi),  E = xi d_xi + eta d_eta,
the sign of the last term chosen so that K_xi = xi d_t - d_eta, K_eta = eta d_t - d_xi (formulas (KD)).
Density action:  L_f = K_f + lam * d_t(f) + mu * c(f),  c(f) = a cocycle found by `cocycles`.

usage:  python3 osp2.py p N     -- closure of osp(2|2), and the multipliers c(f) among three candidates making
                                   K_f + lam d_t f + mu c(f) a representation
"""
import sys, itertools
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank
from math import comb


class Alg:
    def __init__(self, p, N):
        self.p, self.N = p, N
        self.P = P = p ** N
        self.n = n = 4 * P
        self.par = [((i % 4) // 2 + i % 2) % 2 for i in range(n)]
        Z = self.zero
        Dt, Dxi, Deta, Mxi, Meta, I = Z(), Z(), Z(), Z(), Z(), Z()
        self.Mt = {}
        for r in range(P):
            for e1 in (0, 1):
                for e2 in (0, 1):
                    j = self.ix(r, e1, e2)
                    I[j][j] = 1
                    if r > 0:
                        Dt[self.ix(r - 1, e1, e2)][j] = 1
                    if e1:
                        Dxi[self.ix(r, 0, e2)][j] = 1
                    if e2:
                        Deta[self.ix(r, e1, 0)][j] = (-1) ** e1 % p
                    if not e1:
                        Mxi[self.ix(r, 1, e2)][j] = 1
                    if not e2:
                        Meta[self.ix(r, e1, 1)][j] = (-1) ** e1 % p
        for k in range(P):
            M = Z()
            for r in range(P - k):
                for e in range(4):
                    M[4 * (r + k) + e][4 * r + e] = comb(r + k, k) % p
            self.Mt[k] = M
        self.I, self.Dt, self.Dxi, self.Deta, self.Mxi, self.Meta = I, Dt, Dxi, Deta, Mxi, Meta
        self.E = self.add(self.mul(Mxi, Dxi), self.mul(Meta, Deta))
        # D's of (KD): D_xi = eta d_t + d_xi, D_eta = xi d_t + d_eta
        self.DXI = self.add(self.mul(Meta, Dt), Dxi)
        self.DETA = self.add(self.mul(Mxi, Dt), Deta)

    def ix(self, r, e1, e2):
        return 4 * r + 2 * e1 + e2

    def zero(self):
        return [[0] * self.n for _ in range(self.n)]

    def mul(self, A, B):
        p, n = self.p, self.n
        C = self.zero()
        for i in range(n):
            Ai, Ci = A[i], C[i]
            for k in range(n):
                a = Ai[k]
                if a:
                    Bk = B[k]
                    for j in range(n):
                        if Bk[j]:
                            Ci[j] = (Ci[j] + a * Bk[j]) % p
        return C

    def add(self, *Ms, coef=None):
        p, n = self.p, self.n
        C = self.zero()
        coef = coef or [1] * len(Ms)
        for c, M in zip(coef, Ms):
            for i in range(n):
                Mi, Ci = M[i], C[i]
                for j in range(n):
                    if Mi[j]:
                        Ci[j] = (Ci[j] + c * Mi[j]) % p
        return C

    def apply(self, M, v):
        p, n = self.p, self.n
        return [sum(M[i][j] * v[j] for j in range(n) if v[j]) % p for i in range(n)]

    def fn(self, r=0, e1=0, e2=0):
        v = [0] * self.n
        v[self.ix(r, e1, e2)] = 1
        return v

    def M(self, g):
        """multiplication by the function g"""
        out = self.zero()
        for idx, c in enumerate(g):
            if not c:
                continue
            r, e1, e2 = idx // 4, (idx % 4) // 2, idx % 2
            A = self.Mt[r]
            if e1:
                A = self.mul(A, self.Mxi)
            if e2:
                A = self.mul(A, self.Meta)
            out = self.add(out, A, coef=[1, c])
        return out

    def K(self, f, pf):
        Ef = self.apply(self.E, f)
        A = [(2 * a - b) % self.p for a, b in zip(f, Ef)]
        s = -1 if pf else 1
        X = self.add(self.mul(self.M(A), self.Dt), self.mul(self.M(self.apply(self.Dt, f)), self.E))
        X = self.add(X, self.mul(self.M(self.apply(self.Dxi, f)), self.Deta),
                     self.mul(self.M(self.apply(self.Deta, f)), self.Dxi), coef=[1, s, s])
        return X


def osp22(A):
    """generating functions of osp(2|2): 1, t, t^(2) [= t^2/2], xi eta (even); xi, eta, t xi, t eta (odd)"""
    F = A.fn
    return [("1", F(0), 0), ("t", F(1), 0), ("t2", F(2), 0), ("xieta", F(0, 1, 1), 0),
            ("xi", F(0, 1, 0), 1), ("eta", F(0, 0, 1), 1), ("txi", F(1, 1, 0), 1), ("teta", F(1, 0, 1), 1)]


def br(A, X, px, Y, py):
    sg = -1 if (px and py) else 1
    return A.add(A.mul(X, Y), A.mul(Y, X), coef=[1, -sg])


flat = lambda M: [x for r in M for x in r]


def structure(A, gens):
    """structure constants of the K's: [K_f,K_g] = sum a_h K_h (solved mod p)"""
    Ks = [A.K(f, pf) for _, f, pf in gens]
    basis = [flat(K) for K in Ks]
    sc = {}
    for i, j in itertools.product(range(len(gens)), repeat=2):
        Bv = flat(br(A, Ks[i], gens[i][2], Ks[j], gens[j][2]))
        ns = nullspace([list(col) for col in zip(*(basis + [Bv]))], A.p)
        assert len(ns) == 1 and ns[0][-1] % A.p, "not closed"
        v = ns[0]
        inv = pow(v[-1], A.p - 2, A.p)
        sc[(i, j)] = [(-x * inv) % A.p for x in v[:-1]]
    return Ks, sc


def cocycles(A, gens, Ks, sc, cands):
    """mu-coefficients x of c = sum x_k cand_k making L = K + lam d_t f + mu c a representation (all lam)"""
    p = A.p
    good = []
    for coeffs in itertools.product(range(p), repeat=len(cands)):
        if not any(coeffs):
            continue
        ok = True
        for lam in (0, 1):
            L = []
            for (name, f, pf), K in zip(gens, Ks):
                c = [0] * A.n
                for x, cand in zip(coeffs, cands):
                    c = [(a + x * b) % p for a, b in zip(c, cand(f))]
                L.append(A.add(K, A.M(A.apply(A.Dt, f)), A.M(c), coef=[1, lam, 1]))
            for (i, j), a in sc.items():
                lhs = br(A, L[i], gens[i][2], L[j], gens[j][2])
                rhs = A.zero()
                for h, ah in enumerate(a):
                    if ah:
                        rhs = A.add(rhs, L[h], coef=[1, ah])
                if lhs != rhs:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(coeffs)
    return good


def reps(A, gens, Ks, c, lam, mu):
    p = A.p
    return [(A.add(K, A.M(A.apply(A.Dt, f)), A.M(c(f)), coef=[1, lam % p, mu % p]), pf)
            for (_, f, pf), K in zip(gens, Ks)]


def equivariant(A, La, Lb, par):
    p, n = A.p, A.n
    pa = A.par
    unk = [(i, j) for i in range(n) for j in range(n) if (pa[i] + pa[j]) % 2 == par]
    pos = {u: k for k, u in enumerate(unk)}
    rows = []
    for (X, pX), (Y, _) in zip(La, Lb):
        sg = -1 if (par and pX) else 1
        for i in range(n):
            for j in range(n):
                row = {}
                for k in range(n):
                    if Y[i][k] and (k, j) in pos:
                        row[pos[(k, j)]] = (row.get(pos[(k, j)], 0) + Y[i][k]) % p
                    if X[k][j] and (i, k) in pos:
                        row[pos[(i, k)]] = (row.get(pos[(i, k)], 0) - sg * X[k][j]) % p
                if any(row.values()):
                    rr = [0] * len(unk)
                    for cc, v in row.items():
                        rr[cc] = v
                    rows.append(rr)
    return nullspace(rows, p), unk


if __name__ == "__main__":
    p, N = int(sys.argv[1]), int(sys.argv[2])
    A = Alg(p, N)
    gens = osp22(A)
    Ks, sc = structure(A, gens)
    print("closure OK")
    cands = [lambda f: A.apply(A.Dt, f),
             lambda f: A.apply(A.mul(A.DXI, A.DETA), f),
             lambda f: A.apply(A.mul(A.Dxi, A.Deta), f)]
    good = cocycles(A, gens, Ks, sc, cands)
    print("cocycle coefficient vectors (d_t f, D_xi D_eta f, d_xi d_eta f):", good)
