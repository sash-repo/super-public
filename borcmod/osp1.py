"""osp(1|2) in k(1;N|1) over F_p acting on weighted densities F_lam = O(1;N|1) alpha^{lam/2};
all osp(1|2)-equivariant linear maps F_a -> F_b (even and odd), a,b in F_p.
Action: L_{K_f}(phi) = K_f(phi) + lam * d_t(f) * phi  (weighted densities, alpha^{lam/2}).
Usage: python3 osp1.py p N"""
import sys, itertools
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank
from math import comb

def setup(p, N):
    P = p**N; n = 2*P
    idx = lambda r, e: 2*r + e
    def zero(): return [[0]*n for _ in range(n)]
    def mul(A, B):
        C = zero()
        for i in range(n):
            Ai = A[i]
            for k in range(n):
                if Ai[k]:
                    a = Ai[k]; Bk = B[k]
                    for j in range(n):
                        if Bk[j]: C[i][j] = (C[i][j] + a*Bk[j]) % p
        return C
    def add(*Ms, coef=None):
        C = zero(); coef = coef or [1]*len(Ms)
        for c, M in zip(coef, Ms):
            for i in range(n):
                for j in range(n):
                    C[i][j] = (C[i][j] + c*M[i][j]) % p
        return C
    # matrices act on column vectors: M[row=out][col=in]
    Dt = zero(); Dth = zero(); T = zero(); T2 = zero(); TH = zero(); I = zero()
    for r in range(P):
        for e in range(2):
            j = idx(r, e); I[j][j] = 1
            if r > 0: Dt[idx(r-1, e)][j] = 1
            if e == 1: Dth[idx(r, 0)][j] = 1          # d_theta(u theta) = u (theta last; u even)
            if r+1 < P: T[idx(r+1, e)][j] = (r+1) % p
            if r+2 < P: T2[idx(r+2, e)][j] = comb(r+2, 2) % p
            if e == 0: TH[idx(r, 1)][j] = 1          # multiplication by theta (theta*u = u*theta)
    return dict(p=p, P=P, n=n, I=I, Dt=Dt, Dth=Dth, T=T, T2=T2, TH=TH, mul=mul, add=add, zero=zero)

def generators(S, s):
    """K_f = A d_t + (theta d_t f + s d_theta f) d_theta for f = 1, t, t^(2), theta, t*theta.
    Returns list of (vector-field matrix, multiplier-matrix d_t f, parity)."""
    mul, add = S['mul'], S['add']
    I, Dt, Dth, T, T2, TH = (S[k] for k in ('I', 'Dt', 'Dth', 'T', 'T2', 'TH'))
    E = mul(TH, Dth)
    Z = S['zero']()
    K1 = add(Dt, coef=[2]);                       m1 = Z
    Kt = add(mul(T, Dt), E, coef=[2, 1]);         mt = I
    Kt2 = add(mul(T2, Dt), mul(T, E), coef=[2, 1]); mt2 = T     # A=2t^(2), B = theta*t
    Kth = add(mul(TH, Dt), Dth, coef=[1, s]);     mth = Z
    Ktth = mul(T, Kth);                           mtth = TH     # K_{t theta} = t K_theta, d_t f = theta
    return [(K1, m1, 0), (Kt, mt, 0), (Kt2, mt2, 0), (Kth, mth, 1), (Ktth, mtth, 1)]

def rep(S, gens, lam):
    p = S['p']; add = S['add']
    return [(add(K, m, coef=[1, lam % p]), par) for K, m, par in gens]

def parity_of(n):
    return [i % 2 for i in range(n)]

def equivariant(S, La, Lb, par):
    """solve Lb(X) T = (-1)^{par*p(X)} T La(X) for T of parity par; unknowns T[i][j] with par(i)+par(j)=par."""
    p, n = S['p'], S['n']; pa = parity_of(n)
    unk = [(i, j) for i in range(n) for j in range(n) if (pa[i] + pa[j]) % 2 == par]
    pos = {u: k for k, u in enumerate(unk)}
    rows = []
    for (A, pX), (B, _) in zip(La, Lb):
        sg = -1 if (par and pX) else 1
        for i in range(n):
            for j in range(n):
                row = {}
                for k in range(n):     # (B T)_{ij} = sum_k B[i][k] T[k][j]
                    if B[i][k] and (k, j) in pos: row[pos[(k, j)]] = (row.get(pos[(k, j)], 0) + B[i][k]) % p
                    if A[k][j] and (i, k) in pos: row[pos[(i, k)]] = (row.get(pos[(i, k)], 0) - sg*A[k][j]) % p
                if any(row.values()):
                    r = [0]*len(unk)
                    for c, v in row.items(): r[c] = v
                    rows.append(r)
    if not rows: return [[1 if k == m else 0 for k in range(len(unk))] for m in range(len(unk))], unk
    return nullspace(rows, p), unk

def check_closure(S, gens):
    """[K_f,K_g] must lie in span; and L is a representation for every lam (checked lam = 1, 2)."""
    p, mul, add = S['p'], S['mul'], S['add']
    def br(X, px, Y, py):
        sg = -1 if (px and py) else 1
        return add(mul(X, Y), mul(Y, X), coef=[1, (-sg) % p])
    flat = lambda M: [x for r in M for x in r]
    basis = [flat(K) for K, _, _ in gens]
    r0 = rank(basis, p)
    ok = True
    for (X, _, px), (Y, _, py) in itertools.product(gens, gens):
        if rank(basis + [flat(br(X, px, Y, py))], p) != r0: ok = False
    for lam in (1, 2):
        L = rep(S, gens, lam); Lflat = [flat(M) for M, _ in L]
        for (X, px), (Y, py) in itertools.product(L, L):
            if rank(Lflat + [flat(br(X, px, Y, py))], p) != len(Lflat): ok = False
    return ok

if __name__ == "__main__":
    p, N = int(sys.argv[1]), int(sys.argv[2])
    S = setup(p, N)
    for s in (1, -1):
        g = generators(S, s)
        print(f"sign s={s}: closure+rep {'OK' if check_closure(S, g) else 'FAIL'}")
    gens = generators(S, -1)
    for a in range(p):
        La = rep(S, gens, a)
        for b in range(p):
            Lb = rep(S, gens, b)
            for par in (0, 1):
                ns, unk = equivariant(S, La, Lb, par)
                d = len(ns)
                if d and not (a == b and par == 0 and d == 1):
                    print(f"p={p} N={N}  F_{a} -> F_{b}  {'odd' if par else 'even'}: dim {d}")
