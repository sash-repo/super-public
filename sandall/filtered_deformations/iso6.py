#!/usr/bin/env python3
"""Torus-free isomorphism invariants of k^(1)(3;1), er^(1)(3;1) and the deform of
k^(1)(3;1) along the weight-(-6;0) cocycle.  Tables from iso6_tables.py."""
import sys, time
import numpy as np
from collections import Counter
P = 3
T = dict(np.load("iso6_tables.npz"))
NAMES = ["k1", "E6_1", "E6_2", "erdef_p", "erdef_m", "er_gr"]
LABEL = {"k1": "k^(1)(3;1)", "E6_1": "deform (-6;0), lam=1", "E6_2": "deform (-6;0), lam=2",
         "erdef_p": "deform (-3;3) [=er]", "erdef_m": "deform (-3;-3) [=er]", "er_gr": "gr of (-3;3) deform"}


def rank3(M):
    M = np.array(M, dtype=np.int64) % P
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        if M[r, c] == 2:
            M[r] = (2 * M[r]) % P
        f = M[:, c].copy(); f[r] = 0
        M = (M - np.outer(f, M[r])) % P
        r += 1
        if r == rows:
            break
    return r


def ad(C, x):
    return np.einsum("i,ijk->kj", x, C) % P


def adbasis(C):
    n = C.shape[0]
    return [C[i].T % P for i in range(n)]      # ad e_i [k,j] = C[i,j,k]


def envelope(C):
    A = adbasis(C)
    n = len(A)
    span = [a.flatten() for a in A]
    r0 = rank3(np.array(span))
    cur = A
    dims = [r0]
    restricted = None
    while True:
        cur = [np.linalg.matrix_power(a, 3) % P for a in cur]
        span += [a.flatten() for a in cur]
        r = rank3(np.array(span))
        if restricted is None:
            restricted = (r == r0)
        if r == dims[-1]:
            break
        dims.append(r)
    return dims[-1], restricted, dims


def killing_rank(C):
    A = adbasis(C)
    n = len(A)
    K = np.array([[int(np.trace(A[i] @ A[j]) % P) for j in range(n)] for i in range(n)])
    return rank3(K)


def jordan_sig(M):
    """rank of M and dims of the generalized eigenspaces for eigenvalues 0,1,2."""
    n = M.shape[0]
    I = np.eye(n, dtype=np.int64)
    out = [rank3(M)]
    for lam in (0, 1, 2):
        B = (M - lam * I) % P
        Bk = np.linalg.matrix_power(B, n) % P   # exact: entries reduced? keep mod
        out.append(n - rank3(Bk))
    return tuple(out)


def matpow_mod(B, e):
    R = np.eye(B.shape[0], dtype=np.int64)
    while e:
        if e & 1:
            R = R @ B % P
        B = B @ B % P
        e >>= 1
    return R


def jordan_sig2(M):
    n = M.shape[0]
    I = np.eye(n, dtype=np.int64)
    r1 = rank3(M)
    r2 = rank3(M @ M % P)
    ge = []
    for lam in (0, 1, 2):
        ge.append(n - rank3(matpow_mod((M - lam * I) % P, n)))
    return (r1, r2) + tuple(ge)


# F_27 = F_3[a]/(a^3 - a - 1): multiplication matrices
COMP = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=np.int64)   # mult by a


def mulmat(c):
    return (c[0] * np.eye(3, dtype=np.int64) + c[1] * COMP + c[2] * (COMP @ COMP)) % P


def generic_rank_F27(C, rng, trials=20):
    A = adbasis(C)
    n = len(A)
    best = 0
    for _ in range(trials):
        X = rng.integers(0, P, size=(n, 3))
        M = sum(np.kron(A[i], mulmat(X[i])) for i in range(n)) % P
        best = max(best, rank3(M) // 3)
    return best


if __name__ == "__main__":
    which = sys.argv[1:] or NAMES
    rng = np.random.default_rng(20260930)
    NS = int(__import__("os").environ.get("NS", "30000"))
    xs = rng.integers(0, P, size=(NS, 26))
    for nm in which:
        C = T[nm]
        t0 = time.time()
        env, restr, chain = envelope(C)
        kr = killing_rank(C)
        gr = generic_rank_F27(C, np.random.default_rng(1))
        sig = Counter()
        rk = Counter()
        for x in xs:
            M = ad(C, x)
            s = jordan_sig2(M)
            sig[s] += 1
            rk[s[0]] += 1
        print("=" * 72)
        print(f"{LABEL[nm]}  [{nm}]")
        print(f"  3-envelope dim {env} (chain {chain}), restricted: {restr}")
        print(f"  Killing form rank {kr}")
        print(f"  generic rank ad x over F_27: {gr}  (min dim centralizer {26-gr})")
        print(f"  rank ad x over {NS} random F_3 points: {sorted(rk.items())}")
        nil = sum(v for s, v in sig.items() if s[2] == 26)
        print(f"  ad-nilpotent fraction: {nil}/{NS}")
        top = sig.most_common(12)
        print("  top signatures (rk, rk^2, gen.eig dims 0/1/2): ")
        for s, v in top:
            print(f"     {s}: {v}")
        print(f"  ({time.time()-t0:.1f}s)")
        sys.stdout.flush()
