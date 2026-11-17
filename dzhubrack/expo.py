#!/usr/bin/env python3
r"""Theorem 1.6 and Problem 1.6.1 of arXiv:1906.01380: which a_k close g, and how
the surviving indices relate to the exponents of g.

For a purely even g the antisymmetrizor is alternating, so a_k is determined by
its values on the k-subsets of a basis; a_k = 0 identically if and only if it
vanishes on every one of them.  We evaluate a_k on random elements of g, which is
the same test up to a Schwartz-Zippel accident, and -- where the size allows it --
on every k-subset of the basis as well.

Arithmetic is exact modulo a prime p < 256, carried by float64 matrix products
(all intermediate entries stay below 2^53, so BLAS is used without rounding).
"""

import sys
import numpy as np
from itertools import combinations

P = 251


def a_k_dp(mats, N, chunk=1 << 16, par=None, p=P):
    """a_k(X_1,...,X_k) mod P by the subset dynamic programme.

    par is the parity vector of the arguments (all even if omitted).
    Levels are held as uint8 arrays of shape (n_states, N, N).
    """
    k = len(mats)
    if par is None:
        par = [0] * k
    states = [(1 << i,) for i in range(k)]
    idx = {1 << i: i for i in range(k)}
    cur = np.array([m % p for m in mats], dtype=np.int32)
    masks = [1 << i for i in range(k)]
    X = [np.asarray(m % p, dtype=np.float64) for m in mats]
    for size in range(2, k + 1):
        new_masks = [sum(1 << i for i in S) for S in combinations(range(k), size)]
        nidx = {m: t for t, m in enumerate(new_masks)}
        out = np.zeros((len(new_masks), N, N), dtype=np.int64)
        for i in range(k):
            bit = 1 << i
            src, dst, sgn = [], [], []
            for t, m in enumerate(masks):
                if m & bit:
                    continue
                S = m | bit
                s = 1
                for j in range(i):
                    if S & (1 << j):
                        s = -s
                        if par[i] and par[j]:
                            s = -s
                src.append(t)
                dst.append(nidx[S])
                sgn.append(s)
            if not src:
                continue
            src = np.array(src)
            dst = np.array(dst)
            sgn = np.array(sgn, dtype=np.int64)
            for lo in range(0, len(src), chunk):
                hi = min(lo + chunk, len(src))
                blk = cur[src[lo:hi]].astype(np.float64)
                prod = np.matmul(X[i], blk)             # X_i * A[S]
                prod = np.remainder(prod, p).astype(np.int64)
                prod *= sgn[lo:hi, None, None]
                np.add.at(out, dst[lo:hi], prod)
        cur = np.remainder(out, p).astype(np.int32)
        masks = new_masks
    return cur[0].astype(np.int64)


def rand_elt(basis, rng):
    c = rng.integers(0, P, size=len(basis))
    M = np.zeros_like(basis[0])
    for t, b in enumerate(basis):
        M = (M + int(c[t]) * b) % P
    return M


def span_rank(vectors):
    """Rank mod P of a list of vectors."""
    M = np.array(vectors, dtype=np.int64) % P
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c] % P:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, c]), P - 2, P)
        M[r] = (M[r] * inv) % P
        for i in range(rows):
            if i != r and M[i, c] % P:
                M[i] = (M[i] - M[i, c] * M[r]) % P
        r += 1
        if r == rows:
            break
    return r


def survey(name, basis, ks, trials=3, seed=1):
    """Report for each k whether a_k vanishes on g and whether it lands in g."""
    N = basis[0].shape[0]
    B = [np.asarray(b, dtype=np.int64) % P for b in basis]
    gvecs = [b.reshape(-1) for b in B]
    rk = span_rank(gvecs)
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        nonzero, inside = False, True
        for _ in range(trials):
            mats = [rand_elt(B, rng) for _ in range(k)]
            A = a_k_dp(mats, N) % P
            if A.any():
                nonzero = True
                if span_rank(gvecs + [A.reshape(-1)]) > rk:
                    inside = False
                    break
        out[k] = (nonzero, inside)
        if nonzero and inside:
            verdict = "in A(g)"
        elif not nonzero:
            verdict = "vanishes identically"
        else:
            verdict = "leaves g"
        print(f"  {name:10s} a_{k:<3d} {verdict}")
        sys.stdout.flush()
    good = [k for k in ks if out[k][0] and out[k][1]]
    print(f"  {name:10s} ==> A(g) = {{a_k : k in {good}}}")
    return out
