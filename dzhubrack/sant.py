#!/usr/bin/env python3
r"""Super-antisymmetrizors a_k of Lebedev-Leites arXiv:1906.01380, eq. (14)-(15).

    a_k(X_1,...,X_k) = sum_{s in S_k} sign(s) sign(s,P) X_{s(1)}...X_{s(k)},
    sign(s,P) = prod_{i<j, s(i)>s(j)} (-1)^{p_i p_j}.

Everything here is exact: entries are Fractions or ints, never floats.

The sum over S_k is computed by a subset dynamic programme costing k*2^(k-1)
products instead of k! ones.  Writing c(w) for the coefficient of the word
w = (w_1,...,w_k),

    c(w) = prod over pairs (i<j) inverted in w of (-1)^{1 + p_i p_j},

and splitting off the first letter gives

    A[S] = sum_{i in S} (prod_{j in S, j<i} (-1)^{1+p_i p_j}) X_i A[S\{i}].

`amult` implements this for any associative product `mul` (matrices, words of
a free superalgebra, differential operators, ...).
"""

from fractions import Fraction
from itertools import combinations


# ---------------------------------------------------------------- the a_k

def amult(elts, par, mul, add, zero):
    """a_k(elts) for an arbitrary associative product.

    elts : list of k elements; par : list of their parities (0/1);
    mul(x,y), add(x,y), zero: the ambient associative superalgebra.
    """
    k = len(elts)
    if k == 0:
        raise ValueError("a_0 undefined")
    # A[mask] over subsets, built by increasing popcount
    A = {}
    for i in range(k):
        A[1 << i] = elts[i]
    for size in range(2, k + 1):
        for S in combinations(range(k), size):
            mask = 0
            for i in S:
                mask |= 1 << i
            acc = zero
            for i in S:
                sgn = 1
                for j in S:
                    if j < i:
                        sgn = -sgn                      # the 1 in (-1)^{1+p_i p_j}
                        if par[i] and par[j]:
                            sgn = -sgn
                sub = A[mask ^ (1 << i)]
                term = mul(elts[i], sub)
                acc = add(acc, term if sgn > 0 else neg(term))
            A[mask] = acc
    return A[(1 << k) - 1]


def neg(x):
    """Negation that works for matrices (list of lists) and dicts (free algebra)."""
    if isinstance(x, dict):
        return {k: -v for k, v in x.items()}
    if isinstance(x, list):
        return [[-e for e in row] for row in x]
    return -x


# --------------------------------------------------------- matrices over Q

def mzero(n):
    return [[0] * n for _ in range(n)]


def madd(A, B):
    if A is None:
        return B
    return [[A[i][j] + B[i][j] for j in range(len(A))] for i in range(len(A))]


def mmul(A, B):
    n = len(A)
    return [[sum(A[i][t] * B[t][j] for t in range(n)) for j in range(n)] for i in range(n)]


def miszero(A):
    return all(all(e == 0 for e in row) for row in A)


def a_matrix(mats, par):
    n = len(mats[0])
    return amult(mats, par, mmul, madd, mzero(n))


# ------------------------------------------------- the free superalgebra

def wmul(u, v):
    """Product of two elements of the free associative superalgebra.

    An element is a dict {word (tuple of generator indices): coefficient}.
    """
    out = {}
    for w1, c1 in u.items():
        for w2, c2 in v.items():
            w = w1 + w2
            out[w] = out.get(w, 0) + c1 * c2
    return {w: c for w, c in out.items() if c != 0}


def wadd(u, v):
    out = dict(u)
    for w, c in v.items():
        out[w] = out.get(w, 0) + c
        if out[w] == 0:
            del out[w]
    return out


def gen(i):
    return {(i,): 1}


def a_word(idx, par):
    """a_k of the generators listed in idx, with parity vector par (indexed by generator)."""
    elts = [gen(i) for i in idx]
    return amult(elts, [par[i] for i in idx], wmul, wadd, {})


# ----------------------------------------------------------- linear algebra

def in_span(vec, basis):
    """Is vec in the Q-span of basis?  Exact Gaussian elimination."""
    rows = [list(map(Fraction, b)) for b in basis]
    tgt = list(map(Fraction, vec))
    piv = []
    for r in rows:
        r = r[:]
        for (p, pr) in piv:
            if r[p] != 0:
                f = r[p] / pr[p]
                r = [a - f * b for a, b in zip(r, pr)]
        nz = next((i for i, a in enumerate(r) if a != 0), None)
        if nz is not None:
            piv.append((nz, r))
    r = tgt[:]
    for (p, pr) in piv:
        if r[p] != 0:
            f = r[p] / pr[p]
            r = [a - f * b for a, b in zip(r, pr)]
    return all(a == 0 for a in r)


def flat(A):
    return [e for row in A for e in row]
