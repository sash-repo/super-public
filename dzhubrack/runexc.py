#!/usr/bin/env python3
"""Problem 1.6.1 2) of arXiv:1906.01380: the conjectural A(f4) and A(e6).

    conjecture:  A(f4) = {a_2, a_10, a_14, a_22} on the 26-dimensional module,
                 A(e6) = {a_2, a_8, a_10, a_14, a_16, a_22} on the 27-dimensional one.

f4 = Der(J) and e6 = Der(J) + {L_X : tr X = 0} are built in albert.py from the
octonions upwards.  Arithmetic is exact modulo p = 65521.
"""

import sys
import numpy as np

from expo import a_k_dp
import albert

PP = 65521


def rank_mod(vectors, p=PP):
    M = np.array(vectors, dtype=np.int64) % p
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c] % p), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        nz = np.nonzero(M[:, c] % p)[0]
        for i in nz:
            if i != r:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
        if r == rows:
            break
    return r


def survey(name, basis, ks, trials=2, seed=3, p=PP):
    N = basis[0].shape[0]
    B = [np.asarray(b, dtype=np.int64) % p for b in basis]
    gv = [b.reshape(-1) for b in B]
    rk = rank_mod(gv)
    print(f"{name}: dim {len(B)} (independent: {rk}), module of size {N}")
    rng = np.random.default_rng(seed)
    good = []
    for k in ks:
        nonzero, inside = False, True
        for _ in range(trials):
            c = rng.integers(0, p, size=(k, len(B)))
            mats = []
            for t in range(k):
                M = np.zeros((N, N), dtype=np.int64)
                for s in range(len(B)):
                    M = (M + int(c[t, s]) * B[s]) % p
                mats.append(M)
            A = a_k_dp(mats, N, p=p) % p
            if A.any():
                nonzero = True
                if rank_mod(gv + [A.reshape(-1)]) > rk:
                    inside = False
                    break
        v = "in A(g)" if (nonzero and inside) else \
            ("vanishes identically" if not nonzero else "leaves g")
        if nonzero and inside:
            good.append(k)
        print(f"  {name:6s} a_{k:<3d} {v}")
        sys.stdout.flush()
    print(f"  {name:6s} ==> A(g) contains exactly {good} in the range {list(ks)[0]}..{list(ks)[-1]}")
    return good


def f4_basis(restrict=True):
    d = albert.der_J()
    if restrict:
        d = albert.restrict_to_J0(d)
    return [np.array([[int(x) for x in row] for row in M], dtype=np.int64) for M in d]


def e6_basis():
    d = albert.der_J()
    L = albert.L_operators()
    # traceless combinations of the L_X: L_{b_i} - (tr b_i / 3) L_1
    n = 27
    out = [np.array([[int(x) for x in row] for row in M], dtype=np.int64) for M in d]
    tr = albert.trace_form_coeffs()
    from fractions import Fraction
    Lid = [[sum(L[t][r][c] for t in range(3)) for c in range(n)] for r in range(n)]
    for i in range(n):
        M = [[Fraction(L[i][r][c]) - Fraction(tr[i], 3) * Lid[r][c]
              for c in range(n)] for r in range(n)]
        den = 1
        for row in M:
            for x in row:
                den = den * x.denominator // __import__("math").gcd(den, x.denominator)
        out.append(np.array([[int(x * den) for x in row] for row in M], dtype=np.int64))
    return out


if __name__ == "__main__":
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    if which in ("both", "f4"):
        survey("f4", f4_basis(which != "f4_27"), range(2, top + 1))
    if which in ("both", "e6"):
        survey("e6", e6_basis(), range(2, top + 1))
