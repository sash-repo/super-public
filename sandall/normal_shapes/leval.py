#!/usr/bin/env python3
"""Controls for `leiso.py`: the derivation solver and the square-zero count.

1) DERIVATIONS.  For n = 1, u = One the unknowns are few enough to enumerate
   every parity-homogeneous linear map and test the Leibniz rule directly:
   3^13 even ones and 3^12 odd ones.  The number of derivations must be
   p^(dim Der), which checks the linear solver of `leiso.derivations`.
   Also, for every case, a basis of the computed space is reconstructed and the
   Leibniz rule verified on ALL pairs of basis elements.

2) THE COUNT.  The affine method (fiber over the xi-free part) is compared with
   plain enumeration on sub-cubes of dimension up to 14, for both shapes.

Usage:  python3 leval.py
"""
import sys, os, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leiso import (LeAlg, derivations, count_sqzero_affine,          # noqa: E402
                   count_sqzero_bruteforce, odd_split, echelon_add)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import nullspace as _nullspace                        # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def der_space(L, par):
    """basis of the space of derivations of parity par, as dim x dim matrices."""
    p, d = L.p, L.dim
    T = L.table()
    unk = {}
    for j in range(d):
        for i in range(d):
            if (L.par[i] + L.par[j]) % 2 == par % 2:
                unk[(j, i)] = len(unk)
    N = len(unk)
    E = []
    rows = []
    for j in range(d):
        for k in range(j, d):
            s = (-1) ** (par * L.par[j])
            for t in range(d):
                row = [0] * N
                for i in range(d):
                    c = T[j][k][i] % p
                    if c and (i, t) in unk:
                        row[unk[(i, t)]] = (row[unk[(i, t)]] + c) % p
                for i in range(d):
                    if (j, i) in unk:
                        c = T[i][k][t] % p
                        if c:
                            row[unk[(j, i)]] = (row[unk[(j, i)]] - c) % p
                    if (k, i) in unk:
                        c = (s * T[j][i][t]) % p
                        if c:
                            row[unk[(k, i)]] = (row[unk[(k, i)]] - c) % p
                if any(row):
                    rows.append(row)
        if len(rows) >= 3000:
            E = echelon_add(E, rows, p, N)
            rows = []
    E = echelon_add(E, rows, p, N)
    ns = _nullspace(E, p) if E else [[1 if i == j else 0 for i in range(N)]
                                     for j in range(N)]
    out = []
    inv = {v: k for k, v in unk.items()}
    for v in ns:
        M = [[0] * d for _ in range(d)]
        for c in range(N):
            val = int(v[c]) % p
            if val:
                j, i = inv[c]
                M[j][i] = val
        out.append(M)
    return out


def is_derivation(L, M, par):
    """check D[x,y] = [Dx,y] + (-1)^{par p(x)} [x,Dy] on all pairs."""
    p, d = L.p, L.dim
    T = L.table()

    def apply(vec):
        out = [0] * d
        for j, c in enumerate(vec):
            c %= p
            if not c:
                continue
            for i in range(d):
                if M[j][i]:
                    out[i] = (out[i] + c * M[j][i]) % p
        return out

    for j in range(d):
        for k in range(d):
            s = (-1) ** (par * L.par[j])
            lhs = apply(T[j][k])
            rhs = [0] * d
            Dj, Dk = M[j], M[k]
            for i in range(d):
                if Dj[i]:
                    for t in range(d):
                        if T[i][k][t]:
                            rhs[t] = (rhs[t] + Dj[i] * T[i][k][t]) % p
                if Dk[i]:
                    for t in range(d):
                        if T[j][i][t]:
                            rhs[t] = (rhs[t] + s * Dk[i] * T[j][i][t]) % p
            if any((lhs[t] - rhs[t]) % p for t in range(d)):
                return False
    return True


def brute_der_count(L, par):
    """enumerate every parity-homogeneous map and count the derivations."""
    p, d = L.p, L.dim
    slots = [(j, i) for j in range(d) for i in range(d)
             if (L.par[i] + L.par[j]) % 2 == par % 2]
    cnt = 0
    for co in itertools.product(range(p), repeat=len(slots)):
        M = [[0] * d for _ in range(d)]
        for (j, i), c in zip(slots, co):
            M[j][i] = c
        if is_derivation(L, M, par):
            cnt += 1
    return cnt, len(slots)


def main():
    global LOG
    here = os.path.dirname(os.path.abspath(__file__))
    LOG = open(os.path.join(here, "leval.log"), "w")

    say("=== 1) the derivation solver ===")
    for sh in ("0", "Xi"):
        L = LeAlg(1, 3, (1,), sh)
        for par in (0, 1):
            dim, N = derivations(L, par)
            basis = der_space(L, par)
            ok = all(is_derivation(L, M, par) for M in basis)
            say(f"n=1 u=One shape={sh} parity {par}: dim = {dim} (unknowns {N}), "
                f"basis of {len(basis)} all derivations: {ok}")
            cnt, nsl = brute_der_count(L, par)
            say(f"    enumeration of all 3^{nsl} maps: {cnt} derivations "
                f"= 3^{dim}? {cnt == 3 ** dim}")
    for sh in ("0", "Xi"):
        L = LeAlg(2, 3, (1, 1), sh)
        for par in (0, 1):
            dim, N = derivations(L, par)
            basis = der_space(L, par)
            ok = all(is_derivation(L, M, par) for M in basis)
            say(f"n=2 u=One shape={sh} parity {par}: dim = {dim} (unknowns {N}), "
                f"basis of {len(basis)} all derivations: {ok}")

    say("")
    say("=== 2) the square-zero count: affine method vs enumeration ===")
    for sh in ("0", "Xi"):
        L = LeAlg(2, 3, (1, 1), sh)
        A, C = odd_split(L)
        for k in range(0, 6):
            sub = A + C[:k]
            if len(sub) > 13:
                break
            nb = count_sqzero_bruteforce(L, sub)
            na = count_sqzero_affine(L, A, C[:k]) if k else nb
            say(f"shape={sh} sub-cube A u C[:{k}] (dim {len(sub)}): "
                f"enumeration {nb}, affine {na}, agree = {nb == na}")
    LOG.close()


if __name__ == "__main__":
    main()
