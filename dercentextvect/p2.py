#!/usr/bin/env python3
"""Characteristic 2: derivations must respect the squaring as well.

At p = 2 a Lie superalgebra is a triple (g, bracket, squaring s: g_odd -> g_even),
and a derivation D (of either parity) has to satisfy, besides the Leibniz rule,

    D(s(x)) = [x, D(x)]     for every odd x

(both parities of D give this same condition; polarizing it in x gives back the
Leibniz rule, so imposing it on a basis of g_odd suffices).

Here s(x) = x o x, the composition of x with itself as an operator on O; it is a
derivation of O in characteristic 2, so it lies in Der O again.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import rank, nullspace                              # noqa: E402
from dervect import (Super, Echelon, echelon, homogeneous_basis,  # noqa: E402
                     Struct, normalizer)


def op_matrix(S, X):
    """the matrix of X in Der O acting on O (columns = images of monomials)."""
    M = [[0] * S.dimO for _ in range(S.dimO)]
    for c, u in enumerate(S.mons):
        for w, v in S.act(X, u).items():
            M[S.mind[w]][c] = v % S.p
    return M


_BASIS_OPS = {}


def basis_ops(S):
    key = id(S)
    if key not in _BASIS_OPS:
        _BASIS_OPS[key] = [op_matrix(S, {b: 1}) for b in S.dbasis]
    return _BASIS_OPS[key]


def from_operator(S, M):
    """express a derivation of O, given by its matrix on O, in the uD basis."""
    ops = basis_ops(S)
    rows = []
    rhs = []
    for r in range(S.dimO):
        for c in range(S.dimO):
            rows.append([ops[k][r][c] for k in range(S.dimD)])
            rhs.append(M[r][c] % S.p)
    # solve rows * z = rhs by elimination on the augmented matrix
    aug = [rows[i] + [rhs[i]] for i in range(len(rows))]
    R, piv = echelon(aug, S.p)
    z = [0] * S.dimD
    for row, c in zip(R, piv):
        if c == S.dimD:
            raise ValueError("not in Der O")
        z[c] = row[-1] % S.p
        for cc in range(c + 1, S.dimD):
            if row[cc] % S.p:
                raise ValueError("underdetermined -- basis is not independent")
    # verify
    chk = [[sum(z[k] * ops[k][r][c] for k in range(S.dimD)) % S.p
            for c in range(S.dimO)] for r in range(S.dimO)]
    assert chk == [[x % S.p for x in row] for row in M], "conversion failed"
    return S.unvec(z)


def square(S, X):
    """s(X) = X o X for odd X, as an element of Der O (p = 2 only)."""
    assert S.p == 2
    M = op_matrix(S, X)
    n = S.dimO
    M2 = [[sum(M[r][k] * M[k][c] for k in range(n)) % 2 for c in range(n)]
          for r in range(n)]
    return from_operator(S, M2)


def der_dim_p2(S, L, want_blocks=False):
    """dim Der L at p = 2, with the squaring condition imposed."""
    hom = homogeneous_basis(S, L)
    st = Struct(S, hom)
    N, p = st.N, S.p
    assert p == 2
    sq = {}
    for i in range(N):
        if st.par[i] == 1:
            c = st.coords(S.vec(square(S, st.B[i])))
            assert c is not None, "the squaring leaves L"
            sq[i] = c
    lay = {}
    for i in range(N):
        lay.setdefault((st.deg[i], st.par[i]), []).append(i)
    degs = sorted({st.deg[i] for i in range(N)})
    total, blocks = 0, {}
    for d in range(min(degs) - max(degs), max(degs) - min(degs) + 1):
        for q in (0, 1):
            slots = [(i, j) for i in range(N)
                     for j in lay.get((st.deg[i] + d, (st.par[i] + q) % 2), [])]
            if not slots:
                continue
            sind = {s: k for k, s in enumerate(slots)}
            nvar = len(slots)
            eqs = []
            for i in range(N):
                for j in range(i, N):
                    cij = st.cc(i, j)
                    tgt = lay.get((st.deg[i] + st.deg[j] + d,
                                   (st.par[i] + st.par[j] + q) % 2), [])
                    for t in tgt:
                        row = [0] * nvar
                        for k in range(N):
                            if cij[k] % p and (k, t) in sind:
                                row[sind[(k, t)]] ^= cij[k] % p
                        for (a, b) in slots:
                            if a == i:
                                v = st.cc(b, j)[t] % p
                                if v:
                                    row[sind[(a, b)]] ^= v
                            if a == j:
                                v = st.cc(i, b)[t] % p
                                if v:
                                    row[sind[(a, b)]] ^= v
                        if any(row):
                            eqs.append(row)
            # the squaring condition: D(s(e_i)) = [e_i, D e_i] for odd e_i
            for i in sq:
                # degree of s(e_i) is 2 deg(e_i); D(s(e_i)) has degree
                # 2 deg e_i + d and parity q
                tgt = lay.get((2 * st.deg[i] + d, q), [])
                for t in tgt:
                    row = [0] * nvar
                    for k in range(N):
                        if sq[i][k] % p and (k, t) in sind:
                            row[sind[(k, t)]] ^= sq[i][k] % p
                    for (a, b) in slots:
                        if a == i:
                            v = st.cc(i, b)[t] % p
                            if v:
                                row[sind[(a, b)]] ^= v
                    if any(row):
                        eqs.append(row)
            k = nvar - (rank(eqs, p=p) if eqs else 0)
            if k:
                blocks[(d, q)] = k
                total += k
    return (total, blocks) if want_blocks else total


def normalizer_p2(S, L):
    """the normalizer in Der O, restricted to the elements that also respect
    the squaring of L (i.e. that are derivations of the p=2 superalgebra)."""
    NN = normalizer(S, L)
    E = Echelon([S.vec(X) for X in L], S.p)
    B = [S.unvec(r) for r in E.rows]
    # ad D respects the squaring automatically: [D, s(x)] = [x,[D,x]] holds in
    # any Lie superalgebra with a 2-structure, so nothing to impose.
    return NN
