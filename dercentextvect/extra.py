#!/usr/bin/env python3
"""Which derivations of L are NOT induced by Der O?

Skryabin's theorem says Der L = N_{Der O}(L) for Lie algebras of Cartan type.
This module measures the failure of that identity in the super case:
    outer-to-O = dim Der L - (dim N_{Der O}(L) - dim C_{Der O}(L))
and prints a representative of the quotient by its action on L.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import rank, nullspace                              # noqa: E402
from dervect import (Super, Echelon, echelon, homogeneous_basis,  # noqa: E402
                     Struct, normalizer)


def der_block(S, L, st=None, hom=None):
    """all blocks: returns (st, {(d,q): list of solution vectors over slots},
    {(d,q): slots})."""
    if hom is None:
        hom = homogeneous_basis(S, L)
    if st is None:
        st = Struct(S, hom)
    N, p = st.N, S.p
    lay = {}
    for i in range(N):
        lay.setdefault((st.deg[i], st.par[i]), []).append(i)
    degs = sorted({st.deg[i] for i in range(N)})
    sols, slotmap = {}, {}
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
                    sgn = (-1) ** (q * st.par[i])
                    for t in tgt:
                        row = [0] * nvar
                        for k in range(N):
                            if cij[k] % p and (k, t) in sind:
                                row[sind[(k, t)]] = (row[sind[(k, t)]] + cij[k]) % p
                        for (a, b) in slots:
                            if a == i:
                                v = st.cc(b, j)[t]
                                if v % p:
                                    row[sind[(a, b)]] = (row[sind[(a, b)]] - v) % p
                            if a == j:
                                v = st.cc(i, b)[t]
                                if v % p:
                                    row[sind[(a, b)]] = (row[sind[(a, b)]] - sgn * v) % p
                        if any(row):
                            eqs.append(row)
            ns = nullspace(eqs, p=p) if eqs else [
                [1 if i == k else 0 for i in range(nvar)] for k in range(nvar)]
            ns = [[int(x) % p for x in v] for v in ns]
            if ns:
                sols[(d, q)] = ns
                slotmap[(d, q)] = slots
    return st, hom, sols, slotmap


def induced(S, L, st, hom, slots, d, q):
    """the vectors (over `slots`) of the derivations ad D | L, D in N_{Der O}(L),
    D homogeneous of bidegree (d, q)."""
    NN = normalizer(S, L)
    # split N into homogeneous pieces
    NNh = homogeneous_basis(S, NN)
    out = []
    sind = {s: k for k, s in enumerate(slots)}
    for (dd, qq, D) in NNh:
        if (dd, qq) != (d, q):
            continue
        v = [0] * len(slots)
        for i, X in enumerate(st.B):
            c = st.coords(S.vec(S.bracket(D, X)))
            assert c is not None
            for j, val in enumerate(c):
                if val % S.p:
                    v[sind[(i, j)]] = val % S.p
        out.append(v)
    return out


def report(tag, S, L):
    st, hom, sols, slotmap = der_block(S, L)
    NN = normalizer(S, L)
    dN = Echelon([S.vec(X) for X in NN], S.p).dim
    # centralizer of L in Der O
    rows = []
    for X in st.B:
        for t in range(S.dimD):
            row = [S.vec(S.bracket({b: 1}, X))[t] for b in S.dbasis]
            if any(x % S.p for x in row):
                rows.append([x % S.p for x in row])
    dC = len(nullspace(rows, p=S.p)) if rows else S.dimD
    tot = sum(len(v) for v in sols.values())
    print(f"{tag}: dim L={st.N}  dim Der L={tot}  dim N_DerO(L)={dN} "
          f"dim C_DerO(L)={dC}  gap={tot - (dN - dC)}")
    for key in sorted(sols):
        d, q = key
        got = len(sols[key])
        ind = induced(S, L, st, hom, slotmap[key], d, q)
        di = Echelon(ind, S.p).dim if ind else 0
        if got != di:
            print(f"    block (deg {d}, par {q}): dim Der = {got}, "
                  f"induced from Der O = {di}  <-- gap {got - di}")
            # a representative
            E = Echelon(ind, S.p)
            for v in sols[key]:
                if not E.contains(v):
                    show(S, st, slotmap[key], v)
                    break
    return tot, dN, dC


def show(S, st, slots, v, maxterms=12):
    from collections import defaultdict
    act = defaultdict(dict)
    for k, (i, j) in enumerate(slots):
        if v[k] % S.p:
            act[i][j] = v[k] % S.p
    shown = 0
    for i in sorted(act):
        print(f"        D({fmt(S, st.B[i])}) = "
              + " + ".join(f"{c}*{fmt(S, st.B[j])}" for j, c in act[i].items()))
        shown += 1
        if shown >= maxterms:
            print("        ...")
            break


def fmt(S, X):
    parts = []
    for (u, g), c in sorted(X.items(), key=str):
        a, b = u
        s = ""
        for i in range(S.m):
            if a[i]:
                s += f"x{i}^({a[i]})"
        for j in range(S.n):
            if b[j]:
                s += f"y{j}"
        if not s:
            s = "1"
        if g[0] == 'x':
            s += f".d{g[1]}" + (f"^{S.p**g[2]}" if g[2] else "")
        else:
            s += f".e{g[1]}"
        parts.append(("" if c == 1 else f"{c}") + s)
    return "(" + "+".join(parts) + ")"
