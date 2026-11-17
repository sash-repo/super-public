#!/usr/bin/env python3
"""Brute-force dim Der L (no grading used) -- an independent check of der_dim.

Unknowns: the full matrix of D in a basis of L, split only by parity of D.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import rank                                        # noqa: E402
from dervect import Echelon, homogeneous_basis, Struct           # noqa: E402


def der_dim_brute(S, L):
    hom = homogeneous_basis(S, L)
    st = Struct(S, hom)
    N, p = st.N, S.p
    total = 0
    for q in (0, 1):
        nvar = N * N
        eqs = []
        for i in range(N):
            for j in range(i, N):
                cij = st.cc(i, j)
                sgn = (-1) ** (q * st.par[i])
                for t in range(N):
                    # D must be parity-homogeneous of shift q:
                    # D_{a,b} allowed only if par[b] = par[a] + q
                    row = [0] * nvar
                    for k in range(N):
                        if cij[k] % p and (st.par[t] - st.par[k]) % 2 == q:
                            row[k * N + t] = (row[k * N + t] + cij[k]) % p
                    for b in range(N):
                        if (st.par[b] - st.par[i]) % 2 == q:
                            v = st.cc(b, j)[t]
                            if v % p:
                                row[i * N + b] = (row[i * N + b] - v) % p
                        if (st.par[b] - st.par[j]) % 2 == q:
                            v = st.cc(i, b)[t]
                            if v % p:
                                row[j * N + b] = (row[j * N + b] - sgn * v) % p
                    if any(row):
                        eqs.append(row)
        # variables forced to zero by the parity shift
        forced = [k * N + t for k in range(N) for t in range(N)
                  if (st.par[t] - st.par[k]) % 2 != q]
        for f in forced:
            row = [0] * nvar
            row[f] = 1
            eqs.append(row)
        total += nvar - rank(eqs, p=p)
    return total
