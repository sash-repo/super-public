#!/usr/bin/env python3
"""Which cocycle is the one in weight -2?

For the Hamiltonian series the expected answer is the Poisson extension
    0 --> K --> po(2r|n) --> h(2r|n) --> 0,   c(H_f, H_g) = const term of {f, g},
and for the odd Hamiltonian (Buttin) series the expected answer is its odd
analogue
    0 --> K M_1 --> b(r) --> le(r|r) --> 0,   c(Le_f, Le_g) = const term of [f, g],
whose center is ODD.  Here we build these cochains explicitly and check that
each is a cocycle and that its class is nonzero -- i.e. that it is the class
`cecensus.py` finds in weight -2.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exactlin import rank                                          # noqa: E402
from dervect import Super, Echelon, poisson, homogeneous_basis, Struct  # noqa: E402
from centext import Cocycles                                       # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "named.log")


def constant_term_cocycle(S, kind, r):
    """The bracket {f, g} of generating functions, its constant term as a
    2-cochain on L = {H_f} (resp. {Le_f}), written in the basis of L."""
    L, fields = poisson(S, kind, r)
    hom = homogeneous_basis(S, L)
    st = Struct(S, hom)
    N = st.N

    # express each basis element of L as a combination of the H_f's
    rows = [S.vec(fields[u]) for u in fields]
    keys = list(fields)
    E = Echelon(rows, S.p)
    aug = [rows[i][:] + [1 if j == i else 0 for j in range(len(rows))]
           for i in range(len(rows))]
    A, piv = __import__("dervect").echelon(aug, S.p)

    def as_functions(X):
        """write X in Der O as sum c_u H_u; returns {monomial: coeff}."""
        w = [x % S.p for x in S.vec(X)] + [0] * len(rows)
        for row, c in zip(A, piv):
            if c < S.dimD and w[c]:
                f = w[c]
                w = [(a - f * b) % S.p for a, b in zip(w, row)]
        assert not any(w[:S.dimD]), "not in the span of the H_f"
        out = {}
        for i, c in enumerate(w[S.dimD:]):
            c = (-c) % S.p
            if c:
                out[keys[i]] = c
        return out

    F = [as_functions(st.B[i]) for i in range(N)]
    one = ((0,) * S.m, (0,) * S.n)          # the constant function 1

    def brk(u, v):
        """{u, v} of two monomials, as a dict on O -- read off H_u(v)."""
        return S.act(fields[u], v) if u in fields else {}

    c = {}
    for i in range(N):
        for j in range(i, N):
            tot = 0
            for u, cu in F[i].items():
                for v, cv in F[j].items():
                    tot += cu * cv * brk(u, v).get(one, 0)
            tot %= S.p
            if tot:
                c[(i, j)] = tot
    return L, st, c


def report(fh, tag, S, kind, r):
    L, st, c = constant_term_cocycle(S, kind, r)
    C = Cocycles(S, L)
    assert C.N == st.N
    # the parity and weight of the cochain
    ws = {C.deg[i] + C.deg[j] for (i, j) in c}
    ps = {(C.par[i] + C.par[j]) % 2 for (i, j) in c}
    if not c:
        print(f"{tag}: the constant-term cochain is ZERO", file=fh)
        return
    w, eps = ws.pop(), ps.pop()
    assert not ws and not ps, "inhomogeneous"
    P = C.pairs(w, eps)
    idx = {q: t for t, q in enumerate(P)}
    vec = [0] * len(P)
    for (i, j), val in c.items():
        vec[idx[(i, j)]] = val % S.p

    # is it a cocycle?  is it a coboundary?
    dZ, dB = C.block(w, eps)
    cob = []
    for l in range(C.N):
        if C.deg[l] != w or C.par[l] != eps:
            continue
        row = [st.cc(i, j)[l] % S.p for (i, j) in P]
        if any(row):
            cob.append(row)
    is_cob = rank(cob + [vec], p=S.p) == rank(cob, p=S.p) if cob else not any(vec)
    # cocycle test: plug into the defining identity
    eqs = []
    for i in range(C.N):
        for j in range(i, C.N):
            for k in range(j, C.N):
                if C.deg[i] + C.deg[j] + C.deg[k] != w:
                    continue
                if (C.par[i] + C.par[j] + C.par[k]) % 2 != eps:
                    continue
                tot = 0
                for l in range(C.N):
                    t, sg = C.coord_of(idx, l, k)
                    if st.cc(i, j)[l] and t is not None:
                        tot += st.cc(i, j)[l] * sg * vec[t]
                    t, sg = C.coord_of(idx, i, l)
                    if st.cc(j, k)[l] and t is not None:
                        tot -= st.cc(j, k)[l] * sg * vec[t]
                    t, sg = C.coord_of(idx, j, l)
                    if st.cc(i, k)[l] and t is not None:
                        tot += ((-1) ** (C.par[i] * C.par[j])) * st.cc(i, k)[l] * sg * vec[t]
                if tot % S.p:
                    eqs.append((i, j, k, tot % S.p))
    print(f"{tag}: weight={w} parity={'odd' if eps else 'even'} "
          f"cocycle={'YES' if not eqs else 'NO ' + str(eqs[:3])} "
          f"coboundary={'yes' if is_cob else 'NO'} "
          f"| dim H^2_({w},{eps})={dZ - dB}", file=fh)


def main():
    with open(LOG, "w") as fh:
        print("=== the Poisson class of h(2r|n) ===", file=fh)
        for p in (3, 5):
            for (r, n) in [(1, 0), (1, 1), (1, 2), (2, 0)]:
                if p == 5 and r == 2:
                    continue
                S = Super(p, 2 * r, n, [1] * (2 * r))
                report(fh, f"p={p} h({2*r}|{n})", S, 'h', r)
        print("", file=fh)
        print("=== the Buttin class of le(r|r) ===", file=fh)
        for p in (3, 5):
            for r in (1, 2):
                if p == 5 and r == 2:
                    continue
                S = Super(p, r, r, [1] * r)
                report(fh, f"p={p} le({r}|{r})", S, 'le', r)
    print(open(LOG).read())


if __name__ == "__main__":
    main()
