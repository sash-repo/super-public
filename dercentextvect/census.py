#!/usr/bin/env python3
"""The census behind the tables: Der L vs N_{Der O}(L) for modular vectorial
Lie superalgebras.  Writes census.log incrementally (a killed run leaves usable
partial results)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import nullspace                                    # noqa: E402
from dervect import (Super, Echelon, svect, derived, poisson,     # noqa: E402
                     contact, check_closed, normalizer, ad_dim, der_dim)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "census.log")


def line(s, fh):
    print(s)
    sys.stdout.flush()
    fh.write(s + "\n")
    fh.flush()


def centralizer_dim(S, L):
    rows = []
    for X in L:
        for t in range(S.dimD):
            row = [S.vec(S.bracket({b: 1}, X))[t] for b in S.dbasis]
            if any(x % S.p for x in row):
                rows.append([x % S.p for x in row])
    return len(nullspace(rows, p=S.p)) if rows else S.dimD


def one(tag, S, L, fh, limit=90):
    t = time.time()
    E = Echelon([S.vec(X) for X in L], S.p)
    B = [S.unvec(r) for r in E.rows]
    dL = len(B)
    if dL > limit:
        line(f"{tag}: dim L={dL}  SKIPPED (over the size limit {limit})", fh)
        return None
    ok, _ = check_closed(S, B)
    NN = normalizer(S, B)
    dN = Echelon([S.vec(X) for X in NN], S.p).dim
    dC = centralizer_dim(S, B)
    dad, ctr = ad_dim(S, B)
    dD, blocks = der_dim(S, B, want_blocks=True)
    neg = {k: v for k, v in blocks.items() if k[0] < 0}
    line(f"{tag}: dimL={dL} closed={ok} centre={ctr} dim ad={dad} "
         f"dim Der={dD} out={dD - dad} | dim N={dN} dim C={dC} "
         f"gap={dD - (dN - dC)} | neg blocks={dict(sorted(neg.items()))} "
         f"[{time.time() - t:.0f}s]", fh)
    return dict(dim=dL, der=dD, ad=dad, N=dN, C=dC, gap=dD - (dN - dC),
                blocks=blocks)


def chain(tag, S, L, fh, depth=2, limit=90):
    one(tag, S, L, fh, limit)
    cur = L
    for k in range(1, depth + 1):
        nxt = derived(S, cur)
        if Echelon([S.vec(X) for X in nxt], S.p).dim == \
           Echelon([S.vec(X) for X in cur], S.p).dim:
            break
        one(f"{tag}^({k})", S, nxt, fh, limit)
        cur = nxt


def main():
    with open(LOG, "w") as fh:
        line("=== vect(m|n; N) ===", fh)
        for p in (3, 5):
            for (m, n, N) in [(1, 0, [2]), (0, 2, None), (0, 3, None),
                              (1, 1, [1]), (1, 2, [1]), (2, 1, [1, 1]),
                              (1, 1, [2])]:
                S = Super(p, m, n, N)
                one(f"p={p} vect({m}|{n};{N or '-'})", S, S.vect(), fh)

        line("", fh)
        line("=== svect(m|n; N) and its derived algebras ===", fh)
        for p in (3, 5):
            for (m, n, N) in [(3, 0, [1, 1, 1]), (0, 3, None), (0, 4, None),
                              (1, 1, [1]), (1, 2, [1]), (2, 1, [1, 1]),
                              (1, 1, [2]), (1, 3, None), (2, 2, None)]:
                S = Super(p, m, n, N)
                chain(f"p={p} svect({m}|{n};{N or '-'})", S, svect(S), fh)

        line("", fh)
        line("=== h(2r|n) (even Hamiltonian) and its derived algebras ===", fh)
        for p in (3, 5):
            for (m, n, r) in [(2, 0, 1), (2, 1, 1), (2, 2, 1), (0, 3, 0),
                              (0, 4, 0), (2, 3, 1), (4, 0, 2)]:
                S = Super(p, m, n)
                L, _ = poisson(S, 'h', r)
                chain(f"p={p} h({m}|{n})", S, L, fh)

        line("", fh)
        line("=== le(n|n) (odd Hamiltonian / Buttin) ===", fh)
        for p in (3, 5):
            for r in (1, 2):
                S = Super(p, r, r)
                L, _ = poisson(S, 'le', r)
                chain(f"p={p} le({r}|{r})", S, L, fh)

        line("", fh)
        line("=== k(1+2r|n) (contact) ===", fh)
        for p in (3, 5):
            for (r, n) in [(1, 0), (0, 2), (1, 1), (0, 3), (1, 2)]:
                S = Super(p, 1 + 2 * r, n,
                          wt=[2] + [1] * (2 * r) + [1] * n)
                L, _ = contact(S, r)
                chain(f"p={p} k({1 + 2 * r}|{n})", S, L, fh)


if __name__ == "__main__":
    main()
