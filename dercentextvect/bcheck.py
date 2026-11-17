#!/usr/bin/env python3
"""The family b_{a,b}(n) = b_lambda(n) inside m(n): the full census.

For every (a:b) in P^1(F_p) -- i.e. every lambda = 2a/(n(a-b)) in P^1 -- this
computes

    dim L,  dim ad L,  dim Der L,  dim N_{Der O}(L),  dim C_{Der O}(L)

and the gap  dim Der L - (dim N - dim C)  by which Skryabin's conclusion fails.
The centralizer matters here, unlike in the census of the four classical series
where it always vanishes: b_{a,a} (lambda = infinity) has a nonzero one.

lambda = 0, i.e. (a:b) = (0:1), is the Buttin superalgebra b(n), the central
extension of le(n) by K M_1 -- see the exact sequence in the text; that is the
member whose gap the odd Laplacian accounts for.
"""
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import nullspace                                     # noqa: E402
from dervect import (Echelon, check_closed, normalizer, der_dim,   # noqa: E402
                     ad_dim, derived)
import mvect                                                       # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bcheck.log")


def lam(p, n, a, b):
    """lambda = 2a/(n(a-b)) in P^1(F_p)."""
    num, den = (2 * a) % p, (n * (a - b)) % p
    if den == 0:
        return "oo" if num else "0/0"
    return str(num * pow(den, p - 2, p) % p)


def centralizer_dim(S, L):
    rows = []
    for X in L:
        for t in range(S.dimD):
            row = [S.vec(S.bracket({b: 1}, X))[t] for b in S.dbasis]
            if any(x % S.p for x in row):
                rows.append([x % S.p for x in row])
    return len(nullspace(rows, p=S.p)) if rows else S.dimD


def main():
    with open(LOG, "w") as fh:
        def say(s):
            print(s)
            fh.write(s + "\n")
            fh.flush()

        say("p n (a:b) lambda  algebra      dimL centre ad   Der  N    C   gap")
        for p, n in ((3, 1), (5, 1), (3, 2)):
            S = mvect.make(p, n)
            M, fields = mvect.pericontact(S, n)
            todo = [("m", M)]
            for (a, b) in [(1, x) for x in range(p)] + [(0, 1)]:
                todo.append((f"b_({a}:{b}) lam={lam(p, n, a, b)}",
                             mvect.buttin(S, n, a, b, fields)))
            for tag, L in todo:
                for suf, X in ((" ", L), ("^(1)", derived(S, L))):
                    d = Echelon([S.vec(Y) for Y in X], S.p).dim
                    if d == 0:
                        continue
                    t0 = time.time()
                    closed, _ = check_closed(S, X)
                    da, dz = ad_dim(S, X)
                    dD = der_dim(S, X)
                    NN = normalizer(S, X)
                    dN = Echelon([S.vec(Y) for Y in NN], S.p).dim
                    dC = centralizer_dim(S, X)
                    say(f"p={p} n={n} {tag}{suf}: dimL={d} closed={closed} "
                        f"centre={dz} ad={da} Der={dD} N={dN} C={dC} "
                        f"gap={dD - (dN - dC)} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
