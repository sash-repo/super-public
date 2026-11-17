#!/usr/bin/env python3
"""Statement 5.9 for p = 2: Der L against N_{Der O}(L) for vect(m|n;N) and
svect(m|n;N), and the squaring condition D(x^2) = [x, D(x)] for odd x.

For each algebra prints dim L, dim N_{Der O}(L), dim Der L computed from the
bracket alone and with the squaring condition imposed, the number of outer
derivations and the gap dim Der L - dim N_{Der O}(L)."""
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
from dervect import (Super, Echelon, svect, check_closed, normalizer,  # noqa: E402
                     ad_dim, der_dim)
from p2 import der_dim_p2                                              # noqa: E402

CASES = [(1, 1, [1]), (1, 1, [2]), (1, 2, [1]), (2, 1, [1, 1]), (0, 2, None),
         (0, 3, None), (0, 4, None), (1, 1, [3]), (2, 2, None), (1, 3, None)]


def one(tag, S, L):
    t = time.time()
    E = Echelon([S.vec(X) for X in L], S.p)
    B = [S.unvec(r) for r in E.rows]
    ok, _ = check_closed(S, B)
    dN = Echelon([S.vec(X) for X in normalizer(S, B)], S.p).dim
    dD = der_dim(S, B)
    dD2 = der_dim_p2(S, B)
    dad, _ = ad_dim(S, B)
    print(f"{tag}: dimL={len(B)} closed={ok} dimN={dN} Der(bracket)={dD} "
          f"Der(+squaring)={dD2} out={dD2 - dad} gap={dD2 - dN} "
          f"[{time.time() - t:.0f}s]", flush=True)


def main():
    for (m, n, N) in CASES:
        S = Super(2, m, n, N)
        one(f"p=2 vect({m}|{n};{N})", S, S.vect())
        one(f"p=2 svect({m}|{n};{N})", S, svect(S))


if __name__ == "__main__":
    main()
