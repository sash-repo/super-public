#!/usr/bin/env python3
r"""Which a_k close a given Lie (super)algebra?  Lemmas 2.1-2.3 and Problem 2.2.1.

For a matrix Lie superalgebra g with a homogeneous basis, a_k is super-alternating:
it is skew in the even arguments and symmetric in the odd ones, so it is determined
by its values on multisets

    (a subset of the even basis) + (a multiset of the odd basis).

We run over all of them -- this is a complete test, not a sample -- and report

    CLOSED  : a_k(g,...,g) is contained in g,
    NONZERO : a_k does not vanish identically on g.
"""

import sys
from itertools import combinations, combinations_with_replacement

from sant import a_matrix, in_span, flat, miszero
import salg


def multisets(basis, k):
    ev = [i for i, (_, p) in enumerate(basis) if p == 0]
    od = [i for i, (_, p) in enumerate(basis) if p == 1]
    for j in range(min(k, len(ev)) + 1):
        for E in combinations(ev, j):
            for O in combinations_with_replacement(od, k - j):
                yield list(E) + list(O)


def survey(name, basis, pv, ks, limit=None, verbose=False):
    vecs = [flat(M) for M, _ in basis]
    res = {}
    for k in ks:
        closed, nonzero, witness = True, False, None
        cnt = 0
        for idx in multisets(basis, k):
            cnt += 1
            if limit and cnt > limit:
                break
            mats = [basis[i][0] for i in idx]
            par = [basis[i][1] for i in idx]
            A = a_matrix(mats, par)
            if not miszero(A):
                nonzero = True
                if closed and not in_span(flat(A), vecs):
                    closed = False
                    witness = idx
                    break
        res[k] = (closed, nonzero, cnt, witness)
        tag = ("CLOSED " if closed else "not closed") + \
              (", nonzero" if nonzero else ", VANISHES identically")
        print(f"  {name:12s} a_{k:<3d} {tag}"
              f"   ({cnt} multisets{' [truncated]' if limit and cnt > limit else ''})")
        sys.stdout.flush()
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "sanity"):
        print("Lemma 2.1 (sl closed under a_{2l}), 2.3 (q under all a_k, sq under a_{2k}):")
        for nm, (b, pv) in [("sl(2|1)", salg.sl(2, 1)), ("sl(1|1)", salg.sl(1, 1)),
                            ("q(2)", salg.q(2)), ("sq(2)", salg.sq(2))]:
            survey(nm, b, pv, [2, 3, 4, 5])

    if which in ("all", "pe"):
        print("\nLemma 2.2 and Problem 2.2.1 (pe / spe):")
        for nm, (b, pv) in [("pe(1)", salg.pe(1)), ("spe(1)", salg.spe(1)),
                            ("pe(2)", salg.pe(2)), ("spe(2)", salg.spe(2)),
                            ("pe(3)", salg.pe(3)), ("spe(3)", salg.spe(3))]:
            survey(nm, b, pv, [2, 3, 4, 5, 6])

    if which in ("all", "osp"):
        print("\nLemma 2.2 (osp):")
        for nm, (b, pv) in [("osp(1|2)", salg.osp(1, 2)), ("osp(2|2)", salg.osp(2, 2)),
                            ("osp(3|2)", salg.osp(3, 2))]:
            survey(nm, b, pv, [2, 3, 4, 5, 6])
