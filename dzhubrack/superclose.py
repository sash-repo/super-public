#!/usr/bin/env python3
r"""Problem 2.2.1 of arXiv:1906.01380 (and a re-run of Lemmas 2.1-2.3).

For every parity pattern P in {0,1}^k we feed a_k random homogeneous elements of
g of those parities and ask (i) whether the result is 0, (ii) whether it lies in
g.  Since a_k is multilinear in each slot, a random argument tests the whole
slot at once (Schwartz-Zippel); the trials are independent, and the answers are
cross-checked against the exhaustive Fraction engine of closure.py for pe(1),
spe(1), pe(2), spe(2).

Output per k:  the set of parity patterns on which a_k leaves g, if any.
"""

import sys
import numpy as np

from expo import a_k_dp, span_rank, P
import salg


def to_np(basis):
    N = len(basis[0][0])
    out = []
    for M, p in basis:
        A = np.array([[int(x) % P for x in row] for row in M], dtype=np.int64)
        out.append((A, p))
    return out, N


def rand_hom(basis, parity, rng):
    sel = [M for M, p in basis if p == parity]
    if not sel:
        return None
    c = rng.integers(0, P, size=len(sel))
    A = np.zeros_like(sel[0])
    for t, M in enumerate(sel):
        A = (A + int(c[t]) * M) % P
    return A


def survey(name, basis, ks, trials=4, seed=7, str_report=False):
    B, N = to_np(basis)
    gvecs = [M.reshape(-1) for M, _ in B]
    rk = span_rank(gvecs)
    rng = np.random.default_rng(seed)
    for k in ks:
        bad, nonzero = [], False
        for bits in range(1 << k):
            par = [(bits >> i) & 1 for i in range(k)]
            if any(rand_hom(B, p, rng) is None for p in set(par)):
                continue
            for _ in range(trials):
                mats = [rand_hom(B, p, rng) for p in par]
                A = a_k_dp(mats, N, par=par) % P
                if A.any():
                    nonzero = True
                    if span_rank(gvecs + [A.reshape(-1)]) > rk:
                        bad.append(tuple(par))
                        break
        if not nonzero:
            v = "vanishes identically"
        elif not bad:
            v = "CLOSED (all parity patterns)"
        else:
            v = f"leaves g on {len(bad)}/{1 << k} parity patterns, e.g. {bad[0]}"
        print(f"  {name:9s} a_{k:<3d} {v}")
        sys.stdout.flush()


if __name__ == "__main__":
    todo = sys.argv[1:] or ["sanity", "pe", "osp"]
    if "sanity" in todo:
        print("cross-check of the engine (Lemmas 2.1 and 2.3):")
        for nm, (b, pv) in [("sl(2|1)", salg.sl(2, 1)), ("sl(2|2)", salg.sl(2, 2)),
                            ("q(3)", salg.q(3)), ("sq(3)", salg.sq(3))]:
            survey(nm, b, range(2, 7))
    if "pe" in todo:
        print("\nLemma 2.2 and Problem 2.2.1:")
        for nm, (b, pv) in [("pe(1)", salg.pe(1)), ("spe(1)", salg.spe(1)),
                            ("pe(2)", salg.pe(2)), ("spe(2)", salg.spe(2)),
                            ("pe(3)", salg.pe(3)), ("spe(3)", salg.spe(3)),
                            ("pe(4)", salg.pe(4)), ("spe(4)", salg.spe(4))]:
            survey(nm, b, range(2, 8))
    if "osp" in todo:
        print("\nLemma 2.2 (osp):")
        for nm, (b, pv) in [("osp(1|2)", salg.osp(1, 2)), ("osp(2|2)", salg.osp(2, 2)),
                            ("osp(3|2)", salg.osp(3, 2)), ("osp(1|4)", salg.osp(1, 4))]:
            survey(nm, b, range(2, 8))
