"""Dimension of the space of Diff^omega-invariant bilinear operators

        T(lambda) x T(mu) --> T(nu)

of a given order on a 2m-dimensional symplectic manifold.

Usage:
    python3 rungen.py --m 2 --lam 2,1 --mu 2 --nu 2,1 --order 2 [--samples 4]
                      [--fdeg 3] [--nterms 3] [--cubics 1] [--seed 1]
"""

import argparse
import os
import pickle
import random
import sys
import time
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend  # noqa: E402

from symp import Symp, _monomials  # noqa: E402
from fiber import Fiber  # noqa: E402
from gen import Problem  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_part(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def rand_field(F, rng, deg, nterms, spread=3):
    """sum_s f_s(x) v_s with sparse random f_s."""
    n = F.n
    mons = [e for d in range(deg + 1) for e in _monomials(n, d)]
    polys = []
    for _ in F.basis:
        p = {}
        for e in rng.sample(mons, min(nterms, len(mons))):
            c = rng.randint(-spread, spread)
            if c:
                p[e] = Fraction(c)
        polys.append(p)
    return F.field_from([Fraction(1)] * len(F.basis), polys)


def cubic_list(S, how_many):
    """Monomial cubic Hamiltonians.  h_1 = S^3(W) is an irreducible sp(2m)-module
    and the ansatz is Sp-covariant, so one weight vector generating S^3(W) --
    x_1^3 -- already gives the full condition; more are a cross-check."""
    n = S.n
    out = []
    for e in _monomials(n, 3):
        out.append({e: Fraction(1)})
    if how_many:
        return out[:how_many]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--lam", default="2")
    ap.add_argument("--mu", default="2")
    ap.add_argument("--nu", default="2")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--fdeg", type=int, default=3)
    ap.add_argument("--nterms", type=int, default=3)
    ap.add_argument("--cubics", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    S = Symp(args.m)
    lam, mu, nu = parse_part(args.lam), parse_part(args.mu), parse_part(args.nu)
    Fl, Fm, Fn = Fiber(S, lam), Fiber(S, mu), Fiber(S, nu)
    t0 = time.time()
    P = Problem(S, Fl, Fm, Fn, args.order)
    print(f"backend {backend()};  m = {args.m}, n = {S.n}")
    print(f"T{lam} (sdim {Fl.dim}) x T{mu} (sdim {Fm.dim}) --> T{nu} (sdim {Fn.dim}), "
          f"order {args.order}")
    print(f"ansatz schemes: {len(P.terms)}  [{time.time()-t0:.1f}s]")
    if not P.terms:
        print("no schemes: the slot count is odd, so there is NO operator of this "
              "order in this configuration")
        return

    rng = random.Random(args.seed)
    cubics = cubic_list(S, args.cubics)
    rows = []
    for s in range(args.samples):
        A = rand_field(Fl, rng, args.fdeg, args.nterms)
        B = rand_field(Fm, rng, args.fdeg, args.nterms)
        rows += P.membership_rows(A, B)
        for f in cubics:
            rows += P.defect_rows(A, B, f)
        print(f"  sample {s+1}: {len(rows)} equations  [{time.time()-t0:.1f}s]")

    ns = nullspace(rows) if rows else [[Fraction(1) if i == j else Fraction(0)
                                        for i in range(len(P.terms))]
                                       for j in range(len(P.terms))]
    print(f"coefficient kernel: {len(ns)} of {len(P.terms)}  [{time.time()-t0:.1f}s]")

    rng2 = random.Random(args.seed + 977)
    probes = [(rand_field(Fl, rng2, max(2, args.order), 2),
               rand_field(Fm, rng2, max(2, args.order), 2))
              for _ in range(2)]
    r, _M = P.operator_rank(ns, probes)
    print(f"DIMENSION of the space of invariant operators of order {args.order}: {r}")
    print(f"(relations among ansatz schemes inside the kernel: {len(ns)-r})")
    print(f"total {time.time()-t0:.1f}s")

    tag = args.tag or f"m{args.m}_{'-'.join(map(str,lam))}_{'-'.join(map(str,mu))}_" \
                      f"{'-'.join(map(str,nu))}_N{args.order}"
    with open(os.path.join(HERE, f"gen_{tag}.pkl"), "wb") as fh:
        pickle.dump({"m": args.m, "lam": lam, "mu": mu, "nu": nu,
                     "order": args.order, "labels": P.labels,
                     "kernel": [[str(c) for c in v] for v in ns],
                     "opdim": r}, fh)
    print(f"saved gen_{tag}.pkl")


if __name__ == "__main__":
    main()
