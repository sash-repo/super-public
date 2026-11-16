#!/usr/bin/env python3
"""C) The two brackets compared: is the p = 3 deform a TRUE deform?

The two shapes have preserving algebras of equal superdimension, so the shapes
being inequivalent (their classes in H^2 differ) does not by itself say that the
algebras are non-isomorphic.  Here we compute isomorphism invariants of
    le(n;u)      = Span{H_f} for omega_0,
    le_Xi(n;u)   = Span{H_f} for omega_Xi = omega_0 + xi_1 (dxi_1)^{(2)},
namely the derived series, the lower central series, the center, and the
dimensions of the members --- all as superdimensions.

Usage: python3 peribr.py
"""
import sys, os, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from peri import Omega, omega0, omegaXi, vec, rank_p                       # noqa
from perirun import Ham, partials                                          # noqa
from exactlin import rref                                                  # noqa


def bracket_table(alg, deformed):
    """{f,g} on O(n|n;u) by the closed formula (checked in perirun.py C)."""
    par = partials(alg)
    n = alg.n

    def br(f, g, pf):
        r = alg.zero()
        for i in range(n):
            r = alg.add(r, alg.mul(par[i](f), par[n + i](g)))
            r = alg.add(r, alg.smul((-1) ** pf, alg.mul(par[n + i](f), par[i](g))))
        if deformed:
            r = alg.add(r, alg.smul(-((-1) ** pf),
                                    alg.mul(alg.xi(0),
                                            alg.mul(par[0](f), par[0](g)))))
        return r
    return br


def sdim_span(alg, elts, idx):
    """superdimension of the span of a set of functions (parity graded)."""
    ev = [vec(alg, x, idx) for x in elts
          if all(alg.parity(m) == 0 for m in x)]
    od = [vec(alg, x, idx) for x in elts
          if all(alg.parity(m) == 1 for m in x)]
    return rank_p(ev, alg.p), rank_p(od, alg.p)


def span_vectors(alg, elts, idx):
    return [vec(alg, x, idx) for x in elts]


def main():
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "peribr.log"), "w")

    def out(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()

    out("Invariants of the two brackets (parity of H_f is p(f)+1, so the")
    out("superdimensions below are those of the algebra Span{H_f} = O/K with")
    out("the shifted parity).")
    for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 3, (2,))]:
        alg = Omega(n, p, u)
        fb = alg.fun_basis()
        idx = {m: i for i, m in enumerate(fb)}
        const = (tuple([0] * alg.nev), tuple([0] * alg.nod))
        gens = [{m: 1} for m in fb if m != const]
        out(f"\n  n={n} p={p} u={u}:  dim O = {len(fb)}, "
            f"dim g = dim O/K = {len(gens)}")
        for name, dfm in [("omega_0 ", False), ("omega_Xi", True)]:
            br = bracket_table(alg, dfm)
            # derived series
            cur = gens
            out(f"    {name}:")
            desc = []
            for step in range(4):
                prods = []
                for f in cur:
                    pf = set(alg.parity(m) for m in f)
                    pf = pf.pop() if len(pf) == 1 else 0
                    for g in cur:
                        prods.append(br(f, g, pf))
                V = [v for v in span_vectors(alg, prods, idx) if any(v)]
                r = rank_p(V, alg.p) if V else 0
                # a basis of the new space, as functions
                R, piv = rref(V, alg.p) if V else ([], [])
                nxt = []
                for row in R[:r]:
                    nxt.append({fb[j]: row[j] % p for j in range(len(fb))
                                if row[j] % p})
                pe = sum(1 for x in nxt if all(alg.parity(m) == 0 for m in x))
                # superdimension of the derived algebra
                se, so = sdim_span(alg, nxt, idx)
                desc.append((r, se, so))
                out(f"      g^({step + 1}): dim = {r}  sdim (as functions) "
                    f"= ({se}|{so})")
                if r == 0 or (step and desc[-1] == desc[-2]):
                    break
                cur = nxt
            # the center of g
            cen = []
            allb = gens
            M = []
            for f in allb:
                pf = set(alg.parity(m) for m in f)
                pf = pf.pop() if len(pf) == 1 else 0
                row = []
                for g in allb:
                    row += vec(alg, br(f, g, pf), idx)
                M.append(row)
            from exactlin import nullspace
            cols = [[M[i][k] for i in range(len(M))] for k in range(len(M[0]))]
            ns = nullspace(cols, p)
            out(f"      center: dim = {len(ns)}")
    log.close()


if __name__ == "__main__":
    main()
