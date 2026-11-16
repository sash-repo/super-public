#!/usr/bin/env python3
"""D) The non-degeneracy criterion for ODD 2-forms.

Claim: an odd 2-form omega over O(n|n;u) is non-degenerate (i.e. D -> iota_D
omega is bijective) if and only if the constant term of its q-xi block
    B_{ij} = omega(d/dq_i, d/dxi_j)
is an invertible n x n matrix over K.  The two diagonal blocks are irrelevant:
being odd, their entries lie in O_odd, hence in the maximal ideal.

Tested on random odd 2-forms, closed and not.

Usage: python3 perindeg.py
"""
import sys, os, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from peri import Omega, nondegenerate, is_closed, vfield_basis, rank_p     # noqa


def qxi_block(alg, om):
    """the matrix B_{ij} = iota_{d/dxi_j} iota_{d/dq_i} omega, as functions."""
    n = alg.n
    ders = []
    for i in range(n):
        fs = [alg.zero()] * n; fs = list(fs); fs[i] = alg.one()
        ders.append((tuple(fs), tuple([alg.zero()] * n), 0))
    for i in range(n):
        gs = [alg.zero()] * n; gs = list(gs); gs[i] = alg.one()
        ders.append((tuple([alg.zero()] * n), tuple(gs), 1))
    B = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            B[i][j] = alg.iota(ders[n + j])(alg.iota(ders[i])(om))
    return B


def const_term(alg, f):
    one = (tuple([0] * alg.nev), tuple([0] * alg.nod))
    return f.get(one, 0) % alg.p


def main():
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "perindeg.log"), "w")

    def out(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()

    out("D) the non-degeneracy criterion for odd 2-forms")
    bad = 0
    for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 5, (1,)), (2, 5, (1, 1)),
                      (1, 7, (1,)), (1, 3, (2,)), (3, 3, (1, 1, 1))]:
        alg = Omega(n, p, u)
        rng = random.Random(23)
        B2 = alg.basis(2, par=1)
        agree = tot = ncl = nnd = 0
        from peri import omega0
        om0 = omega0(alg)
        for trial in range(240 if n < 3 else 60):
            # half of the sample: perturbations of omega_0, so that the
            # non-degenerate forms are well represented for n >= 2 as well
            om = om0 if trial % 2 else alg.zero()
            for _ in range(rng.randrange(1, 8)):
                om = alg.add(om, alg.smul(rng.randrange(1, p),
                                          {rng.choice(B2): 1}))
            if not om:
                continue
            tot += 1
            nd = nondegenerate(alg, om)
            M = [[const_term(alg, x) for x in row] for row in qxi_block(alg, om)]
            crit = rank_p(M, p) == n
            if nd == crit:
                agree += 1
            if is_closed(alg, om):
                ncl += 1
            if nd:
                nnd += 1
        out(f"    n={n} p={p} u={u}: criterion agrees on {agree}/{tot} random odd "
            f"2-forms ({nnd} non-degenerate, {ncl} closed)")
        if agree != tot:
            bad += 1
    out(f"\n  mismatches: {bad}")
    log.close()


if __name__ == "__main__":
    main()
