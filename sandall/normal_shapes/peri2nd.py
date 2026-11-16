#!/usr/bin/env python3
"""The odd forms of the 2nd type: is cohomology a complete invariant?

The question Section 12 leaves open.  Proposition "Ph" of the text says that for
EVERY non-degenerate closed 2-form over O(n|n;u) the preserving algebra is
Pi^{p(omega)+1} Z^1, whose superdimension does not depend on omega.  So a form of
the 2nd type, if there is one, must give an algebra of the SAME superdimension as
the standard one; the smaller algebras of the transported construction therefore
prove only that the twisted objects are not forms over O(n|n;u) at all.

What settles the question in a given case is the census: the orbits of Aut O on
the non-degenerate closed odd 2-forms, against the classes cohomology sees.  Here
the census is taken exhaustively for n = 1 and u = One, for p = 3 (where H^2_odd
is 1-dimensional, so 2 classes are predicted) and for p = 5, 7 (where H^2_odd = 0,
so 1 class is predicted).

Usage: python3 peri2nd.py
"""
import sys, os, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))

from peri import (Omega, omega0, omegaXi, is_closed, is_exact, nondegenerate,   # noqa
                  cohomology, vec, rank_p, nullspace_p)
from perirun import pullback                                                    # noqa


def closed_odd_2forms_basis(alg):
    """a basis of Z^2_odd, as coordinate vectors on the odd 2-form monomials."""
    B2 = [m for m in alg.basis(2) if alg.parity(m) == 1]
    B3 = alg.basis(3)
    i3 = {m: i for i, m in enumerate(B3)}
    d = alg.d()
    rows = [vec(alg, d({m: 1}), i3) for m in B2]        # row per basis form
    # we want {v | sum_i v_i d(m_i) = 0}: the equations are the COLUMNS
    cols = [[rows[j][i] for j in range(len(rows))] for i in range(len(B3))]
    ns = nullspace_p(cols, alg.p)                        # rows = coefficients
    return B2, ns


def form_of(alg, B2, v):
    out = alg.zero()
    for c, m in zip(v, B2):
        if c % alg.p:
            out = alg.add(out, alg.smul(c, {m: 1}))
    return out


def key_of(alg, B2, om):
    i2 = {m: i for i, m in enumerate(B2)}
    return tuple(vec(alg, om, i2))


def generators_n1(alg):
    """a generating set of Aut O(1|1;One): the two scalings and the elementary
    substitutions x -> x + x^(k), xi -> xi + x^(k) xi."""
    p = alg.p
    assert alg.n == 1 and alg.u == (1,)
    x, xi = alg.q(0, 1), alg.xi(0)
    gens = []
    for a in range(2, p):                       # x -> a x
        gens.append(([alg.smul(a, x)], [xi]))
    for b in range(2, p):                       # xi -> b xi
        gens.append(([x], [alg.smul(b, xi)]))
    for k in range(2, p):                       # x -> x + x^(k)
        gens.append(([alg.add(x, alg.q(0, k))], [xi]))
    for k in range(1, p):                       # xi -> xi + x^(k) xi
        gens.append(([x], [alg.add(xi, alg.mul(alg.q(0, k), xi))]))
    return gens


def automorphisms_n1(alg):
    """all of Aut O(1|1;One): x -> a1 x + a2 x^(2), xi -> b0 xi + b1 x xi +
    b2 x^(2) xi, with a1, b0 invertible."""
    p = alg.p
    assert alg.n == 1 and alg.u == (1,)
    xs = [alg.q(0, k) for k in range(1, p)]              # x, x^(2), ..., x^(p-1)
    xis = [alg.mul(alg.q(0, k), alg.xi(0)) for k in range(p)]
    out = []
    for a in itertools.product(range(p), repeat=p - 1):
        if a[0] == 0:
            continue
        F = alg.zero()
        for c, y in zip(a, xs):
            F = alg.add(F, alg.smul(c, y))
        for b in itertools.product(range(p), repeat=p):
            if b[0] == 0:
                continue
            G = alg.zero()
            for c, y in zip(b, xis):
                G = alg.add(G, alg.smul(c, y))
            out.append(([F], [G]))
    return out


def orbits_under(alg, B2, forms, group):
    """the partition of `forms` into orbits of the given set of substitutions,
    closed up (so a generating set is enough)."""
    todo = set(forms)
    orbits = []
    while todo:
        v0 = next(iter(todo))
        front = [v0]
        orb = {v0}
        while front:
            v = front.pop()
            om = form_of(alg, B2, list(v))
            for F, G in group:
                k = key_of(alg, B2, pullback(alg, F, G, om))
                assert k in todo or k in orb, "the orbit left the set"
                if k not in orb:
                    orb.add(k)
                    front.append(k)
        orbits.append((v0, orb))
        todo -= orb
    return orbits


def census(p, log, full=False):
    alg = Omega(1, p, (1,))
    B2, ns = closed_odd_2forms_basis(alg)
    log(f"\np = {p}, O(1|1;One): dim Omega^2_odd = {len(B2)}, "
        f"dim Z^2_odd = {len(ns)}, so {p**len(ns)} closed odd 2-forms")

    h2 = cohomology(alg, 2, par=1)[0]
    log(f"  dim H^2_odd = {h2}")

    forms = []
    for coef in itertools.product(range(p), repeat=len(ns)):
        v = [0] * len(B2)
        for c, row in zip(coef, ns):
            if c:
                v = [(v[i] + c * row[i]) % p for i in range(len(B2))]
        om = form_of(alg, B2, v)
        if om and nondegenerate(alg, om):
            forms.append(tuple(v))
    log(f"  non-degenerate among them: {len(forms)}")

    gens = generators_n1(alg)
    log(f"  {len(gens)} generators of Aut O(1|1;One)")
    orbits = orbits_under(alg, B2, forms, gens)
    log(f"  ORBITS: {len(orbits)}  sizes {sorted(len(o) for _, o in orbits)}")
    for v0, orb in orbits:
        om = form_of(alg, B2, list(v0))
        log(f"    size {len(orb):5d}   exact: {is_exact(alg, om, 2)}   "
            f"contains omega_0: {key_of(alg, B2, omega0(alg)) in orb}"
            + (f"   contains omega_Xi: "
               f"{key_of(alg, B2, omegaXi(alg)) in orb}" if p == 3 else ""))

    if full:
        auts = automorphisms_n1(alg)
        orb2 = orbits_under(alg, B2, forms, auts)
        log(f"  CONTROL, the whole group ({len(auts)} elements): "
            f"{len(orb2)} orbits, sizes {sorted(len(o) for _, o in orb2)} --- "
            f"{'same' if len(orb2) == len(orbits) else 'DIFFERENT'}")
    return len(orbits)


def main():
    fh = open(os.path.join(HERE, "peri2nd.log"), "w")

    def log(s=""):
        print(s)
        fh.write(s + "\n")
        fh.flush()

    log("The census of the non-degenerate closed ODD 2-forms over O(1|1;One),")
    log("exhaustive, and the orbits of Aut O on them.")
    res = {}
    res[3] = census(3, log, full=True)
    res[5] = census(5, log)
    log("\nSUMMARY")
    for p, k in sorted(res.items()):
        pred = 2 if p == 3 else 1
        log(f"  p = {p}: {k} orbits, cohomology predicts {pred} --- "
            f"{'AGREE' if k == pred else 'DISAGREE'}")
    fh.close()


if __name__ == "__main__":
    main()
