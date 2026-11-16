#!/usr/bin/env python3
"""Smoke tests for peri.py: the algebra, d, the Cartan formula, omega_0."""
import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peri import *          # noqa
from peri import Omega      # noqa

fails = 0


def check(name, ok):
    global fails
    if not ok:
        fails += 1
    print(("  ok   " if ok else "  FAIL ") + name)


def rnd_elt(alg, deg, rng, terms=3):
    B = alg.basis(deg)
    out = {}
    for _ in range(terms):
        m = rng.choice(B)
        out = alg.add(out, alg.smul(rng.randrange(1, alg.p), {m: 1}))
    return out


for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 5, (1,)), (2, 5, (1, 1)), (1, 3, (2,))]:
    alg = Omega(n, p, u)
    rng = random.Random(17)
    print(f"n={n} p={p} u={u}  dim O = {len(alg.fun_basis())} "
          f"dim Omega^1 = {len(alg.basis(1))} dim Omega^2 = {len(alg.basis(2))}")
    d = alg.d()

    # associativity + supercommutativity on random monomials
    ok = True
    B = alg.basis(0) + alg.basis(1) + alg.basis(2)
    for _ in range(200):
        a, b, c = ({rng.choice(B): 1} for _ in range(3))
        if alg.mul(alg.mul(a, b), c) != alg.mul(a, alg.mul(b, c)):
            ok = False
        pa = alg.parity(list(a)[0]); pb = alg.parity(list(b)[0])
        if alg.mul(a, b) != alg.smul((-1) ** (pa * pb), alg.mul(b, a)):
            ok = False
    check("associative + supercommutative", ok)

    # d^2 = 0 and d is an odd derivation
    ok = True
    for _ in range(200):
        x = rnd_elt(alg, rng.choice([0, 1, 2]), rng)
        if d(d(x)):
            ok = False
    check("d^2 = 0", ok)
    ok = True
    for _ in range(200):
        a = {rng.choice(B): 1}
        b = {rng.choice(B): 1}
        pa = alg.parity(list(a)[0])
        lhs = d(alg.mul(a, b))
        rhs = alg.add(alg.mul(d(a), b), alg.smul((-1) ** pa, alg.mul(a, d(b))))
        if lhs != rhs:
            ok = False
    check("d is an odd derivation", ok)

    # Cartan formula L_D = iota_D d + (-1)^{p(D)} d iota_D = [iota_D, d]
    vfs = vfield_basis(alg)
    ok = True
    for _ in range(120):
        D = rng.choice(vfs)
        x = rnd_elt(alg, rng.choice([0, 1, 2]), rng)
        iD = alg.iota(D)
        lhs = alg.lie(D)(x)
        rhs = alg.add(iD(d(x)), alg.smul((-1) ** D[2], d(iD(x))))
        if lhs != rhs:
            ok = False
            break
    check("Cartan formula L_D = [iota_D, d]", ok)

    # L_D commutes with d in the super sense: L_D d = (-1)^{p(D)} d L_D
    ok = True
    for _ in range(120):
        D = rng.choice(vfs)
        x = rnd_elt(alg, rng.choice([0, 1, 2]), rng)
        LD = alg.lie(D)
        if LD(d(x)) != alg.smul((-1) ** D[2], d(LD(x))):
            ok = False
            break
    check("L_D super-commutes with d", ok)

    # L_D is a derivation of parity p(D)
    ok = True
    for _ in range(120):
        D = rng.choice(vfs)
        a = {rng.choice(B): 1}
        b = {rng.choice(B): 1}
        pa = alg.parity(list(a)[0])
        LD = alg.lie(D)
        if LD(alg.mul(a, b)) != alg.add(alg.mul(LD(a), b),
                                        alg.smul((-1) ** (D[2] * pa), alg.mul(a, LD(b)))):
            ok = False
            break
    check("L_D is a derivation of parity p(D)", ok)

    # omega_0 is closed, non-degenerate, and exact
    om = omega0(alg)
    check("omega_0 closed", is_closed(alg, om))
    check("omega_0 non-degenerate", nondegenerate(alg, om))
    check("omega_0 exact", is_exact(alg, om))
    a0 = alpha0(alg)
    check("d(alpha_0) = +- omega_0",
          d(a0) == om or d(a0) == alg.smul(-1, om))
    print()

print("FAILURES:", fails)
