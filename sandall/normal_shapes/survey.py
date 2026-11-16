#!/usr/bin/env python3
"""m = 4, 5: a randomized survey (exhaustive enumeration stops at m = 3, since
dim Z^2 = 55 already for m = 4).

Samples random closed forms with diagonal (1,0,...,0), keeps the non-degenerate
ones, and tallies the derived series of h(omega) -- an Aut-invariant, so the
number of distinct values is a LOWER bound for the number of normal shapes.

Also prints the invariants of the natural candidates
    omega_0 + dx_1^{(2)},   exp-type,   and the alternating omega_0 itself.
"""
import sys, random, time
from collections import Counter
from nalt2 import *
from run1 import dims, diag_of, show_form


def sample(alg, Z, fix_b, rng):
    while True:
        om = {}
        for b in Z:
            if rng.getrandbits(1):
                addform(om, b)
        d = diag_of(alg, om)
        if d != tuple(fix_b):
            continue
        if nondeg(alg, om):
            return om


def invariant(alg, om):
    h = preserving_algebra(alg, om)
    return tuple(derived_series(alg, h))


def run(m, n=150, seed=20260816):
    rng = random.Random(seed)
    alg = Alg(m, (1,) * m)
    t0 = time.time()
    dO, dZ, dB, dH, Z = dims(alg)
    print("=" * 78)
    print("m = %d, u = 1 : dim O = %d, dim vect = %d, dim Omega^2 = %d, "
          "dim Z^2 = %d, dim H^2 = %d" % (m, alg.dim, m * alg.dim, dO, dZ, dH))
    fix_b = tuple([1] + [0] * (m - 1))

    # the natural candidates.  For odd m the standard shape is
    # dx_1^{(2)} + dx_2 dx_3 + dx_4 dx_5 + ... (Pi_m, antidiagonal), for even m
    # it is the alternating dx_1 dx_{1+m/2} + ... , with or without dx_1^{(2)}.
    if m % 2 == 0:
        half = m // 2
        om0 = {('a', i, i + half): alg.one for i in range(half)}
        base = "omega_0 = sum dx_i dx_{i+%d}" % half
    else:
        om0 = {('a', 1 + 2 * i, 2 + 2 * i): alg.one for i in range((m - 1) // 2)}
        base = "omega_0 = dx_2dx_3 + dx_4dx_5 + ..."
    cands = [("%s (alternating)" % base, dict(om0)),
             ("the same + dx_1^{(2)}", dict(list(om0.items()) +
                                            [(('b', 0), alg.one)]))]
    e = alg.one ^ alg.gen(m - 1)              # exp(x_m) = 1 + x_m at p = 2, u = 1
    omx = {k: alg.mul(e, v) for k, v in om0.items()}
    cands.append(("exp(x_%d) times it + dx_1^{(2)}" % m,
                  dict(list(omx.items()) + [(('b', 0), alg.one)])))
    for name, om in cands:
        ok = nondeg(alg, om)
        cl = not d2(alg, om)
        print("  %-32s closed: %-5s non-degenerate: %-5s  dim h, h^(1), ... = %s"
              % (name, cl, ok, list(invariant(alg, om)) if cl and ok else "-"))

    tally = Counter()
    for i in range(n):
        om = sample(alg, Z, fix_b, rng)
        tally[invariant(alg, om)] += 1
    print("  %d random non-degenerate non-alternating closed forms -> %d "
          "distinct invariants:" % (n, len(tally)))
    for inv, c in tally.most_common():
        print("      %-30s  %d samples" % (list(inv), c))
    print("  [%.1f s]" % (time.time() - t0))


if __name__ == "__main__":
    args = sys.argv[1:]
    m = int(args[0]) if args else 4
    n = int(args[1]) if len(args) > 1 else 150
    run(m, n)
