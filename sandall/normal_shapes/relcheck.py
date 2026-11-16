#!/usr/bin/env python3
"""What does the non-alternating term do to the classification?

Over O(m;1), p = 2, with the diagonal normalized to (1,0,...,0), compare

  (a) the orbits, under the FULL automorphism group of O(m;1), of the closed
      ALTERNATING forms omega' for which dx_1^{(2)} + omega' is non-degenerate;
  (b) the orbits, under the same group, of the NON-ALTERNATING forms
      omega = dx_1^{(2)} + omega' themselves.

Any phi carrying one omega to another fixes the (constant) diagonal, so (b) is
computed with the stabilizer of the diagonal.  The two counts need not agree:
a phi in that stabilizer pulls dx_1^{(2)} back to dx_1^{(2)} PLUS alternating
cross terms, so it moves omega' by more than its linear action in (a).

Prints for each orbit of (b) the derived series of h(omega) and whether the
last term is simple.
"""
import sys, itertools
from nalt2 import *
from run1 import dims, enumerate_forms, key_of, form_of, show_form, stab_gens


def full_gens(alg):
    """generators of Aut O(m;1): the linear ones and x_i -> x_i + h, h in m^2."""
    m = alg.m
    quad = [1 << k for k, a in enumerate(alg.mons) if sum(a) >= 2]
    gens = []
    lin = []
    for bits in range(1 << m):
        v = 0
        for i in range(m):
            if (bits >> i) & 1:
                v ^= alg.gen(i)
        lin.append((bits, v))
    for phi in itertools.product(lin, repeat=m):
        L = [[(p[0] >> k) & 1 for k in range(m)] for p in phi]
        if rank(L, p=2) == m:
            gens.append([p[1] for p in phi])
    for i in range(m):
        for h in quad:
            g = [alg.gen(j) for j in range(m)]
            g[i] ^= h
            gens.append(g)
    return gens


def orbits_under(alg, keys, gens, b2):
    from run3 import linearize, apply_lin
    mats = [linearize(alg, g, b2) for g in gens]
    todo = set(keys)
    reps = []
    while todo:
        start = min(todo)
        orb, stack = {start}, [start]
        while stack:
            k = stack.pop()
            for img in mats:
                k2 = apply_lin(img, k)
                if k2 not in orb:
                    orb.add(k2)
                    stack.append(k2)
        reps.append((start, len(orb)))
        todo -= orb
    return reps


def run(m):
    alg = Alg(m, (1,) * m)
    b2 = g2_basis(m)
    _, _, _, _, Z = dims(alg)
    alt = enumerate_forms(alg, Z, fix_b=(0,) * m)
    dg = ('b', 0)

    good = [om for om in alt
            if nondeg(alg, dict(list(om.items()) + [(dg, alg.one)]))]
    print("=" * 78)
    print("m = %d : closed alternating forms %d; of them with dx_1^{(2)} + "
          "omega' non-degenerate: %d" % (m, len(alt), len(good)))

    full = full_gens(alg)
    stab = stab_gens(alg, tuple([1] + [0] * (m - 1)))
    print("  |generators of Aut O(%d;1)| = %d, |generators of the stabilizer "
          "of the diagonal| = %d" % (m, len(full), len(stab)))

    # (a) the alternating forms alone, under the full group
    oa = orbits_under(alg, [key_of(alg, om, b2) for om in good], full, b2)
    print("  (a) orbits of omega' alone, under the full Aut : %d" % len(oa))
    for k, s in sorted(oa, key=lambda r: -r[1]):
        print("        |orbit| = %-6d omega' = %s"
              % (s, show_form(alg, form_of(alg, k, b2))))

    # (b) the non-alternating forms dx_1^{(2)} + omega'
    keys = [key_of(alg, dict(list(om.items()) + [(dg, alg.one)]), b2)
            for om in good]
    ob = orbits_under(alg, keys, stab, b2)
    print("  (b) orbits of dx_1^{(2)} + omega' : %d" % len(ob))
    for k, s in sorted(ob, key=lambda r: -r[1]):
        om = form_of(alg, k, b2)
        h = preserving_algebra(alg, om)
        ds = derived_series(alg, h)
        last = h
        for _ in range(4):
            nxt = reduce_basis(alg, derived(alg, last))
            if len(nxt) == len(last):
                break
            last = nxt
        simp = is_simple(alg, last) if len(last) <= 12 else "?"
        print("        |orbit| = %-6d omega = %s" % (s, show_form(alg, om)))
        print("           dim h, h^(1), ... = %s ; last term dim %d, simple: %s"
              % (ds, len(last), simp))


if __name__ == "__main__":
    for m in (int(a) for a in (sys.argv[1:] or ["2", "3"])):
        run(m)
