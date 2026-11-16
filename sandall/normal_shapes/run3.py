#!/usr/bin/env python3
"""m = 3, u = 1: the full Aut-orbit census of the closed non-degenerate
NON-ALTERNATING 2-forms over O(3;1), p = 2.

By the reduction lemma the diagonal of a closed form is constant and a linear
change of variables brings a non-zero one to (1,0,0); so we enumerate
    omega = dx_1^{(2)} + (closed alternating form)
and act by the stabilizer of the diagonal (1,0,0).

phi^* is F_2-LINEAR on Omega^2, so each generator is precomputed as its matrix
(a list of images of the basis bits) and applied by XOR-ing.

Output is written to run3.log in this directory, incrementally.
"""
import sys, time, random
from nalt2 import *
from run1 import (dims, enumerate_forms, key_of, form_of, show_form,
                  stab_gens, diag_of)

LOG = None                       # opened by main(); importing must not truncate


def out(s):
    print(s)
    if LOG is not None:
        LOG.write(s + "\n")


def linearize(alg, phi, b2):
    """matrix of phi^* on Omega^2, as images of the single-bit basis forms."""
    img = []
    nb = len(b2) * alg.dim
    for s in range(nb):
        k, t = divmod(s, alg.dim)
        om = {b2[k]: 1 << t}
        img.append(key_of(alg, pullback(alg, phi, om), b2))
    return img


def apply_lin(img, key):
    r = 0
    k = key
    while k:
        lo = k & -k
        s = lo.bit_length() - 1
        k ^= lo
        r ^= img[s]
    return r


def main(m=3):
    global LOG
    LOG = open("run3.log", "w", buffering=1)
    alg = Alg(m, (1,) * m)
    b2 = g2_basis(m)
    t0 = time.time()
    dO, dZ, dB, dH, Z = dims(alg)
    out("m = %d, u = 1 : dim O = %d, dim Omega^2 = %d, dim Z^2 = %d, "
        "dim B^2 = %d, dim H^2 = %d" % (m, alg.dim, dO, dZ, dB, dH))

    fix_b = tuple([1] + [0] * (m - 1))
    forms = enumerate_forms(alg, Z, fix_b=fix_b)
    out("closed forms with diagonal %s : %d" % (str(fix_b), len(forms)))

    nd = [om for om in forms if nondeg(alg, om)]
    out("of them non-degenerate : %d" % len(nd))

    # structural check: for m odd, det(A + E_11) = det of A with row/col 1
    # deleted; for m = 3 that is a_{23}^2, so non-degeneracy <=> a_{23}(0) != 0
    if m == 3:
        bad = sum(1 for om in forms
                  if nondeg(alg, om) != (alg.const_term(om.get(('a', 1, 2), 0)) == 1))
        out("check  non-degenerate <=> a_23 has non-zero constant term : "
            "%d mismatches (must be 0)" % bad)

    gens = stab_gens(alg, fix_b)
    out("|generating set of the stabilizer of the diagonal| = %d" % len(gens))

    mats = [linearize(alg, g, b2) for g in gens]
    # verify the linearization against the symbolic pullback
    bad = 0
    for _ in range(200):
        om = random.choice(nd)
        gi = random.randrange(len(gens))
        if apply_lin(mats[gi], key_of(alg, om, b2)) != \
                key_of(alg, pullback(alg, gens[gi], om), b2):
            bad += 1
    out("linearized pullback vs symbolic pullback : %d mismatches "
        "(must be 0)" % bad)

    todo = {key_of(alg, om, b2) for om in nd}
    out("orbit computation over %d forms with %d generators ..." % (len(todo),
                                                                    len(mats)))
    reps = []
    while todo:
        start = min(todo)
        orb = {start}
        stack = [start]
        while stack:
            k = stack.pop()
            for img in mats:
                k2 = apply_lin(img, k)
                if k2 not in orb:
                    orb.add(k2)
                    stack.append(k2)
        reps.append((start, len(orb)))
        todo -= orb
        out("  ... %d orbits so far, %d forms left [%.1f s]"
            % (len(reps), len(todo), time.time() - t0))

    out("")
    out("ORBITS: %d" % len(reps))
    for k, size in sorted(reps, key=lambda r: -r[1]):
        om = form_of(alg, k, b2)
        h = preserving_algebra(alg, om)
        ds = derived_series(alg, h)
        out("  |orbit| = %-7d omega = %s" % (size, show_form(alg, om)))
        out("     dim h, h^(1), h^(2), ... = %s   (dim vect(%d;1) = %d)"
            % (ds, m, m * alg.dim))
    out("[total %.1f s]" % (time.time() - t0))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
