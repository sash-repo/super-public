#!/usr/bin/env python3
"""p = 2: the classification of the HOMOGENEOUS closed non-degenerate 2-forms
on O(m|n;1) -- the ones a superspace actually carries.

scensus.py shows that the whole set of closed non-degenerate forms breaks into
thousands of classes under the parity-preserving group.  Almost all of them
consist of forms that are neither even nor odd, and such a form is not a
geometric object on a superspace: it is a sum of two forms of different
parities.  Restricted to the homogeneous forms the classification is small
again, and this script computes it, together with the invariants that separate
the classes:

    * the type of the two diagonal blocks of the constant Gram matrix
      (alternating = Lebedev's Pi_{2k}, or not = his One_r);
    * the class of omega in H^2, up to the induced action of Aut;
    * sdim h(omega).
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S
from scensus import (build, elementary_autos, all_autos, gen_matrix,
                     apply_map, orbits, show, name, diag_type,
                     invertible_table, gen_index, dgen_index)

LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "shomog.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def homogeneous_nondeg(A, zbas, gram_bits, G, par):
    """closed non-degenerate forms of the given parity, as ints."""
    B2 = A.basis(2)
    tab = invertible_table(G)
    # restrict Z^2 to the given parity: keep only the basis vectors whose
    # support is homogeneous of that parity, after splitting each basis vector
    parmask = 0
    for i, mm in enumerate(B2):
        if A.parity(mm) == par:
            parmask |= 1 << i
    zpar = []
    for z in zbas:
        a, b = z & parmask, z & ~parmask
        # a closed form splits into its two parity components, each closed
        if a:
            zpar.append(a)
        if b:
            zpar.append(b)
    # re-extract a basis of the parity-par part of Z^2
    zpar = [z & parmask for z in zpar if z & parmask]
    basis = []
    for z in zpar:
        v = z
        for b in basis:
            hb = b.bit_length() - 1
            if (v >> hb) & 1:
                v ^= b
        if v:
            basis.append(v)
            basis.sort(key=lambda t: -t)
    d = A.d()
    for z in basis:                       # control: still closed
        om = show(A, z, B2)
        assert not d(om), "the parity component of a closed form is not closed"
    zg = []
    for z in basis:
        g = 0
        b = z
        while b:
            i = (b & -b).bit_length() - 1
            g ^= gram_bits[i]
            b &= b - 1
        zg.append(g)
    out = []
    D = len(basis)
    cur, curg = 0, 0
    for i in range(1 << D):
        if i:
            k = (i & -i).bit_length() - 1
            cur ^= basis[k]
            curg ^= zg[k]
        if tab[curg]:
            out.append(cur)
    return out, len(basis)


def block_types(A, om):
    """(alternating?, alternating?) for the x-block and the xi-block of C."""
    _, C = S.nondegenerate(A, om)
    m, n = A.m, A.n
    altx = all(C[a][a] % 2 == 0 for a in range(m))
    alte = all(C[m + a][m + a] % 2 == 0 for a in range(n))
    return (altx, alte)


def h2_classes(A, B2, idx, elems, maps):
    """the class of each form in H^2, and the orbit of that class under Aut."""
    d = A.d()
    # coboundaries B^2 as a bit space
    B1 = A.basis(1)
    i2 = {mm: i for i, mm in enumerate(B2)}
    rows = []
    for mm in B1:
        img = d({mm: 1})
        bits = 0
        for nn, c in img.items():
            if c % 2:
                bits |= 1 << i2[nn]
        if bits:
            rows.append(bits)
    ech = []
    for v in rows:
        w = v
        for b in ech:
            hb = b.bit_length() - 1
            if (w >> hb) & 1:
                w ^= b
        if w:
            ech.append(w)
            ech.sort(key=lambda t: -t)

    def reduce(v):
        w = v
        for b in ech:
            hb = b.bit_length() - 1
            if (w >> hb) & 1:
                w ^= b
        return w
    return reduce


def run(m, n, verify=True):
    A = S.Om(m, n, 2, (1,) * m, dh=2, dxh=2)
    B2, idx, zbas, gram_bits, G = build(A)
    gens = elementary_autos(A, graded=True)
    maps = [gen_matrix(A, phi, B2, idx) for phi in gens]
    red = h2_classes(A, B2, idx, None, maps)
    say("")
    say(f"  O({m}|{n};1), p = 2:  dim Omega^2 = {len(B2)}, dim Z^2 = {len(zbas)}")
    for par, pname in ((0, "EVEN"), (1, "ODD ")):
        el, dz = homogeneous_nondeg(A, zbas, gram_bits, G, par)
        if not el:
            say(f"    {pname} forms: none "
                + ("(an odd form needs m = n)" if par == 1 and m != n else ""))
            continue
        t0 = time.time()
        reps, sizes = orbits(el, maps)
        line = (f"    {pname} forms: {len(el)} of 2^{dz} closed ones, "
                f"{len(reps)} class(es)")
        if verify:
            aut = all_autos(A, True) if (m + n) <= 3 and (m == 0 or n > 0) else []
            if aut and len(aut) * len(el) <= 60_000_000:
                full = [gen_matrix(A, phi, B2, idx) for phi in aut]
                r2, _ = orbits(el, full)
                line += (f"  [full group of {len(aut)}: {len(r2)} -- "
                         f"{'ok' if len(r2)==len(reps) else 'MISMATCH'}]")
        say(line + f"  [{time.time()-t0:.1f} s]")
        seen = {}
        for r, s in sorted(zip(reps, sizes), key=lambda t: -t[1]):
            om = show(A, r, B2)
            bt = block_types(A, om)
            cls = red(r)
            ev, od, _ = S.preserving(A, om)
            key = (bt, cls != 0)
            seen.setdefault(key, []).append((s, om, (ev, od)))
        for key in sorted(seen, key=lambda k: (k[0], k[1])):
            bt, nz = key
            tag = (f"x-block {'alternating' if bt[0] else 'non-alternating'}, "
                   f"xi-block {'alternating' if bt[1] else 'non-alternating'}, "
                   f"[omega] {'!= 0' if nz else '= 0'} in H^2")
            say(f"       {len(seen[key])} class(es) with {tag}")
            for s, om, sd in seen[key][:2]:
                say(f"           |orbit| = {s:7d}  sdim h = ({sd[0]}|{sd[1]})  "
                    f"omega = {name(A, om)}")


if __name__ == "__main__":
    say("The homogeneous classification at p = 2, u = 1")
    for M in (1, 2, 3):
        say("")
        say("=" * 78)
        say(f"M = {M}")
        for n in range(0, M + 1):
            run(M - n, n)
    LOG.close()
