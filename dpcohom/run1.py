#!/usr/bin/env python3
"""Census: closed non-alternating 2-forms over O(m;u), p = 2.

Prints, for each (m, u):
  * dim Omega^2, dim Z^2, dim B^2, dim H^2;
  * the check that every closed form has CONSTANT diagonal;
  * the check that ker of all partials in O(m;u) is the constants;
  * for u = 1 and small m: the Aut-orbits of the non-degenerate NON-ALTERNATING
    closed forms, with a representative and the derived series of h(omega).
"""
import sys, itertools, time
from nalt2 import *


def dims(alg):
    b2 = g2_basis(alg.m)
    Z = closed_space(alg)
    # B^2 = d(Omega^1)
    rows = []
    for i in range(alg.m):
        for t in range(alg.dim):
            of = [0] * alg.m
            of[i] = 1 << t
            rows.append(flatten2(alg, d1(alg, of), b2))
    dB = rank(rows, p=2) if rows else 0
    return len(b2) * alg.dim, len(Z), dB, len(Z) - dB, Z


def const_diagonal_check(alg, Z):
    bad = 0
    for om in Z:
        for k, v in om.items():
            if k[0] == 'b' and v not in (0, alg.one):
                bad += 1
    return bad


def kernel_of_partials(alg):
    """dim of { f in O | partial_i f = 0 for all i }."""
    rows = []
    for t in range(alg.dim):
        f = 1 << t
        r = []
        for i in range(alg.m):
            p = alg.partial(i, f)
            r.extend((p >> s) & 1 for s in range(alg.dim))
        rows.append(r)
    return alg.dim - rank(rows, p=2)


def diag_of(alg, om):
    return tuple(1 if om.get(('b', j), 0) else 0 for j in range(alg.m))


def enumerate_forms(alg, Z, fix_b=None):
    """all F_2-combinations of the basis Z of Z^2 with diagonal fix_b.

    Done as  omega_0 + Z_alt  (Z_alt = the closed ALTERNATING forms), so the
    enumeration is over 2^dim(Z_alt) elements, not over all of Z^2.
    """
    if fix_b is None:
        pool, base = Z, {}
    else:
        alt = [om for om in Z if diag_of(alg, om) == (0,) * alg.m]
        # complete alt to Z by Gaussian elimination on the diagonal part
        rest = [om for om in Z if diag_of(alg, om) != (0,) * alg.m]
        piv, base = {}, None
        red = []
        for om in rest:
            d = diag_of(alg, om)
            cur, dcur = dict(om), d
            for j in range(alg.m):
                if dcur[j] and j in piv:
                    addform(cur, piv[j][0])
                    dcur = tuple(a ^ b for a, b in zip(dcur, piv[j][1]))
            j0 = next((j for j in range(alg.m) if dcur[j]), None)
            if j0 is None:
                alt.append(cur)
            else:
                piv[j0] = (cur, dcur)
                red.append((cur, dcur))
        # build a form with diagonal exactly fix_b
        base, dbase = {}, (0,) * alg.m
        for j in range(alg.m):
            if fix_b[j] != dbase[j]:
                if j not in piv:
                    return []                 # this diagonal is not attained
                addform(base, piv[j][0])
                dbase = tuple(a ^ b for a, b in zip(dbase, piv[j][1]))
        if dbase != tuple(fix_b):
            return []
        pool = alt
    n = len(pool)
    out = []
    for bits in range(1 << n):
        om = dict(base)
        b, i = bits, 0
        while b:
            if b & 1:
                addform(om, pool[i])
            b >>= 1
            i += 1
        out.append(om)
    return out


def key_of(alg, om, b2):
    v = 0
    for s, k in enumerate(b2):
        v |= om.get(k, 0) << (s * alg.dim)
    return v


def form_of(alg, key, b2):
    om = {}
    mask = (1 << alg.dim) - 1
    for s, k in enumerate(b2):
        c = (key >> (s * alg.dim)) & mask
        if c:
            om[k] = c
    return om


def orbits(alg, forms, gens):
    b2 = g2_basis(alg.m)
    todo = {key_of(alg, om, b2) for om in forms}
    reps = []
    while todo:
        start = min(todo)
        orb = {start}
        stack = [start]
        while stack:
            k = stack.pop()
            om = form_of(alg, k, b2)
            for g in gens:
                k2 = key_of(alg, pullback(alg, g, om), b2)
                if k2 not in orb:
                    orb.add(k2)
                    stack.append(k2)
        reps.append((start, len(orb)))
        todo -= orb
    return reps


def show_form(alg, om):
    parts = []
    for k in sorted(om, key=str):
        c = om[k]
        if not c:
            continue
        s = alg.show(c)
        if k[0] == 'a':
            parts.append("(%s)dx%ddx%d" % (s, k[1] + 1, k[2] + 1))
        else:
            parts.append("(%s)dx%d^(2)" % (s, k[1] + 1))
    return " + ".join(parts) if parts else "0"


def stab_gens(alg, bvec):
    """automorphisms of O(m;1) fixing the diagonal vector bvec, as generators:
    all of them for m <= 2, a generating set for m = 3."""
    m = alg.m
    maximal = [1 << k for k, a in enumerate(alg.mons) if any(a)]
    quad = [1 << k for k, a in enumerate(alg.mons) if sum(a) >= 2]
    gens = []
    # linear automorphisms whose Frobenius-twisted action fixes bvec
    lin = []
    for bits in range(1 << m):
        v = 0
        for i in range(m):
            if (bits >> i) & 1:
                v ^= alg.gen(i)
        lin.append((bits, v))
    for phi_bits in itertools.product(lin, repeat=m):
        L = [[(pb[0] >> k) & 1 for k in range(m)] for pb in phi_bits]
        if rank(L, p=2) != m:
            continue
        phi = [pb[1] for pb in phi_bits]
        om = {('b', i): alg.one for i in range(m) if bvec[i]}
        if pullback(alg, phi, om) == om:
            gens.append(phi)
    # unipotent: x_i -> x_i + h, h in m^2 (these always fix the diagonal)
    for i in range(m):
        for h in quad:
            phi = [alg.gen(j) for j in range(m)]
            phi[i] = phi[i] ^ h
            gens.append(phi)
    return gens


def report(m, u, do_orbits=True, fix_b=None):
    alg = Alg(m, u)
    t0 = time.time()
    dO, dZ, dB, dH, Z = dims(alg)
    print("=" * 78)
    print("m = %d, u = %s   dim O = %d" % (m, u, alg.dim))
    print("  dim Omega^2 = %-4d dim Z^2 = %-4d dim B^2 = %-4d dim H^2 = %d"
          % (dO, dZ, dB, dH))
    print("  closed forms with non-constant diagonal : %d   <-- must be 0"
          % const_diagonal_check(alg, Z))
    print("  dim { f in O | all partials vanish }    : %d   <-- must be 1"
          % kernel_of_partials(alg))
    if not do_orbits or any(k != 1 for k in alg.u):
        if any(k != 1 for k in alg.u) and do_orbits:
            print("  (orbits need substitution in a divided power algebra: "
                  "implemented for u = 1 only)")
        else:
            print("  (orbits skipped)")
        return
    forms = enumerate_forms(alg, Z, fix_b=fix_b)
    nd = [om for om in forms if nondeg(alg, om) and not alternating(om)]
    print("  closed forms with diagonal %s : %d, of them non-degenerate and "
          "non-alternating: %d" % (fix_b, len(forms), len(nd)))
    if not nd:
        return
    gens = stab_gens(alg, fix_b if fix_b else tuple([1] + [0] * (m - 1)))
    print("  |generating set of the stabilizer| = %d" % len(gens))
    reps = orbits(alg, nd, gens)
    b2 = g2_basis(alg.m)
    print("  ORBITS: %d" % len(reps))
    for k, size in sorted(reps, key=lambda r: -r[1]):
        om = form_of(alg, k, b2)
        h = preserving_algebra(alg, om)
        ds = derived_series(alg, h)
        print("    |orbit| = %-6d  omega = %s" % (size, show_form(alg, om)))
        print("       dim h, h^(1), h^(2), ... = %s   (dim vect = %d)"
              % (ds, m * alg.dim))
    print("  [%.1f s]" % (time.time() - t0))


if __name__ == "__main__":
    print("exactlin backend:", backend())
    report(1, (1,), fix_b=(1,))
    report(2, (1, 1), fix_b=(1, 0))
    report(1, (2,), fix_b=(1,))
    report(2, (2, 1), do_orbits=False)
    report(3, (1, 1, 1), do_orbits=False)
    report(4, (1, 1, 1, 1), do_orbits=False)
