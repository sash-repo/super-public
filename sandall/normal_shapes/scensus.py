#!/usr/bin/env python3
"""p = 2, u = 1: the census of closed non-degenerate 2-forms on O(m|n;1),
under the FULL automorphism group of the underlying algebra and under its
PARITY-PRESERVING subgroup.

The point of the super case.  At p = 2 the superspace O(m|n;1) and the space
O(m+n;1) have the same algebra of functions, the same de Rham complex and the
same 2-forms (see collapse.py): superization adds a Z/2-grading and nothing
else.  It does change the classification, because an automorphism of the
SUPERspace must preserve the grading.  This script measures the difference.

Everything is over F_2 and exhaustive: Z^2 is enumerated by a Gray code, the
Gram matrix is linear in omega and is carried along by the same XOR, and the
orbits are found by breadth-first search with the elementary automorphisms
        g_a -> g_a + g_b        (linear, b != a)
        g_a -> g_a + mu         (mu a monomial of degree >= 2)
as generators (for the graded group: only those with p(g_b) = p(g_a),
p(mu) = p(g_a)).  These generate: the group is linear-part times unipotent,
and the unipotent part is filtered with elementary generators in each layer.
`--check` verifies the generation against nalt2.automorphisms for m <= 3.
"""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scensus.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ------------------------------------------------------------- substitutions
def gen_index(A):
    """the m+n function generators, as indices into the monomial exponents."""
    return list(range(A.m)) + [A.ixi + j for j in range(A.n)]


def dgen_index(A):
    return [A.idx + i for i in range(A.m)] + [A.idxi + j for j in range(A.n)]


def subst(A, phi, f):
    """substitute g_a -> phi[a] into f in O(m|n;1)  (p = 2, u = 1)."""
    gi = gen_index(A)
    out = {}
    for mon, c in f.items():
        t = A.one()
        for a, g in enumerate(gi):
            if mon[g]:
                t = A.mul(t, phi[a])
        out = A.add(out, A.smul(c, t))
    return out


def one_form_coeffs(A, alpha):
    """alpha = sum h_k dg_k  ->  the list of h_k."""
    dgi = dgen_index(A)
    hs = [dict() for _ in dgi]
    for mon, c in alpha.items():
        for k, g in enumerate(dgi):
            if mon[g] == 1:
                red = list(mon)
                red[g] = 0
                hs[k][tuple(red)] = c
                break
    return hs


def divided_square(A, alpha):
    """alpha^{(2)} for a 1-form alpha = sum h_k dg_k, at p = 2:
       = sum_k h_k^2 (dg_k)^{(2)} + sum_{k<l} h_k h_l dg_k dg_l."""
    dgi = dgen_index(A)
    hs = one_form_coeffs(A, alpha)
    out = {}
    for k in range(len(dgi)):
        if not hs[k]:
            continue
        sq = A.mul(hs[k], hs[k])
        out = A.add(out, A.mul(sq, A.gen(dgi[k], 2)))
    for k in range(len(dgi)):
        for l in range(k + 1, len(dgi)):
            if not hs[k] or not hs[l]:
                continue
            out = A.add(out, A.mul(A.mul(hs[k], hs[l]),
                                   A.mul(A.gen(dgi[k]), A.gen(dgi[l]))))
    return out


def pullback2(A, phi, om):
    """phi^*(omega) for omega in Omega^2, p = 2, u = 1."""
    d = A.d()
    dgi = dgen_index(A)
    dphi = [d(f) for f in phi]
    out = {}
    for mon, c in om.items():
        # split the monomial into its function part and its form part
        fun = list(mon)
        for g in dgi:
            fun[g] = 0
        cc = subst(A, phi, {tuple(fun): c})
        if not cc:
            continue
        occ = [(k, mon[g]) for k, g in enumerate(dgi) if mon[g]]
        if len(occ) == 1 and occ[0][1] == 2:
            img = divided_square(A, dphi[occ[0][0]])
        elif len(occ) == 2:
            img = A.mul(dphi[occ[0][0]], dphi[occ[1][0]])
        else:
            raise AssertionError("not a 2-form monomial: %r" % (mon,))
        out = A.add(out, A.mul(cc, img))
    return out


def elementary_autos(A, graded):
    """the elementary automorphisms, as substitution lists."""
    gi = gen_index(A)
    G = len(gi)
    pars = [A.par[g] for g in gi]
    ident = [A.gen(g) for g in gi]
    outs = []
    # linear transvections
    for a in range(G):
        for b in range(G):
            if a == b:
                continue
            if graded and pars[a] != pars[b]:
                continue
            phi = list(ident)
            phi[a] = A.add(A.gen(gi[a]), A.gen(gi[b]))
            outs.append(phi)
    # elementary higher substitutions
    mons = [mm for mm in A.basis(0) if sum(mm) >= 2]
    for a in range(G):
        for mm in mons:
            if graded and A.parity(mm) != pars[a]:
                continue
            phi = list(ident)
            phi[a] = A.add(A.gen(gi[a]), {mm: 1})
            outs.append(phi)
    return outs


# ------------------------------------------------------------- bit machinery
def build(A):
    """basis of Omega^2, the closed subspace, the Gram map, all as bit data."""
    B2 = A.basis(2)
    idx = {mm: i for i, mm in enumerate(B2)}
    B3 = A.basis(3)
    i3 = {mm: i for i, mm in enumerate(B3)}
    d = A.d()
    M = S.matrix_of(A, d, B2, i3)
    # closed forms = left kernel of M (rows indexed by B2)
    T = [[M[r][c] for r in range(len(B2))] for c in range(len(B3))]
    ns = S.nullspace_p(T, 2) if T else []
    zbas = []
    for v in ns:
        bits = 0
        for i, c in enumerate(v):
            if int(c) % 2:
                bits |= 1 << i
        zbas.append(bits)
    # Gram constant matrix, linear in omega: precompute per basis monomial
    G = A.m + A.n
    gram_bits = []
    for mm in B2:
        _, C = S.nondegenerate(A, {mm: 1})
        bits = 0
        for a in range(G):
            for b in range(G):
                if C[a][b] % 2:
                    bits |= 1 << (a * G + b)
        gram_bits.append(bits)
    return B2, idx, zbas, gram_bits, G


def invertible_table(G):
    """table over all G x G bit matrices (G <= 3) of invertibility over F_2."""
    tab = bytearray(1 << (G * G))
    for bits in range(1 << (G * G)):
        rows = [[(bits >> (a * G + b)) & 1 for b in range(G)] for a in range(G)]
        # Gaussian elimination over F_2
        r = 0
        cols = list(range(G))
        Mx = [row[:] for row in rows]
        for c in cols:
            piv = None
            for i in range(r, G):
                if Mx[i][c]:
                    piv = i
                    break
            if piv is None:
                continue
            Mx[r], Mx[piv] = Mx[piv], Mx[r]
            for i in range(G):
                if i != r and Mx[i][c]:
                    for k in range(G):
                        Mx[i][k] ^= Mx[r][k]
            r += 1
        tab[bits] = 1 if r == G else 0
    return tab


def enumerate_nondeg(A, zbas, gram_bits, G):
    """all closed non-degenerate 2-forms, as ints over the Omega^2 basis."""
    tab = invertible_table(G)
    zg = []
    for z in zbas:
        g = 0
        b = z
        while b:
            i = (b & -b).bit_length() - 1
            g ^= gram_bits[i]
            b &= b - 1
        zg.append(g)
    D = len(zbas)
    out = []
    cur, curg = 0, 0
    # Gray code over the 2^D combinations
    for i in range(1 << D):
        if i:
            k = (i & -i).bit_length() - 1
            cur ^= zbas[k]
            curg ^= zg[k]
        if tab[curg]:
            out.append(cur)
    return out


def gen_matrix(A, phi, B2, idx):
    """phi^* as a table: image of each basis monomial, as an int."""
    imgs = []
    for mm in B2:
        img = pullback2(A, phi, {mm: 1})
        bits = 0
        for nn, c in img.items():
            if c % 2:
                bits |= 1 << idx[nn]
        imgs.append(bits)
    return imgs


def apply_map(imgs, v):
    out = 0
    b = v
    while b:
        i = (b & -b).bit_length() - 1
        out ^= imgs[i]
        b &= b - 1
    return out


def orbits(elems, maps):
    """orbit partition of `elems` (a list of ints) under the given maps."""
    left = set(elems)
    reps = []
    sizes = []
    while left:
        s = next(iter(left))
        seen = {s}
        stack = [s]
        while stack:
            v = stack.pop()
            for imgs in maps:
                w = apply_map(imgs, v)
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        reps.append(s)
        sizes.append(len(seen))
        left -= seen
    return reps, sizes


# ------------------------------------------------------------------ reporting
def show(A, v, B2):
    om = {}
    for i, mm in enumerate(B2):
        if (v >> i) & 1:
            om[mm] = 1
    return om


def name(A, om):
    gnames = ([f"x{i+1}" for i in range(A.m)] + [f"e{j+1}" for j in range(A.n)])
    dgi = dgen_index(A)
    gi = gen_index(A)
    terms = []
    for mon, c in sorted(om.items()):
        fun = []
        for a, g in enumerate(gi):
            if mon[g]:
                fun.append(gnames[a])
        form = []
        for k, g in enumerate(dgi):
            if mon[g] == 2:
                form.append(f"(d{gnames[k]})^(2)")
            elif mon[g] == 1:
                form.append(f"d{gnames[k]}")
        terms.append(("".join(fun) + " " if fun else "") + "".join(form))
    return " + ".join(terms) if terms else "0"


def diag_type(A, om):
    """(is the x-part of the diagonal non-zero, is the xi-part non-zero)."""
    dgi = dgen_index(A)
    dx, de = False, False
    for mon, c in om.items():
        for k, g in enumerate(dgi):
            if mon[g] == 2 and c % 2:
                if k < A.m:
                    dx = True
                else:
                    de = True
    return (dx, de)


def all_autos(A, graded):
    """EVERY automorphism of O(m|n;1), by brute force (small cases only).

    This is the control on `elementary_autos`: the orbits it gives must be the
    orbits of the group generated by the elementary ones.
    """
    gi = gen_index(A)
    G = len(gi)
    pars = [A.par[g] for g in gi]
    mons = [mm for mm in A.basis(0) if sum(mm) >= 1]
    pools = []
    for a in range(G):
        pool = [mm for mm in mons
                if (not graded) or A.parity(mm) == pars[a]]
        pools.append(pool)
    out = []
    for choice in itertools.product(*[range(1 << len(p)) for p in pools]):
        phi = []
        L = []
        for a in range(G):
            f = {}
            for i, mm in enumerate(pools[a]):
                if (choice[a] >> i) & 1:
                    f[mm] = 1
            phi.append(f)
            row = []
            for b, g in enumerate(gi):
                e = [0] * A.N
                e[g] = 1
                row.append(f.get(tuple(e), 0))
            L.append(row)
        if S.rank_p(L, 2) == G:
            out.append(phi)
    return out


def summarize(A, reps, sizes, Bz, tag, do_h=True, show_max=4):
    """orbit counts by parity of omega, the diagonal invariant, and samples."""
    bypar = {"even": 0, "odd": 0, "mixed": 0}
    bydiag = {}
    rows = []
    for r, s in zip(reps, sizes):
        om = show(A, r, Bz)
        pars = set(A.parity(mm) for mm in om)
        par = ("even" if pars == {0} else "odd" if pars == {1} else "mixed")
        bypar[par] += 1
        dt = diag_type(A, om)
        bydiag[(par, dt)] = bydiag.get((par, dt), 0) + 1
        rows.append((s, par, dt, om))
    say(f"     orbits by parity of omega: even {bypar['even']}, "
        f"odd {bypar['odd']}, inhomogeneous {bypar['mixed']}")
    ks = sorted(bydiag, key=lambda k: (k[0], k[1]))
    say("     orbits by (parity; diagonal (x-part, xi-part)): "
        + ", ".join(f"{k[0]}/({int(k[1][0])},{int(k[1][1])}): {bydiag[k]}"
                    for k in ks))
    rows.sort(key=lambda t: -t[0])
    shown = 0
    for s, par, dt, om in rows:
        if par == "mixed":
            continue
        line = (f"       |orbit| = {s:8d}  {par:4s} diag=({int(dt[0])},"
                f"{int(dt[1])})  omega = {name(A, om)}")
        if do_h:
            ev, od, _ = S.preserving(A, om)
            line += f"   sdim h = ({ev}|{od})"
        say(line)
        shown += 1
        if shown >= show_max:
            break


def run(M, verify=True, do_h=True):
    say("")
    say("=" * 78)
    say(f"M = {M} indeterminates over F_2, u = 1")
    A0 = S.Om(M, 0, 2, (1,) * M, dh=2, dxh=2)
    B2, idx, zbas, gram_bits, G = build(A0)
    say(f"  dim Omega^2 = {len(B2)}, dim Z^2 = {len(zbas)}")
    t0 = time.time()
    elems = enumerate_nondeg(A0, zbas, gram_bits, G)
    say(f"  closed non-degenerate 2-forms: {len(elems)} of 2^{len(zbas)} "
        f"[{time.time()-t0:.1f} s]")

    # (a) the ungraded group -- the non-alternating case
    t0 = time.time()
    maps = [gen_matrix(A0, phi, B2, idx)
            for phi in elementary_autos(A0, graded=False)]
    reps, sizes = orbits(elems, maps)
    say(f"  UNGRADED, i.e. the non-alternating case over O({M};1): {len(reps)} orbit(s), "
        f"{len(maps)} generators [{time.time()-t0:.1f} s]")
    for r, s in sorted(zip(reps, sizes), key=lambda t: -t[1])[:4]:
        om = show(A0, r, B2)
        line = f"       |orbit| = {s:8d}   omega = {name(A0, om)}"
        if do_h:
            ev, od, _ = S.preserving(A0, om)
            line += f"   dim h = {ev + od}"
        say(line)
    if verify and len(elems) * 24 <= 4_000_000:
        full = [gen_matrix(A0, phi, B2, idx) for phi in all_autos(A0, False)]
        r2, s2 = orbits(elems, full)
        say(f"       control: the FULL group ({len(full)} elements) gives "
            f"{len(r2)} orbit(s) -- {'ok' if len(r2) == len(reps) else 'MISMATCH'}")

    # (b) every splitting m|n of the same M
    for n in range(1, M + 1):
        m = M - n
        A = S.Om(m, n, 2, (1,) * m, dh=2, dxh=2)
        Bz, idz, zb, gb, Gb = build(A)
        assert len(Bz) == len(B2) and len(zb) == len(zbas), "collapse fails!"
        t0 = time.time()
        el = enumerate_nondeg(A, zb, gb, Gb)
        gens = elementary_autos(A, graded=True)
        maps = [gen_matrix(A, phi, Bz, idz) for phi in gens]
        reps, sizes = orbits(el, maps)
        aut = all_autos(A, True)
        say(f"  SUPER {m}|{n}: {len(el)} forms (same number: the collapse), "
            f"|Aut| = {len(aut)}, {len(reps)} orbit(s) "
            f"[{time.time()-t0:.1f} s]")
        say(f"     counting bound: {len(el)}/{len(aut)} = "
            f"{len(el)/len(aut):.1f} <= number of orbits")
        if verify and len(aut) * len(el) <= 100_000_000:
            full = [gen_matrix(A, phi, Bz, idz) for phi in aut]
            r2, s2 = orbits(el, full)
            say(f"     control: the FULL group gives {len(r2)} orbit(s) -- "
                f"{'ok' if len(r2) == len(reps) else 'MISMATCH'}")
        else:
            say("     control: full-group check skipped (too large); the "
                "elementary generators are verified in the smaller cases")
        summarize(A, reps, sizes, Bz, f"{m}|{n}", do_h=do_h)


if __name__ == "__main__":
    for M in (1, 2, 3):
        run(M)
    LOG.close()
