#!/usr/bin/env python3
"""The groupoid of a generalized root system, part B: the wall-crossings and the codimension-2 residues.

For each GRS we verify, exactly:

  (1) every simple root of the base B(C) of a chamber C supports a facet of C, and
      the chamber C' across that facet has base s_gamma B(C) with Serganova's
      s_gamma of [S] section 6 -- i.e. the wall-crossings of the arrangement A(R)
      are precisely the simple reflections (anisotropic walls) and the odd
      reflections (isotropic walls);
  (2) for an isotropic simple gamma the unique linear map B(C) -> B(C') is NOT an
      isometry of the form (so the crossing is not induced by Iso(V): the mirror
      lemma, computationally);
  (3) the codimension-2 strata X of A(R): the roots vanishing on X form a rank-2
      GRS (axioms rechecked on the restriction), the link is an arrangement of k
      lines, the dual cycle has length 2k, and we record how many of the crossings
      are reflections and how many are odd reflections.  These cycles are the
      defining relations of the chamber groupoid.

Also: the defective GRS A(1,1) = psl(2|2), where bases are linearly dependent,
counted by Serganova's Definition 6.1 directly.
"""
from fractions import Fraction as F
from itertools import combinations, product

from check_iv_a import (dot, rref, rank, solve, normalize, matinv, walls, zaslavsky,
                        enumerate_regions, positive_system, indecomposables, is_base,
                        sysA, sysA11, sysB, sysC, sysD, sysD21, sysF4, sysG3)


def refl(G, a, v):
    k = 2 * dot(G, a, v) / dot(G, a, a)
    return tuple(x - k * y for x, y in zip(v, a))


def serganova_s(G, roots, B, g):
    """s_gamma(B) of [S] section 6: gamma |-> -gamma, beta |-> beta if (beta,gamma)=0,
    beta |-> beta +- gamma (the sign making it a root) otherwise."""
    rs = set(roots)
    if dot(G, g, g) != 0:
        return frozenset(refl(G, g, b) for b in B)
    out = []
    for b in B:
        if b == g:
            out.append(tuple(-x for x in g))
        elif dot(G, b, g) == 0:
            out.append(b)
        else:
            p = tuple(x + y for x, y in zip(b, g))
            m = tuple(x - y for x, y in zip(b, g))
            if p in rs and m in rs:
                return None          # ambiguous: would contradict [S] Lemma 1.11
            out.append(p if p in rs else m)
    return frozenset(out)


def linear_map_matrix(G, src, dst):
    """Matrix (in ambient coords) of the linear map sending src[i] |-> dst[i]; both bases of V.
    Returned as the list of images of an arbitrary basis is awkward in a subspace, so we
    instead check isometry directly on the Gram matrices: the map is an isometry iff
    Gram(src) == Gram(dst)."""
    return None


def is_isometry_on_bases(G, src, dst):
    """The unique linear map src -> dst is an isometry of V iff the Gram matrices agree."""
    n = len(src)
    return all(dot(G, src[i], src[j]) == dot(G, dst[i], dst[j]) for i in range(n) for j in range(n))


def crossings_report(sysd):
    G, roots = sysd["G"], sysd["roots"]
    Vb = sysd["Vbasis"] or [tuple(F(1) if i == j else F(0) for j in range(len(G))) for i in range(len(G))]
    dim = len(Vb)
    name = sysd["name"]
    wl = walls(G, roots, Vb)
    funcs = list(wl)
    nreg = zaslavsky(funcs, dim)
    gram = [[dot(G, Vb[i], Vb[j]) for j in range(dim)] for i in range(dim)]
    ginv = matinv(gram)
    regs = enumerate_regions(funcs, dim, nreg, gram_inv=ginv)
    assert len(regs) == nreg, (name, len(regs), nreg)

    # sign vector -> base, and base -> sign vector
    base_of = {}
    for key, u in regs.items():
        v = tuple(sum(Vb[k][i] * u[k] for k in range(dim)) for i in range(len(G)))
        base_of[key] = (frozenset(indecomposables(positive_system(G, roots, v))), v)
    by_base = {b: k for k, (b, v) in base_of.items()}
    assert len(by_base) == nreg

    n_refl = n_odd = 0
    bad = []
    nonisometry = 0
    for key, (B, v) in base_of.items():
        Bl = sorted(B)
        assert rank(Bl) == dim, (name, "base not a basis")
        gm = [[dot(G, Bl[i], Bl[j]) for j in range(dim)] for i in range(dim)]
        gmi = matinv(gm)
        for idx, g in enumerate(Bl):
            # cross the wall g^perp: keep (v',beta) = 1 for beta != g, set (v',g) = -eps
            for den in (4, 16, 64, 256, 1024, 4096):
                c = [F(1)] * dim
                c[idx] = F(-1, den)
                u2 = [sum(gmi[k][i] * c[i] for i in range(dim)) for k in range(dim)]
                v2 = tuple(sum(Bl[k][i] * u2[k] for k in range(dim)) for i in range(len(G)))
                # sign vector in exactly the same normalization as enumerate_regions:
                uu = solve(Vb, v2)
                vals = [sum(f[i] * uu[i] for i in range(dim)) for f in funcs]
                key2 = tuple(1 if x > 0 else (-1 if x < 0 else 0) for x in vals)
                if 0 in key2:
                    continue
                # the sign vector must differ from key exactly on the wall g^perp
                fg = normalize(tuple(dot(G, b, g) for b in Vb))
                diff = [i for i in range(len(funcs)) if key2[i] != key[i]]
                if diff == [funcs.index(fg)]:
                    break
            else:
                bad.append((name, "no clean crossing", g))
                continue
            B2 = base_of[key2][0]
            S = serganova_s(G, roots, B, g)
            if S != B2:
                bad.append((name, "crossing != s_gamma", tuple(g)))
            if dot(G, g, g) != 0:
                n_refl += 1
                assert B2 == frozenset(refl(G, g, b) for b in B)
            else:
                n_odd += 1
                # the crossing map B -> B2 (simple roots matched by s_gamma) is not an isometry
                ordered = []
                for b in Bl:
                    if b == g:
                        ordered.append(tuple(-x for x in g))
                    elif dot(G, b, g) == 0:
                        ordered.append(b)
                    else:
                        p = tuple(x + y for x, y in zip(b, g))
                        m = tuple(x - y for x, y in zip(b, g))
                        ordered.append(p if p in set(roots) else m)
                if not is_isometry_on_bases(G, Bl, ordered):
                    nonisometry += 1
    print(f"\n=== {name} ===  {nreg} chambers")
    print(f"  wall crossings checked: {n_refl} across anisotropic walls (= simple reflections), "
          f"{n_odd} across isotropic walls (= odd reflections)")
    print(f"  all crossings equal Serganova's s_gamma: {not bad}" + (f"  {bad[:3]}" if bad else ""))
    print(f"  odd crossings whose base map is NOT an isometry: {nonisometry}/{n_odd}"
          f"  ({'all' if nonisometry == n_odd else 'SOME ARE'})")
    return wl, funcs, dim, G, roots, Vb


def residues_report(wl, funcs, dim, G, roots, Vb, quiet=False):
    """Codimension-2 strata: the rank-2 residues, i.e. the relations of the groupoid."""
    seen = {}
    for f1, f2 in combinations(funcs, 2):
        X = rref([f1, f2])
        if len(X) != 2:
            continue
        key = X
        if key in seen:
            continue
        # all roots whose wall contains X
        sub = [r for r in roots if solve(list(X), normalize(tuple(dot(G, b, r) for b in Vb))) is not None]
        subwalls = {normalize(tuple(dot(G, b, r) for b in Vb)) for r in sub}
        aniso = {w for w in subwalls if dot(G, wl[w][0][0], wl[w][0][0]) != 0}
        # is the restriction a GRS?  (axioms rechecked on sub) and is the form on its span
        # still nondegenerate?  (span is 2-dimensional: take two independent roots)
        from check_iv_a import check_axioms
        ax = check_axioms(G, sub)
        ind = []
        for r in sub:
            if rank(ind + [r]) > len(ind):
                ind.append(r)
        gr = [[dot(G, ind[i], ind[j]) for j in range(len(ind))] for i in range(len(ind))]
        nondeg = rank(gr) == len(ind)
        seen[key] = (len(subwalls), len(aniso), len(sub), not ax, nondeg)
    from collections import Counter
    cnt = Counter(seen.values())
    if quiet:
        return cnt
    print(f"  codim-2 strata: {len(seen)};  types:")
    for k, v in sorted(cnt.items()):
        w, a, nr, isg, nd = k
        print(f"    {v:3d} x  {w} walls ({a} aniso, {w-a} iso), |R_X| = {nr}, axioms: {isg}, "
              f"form on span nondegenerate: {nd}"
              f"   -> cycle of length {2*w}: {2*a} reflections + {2*(w-a)} odd reflections")


def a11_bases():
    """Serganova Def. 6.1 for the defective A(1,1): bases may be linearly dependent."""
    s = sysA11()
    G, roots = s["G"], s["roots"]
    def reps(B, r, bound=4):
        out = []
        for c in product(range(-bound, bound + 1), repeat=len(B)):
            if not (all(x >= 0 for x in c) or all(x <= 0 for x in c)):
                continue
            v = tuple(sum(F(c[i]) * B[i][j] for i in range(len(B))) for j in range(len(G)))
            if v == r:
                out.append(c)
        return out
    good = []
    for k in range(1, len(roots) + 1):
        for B in combinations(roots, k):
            if rank(list(B)) != 2:
                continue
            ok = True
            for r in roots:
                if len(reps(list(B), r)) != 1:
                    ok = False
                    break
            if ok:
                good.append(B)
    print(f"\n=== A(1,1) = psl(2|2): bases per Serganova Def. 6.1 ===")
    print(f"  number of bases = {len(good)}, sizes = {sorted({len(b) for b in good})}")
    print(f"  chambers of A(R) = 8 (computed in part A)")
    for B in good[:4]:
        print("   sample base:", [tuple(x) for x in B])


def residue_census():
    """The rank-2 residues of every system of the table of Subsec. 8.8 (and the defective
    A(1,1)), collected by type (k walls, a of them anisotropic, |R_X|) -- the table of
    Subsec. 8.16 with its column "occurs in"."""
    from collections import defaultdict
    from check_iv_a import sysA as A
    systems = [A(1, 0), A(2, 0), A(2, 1), A(3, 1), sysB(1, 1), sysB(2, 1), sysB(1, 2),
               sysC(2), sysC(3), sysD(2, 1), sysD(2, 2), sysD21(1), sysD21(2), sysD21(3),
               sysG3(), sysF4(), sysA11()]
    occurs = defaultdict(list)
    for s in systems:
        G, roots = s["G"], s["roots"]
        Vb = s["Vbasis"] or [tuple(F(1) if i == j else F(0) for j in range(len(G)))
                             for i in range(len(G))]
        wl = walls(G, roots, Vb)
        cnt = residues_report(wl, list(wl), len(Vb), G, roots, Vb, quiet=True)
        for (w, a, nr, isg, nd), v in cnt.items():
            occurs[(w, a, nr, nd)].append(s["name"])
    print("residue types over all systems: k walls, a anisotropic, |R_X|, "
          "form on span nondegenerate -> systems where it occurs")
    for key in sorted(occurs, key=lambda k: (k[0], -k[1], k[2])):
        w, a, nr, nd = key
        print(f"  k={w} a={a} |R_X|={nr:2d} nondeg={str(nd):5s}  cycle {2*w}: {2*a} refl + "
              f"{2*(w-a)} odd  in: {', '.join(occurs[key])}")
    print(f"number of types: {len(occurs)}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["census"]:
        residue_census()
        raise SystemExit
    for s in [sysA(1, 0), sysA(2, 0), sysA(2, 1), sysB(1, 1), sysB(2, 1), sysC(3),
              sysD(2, 1), sysD21(3), sysG3(), sysF4()]:
        args = crossings_report(s)
        residues_report(*args)
    a11_bases()
