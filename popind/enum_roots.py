"""Enumeration of the K-invariant full-rank root lattices by Theorem 4.7.1 of [P],
for the rank-two groups with s = n+1 = 3 whose graph is a triangle:
K = G(4,2,2) and K = K_12.  Exact arithmetic; see check_lat.py for the conventions.
"""
from fractions import Fraction as Fr
from itertools import product as iproduct

from lat_core import (F, ZERO, ONE, I, SQ2, fadd, fsub, fmul, fneg, fscal, fconj,
                      finv, fdiv, fis0, herm, fstr, fnum, Lat, lat_sum,
                      lat_from_vecs, apply_mat, scale_lat, line_part,
                      coords_to_vec, vec_to_coords)
from check_lat import (say, check, CHECKS, refl, lat_invariant, root_lattice,
                       vscal, vadd, mat_id, intermediate_lattices, lat_C,
                       cyc_ring, stable_under, scaleC, OUT)
from run_lat import E, lin, Ogens, INVSQ2, ISQ2, HALF


def enum_roots(tag, es, n, global_ring=None, normal_D1=None, table2=None,
               order=None):
    s = len(es)
    say("")
    say("=" * 78)
    say(f"root lattices of {tag} by Theorem 4.7.1 of [P]")
    say("=" * 78)
    Rs = [refl(e, n) for e in es]
    b = {(j, k): fscal(2, herm(es[j], es[k])) for j in range(s) for k in range(s)
         if j != k}
    rings = [cyc_ring(b, j, s) for j in range(s)]
    for j in range(s):
        say(f"  Z[Tr K]_{j+1} = Z-span of " +
            ", ".join(fstr(tuple(x)) for x in rings[j].basis()))
    D1 = normal_D1
    check("Delta_1 is stable under Z[Tr K]_1", stable_under(rings[0], D1))
    if global_ring is not None:
        check("Delta_1 is stable under the global ring", stable_under(global_ring, D1))
    cands = [[D1]]
    for j in range(1, s):
        lo = scaleC(b[(0, j)], D1)
        hi = scaleC(finv(b[(j, 0)]), D1)
        check(f"b_1{j+1} Delta_1 <= b_{j+1}1^(-1) Delta_1", hi.contains(lo))
        say(f"  [b_{j+1}1^(-1)Delta_1 : b_1{j+1}Delta_1] = {lo.index_in(hi)}")
        cs = [D for D in intermediate_lattices(lo, hi, 1)
              if stable_under(rings[j], D)
              and (global_ring is None or stable_under(global_ring, D))]
        say(f"  candidates for Delta_{j+1}: {len(cs)}")
        cands.append(cs)
    good = []
    for combo in iproduct(*cands):
        ok = True
        for j in range(s):
            for k in range(s):
                if j != k and not fis0(b[(j, k)]):
                    if not combo[k].contains(scaleC(b[(j, k)], combo[j])):
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue
        gens = []
        for j in range(s):
            for lam in combo[j].basis():
                gens.append(vscal(tuple(lam), es[j]))
        L = lat_from_vecs(gens)
        if L.rank != 2 * n:          # discreteness (needed because s = n+1)
            continue
        inv = all(lat_invariant(L, R, n) for R in Rs)
        isroot = root_lattice(L, es, n) == L
        parts_ok = all(line_part(L, es[j], n) ==
                       lat_from_vecs([vscal(tuple(x), es[j])
                                      for x in combo[j].basis()])
                       for j in range(s))
        good.append((combo, L, inv, isroot, parts_ok))
    say(f"  tuples (Delta_1,...,Delta_s) satisfying a), b) and discreteness: {len(good)}")
    for (combo, L, inv, isroot, parts_ok) in good:
        desc = " | ".join("Delta_%d = Z-span(%s)" % (
            j + 1, ", ".join(fstr(tuple(x)) for x in combo[j].basis()))
            for j in range(s))
        say(f"    {desc}")
        say(f"       K-invariant: {inv},  root lattice: {isroot},"
            f"  Gamma_j = Delta_j: {parts_ok}")
    # similarity classes
    reps = []
    cands_alpha = []
    z = F(0, 1)
    for k in range(8):
        a = ONE
        for _ in range(k):
            a = fmul(a, z)
        for m in range(-4, 5):
            u = a
            step = fadd(ONE, I)
            if m >= 0:
                for _ in range(m):
                    u = fmul(u, step)
            else:
                for _ in range(-m):
                    u = fmul(u, finv(step))
            cands_alpha.append(u)
            for w in (fadd(ONE, ISQ2), fsub(ONE, ISQ2), ISQ2, F(3), F(2)):
                cands_alpha.append(fmul(u, w))
                cands_alpha.append(fmul(u, finv(w)))
    classes = []
    for (combo, L, inv, isroot, parts_ok) in good:
        hit = None
        for ci, (Lrep, members) in enumerate(classes):
            for a in cands_alpha:
                if scale_lat(a, L, n) == Lrep:
                    hit = ci
                    break
            if hit is not None:
                break
        if hit is None:
            classes.append((L, [L]))
        else:
            classes[hit][1].append(L)
    say(f"  similarity classes among them: {len(classes)}"
        f"  (sizes {[len(m) for (_, m) in classes]})")
    if table2:
        say("  --- comparison with Table 2 of [P] ---")
        for nm, T in table2:
            inclass = [ci for ci, (Lrep, _) in enumerate(classes)
                       if any(scale_lat(a, T, n) == Lrep for a in cands_alpha)]
            check(f"{nm} is similar to exactly one of the classes found "
                  f"(class {inclass})", len(inclass) == 1)
    return good, classes, cands_alpha


def O_lat(alpha):
    """the ring [1, alpha] as a lattice in C"""
    return lat_C([ONE, alpha])


def run_g422():
    n = 2
    e1 = lin(n, (fmul(I, INVSQ2), 0), (fneg(INVSQ2), 1))
    e2 = lin(n, (INVSQ2, 0), (fneg(INVSQ2), 1))
    e3 = lin(n, (ONE, 1))
    es = [e1, e2, e3]
    T = lat_from_vecs(Ogens(I, e1) + Ogens(I, e2))
    T2 = lat_sum(T, lat_from_vecs(Ogens(I, vscal(INVSQ2, e3))))
    T3 = lat_from_vecs(Ogens(I, e1) + Ogens(I, vscal(fadd(ONE, I), e2)))
    return enum_roots("K = G(4,2,2)", es, n, global_ring=O_lat(I),
                      normal_D1=O_lat(I),
                      table2=[("[G(4,2,2)]_1 (T)", T), ("[G(4,2,2)]_2", T2),
                              ("[G(4,2,2)]_3", T3)])


def run_k12():
    n = 2
    a = fscal(Fr(1, 4), fadd(SQ2, fmul(I, fsub(SQ2, F(2)))))
    bb = fscal(Fr(1, 4), fsub(fadd(F(2), SQ2), fmul(I, SQ2)))
    e1 = lin(n, (INVSQ2, 0), (fscal(HALF, fadd(ONE, I)), 1))
    e2 = lin(n, (a, 0), (bb, 1))
    e3 = lin(n, (INVSQ2, 0), (fscal(HALF, fsub(ONE, I)), 1))
    es = [e1, e2, e3]
    T = lat_from_vecs(Ogens(ISQ2, e1) + Ogens(ISQ2, e2))
    return enum_roots("K = K_12", es, n, normal_D1=O_lat(ISQ2),
                      table2=[("[K12] (T)", T)])


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "g422"):
        run_g422()
    if which in ("all", "k12"):
        run_k12()
    say("")
    say(f"checks: {CHECKS[1]}/{CHECKS[0]} passed")
    with open("enum_roots.out", "w") as f:
        f.write("\n".join(OUT) + "\n")
