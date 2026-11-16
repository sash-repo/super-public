"""Driver: the three groups of Sec. 4.8 of [P] left to the reader."""
from fractions import Fraction as Fr

from lat_core import (F, ZERO, ONE, I, SQ2, fis0, fadd, fsub, fmul, fneg, fscal, fconj,
                      finv, fdiv, herm, fstr, fnum, Lat, lat_sum, lat_from_vecs,
                      apply_mat, scale_lat, line_part)
from check_lat import (analyse, say, check, CHECKS, refl, lat_invariant,
                       root_lattice, vscal, vadd, vsub, mat_id, mat_sub,
                       mat_inv, group_order, OUT, intermediate_lattices)

HALF = Fr(1, 2)
INVSQ2 = fscal(HALF, SQ2)          # 1/sqrt 2
ISQ2 = fmul(I, SQ2)                # i sqrt 2


def E(n, k):
    return tuple(ONE if j == k else ZERO for j in range(n))


def lin(n, *pairs):
    v = [ZERO] * n
    for (a, k) in pairs:
        v[k] = fadd(v[k], a)
    return tuple(v)


def Ogens(alpha, e):
    """generators of the rank-2 lattice [1,alpha] e = Z e + Z alpha e"""
    return [e, vscal(alpha, e)]


# ------------------------------------------------------------- G(4,2,2)
def g422():
    n = 2
    e1 = lin(n, (fmul(I, INVSQ2), 0), (fneg(INVSQ2), 1))
    e2 = lin(n, (INVSQ2, 0), (fneg(INVSQ2), 1))
    e3 = lin(n, (ONE, 1))
    es = [e1, e2, e3]
    gens = Ogens(I, e1) + Ogens(I, e2)
    return analyse("K = K_2 of type G(4,2,2)  (n = 2, s = 3): 'the most complicated case'",
                   es, expected_order=16, Kprime_size=8, lattice_gens=gens, n=n,
                   extra=g422_extra)


def g422_extra(D):
    n, es, Rs = D['n'], D['es'], D['Rs']
    e1, e2, e3 = es
    say("")
    say("  --- the scalar argument (Subsec. A.14 of the text) ---")
    R1R2 = [tuple(_m(Rs[0], Rs[1], n, r, c) for c in range(n)) for r in range(n)]
    idm = mat_id(n)
    scal = [tuple(fmul(I, idm[r][c]) for c in range(n)) for r in range(n)]
    prod = [tuple(_m(R1R2, Rs[2], n, r, c) for c in range(n)) for r in range(n)]
    check("R_1 R_2 R_3 = i * id  (so i.id lies in K)", prod == scal)

    # eps-coordinates: L = Z[i]eps_1 + Z[i]eps_2, the four lattices of A.14.2
    ep1, ep2 = E(n, 0), E(n, 1)
    L = lat_from_vecs([ep1, vscal(I, ep1), ep2, vscal(I, ep2)])
    v_B = vscal(fscal(HALF, fadd(ONE, I)), vadd(ep1, ep2))     # (1+i)/2 (e1+e2)
    v_C = vscal(F(HALF), vadd(ep1, ep2))                        # 1/2 (e1+e2)
    v_D = vscal(F(HALF), vadd(ep1, vscal(I, ep2)))              # 1/2 (e1+i e2)
    GA = L
    GB = lat_sum(L, lat_from_vecs([v_B, vscal(I, v_B)]))
    GC = lat_sum(L, lat_from_vecs([v_C, vscal(I, v_C)]))
    GD = lat_sum(L, lat_from_vecs([v_D, vscal(I, v_D)]))
    names = ["Gamma_A = Z[i]eps_1 + Z[i]eps_2",
             "Gamma_B = Gamma_A + Z[i](1+i)/2 (eps_1+eps_2)",
             "Gamma_C = Gamma_A + Z[i]1/2 (eps_1+eps_2)",
             "Gamma_D = Gamma_A + Z[i]1/2 (eps_1+i eps_2)"]
    for L1, nm in zip([GA, GB, GC, GD], names):
        ok = all(lat_invariant(L1, R, n) for R in Rs)
        check(f"{nm} is K-invariant", ok)
        check(f"   ... and is a root lattice for K", root_lattice(L1, es, n) == L1)
    check("[Gamma_B : Gamma_A] = 2", GA.index_in(GB) == 2)
    check("[Gamma_C : Gamma_B] = 2", GB.index_in(GC) == 2)
    check("[Gamma_D : Gamma_B] = 2", GB.index_in(GD) == 2)
    check("Gamma_C != Gamma_D", GC != GD)

    # the normalizer element diag(1,i) carries Gamma_C to Gamma_D
    g = [(ONE, ZERO), (ZERO, I)]
    ginv = mat_inv(g, n)
    conj_ok = True
    Kel = _closure(Rs, n)
    for R in Rs:
        M = _mm(_mm(g, R, n), ginv, n)
        if not any(M == X for X in Kel):
            conj_ok = False
    check("diag(1,i) normalizes K", conj_ok)
    check("diag(1,i) Gamma_C = Gamma_D", apply_mat(g, GC, n) == GD)
    check("diag(1,i) Gamma_A = Gamma_A", apply_mat(g, GA, n) == GA)
    check("diag(1,i) Gamma_B = Gamma_B", apply_mat(g, GB, n) == GB)

    # the six reflections of K and their root lines (step 5 of the proof)
    say("")
    say("  --- the six reflections of K and their root lines ---")
    Kel = _closure(Rs, n)
    idm = mat_id(n)
    negid = [tuple(fneg(x) for x in row) for row in idm]
    refls = [M for M in Kel
             if _mm(M, M, n) == idm and M != idm and M != negid
             and fis0(fadd(M[0][0], M[1][1]))]
    check("K has exactly 6 reflections (order 2, trace 0)", len(refls) == 6)
    lines = set()
    for M in refls:
        Dm = mat_sub(idm, M, n)
        # the root line: the image of id - M, spanned by its first nonzero column
        col = None
        for c in range(n):
            v = tuple(Dm[r][c] for r in range(n))
            if not all(fis0(x) for x in v):
                col = v
                break
        piv = next(x for x in col if not fis0(x))
        col = vscal(finv(piv), col)          # normalise the direction vector
        lines.add(lat_from_vecs([col, vscal(I, col)]))
    check("their root lines are 6 distinct complex lines", len(lines) == 6)
    expect = set()
    for v in [E(n, 0), E(n, 1), vadd(E(n, 0), E(n, 1)), vsub(E(n, 0), E(n, 1)),
              vadd(E(n, 0), vscal(I, E(n, 1))), vsub(E(n, 0), vscal(I, E(n, 1)))]:
        expect.add(lat_from_vecs([v, vscal(I, v)]))
    check("they are C eps_1, C eps_2, C(eps_1 +- eps_2), C(eps_1 +- i eps_2)",
          lines == expect)

    # identification with Table 2 of [P]
    T = lat_from_vecs(Ogens(I, e1) + Ogens(I, e2))
    T2 = lat_sum(T, lat_from_vecs([v for v in Ogens(I, vscal(INVSQ2, e3))]))
    T3 = lat_from_vecs(Ogens(I, e1) + Ogens(I, vscal(fadd(ONE, I), e2)))
    say("")
    say("  --- identification with the rows of Table 2 of [P] ---")
    sq2 = SQ2
    z8 = F(0, 1)
    opi = fadd(ONE, I)
    check("[G(4,2,2)]_1 : T = [1,i]e_1+[1,i]e_2 = zeta_8 Gamma_B",
          scale_lat(z8, GB, n) == T)
    check("[G(4,2,2)]_2 : T + (1/sqrt2)[1,i]e_3 = zeta_8 (1+i)^(-1) Gamma_A",
          scale_lat(fmul(z8, finv(opi)), GA, n) == T2)
    check("[G(4,2,2)]_3 : [1,i]e_1+[1,i](1+i)e_2 = zeta_8 (1+i) Gamma_D",
          scale_lat(fmul(z8, opi), GD, n) == T3)
    check("T, T_2, T_3 are pairwise distinct", len({T, T2, T3}) == 3)
    for L1, nm in [(T, "T"), (T2, "T_2"), (T3, "T_3")]:
        check(f"{nm} is K-invariant and a root lattice",
              all(lat_invariant(L1, R, n) for R in Rs) and root_lattice(L1, es, n) == L1)

    # completeness: every K-invariant lattice with Gamma cap C eps_1 = Z[i]eps_1
    # lies between Gamma_A and (1/2)Gamma_A; brute force over that finite quotient
    say("")
    say("  --- completeness check: all K-invariant lattices between L and (1/2)L ---")
    Lhalf = scale_lat(F(HALF), L, n)
    inter = intermediate_lattices(L, Lhalf, n)
    good = []
    for G in inter:
        if all(lat_invariant(G, R, n) for R in Rs):
            # the normalisation: Gamma cap C eps_1 = Z[i] eps_1
            l1 = line_part(G, ep1, n)
            if l1 == lat_from_vecs([ep1, vscal(I, ep1)]):
                good.append(G)
    check("exactly 4 such lattices", len(good) == 4)
    check("they are Gamma_A, Gamma_B, Gamma_C, Gamma_D",
          set(good) == {GA, GB, GC, GD})
    say(f"  (of the {len(inter)} lattices between L and (1/2)L, "
        f"{len([G for G in inter if all(lat_invariant(G,R,n) for R in Rs)])} are K-invariant, "
        f"{len(good)} of them normalised)")


def _m(A, B, n, r, c):
    s = ZERO
    for k in range(n):
        s = fadd(s, fmul(A[r][k], B[k][c]))
    return s


def _mm(A, B, n):
    return [tuple(_m(A, B, n, r, c) for c in range(n)) for r in range(n)]


def _closure(gens, n):
    idm = mat_id(n)
    G = [idm]
    frontier = [idm]
    while frontier:
        new = []
        for M in frontier:
            for g in gens:
                P = _mm(g, M, n)
                if not any(P == X for X in G):
                    G.append(P)
                    new.append(P)
        frontier = new
    return G


# ------------------------------------------------------------------- K12
def k12():
    n = 2
    a = fscal(Fr(1, 4), fadd(SQ2, fmul(I, fsub(SQ2, F(2)))))     # (sqrt2+(sqrt2-2)i)/4
    bb = fscal(Fr(1, 4), fsub(fadd(F(2), SQ2), fmul(I, SQ2)))    # (2+sqrt2-sqrt2 i)/4
    e1 = lin(n, (INVSQ2, 0), (fscal(HALF, fadd(ONE, I)), 1))
    e2 = lin(n, (a, 0), (bb, 1))
    e3 = lin(n, (INVSQ2, 0), (fscal(HALF, fsub(ONE, I)), 1))
    es = [e1, e2, e3]
    gens = Ogens(ISQ2, e1) + Ogens(ISQ2, e2)
    return analyse("K = K_12  (n = 2, s = 3)", es, expected_order=48,
                   Kprime_size=12, lattice_gens=gens, n=n, extra=k12_extra)


def k12_extra(D):
    n, es, Rs = D['n'], D['es'], D['Rs']
    say("")
    say("  --- Z[Tr K]_1 and the lattice Gamma_1 ---")
    b = D['b']
    # cyclic products c_{1...}: closed walks at 1 of length <= 4
    prods = set()
    for path in [(0, 1, 0), (0, 2, 0), (0, 1, 2, 0), (0, 2, 1, 0)]:
        p = ONE
        for u, v in zip(path, path[1:]):
            p = fmul(p, b[(u, v)])
        prods.add(p)
    for p in sorted(prods, key=lambda x: str(x)):
        say(f"    cyclic product at 1: {fstr(p)}   (= {fnum(p):.4f})")
    say("")
    say("  --- Gamma_1 as a Z[Tr K]_1-module ---")
    L = D['L']
    L1 = line_part(L, es[0], n)
    check("Gamma_1 = [1, i sqrt2] e_1",
          L1 == lat_from_vecs(Ogens(ISQ2, es[0])))
    L2 = line_part(L, es[1], n)
    check("Gamma_2 = [1, i sqrt2] e_2",
          L2 == lat_from_vecs(Ogens(ISQ2, es[1])))
    L3 = line_part(L, es[2], n)
    zeta8inv = finv(F(0, 1))
    check("Gamma_3 = zeta_8^(-1)[1, i sqrt2] e_3",
          L3 == lat_from_vecs(Ogens(ISQ2, vscal(zeta8inv, es[2]))))
    say("")
    say("  --- b-conditions (Theorem 4.7.1 c) of [P]) on Delta_1 ---")
    for (j, k), val in sorted(b.items()):
        say(f"    b_{j+1}{k+1} = {fstr(val)}  (= {fnum(val):.4f}),  |b|^2 = "
            f"{fnum(fmul(val, fconj(val))).real:.4f}")


# ------------------------------------------------------------------- K31
def k31(full_order=False):
    n = 4
    e1 = lin(n, (INVSQ2, 1), (fneg(INVSQ2), 3))
    e2 = lin(n, (fneg(fmul(I, INVSQ2)), 1), (INVSQ2, 2))
    e3 = lin(n, (fneg(INVSQ2), 2), (INVSQ2, 3))
    co = fscal(HALF, fmul(fsub(I, ONE), INVSQ2))       # (-1+i)/(2 sqrt2)
    e4 = lin(n, (co, 0), (co, 1), (co, 2), (co, 3))
    e5 = lin(n, (fmul(fsub(ONE, I), INVSQ2), 3))       # (1-i)/sqrt2 * eps_4
    es = [e1, e2, e3, e4, e5]
    gens = []
    for e in es[:4]:
        gens += Ogens(I, e)
    return analyse("K = K_31  (n = 4, s = 5),  K' = K_29", es,
                   expected_order=46080 if full_order else None,
                   Kprime_size=7680 if full_order else None,
                   lattice_gens=gens, n=n, extra=k31_extra)


def k31_extra(D):
    n, es, Rs, L = D['n'], D['es'], D['Rs'], D['L']
    b = D['b']
    say("")
    say("  --- K' = K_29 is a Case-1 group of Sec. 4.7.4 of [P] ---")
    from check_lat import cyc_ring, stable_under, lat_C
    ring = cyc_ring(b, 0, 4)
    say("    Z[Tr K']_1 = Z-span of " +
        ", ".join(fstr(tuple(x)) for x in ring.basis()))
    check("Z[Tr K']_1 = Z[i]", ring == lat_C([ONE, I]))
    say("    the edges of the graph of K' carry |c_jk| = 1:")
    for (j, k) in [(0, 1), (0, 2), (1, 2), (1, 3)]:
        c = fmul(b[(j, k)], b[(k, j)])
        check(f"|c_{j+1}{k+1}| = 1", fmul(c, fconj(c)) == ONE)
    # Corollary 4.7.2: Gamma_k = (id - R_k)Gamma_j whenever |c_jk| = 1
    idm = mat_id(n)
    for (j, k) in [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4)]:
        c = fmul(b[(j, k)], b[(k, j)])
        if fmul(c, fconj(c)) != ONE:
            continue
        Dk = mat_sub(idm, Rs[k], n)
        Gj = line_part(L, es[j], n)
        Gk = line_part(L, es[k], n)
        check(f"Corollary 4.7.2: Gamma_{k+1} = (id-R_{k+1})Gamma_{j+1}",
              apply_mat(Dk, Gj, n) == Gk)
    say("    hence every K'-invariant lattice is determined by Gamma_1,")
    say("    and Gamma_1 = lambda Z[i]e_1 because Z[i] is a principal ideal domain.")
    check("Lambda*(K) = Lambda too (no larger K-invariant lattice)",
          all(lat_invariant(D['Lstar'], R, n) for R in Rs) and D['Lstar'] == L)


def g422_five():
    """the five lattices between T and T*(K'), and their similarity classes --
    the exact analogue of the list 5 of Example 4.8.1 b) of [P] for n >= 3."""
    n = 2
    e1 = lin(n, (fmul(I, INVSQ2), 0), (fneg(INVSQ2), 1))
    e2 = lin(n, (INVSQ2, 0), (fneg(INVSQ2), 1))
    e3 = lin(n, (ONE, 1))
    es = [e1, e2, e3]
    Rs = [refl(e, n) for e in es]
    ep1, ep2 = E(n, 0), E(n, 1)
    L = lat_from_vecs([ep1, vscal(I, ep1), ep2, vscal(I, ep2)])
    vB = vscal(fscal(HALF, fadd(ONE, I)), vadd(ep1, ep2))
    vC = vscal(F(HALF), vadd(ep1, ep2))
    vD = vscal(F(HALF), vadd(ep1, vscal(I, ep2)))
    GA = L
    GB = lat_sum(L, lat_from_vecs([vB, vscal(I, vB)]))
    GC = lat_sum(L, lat_from_vecs([vC, vscal(I, vC)]))
    GD = lat_sum(L, lat_from_vecs([vD, vscal(I, vD)]))
    T = lat_from_vecs(Ogens(I, e1) + Ogens(I, e2))
    idm = mat_id(n)
    S = [tuple(ZERO for _ in range(n)) for _ in range(n)]
    for j in range(n):
        Dm = mat_sub(idm, Rs[j], n)
        S = [tuple(fadd(S[r][c], Dm[r][c]) for c in range(n)) for r in range(n)]
    Tstar = apply_mat(mat_inv(S, n), T, n)
    say("")
    say("  --- the five lattices between T and T*, as in list 5 of Sec. 4.8.1 b) ---")
    inter = intermediate_lattices(T, Tstar, n)
    check("there are five of them", len(inter) == 5)
    z = F(0, 1)
    cands = []
    for k in range(8):
        a = ONE
        for _ in range(k):
            a = fmul(a, z)
        for m in range(-4, 5):
            u = a
            st = fadd(ONE, I)
            if m >= 0:
                for _ in range(m):
                    u = fmul(u, st)
            else:
                for _ in range(-m):
                    u = fmul(u, finv(st))
            cands.append(u)
    names = [("Gamma_A", GA), ("Gamma_B", GB), ("Gamma_C", GC), ("Gamma_D", GD)]
    tally = []
    for G in inter:
        hit = [nm for (nm, X) in names if any(scale_lat(a, G, n) == X for a in cands)]
        inv = all(lat_invariant(G, R, n) for R in Rs)
        root = root_lattice(G, es, n) == G
        tally.append((T.index_in(G), hit, inv, root))
        say(f"    index {T.index_in(G)} over T: similar to {hit},"
            f" K-invariant {inv}, root lattice {root}")
    check("all five are K-invariant root lattices",
          all(t[2] and t[3] for t in tally))
    check("they represent all four similarity classes",
          len({tuple(t[1]) for t in tally}) == 4)
    check("Gamma_B occurs twice (T itself and T*)",
          sum(1 for t in tally if t[1] == ["Gamma_B"]) == 2)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "g422"):
        g422()
    if which in ("all", "k12"):
        k12()
    if which in ("all", "k31"):
        k31(full_order=("fast" not in sys.argv))
    if which in ("all", "g422"):
        g422_five()
    if which in ("all", "enum"):
        import enum_roots
        enum_roots.run_g422()
        enum_roots.run_k12()
    say("")
    say("=" * 78)
    say(f"checks: {CHECKS[1]}/{CHECKS[0]} passed")
    with open("check_lat.out", "w") as f:
        f.write("\n".join(OUT) + "\n")
