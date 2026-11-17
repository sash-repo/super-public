"""CP^{1|4} with one odd parameter: the classification in degree 3.

Over Spec Cee the retract CP^{1|4} = (1,1,1,1) is classified in [Ba1] by the
orbits of g = H^0(M; T_0), of dimension 19, on H^1(M; T_2), of dimension 10
(and the 1-dimensional H^1(M; T_4)).  With one odd parameter the odd degrees
enter, and for this retract the only non-zero odd layer in the range is

        H^1(M; T_3),   of dimension 8,

so what has to be added is [Ba1, Thm. 4.1] run in degree 3 instead of degree 2:
Statements 15.1, 15.2 and 15.3 of the text.  Four parts:

  A. the module:  H^1(T_3) = V (x) W with V the standard sl(2)-module and
     W = ker e = the span of the x^{-1} xi_ixi_jxi_k d/dx, on which aut(E)
     acts as the WHOLE of gl(W);
  B. the orbits: a class is a 2 x 4 matrix, the rank-1 locus is cut out by the
     six 2 x 2 minors (verified: their span is g-stable), and the orbit
     dimensions are 4 on it and 7 off it -- so there are exactly TWO orbits;
  C. the stabilizers, projected onto the sl(2) of the base: the whole of sl(2)
     for rank 2 (a twisted copy), a Borel for rank 1;
  D. the honest families exp(tau z) over Spec Lambda(tau): sdim_Cee v, which
     separates the two without any cohomology, and homogeneity by a gcd.
"""

from fractions import Fraction as Fr
from itertools import combinations
import sys

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank, nullspace
import oddpar as O
import famtools as F
from ns import SM

KS = [1, 1, 1, 1]
Q = 3
DEG = 8


# --------------------------------------------------------------- the 2x4 matrix
#
# The basis of H^1(T_3) produced by quot.Quot is
#
#   b0..b3 = x^{-1} xi_123, xi_124, xi_134, xi_234 d/dx        (weight  +1)
#   b4..b7 = x^{-1} xi_1234 d/dxi_1 .. d/dxi_4                 (weight  -1)
#
# and f(b0) = b7, f(b1) = -b6, f(b2) = b5, f(b3) = -b4, so that the pairs
# (b_a, f(b_a)) are the four copies of V.  Writing a class in the basis
# (b_a; f(b_a)) gives the matrix below, whose rank is the invariant.

def mat24(c):
    return [[c[0], c[1], c[2], c[3]],
            [c[7], -c[6], c[5], -c[4]]]


def minors(c):
    M = mat24(c)
    return [M[0][a] * M[1][b] - M[0][b] * M[1][a]
            for a, b in combinations(range(4), 2)]


def rk(c):
    return rank([[Fr(x) for x in row] for row in mat24(c)])


# --------------------------------------------------------- quadratic forms in c
#
# A minor is a quadratic form in c_0..c_7; the action of X in g on functions is
# (D_A F)(c) = sum_{i,j} A[i][j] c_j dF/dc_i.  A g-stable locus has a g-stable
# span of equations, which is what part B checks.

def quad_of_minor(a, b):
    """The minor (a,b) as a dict {(i,j): coeff} with i <= j."""
    row0 = {0: 0, 1: 1, 2: 2, 3: 3}                 # M[0][a] = c_{row0[a]}
    row1 = {0: (7, 1), 1: (6, -1), 2: (5, 1), 3: (4, -1)}
    out = {}
    i0, (j0, s0) = row0[a], row1[b]
    out[tuple(sorted((i0, j0)))] = out.get(tuple(sorted((i0, j0))), 0) + s0
    i1, (j1, s1) = row0[b], row1[a]
    k = tuple(sorted((i1, j1)))
    out[k] = out.get(k, 0) - s1
    return {k: Fr(v) for k, v in out.items() if v}


def derivative(A, q, n=8):
    """D_A q, again a quadratic form."""
    out = {}

    def add(k, v):
        k = tuple(sorted(k))
        out[k] = out.get(k, Fr(0)) + v

    for (i, j), coef in q.items():
        # q = coef * c_i c_j  (i <= j, possibly i == j)
        for (p, r) in ((i, j), (j, i)):
            # differentiate in c_p, substitute (A c)_p
            fac = coef if i != j else 2 * coef
            if i == j and (p, r) == (j, i):
                continue                     # counted once, with the factor 2
            for t in range(n):
                if A[p][t]:
                    add((t, r), fac * A[p][t])
    return {k: v for k, v in out.items() if v}


def as_vector(q, keys):
    return [q.get(k, Fr(0)) for k in keys]


def generic_odd_rank(M, deg, chart="U"):
    """The rank of  v_od --> T_p(M)_od  at a point in general position."""
    ev = M.evaluations(deg)[chart]["odd"]
    best = 0
    for x0 in (Fr(1), Fr(2), Fr(3), Fr(-1), Fr(5), Fr(7)):
        rows = []
        for A, B in ev:
            rows.append([sum(Fr(c) * x0 ** e for e, c in (b or {}).items())
                         for b in B])
        best = max(best, rank(rows))
    return best


def image_in_vect(M, deg):
    """The image of v_ev in H^0(CP^1; Theta) = sl(2): which fields of the base
    actually lift.  A field is read on the reduction, where its d/dx
    coefficient is a polynomial in x of degree <= 2."""
    rows = [A.get((), {}) for A, B in M.vector_fields(0, deg)]
    rows = [a for a in rows if a]
    degs = sorted({e for a in rows for e in a})
    mat = [[Fr(a.get(e, 0)) for e in degs] for a in rows]
    return (rank(mat) if mat else 0), degs


def main():
    print(__doc__, flush=True)
    L = O.Layer(KS, Q)
    n = L.dim

    # ------------------------------------------------------------------- A
    print("=" * 74)
    print("A.  The module H^1(M; T_3)")
    print("=" * 74)
    print(f"dim H^1(M; T_3) = {n},   dim g = {len(L.gens)} "
          f"(= 3 + 16 = sl(2) + gl(4))")
    print(f"[e,f] = h, [h,e] = 2e, [h,f] = -2f on it: {L.check_sl2()}")
    print(f"h-weights: {L.h_weights()}")
    print(f"sl(2)-decomposition (top weight -> multiplicity): "
          f"{L.sl2_decomposition()}   i.e. V^(+4), V the standard module")
    hw = nullspace(L.mat("e"))
    print(f"\nW := ker e, of dimension {len(hw)}:")
    for v in hw:
        print("    ", F.show(L.Q, v))
    print("so H^1(M; T_3) = V (x) W with W = Lambda^3 E, and the pairing is")
    Fm = L.mat("f")
    for j in range(4):
        col = [Fm[i][j] for i in range(n)]
        nz = [(i, col[i]) for i in range(n) if col[i]]
        print(f"     f(b{j}) = " + " + ".join(f"{c}*b{i}" for i, c in nz))

    # the weights of the diagonal torus of gl(4) identify W as Lambda^3 E
    print("\nthe weights of the diagonal torus of aut(E) = gl(4) on the basis "
          "(a 1 in the\nslot i means that xi_i d/dxi_i acts by 1):")
    tor = [L.mat(f"x^0 xi_{i} d/dxi_{i}") for i in range(1, 5)]
    for j in range(n):
        w = [tor[i][j][j] for i in range(4)]
        off = any(tor[i][a][j] for i in range(4) for a in range(n) if a != j)
        print(f"    b{j}: {w}{'  (NOT diagonal!)' if off else ''}")
    print("so W = ker e carries the weights eps_i + eps_j + eps_k, i.e. "
          "W = Lambda^3 E,")
    print("and the second row of the matrix carries the same weights, "
          "as it must.")

    # aut(E) acts on W as the whole of gl(W)
    autmats = []
    for nm, _ in L.gens:
        if nm in ("e", "h", "f"):
            continue
        M = L.mat(nm)
        autmats.append([M[i][j] for i in range(4) for j in range(4)])
    r = rank(autmats)
    print(f"\nthe image of aut(E) = gl(4) in gl(W) has dimension {r} "
          f"(= 16 = dim gl(W)): aut(E) acts on W as the WHOLE of gl(W),")
    print("    W being Lambda^3E = E^* (x) det E, so gl(4) --> gl(W) is "
          "X |--> -X^t + tr(X).")
    print("Whence the group acting on P(H^1) is SL(V) x GL(W) and a class is "
          "a 2 x 4 matrix.")

    # ------------------------------------------------------------------- B
    print()
    print("=" * 74)
    print("B.  The orbits: the rank stratification of a 2 x 4 matrix")
    print("=" * 74)
    keys = sorted({k for a, b in combinations(range(4), 2)
                   for k in quad_of_minor(a, b)})
    span = [as_vector(quad_of_minor(a, b), keys)
            for a, b in combinations(range(4), 2)]
    r0 = rank(span)
    print(f"the six 2 x 2 minors are linearly independent: rank {r0} of 6")
    bad = []
    for nm, _ in L.gens:
        A = L.mat(nm)
        for a, b in combinations(range(4), 2):
            d = derivative(A, quad_of_minor(a, b))
            if rank(span + [as_vector(d, keys)]) != r0:
                bad.append((nm, a, b))
    print(f"the span of the minors is g-stable: {not bad} "
          f"({len(bad)} failures over {len(L.gens)} generators x 6 minors)")
    print("hence the rank-1 locus -- the cone over the Segre variety "
          "P^1 x P^3 -- is g-stable.")

    print("\norbit dimensions in P(H^1) = P^7, by the rank of the matrix:")
    samples = []
    for i in range(n):                                   # the basis classes
        samples.append([1 if j == i else 0 for j in range(n)])
    for i, j in combinations(range(n), 2):               # pairs
        samples.append([1 if k in (i, j) else 0 for k in range(n)])
        samples.append([1 if k == i else (2 if k == j else 0) for k in range(n)])
    samples += [[1] * n, [2, 3, 5, 7, 11, 13, 17, 19], [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 2, 3, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 3, 4],
                [3, 1, 4, 1, 5, 9, 2, 6], [1, -1, 1, -1, 1, -1, 1, -1]]
    tally = {}
    for s in samples:
        v = [Fr(c) for c in s]
        if not any(v):
            continue
        tally.setdefault((rk(v), L.orbit_dim_proj(v)), []).append(s)
    for (r1, od), vs in sorted(tally.items()):
        print(f"   rank {r1}:  orbit dimension {od}   "
              f"({len(vs)} of the {len(samples)} classes tested)")
    print(f"\n   {len(samples)} classes, no rank with two different orbit "
          f"dimensions: {len({r1 for r1, _ in tally}) == len(tally)}")
    print("""
   The rank-1 locus is irreducible of projective dimension 1 + 3 = 4 and the
   orbit through each of its points is of dimension 4, so it is ONE orbit;
   its complement is connected and every orbit in it is open, so it, too, is
   ONE orbit.  Hence: exactly two orbits, and no moduli.""")

    # ------------------------------------------------------------------- C
    print("=" * 74)
    print("C.  The stabilizers, and the sl(2) they contain")
    print("=" * 74)
    reps = {"rank 2": [1, 0, 0, 0, 1, 0, 0, 0],
            "rank 1": [1, 0, 0, 0, 0, 0, 0, 0]}
    for nm, s in reps.items():
        v = [Fr(c) for c in s]
        # the stabilizer of the LINE [v]: X.v in Cee v
        rows = []
        for gname, _ in L.gens:
            M = L.mat(gname)
            rows.append([sum(M[i][j] * v[j] for j in range(n)) for i in range(n)])
        # solve  sum_k t_k (X_k.v) = lam v
        aug = [[rows[k][i] for k in range(len(rows))] + [-v[i]] for i in range(n)]
        sol = nullspace(aug)
        dim_stab = len(sol)
        # the projection of the stabilizer onto sl(2) = the first three slots
        proj = [[u[0], u[1], u[2]] for u in sol]
        dp = rank(proj) if proj else 0
        print(f"\n   {nm}:  class {F.show(L.Q, v)}")
        print(f"      dim of the stabilizer of the line = {dim_stab} "
              f"(= {len(L.gens)} - {L.orbit_dim_proj(v)} - 1 + 1)")
        print(f"      its projection to the sl(2) of CP^1 has dimension {dp}"
              f"  ({'the whole of sl(2)' if dp == 3 else 'a Borel: solvable'})")
        if dp == 3:
            print("      so the stabilizer contains a copy of sl(2) mapping "
                  "isomorphically onto")
            print("      the sl(2) of the base -- a TWISTED one: a rank-2 "
                  "element of V (x) W is an")
            print("      injection V^* --> W, and the copy acts on its image "
                  "through that injection,")
            print("      not through the standard action on the base.")
            # is the standard sl(2) in the stabilizer?
            std = []
            for i in range(3):
                e = [Fr(1) if k == i else Fr(0) for k in range(len(L.gens))]
                tgt = [sum(e[k] * rows[k][i2] for k in range(len(rows)))
                       for i2 in range(n)]
                std.append(any(tgt))
            print(f"      the standard sl(2) does NOT stabilize it: "
                  f"e, h, f move the class: {std}")

    # ------------------------------------------------------------------- D
    print()
    print("=" * 74)
    print("D.  The two supermanifolds over Spec Lambda(tau)")
    print("=" * 74)
    models = [("split", SM.split(KS, npar=1))]
    for nm, s in reps.items():
        models.append((nm, F.cocycle_to_family(
            KS, F.vec_to_coc(L.Q, [Fr(c) for c in s]), npar=1)))
    d0 = models[0][1].sdim_vect(DEG)
    for nm, M in models:
        d, t = M.sdim_vect(DEG), M.transitive(DEG)
        print(f"\n   {nm}:  sdim_Cee v = {d},  homogeneous {t},  "
              f"non-split {d != d0}", flush=True)
        M.transitive(DEG, verbose=True)
        print(f"      generic rank of the evaluation onto the "
              f"{len(KS)} odd directions: "
              f"{generic_odd_rank(M, DEG)} of {len(KS)}", flush=True)
        r, degs = image_in_vect(M, DEG)
        print(f"      the image of v_ev in H^0(CP^1; Theta) = sl(2) is of "
              f"dimension {r}: the fields x^e d/dx with e in {degs}"
              f"{'  = the whole of sl(2)' if r == 3 else '  = a Borel'}",
              flush=True)
    print("""
   So the odd degree of the retract CP^{1|4} carries exactly two supermanifolds
   over Spec Lambda(tau) besides the split one, told apart by sdim_Cee v alone
   -- the same shape as the retract (3,3) in odd dimension 2, with the 2 x 2
   matrix replaced by a 2 x 4 one.

   But NEITHER is homogeneous, and the rank-2 one is the instructive case: its
   stabilizer contains a twisted copy of sl(2) mapping isomorphically onto the
   sl(2) of the base, that copy does lift -- the EVEN part of the evaluation is
   onto at every point of CP^1, so the supermanifold is ev-homogeneous -- and
   the odd part collapses to rank 1 of 4 everywhere.  So the criterion of
   Section 10 is not an equivalence (Remark 10.2): a lifted sl(2) is necessary
   and not sufficient, and the lifting of the fields of degree -1
   is independent of it.  Over (3,3) they happened to; here they do not.""")


if __name__ == "__main__":
    sys.exit(main())
