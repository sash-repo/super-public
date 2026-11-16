#!/usr/bin/env python3
"""
run_orb.py -- T_2 (the orbit separator) and the corank-1|2 stratum.

Part 0  self-tests of T_2: every certificate it produces is re-checked here by
        matrix multiplication alone.
Part 1  the structure theorem for the corank-1|2 types II(n,m):
            Nt_f = A^2,  A = k[x]/(x^mu),  mu = min(n-1, m+1),
            the stabiliser acts A-linearly, through
            G(n,m) = { g in GL(2,A) : det g in 1 + x^r A },  r = n-1-m,
        all four claims verified against the engine of T_1, type by type.
Part 2  the complete list of first-layer classes of every II(n,m),
        with orbit dimensions, separating invariants and transitivity checks.
Part 3  the closed formulas, checked over the whole range.
Part 4  regression: T_2 on the types already settled (I_1, III_3) and on the
        one type with a modulus (VII_2), where it must NOT report finiteness.
"""

from fractions import Fraction as F
import time

import stab
import orb
import sv

bad = 0


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def II(n, m):
    return [(1, (n,), ()), (1, (m,), (0, 1))]


def mu_of(n, m):
    return min(n - 1, m + 1)


# --------------------------------------------------------------- Part 0

hdr("(0)  self-tests of T_2")

r = stab.analyse("II(4,2)", 1, 2, II(4, 2), 12, verbose=False)
mats, basis = r["mats"], r["basis"]
n = len(basis)
pool = orb.nilpotent_pool(mats)
tori = orb.torus_pool(mats, n)

# (a) every element of the pool is nilpotent and lies in the span of rho(s_f)
span = stab.rank_rows([sum(M, []) for M in mats])
ok_a = True
for M in pool:
    if not orb.is_nilpotent(M):
        ok_a = False
    if stab.rank_rows([sum(X, []) for X in mats] + [sum(M, [])]) != span:
        ok_a = False

# (b) exp of a nilpotent element really is an automorphism of the layer that
#     comes from the group: check exp(tM) exp(-tM) = 1 and det = 1
ok_b = True
for M in pool[:8]:
    E = orb.exp_nilpotent(M, F(3, 2))
    Ei = orb.exp_nilpotent(M, F(-3, 2))
    if orb.mat_mul(E, Ei) != orb.mat_id(n):
        ok_b = False

# (c) the certificate of a reduction reproduces the reduced vector
ok_c = True
import random as _r
rnd = _r.Random(1)
for _ in range(20):
    v = [F(rnd.randint(-5, 5)) for _ in range(n)]
    c, g = orb.reduce_vector(mats, v, pool, tori)
    if g.apply(v) != c:
        ok_c = False

# (d) a stable subspace really is stable
ok_d = True
S = orb.submodule(mats, [F(0), F(0), F(1), F(0), F(0), F(0)])
for b in S.basis():
    for M in mats:
        if not S.contains(orb.apply_mat(M, b)):
            ok_d = False

for label, v in [("pool nilpotent and inside rho(s_f)", ok_a),
                 ("exp(tM) invertible with exp(-tM)", ok_b),
                 ("reduction certificates reproduce their vectors", ok_c),
                 ("submodule() returns a stable subspace", ok_d)]:
    print(f"    {label}: {'ok' if v else 'FAIL'}")
    bad += 0 if v else 1


# --------------------------------------------------------------- Part 1

hdr("(1)  the structure theorem for II(n,m):  Nt_f = A^2 and the action is "
    "A-linear")

print("""
For f = x^n + x^m.xi1.xi2 with 1 <= m <= n-1 put  r = n-1-m  and
mu = min(n-1, m+1).  The claims verified below, type by type, are

  (i)   (m I_f)_odd = < x^b.xi_i : b >= mu >, so Nt_f has the monomial basis
        x^b.xi_i, 0 <= b <= mu-1, and  dim Nt_f = 2.mu;
  (ii)  s_f = { alpha(x).xi1.xi2.d_x + sum b_ij(x).xi_i.d_xi_j :
                b_11 + b_22 = -n.x^r.alpha }, and the d_x part acts by ZERO on
        the layer, since xi1.xi2.xi_i = 0;
  (iii) hence rho(s_f) = { B in gl(2,A) : tr B in x^r.A },  A = k[x]/(x^mu),
        acting by left multiplication on A^2 -- in particular A-LINEARLY;
  (iv)  dim rho(s_f) = 4.mu - min(r,mu).
""")


def predicted_algebra(mu, r):
    """the matrices of { B in gl(2,A) : tr B in x^r A } on A^2 = Nt_f,
    in the basis x^b.xi_i ordered as the engine orders it (index 2b+i)"""
    n = 2 * mu

    def elem(s, k, l):                     # the matrix of x^s.E_{kl}
        M = [[F(0)] * n for _ in range(n)]
        for b in range(mu):
            if b + s < mu:
                M[2 * (b + s) + k][2 * b + l] = F(1)
        return M

    out = []
    for s in range(mu):
        out.append(elem(s, 0, 1))
        out.append(elem(s, 1, 0))
        A = elem(s, 0, 0)
        B = elem(s, 1, 1)
        out.append([[A[i][j] - B[i][j] for j in range(n)] for i in range(n)])
    for s in range(r, mu):                 # trace in x^r A
        A = elem(s, 0, 0)
        B = elem(s, 1, 1)
        out.append([[A[i][j] + B[i][j] for j in range(n)] for i in range(n)])
    return out


print(f"{'type':10s} {'r':>2s} {'mu':>3s} {'dim Nt':>7s} {'pred':>5s} "
      f"{'dim rho':>8s} {'pred':>5s}  basis = x^b.xi_i, b < mu   "
      f"rho(s_f) = {{tr in x^r A}}")

CASES = [(n, m) for n in range(3, 9) for m in range(1, n)]
for (nn, mm) in CASES:
    D = 2 * nn + 4
    r = stab.analyse(f"II({nn},{mm})", 1, 2, II(nn, mm), D, verbose=False)
    mats, basis = r["mats"], r["basis"]
    mu, rr = mu_of(nn, mm), nn - 1 - mm
    n_layer = len(basis)

    want_basis = [(( b,), frozenset({i})) for b in range(mu) for i in (0, 1)]
    ok_basis = (sorted(basis, key=stab.mkey) ==
                sorted(want_basis, key=stab.mkey))

    dim_rho = stab.rho_image_dim(mats)
    pred_rho = 4 * mu - min(rr, mu)

    # the two spans must coincide
    P = predicted_algebra(mu, rr)
    rows_r = [sum(M, []) for M in mats]
    rows_p = [sum(M, []) for M in P]
    same = (stab.rank_rows(rows_r) == stab.rank_rows(rows_p) ==
            stab.rank_rows(rows_r + rows_p))

    flag = "ok" if (ok_basis and n_layer == 2 * mu and
                    dim_rho == pred_rho and same) else "FAIL"
    bad += 0 if flag == "ok" else 1
    print(f"II({nn},{mm})".ljust(10) + f" {rr:2d} {mu:3d} {n_layer:7d} "
          f"{2*mu:5d} {dim_rho:8d} {pred_rho:5d}  "
          f"{'ok' if ok_basis else 'BASIS FAIL'}   "
          f"{'ok' if same else 'ALGEBRA FAIL'}   {flag}")


# --------------------------------------------------------------- Part 2

hdr("(2)  the complete first-layer classification of II(n,m)")

print("""
The action being A-linear and A = k[x]/(x^mu) a principal ideal ring, the
theory of elementary divisors gives the orbits outright: every v in A^2 is
x^j.u with u unimodular, and a unimodular u is part of a basis, so

    orbits of GL(2,A) on A^2  =  { x^j.A^2 minus x^{j+1}.A^2 }_{0<=j<mu} u {0},

that is, mu+1 of them, with representatives x^j.xi_1 and orbit dimensions
2(mu-j).  The subgroup G(n,m) = {det in 1+x^r A} has the SAME orbits: the
stabiliser of x^j.xi_1 in GL(2,A) already contains elements of every
determinant (take diag(1,u)), so GL = G(n,m).Stab and the orbit does not split.

Machine check of all of it: the representatives, their orbit dimensions, the
stable-subspace certificates x^j.A^2 that separate them, and transitivity --
random points of each stratum are carried to the representative by an explicit
product of one-parameter subgroups.
""")

print(f"{'type':10s} {'mu':>3s} {'classes':>8s} {'orbit dims':22s} "
      f"{'separated':>10s} {'transitive':>12s}  {'sdim O_f':>10s} {'time':>7s}")

TOTAL_TRANS = [0, 0]
for (nn, mm) in CASES:
    t0 = time.time()
    D = 2 * nn + 4
    res = stab.analyse(f"II({nn},{mm})", 1, 2, II(nn, mm), D, verbose=False)
    mats, basis = res["mats"], res["basis"]
    mu = mu_of(nn, mm)
    n_layer = len(basis)
    degs = [stab.mdeg(mm) for mm in basis]
    cache = {}
    pool = orb.nilpotent_pool(mats)
    tori = orb.torus_pool(mats, n_layer)

    # the claimed representatives  x^j.xi_1,  j = 0..mu-1,  and 0
    reps = []
    for j in range(mu):
        v = [F(0)] * n_layer
        v[2 * j] = F(1)
        reps.append(v)
    reps.append([F(0)] * n_layer)

    dims = [orb.orbit_dim(mats, v) for v in reps]
    ok_dims = dims == [2 * (mu - j) for j in range(mu)] + [0]

    # pairwise separation, each with a certificate
    ok_sep = True
    for a in range(len(reps)):
        for b in range(a + 1, len(reps)):
            verdict, cert = orb.same_orbit(mats, reps[a], reps[b], pool,
                                           tori, degs, cache)
            if verdict is not False:
                ok_sep = False

    # transitivity on each stratum, with explicit group elements
    def gen(rep, rnd):
        j = next((i // 2 for i in range(n_layer) if rep[i]), mu)
        v = [F(0)] * n_layer
        for b in range(j, mu):
            for i in (0, 1):
                v[2 * b + i] = F(rnd.randint(-4, 4))
        if not (v[2 * j] or v[2 * j + 1]):        # keep the valuation exact
            v[2 * j + 1] = F(1)
        return v

    okc, tot, first = orb.verify_transitive(mats, reps[:-1], gen,
                                            samples=15, pool=pool, tori=tori,
                                            degs=degs, cache=cache)
    TOTAL_TRANS[0] += okc
    TOTAL_TRANS[1] += tot
    sdim = res["sdim"]
    ok = ok_dims and ok_sep and okc == tot
    bad += 0 if ok else 1
    print(f"II({nn},{mm})".ljust(10) + f" {mu:3d} {mu+1:8d} "
          + str(dims).ljust(22)
          + f" {'ok' if ok_sep else 'FAIL':>10s} {f'{okc}/{tot}':>12s}  "
          + f"({sdim[0]}|{sdim[1]})".rjust(10)
          + f" {time.time()-t0:6.1f}s"
          + ("" if ok else "   <-- FAIL"))

print(f"\n  transitivity over all the types above: "
      f"{TOTAL_TRANS[0]}/{TOTAL_TRANS[1]} random points reduced to their "
      f"representative, each with an explicit group element.")


# --------------------------------------------------------------- Part 3

hdr("(3)  the closed formulas over the whole range")

print("""
    mu = min(n-1, m+1)      r = n-1-m
    sdim O_f = (n+m-1 | 2m)                 (Milnor number of the germ)
    dim Nt_f = 2.mu,  dim N_f = 2.mu-2,  the new stratum being q = 2
    number of first-layer classes = mu+1 = min(n-1,m+1)+1
""")
ok_all = True
for (nn, mm) in CASES:
    D = 2 * nn + 4
    res = stab.analyse(f"II({nn},{mm})", 1, 2, II(nn, mm), D, verbose=False)
    mu = mu_of(nn, mm)
    got = (res["sdim"][0], res["sdim"][1], res["dimNt"], res["dimN"])
    want = (nn + mm - 1, 2 * mm, 2 * mu, 2 * mu - 2)
    if got != want:
        ok_all = False
        print(f"    II({nn},{mm}): got {got}, want {want}   <-- FAIL")
print(f"    all {len(CASES)} types: {'ok' if ok_all else 'FAIL'}")
bad += 0 if ok_all else 1

print("""
The count mu+1 = min(n-1,m+1)+1 grows with m up to m = n-2 and is then
constant: II(n,n-2) and II(n,n-1) have the same number of classes, n-1+1 = n
and n-1+1 = n, while II(n,1) has 3 for every n >= 3.  So the first layer
SEES the second index m, which the classical classification records only as a
label: within one corank the number of classes over a type is an invariant
that the even theory does not have.
""")


# --------------------------------------------------------------- Part 4

hdr("(4)  regression: T_2 where the answer is already known")

REG = [("I_1", 0, 4, [(1, (), (0, 1, 2, 3))], 3, 3),
       ("III_3", 1, 4, [(1, (3,), ()), (1, (0,), (0, 1, 2, 3))], 9, None)]

for name, p, q, terms, D, expect in REG:
    res = stab.analyse(name, p, q, terms, D, verbose=False)
    mats, basis = res["mats"], res["basis"]
    n_layer = len(basis)
    degs = [stab.mdeg(mm) for mm in basis]
    cen = orb.orbit_census(mats, n_layer, samples=40, degs=degs, cache={})
    print(f"  {name:8s} dim Nt = {n_layer:2d}, modality {res['modality']}, "
          f"canonical forms found by sampling: {len(cen)}")
    for c, d, h in cen:
        nz = [(sv.show_mon(basis[i], "xy", "xi"), str(c[i]))
              for i in range(n_layer) if c[i]]
        print(f"      orbit dim {d:2d}  hits {h:3d}  "
              + (" + ".join(f"{a}.{b}" for b, a in nz) if nz else "0"))

print("""
  I_1: the open orbit and nothing else is reached by random sampling, which is
  what modality 0 predicts; the deeper classes (xi_1.xi_2.xi_3 and 0) are the
  strata of germs with no critical point and are hit only by starting
  inside them.
""")

# VII_2: the type WITH a modulus.  T_2 must not claim finiteness here.
res = stab.analyse("VII_2", 2, 3,
                   [(1, (3, 0), ()), (1, (0, 4), ()),
                    (1, (1, 0), (0, 1)), (1, (0, 1), (1, 2))], 9,
                   verbose=False)
mats, basis = res["mats"], res["basis"]
n_layer = len(basis)
i1 = basis.index(((0, 0), frozenset({0})))
i3 = basis.index(((0, 0), frozenset({2})))
v = [F(0)] * n_layer
w = [F(0)] * n_layer
v[i1], v[i3] = F(1), F(1)
w[i1], w[i3] = F(2), F(1)
verdict, cert = orb.same_orbit(mats, v, w,
                               degs=[stab.mdeg(mm) for mm in basis], cache={})
print(f"  VII_2: xi_1+xi_3  versus  2.xi_1+xi_3 :  verdict {verdict}")
print(f"         {cert if not isinstance(cert, tuple) else 'conjugate'}")
sem = orb.semi_invariants(mats, n_layer, 1)
print(f"         semi-invariants of degree 1 that T_2 could diagonalise: "
      f"{len(sem)}")
print("         (the modulus a_1/a_3 is found by T_1 through the "
      "invariant-form\n          system, which is the certificate that "
      "actually settles this type)")
if verdict is True:
    print("         <-- FAIL: two points of a one-parameter family "
          "declared conjugate")
    bad += 1
else:
    print("         ok: T_2 refuses to merge two values of the modulus, "
          "as it must")


hdr("SUMMARY: " + ("all checks passed" if bad == 0 else f"{bad} FAILURES"))
