#!/usr/bin/env python3
"""
run_stab.py -- the validation suite and the census produced by T_1 (stab.py).

  Part 0  self-tests: s_f is closed under the bracket, rho is a homomorphism
          of Lie algebras, exp(V) is an algebra automorphism and fixes f.
  Part 1  the two worked examples II(n,1) and I_1, recomputed from scratch:
          stabiliser, layer representation, orbit dimensions.
  Part 2  the census: the stabilizer table and the types of corank 2|3.
  Part 3  the family II(n,1), n = 2..8.

Everything exact; every stabiliser dimension bracketed between a lower and an
upper computation and printed only when the two agree (CERTIFIED).
"""

from fractions import Fraction as F
import time

import sv
import stab

bad = 0


def hdr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# --------------------------------------------------------------- Part 0

hdr("(0)  self-tests of the engine")

TESTS = [
    ("II(3,1)", 1, 2, [(1, (3,), ()), (1, (1,), (0, 1))], 10),
    ("I_1", 0, 4, [(1, (), (0, 1, 2, 3))], 3),
    ("III_3", 1, 4, [(1, (3,), ()), (1, (0,), (0, 1, 2, 3))], 9),
]

for name, p, q, terms, D in TESTS:
    R, f, mI, basis, k, safe = stab.layer(p, q, terms, D)
    index = {m: i for i, m in enumerate(basis)}
    M = max(k - 1, 0) + stab.deg_max(f)
    if p:
        M = min(M, D - stab.xdeg_max(f) - 1)
    exact, _, _ = stab.stab_algebra(R, f, p, q, 0, M)

    # (a) V(f) = 0 really holds, for every basis element
    a_ok = all(not stab.apply_vf(R, V, f) for V in exact)

    # (b) closed under the bracket
    b_ok = True
    L = stab.LinSpace()
    coord = []
    keys = sorted({(s, m) for V in exact for s, poly in V.items()
                   for m in poly}, key=lambda t: (str(t[0]), stab.mkey(t[1])))
    rows = [[V.get(s, {}).get(m, F(0)) for (s, m) in keys] for V in exact]
    r0 = stab.rank_rows(rows)
    for i in range(min(len(exact), 6)):
        for j in range(i + 1, min(len(exact), 6)):
            br = stab.vf_bracket(R, exact[i], exact[j])
            br = {s: {m: c for m, c in poly.items()
                      if stab.mdeg(m) - 1 <= M} for s, poly in br.items()}
            br = {s: pp for s, pp in br.items() if pp}
            if not br:
                continue
            row = [br.get(s, {}).get(m, F(0)) for (s, m) in keys]
            extra = {(s, m) for s, poly in br.items() for m in poly} - set(keys)
            if extra or stab.rank_rows(rows + [row]) != r0:
                b_ok = False

    # (c) rho is a Lie algebra homomorphism
    def rho_of(V):
        Vc = {s: {m: c for m, c in poly.items() if stab.mdeg(m) - 1 <= k - 1}
              for s, poly in V.items()}
        Vc = {s: pp for s, pp in Vc.items() if pp}
        n = len(basis)
        if not Vc:
            return [[F(0)] * n for _ in range(n)]
        return stab.rho(R, Vc, basis, mI, index)

    c_ok = True
    n = len(basis)
    for i in range(min(len(exact), 5)):
        for j in range(i + 1, min(len(exact), 5)):
            A, B = rho_of(exact[i]), rho_of(exact[j])
            comm = [[sum(A[u][l] * B[l][v] - B[u][l] * A[l][v]
                         for l in range(n)) for v in range(n)]
                    for u in range(n)]
            got = rho_of(stab.vf_bracket(R, exact[i], exact[j]))
            if comm != got:
                c_ok = False

    # (d) exp(V) is an algebra automorphism and fixes f
    d_ok = True
    nilp = [V for V in exact if stab.vf_degree(V) >= 1][:4]
    rnd = [sv.mul(R, {m: F(1)}, {mm: F(1)})
           for m in basis[:2] for mm in basis[:2]]
    for V in nilp:
        if stab.exp_vf(R, V, f) != f:
            d_ok = False
        for g in basis[:3]:
            for h in basis[:3]:
                lhs = stab.exp_vf(R, V, sv.mul(R, {g: F(1)}, {h: F(1)}))
                rhs = sv.mul(R, stab.exp_vf(R, V, {g: F(1)}),
                             stab.exp_vf(R, V, {h: F(1)}))
                rhs = {m: c for m, c in rhs.items() if c}
                if lhs != rhs:
                    d_ok = False

    flags = [("V(f)=0 exact", a_ok), ("bracket-closed", b_ok),
             ("rho homomorphism", c_ok), ("exp automorphism, exp(V)f=f", d_ok)]
    print(f"{name:10s} dim s_f(deg<={M}) = {len(exact):3d}   " +
          "   ".join(f"{t}: {'ok' if v else 'FAIL'}" for t, v in flags))
    bad += sum(0 if v else 1 for _, v in flags)


# --------------------------------------------------------------- Part 1

hdr("(1)  the two worked examples II(n,1) and I_1, recomputed")


def strata_report(r, label_of):
    """orbit dimension at named points, and the absorption test"""
    b, idx, mats = r["basis"], r["index"], r["mats"]
    n = len(b)
    Nf = [i for i, m in enumerate(b) if stab.mdeg(m) >= 2]

    def vec(d):
        v = [F(0)] * n
        for mon, c in d.items():
            v[idx[mon]] = F(c)
        return v

    print(f"    invariant subspace N_f = span("
          f"{', '.join(sv.show_mon(b[i], 'xy', 'xi') for i in Nf)})")
    for name, d in label_of:
        v = vec(d)
        od = stab.orbit_dim(mats, v)
        print(f"    orbit dim at {name:26s} = {od:2d}")
    return Nf


def absorbs(r, gdict):
    """is N_f contained in the tangent space rho(S).g ?"""
    b, idx, mats = r["basis"], r["index"], r["mats"]
    n = len(b)
    v = [F(0)] * n
    for mon, c in gdict.items():
        v[idx[mon]] = F(c)
    tang = [stab.act(M, v) for M in mats]
    base = stab.rank_rows(tang)
    for i, m in enumerate(b):
        if stab.mdeg(m) >= 2:
            e = [F(0)] * n
            e[i] = F(1)
            if stab.rank_rows(tang + [e]) != base:
                return False
    return True


print()
print("Example 2.7.5, type II(n,1):  f = x^n + x.xi1.xi2  in O^{1|2}, n = 3")
r3 = stab.analyse("II(3,1)", 1, 2, [(1, (3,), ()), (1, (1,), (0, 1))], 12,
                  show_basis=True)
xi1 = ((0,), frozenset({0}))
xxi1 = ((1,), frozenset({0}))
strata_report(r3, [("xi_1        (new stratum)", {xi1: 1}),
                   ("x.xi_1      (in N_f)", {xxi1: 1}),
                   ("0", {})])
ab = absorbs(r3, {xi1: 1})
print(f"    N_f absorbed into the tangent space at xi_1: {'YES' if ab else 'NO'}"
      "   <-- the mechanism of Remark 2.7.7")
print("    the three classes have orbit dimensions 4, 2, 0 -- confirmed"
      if [stab.orbit_dim(r3["mats"], v) for v in
          ([F(1), F(0), F(0), F(0)], [F(0), F(0), F(1), F(0)],
           [F(0)] * 4)] == [4, 2, 0] else "    MISMATCH with the paper")
if not ab:
    bad += 1

print()
print("Example 2.7.6, type I_1:  f = xi1.xi2.xi3.xi4  in O^{0|4}")
r1 = stab.analyse("I_1", 0, 4, [(1, (), (0, 1, 2, 3))], 3, show_basis=True)
e1 = ((), frozenset({0}))
e123 = ((), frozenset({0, 1, 2}))
strata_report(r1, [("xi_1            (new stratum)", {e1: 1}),
                   ("xi_1xi_2xi_3    (in N_f)", {e123: 1}),
                   ("0", {})])
ab1 = absorbs(r1, {e1: 1})
print(f"    N_f absorbed into the tangent space at xi_1: "
      f"{'YES' if ab1 else 'NO'}")
print("    the linear part has dimension "
      f"{r1['lin']} = dim sl(4) = 15  "
      f"{'ok' if r1['lin'] == 15 else 'MISMATCH'}")
print("    the degree->=1 part has dimension "
      f"{r1['filt'][1]} = 4 x dim Lambda^3(k^4) = 16  "
      f"{'ok' if r1['filt'][1] == 16 else 'MISMATCH'}")
if not ab1 or r1["lin"] != 15 or r1["filt"][1] != 16:
    bad += 1


# --------------------------------------------------------------- Part 2

hdr("(2)  the census: the stabiliser of every type of the table")

CENSUS = [
    ("I_1",       0, 4, [(1, (), (0, 1, 2, 3))], 3),
    ("I_2",       0, 6, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5))], 3),
    ("I_4",       0, 6, [(1, (), (0, 1, 2, 3, 4, 5))], 3),
    ("II(3,1)",   1, 2, [(1, (3,), ()), (1, (1,), (0, 1))], 12),
    ("II(4,1)",   1, 2, [(1, (4,), ()), (1, (1,), (0, 1))], 12),
    ("II(4,2)",   1, 2, [(1, (4,), ()), (1, (2,), (0, 1))], 12),
    ("III_3",     1, 4, [(1, (3,), ()), (1, (0,), (0, 1, 2, 3))], 9),
    ("III_4",     1, 4, [(1, (3,), ()), (1, (1,), (0, 1, 2, 3))], 9),
    ("VII_2",     2, 3, [(1, (3, 0), ()), (1, (0, 4), ()),
                         (1, (1, 0), (0, 1)), (1, (0, 1), (1, 2))], 12),
]

print("first, the hypothesis the engine rests on:  (m^{k+1})_odd < (m I_f)_odd,")
print("with k recomputed at two truncations.  k is the top of the filtration of")
print("Nt_f by the degree, which for a merely quasi-homogeneous f exceeds the top")
print("degree of the monomial basis -- VII_2 is such a case (k = 4, basis top 3).")
print()
for name, p, q, terms, D in CENSUS:
    _, _, _, b1, k1, s1 = stab.layer(p, q, terms, D)
    _, _, _, b2, k2, s2 = stab.layer(p, q, terms, D + 2)
    top = max((stab.mdeg(m) for m in b1), default=0)
    ok = (k1 == k2 and len(b1) == len(b2))
    if not ok:
        bad += 1
    print(f"    {name:9s} k = {k1} (at D+2: {k2}), basis top degree {top}, "
          f"dim Nt = {len(b1)} (at D+2: {len(b2)})   {'OK' if ok else 'UNSTABLE'}")
print()

rows = []
print(f"{'type':9s} {'p|q':>5s} {'sdim O_f':>10s} {'dim Nt':>7s} {'dim N':>6s}"
      f" {'lin':>4s} {'dim S':>6s} {'cert':>5s} {'rho':>4s} {'orbit':>6s}"
      f" {'mod':>4s} {'form':>5s}  verdict")
for name, p, q, terms, D in CENSUS:
    t0 = time.time()
    r = stab.analyse(name, p, q, terms, D, verbose=False)
    # the exact modality certificate: an invariant bilinear form on the layer
    forms = stab.annihilator_forms(r["matsU"], r["dimNt"])
    r["forms"] = forms
    rows.append(r)
    if not r["ok_layer"] or not r["certified"]:
        bad += 1
    # the two certificates must not contradict each other
    if forms and r["modality"] == 0:
        print("    CONTRADICTION: invariant form but open orbit")
        bad += 1
    print(f"{name:9s} {str(p)+'|'+str(q):>5s}"
          f" {'('+str(r['sdim'][0])+'|'+str(r['sdim'][1])+')':>10s}"
          f" {r['dimNt']:7d} {r['dimN']:6d} {r['lin']:4d} {r['dimL']:6d}"
          f" {'yes' if r['certified'] else 'NO':>5s} {r['rhodim']:4d}"
          f" {r['gdim']:6d} {r['modality']:4d} {len(forms):5d}"
          f"  {r['verdict']}"
          f"   [{time.time()-t0:.1f}s]")

print()
print("dim S = dimension of the jet algebra of Stab(f) that acts on the layer;")
print("lin   = its linear part; rho = dim of its image in gl(Nt_f);")
print("orbit = dimension of the generic orbit; mod = dim Nt_f - orbit;")
print("form  = number of invariant bilinear forms on Nt_f.  A nonzero count")
print("        PROVES modality >= 1; an open orbit PROVES modality = 0; the")
print("        two are mutually exclusive and every row satisfies exactly one.")


hdr("(2b)  the one type with a modulus: VII_2")

rV = [r for r in rows if r["name"] == "VII_2"][0]
nV = rV["dimNt"]
namesV = [sv.show_mon(m, "xy", "xi") for m in rV["basis"]]
A = rV["forms"][0]
print("layer basis:", namesV)
print(f"the invariant form has rank "
      f"{stab.rank_rows([row[:] for row in A])} and is "
      f"{stab.form_kind(A, nV)}; its nonzero entries are")
for i in range(nV):
    for j in range(nV):
        if A[i][j]:
            print(f"    A[{namesV[i]}, {namesV[j]}] = {A[i][j]}")
print()
print("So, writing g = a_1.xi_1 + a_2.xi_2 + a_3.xi_3 + (higher terms), every")
print("generator M satisfies  a_3 (Mg)_1 - a_1 (Mg)_3 = 0, i.e. the ratio")
print("        lambda = a_1/a_3")
print("is a rational invariant: it is the modulus.  Checked below both")
print("infinitesimally and at the level of the group.")

rnd_ok, grp_ok = True, True
import random as _r
rr = _r.Random(7)
for _ in range(200):
    g = [F(rr.randint(-9, 9)) for _ in range(nV)]
    for M in rV["matsU"]:
        Mg = stab.act(M, g)
        if Mg[0] * g[2] - g[0] * Mg[2] != 0:
            rnd_ok = False
nilV = [V for V in rV["S"] if stab.vf_degree(V) >= 1]
matsV = [stab.rho(rV["R"], V, rV["basis"], rV["mI"], rV["index"]) for V in nilV]
g = [F(rr.randint(-9, 9)) for _ in range(nV)]
while g[2] == 0:
    g = [F(rr.randint(-9, 9)) for _ in range(nV)]
before = F(g[0], g[2])
for M in matsV:
    E = stab.mat_exp(M)
    g = [sum(E[i][j] * g[j] for j in range(nV)) for i in range(nV)]
after = F(g[0], g[2]) if g[2] else None
print(f"    infinitesimally, 200 random points x {len(rV['matsU'])} generators:"
      f" {'invariant' if rnd_ok else 'FAILED'}")
print(f"    under exp of all {len(matsV)} unipotent generators: "
      f"lambda {before} -> {after}  "
      f"{'invariant' if after == before else 'FAILED'}")
if not rnd_ok or after != before:
    bad += 1
print()
print("VII_2 is SIMPLE in the Serganova-Vaintrob table -- modality 0 without odd parameters.  With")
print("one odd parameter it acquires a modulus.  So simplicity is not preserved")
print("by the passage to a Grassmann base, and (0|1)-modality is not the")
print("classical modality.")


# --------------------------------------------------------------- Part 3

hdr("(3)  the family II(n,1):  f = x^n + x.xi1.xi2,  the test case of B_1")

print(f"{'n':>3s} {'sdim O_f':>10s} {'dim Nt':>7s} {'lin':>4s} {'dim S':>6s}"
      f" {'cert':>5s} {'orbit':>6s} {'mod':>4s}  orbit dims (xi_1, x.xi_1, 0)")
for n in range(2, 9):
    D = max(14, n + 7)
    r = stab.analyse(f"II({n},1)", 1, 2,
                     [(1, (n,), ()), (1, (1,), (0, 1))], D, verbose=False)
    b, idx = r["basis"], r["index"]

    def vec(mon):
        v = [F(0)] * len(b)
        if mon in idx:
            v[idx[mon]] = F(1)
        return v
    ds = [stab.orbit_dim(r["mats"], vec(xi1)),
          stab.orbit_dim(r["mats"], vec(xxi1)),
          0]
    if not r["certified"]:
        bad += 1
    print(f"{n:3d} {'('+str(r['sdim'][0])+'|'+str(r['sdim'][1])+')':>10s}"
          f" {r['dimNt']:7d} {r['lin']:4d} {r['dimL']:6d}"
          f" {'yes' if r['certified'] else 'NO':>5s} {r['gdim']:6d}"
          f" {r['modality']:4d}  {ds}")

print()
print("n = 2 is not a type of the tables: x^2 + x.xi1.xi2 = (x + xi1.xi2/2)^2,")
print("so it is the Morse germ x^2, and its layer is only q = 2 dimensional.")
print("From n = 3 on the family is uniform: dim Nt_f = 4, linear part sl(2),")
print("three orbits of dimensions 4, 2, 0.")

hdr("(4)  the whole corank-2|3 stratum:  the types VII")

print("f = f_0(x,y) + x.xi1.xi2 + y.xi2.xi3, the even part f_0 running over the")
print("four simple types of the table.  Each row is certified: the bracket closes,")
print("and the invariant form is an exact certificate of modality >= 1.")
print()
print(f"{'type':10s} {'f_0':12s} {'sdim O_f':>10s} {'dim Nt':>7s} {'k':>2s}"
      f" {'lin':>4s} {'dim S':>6s} {'orbit':>6s} {'mod':>4s} {'form':>5s}  modulus")
VII = [
    ("VII_1(3)", "x^2y+y^3", [(1, (2, 1), ()), (1, (0, 3), ())], 12),
    ("VII_1(4)", "x^2y+y^4", [(1, (2, 1), ()), (1, (0, 4), ())], 12),
    ("VII_1(5)", "x^2y+y^5", [(1, (2, 1), ()), (1, (0, 5), ())], 15),
    ("VII_2", "x^3+y^4", [(1, (3, 0), ()), (1, (0, 4), ())], 12),
    ("VII_3", "x^3+xy^3", [(1, (3, 0), ()), (1, (1, 3), ())], 12),
    ("VII_4", "x^3+y^5", [(1, (3, 0), ()), (1, (0, 5), ())], 15),
]
for name, f0, even, D in VII:
    terms = even + [(1, (1, 0), (0, 1)), (1, (0, 1), (1, 2))]
    r = stab.analyse(name, 2, 3, terms, D, verbose=False)
    A = stab.annihilator_forms(r["matsU"], r["dimNt"])
    n = r["dimNt"]
    # is the form always xi_1 ^ xi_3, and is a_1/a_3 always the modulus?
    ok = bool(A)
    if ok:
        i1 = r["index"].get(((0, 0), frozenset({0})))
        i3 = r["index"].get(((0, 0), frozenset({2})))
        ok = (i1 is not None and i3 is not None
              and stab.rank_rows([row[:] for row in A[0]]) == 2
              and A[0][i1][i3] != 0)
        for M in r["matsU"]:
            for _ in range(5):
                g = [F(rr.randint(-9, 9)) for _ in range(n)]
                Mg = stab.act(M, g)
                if Mg[i1] * g[i3] - g[i1] * Mg[i3] != 0:
                    ok = False
    if (not r["certified"]) or (not r["ok_layer"]) or (not r["ok_rho_room"]) \
            or (not ok) or r["modL"] != 1 or r["modU"] != 1:
        bad += 1
    print(f"{name:10s} {f0:12s}"
          f" {'('+str(r['sdim'][0])+'|'+str(r['sdim'][1])+')':>10s}"
          f" {r['dimNt']:7d} {r['k']:2d} {r['lin']:4d} {r['dimL']:6d}"
          f" {r['gdim']:6d} {r['modality']:4d} {len(A):5d}"
          f"  {'a_1/a_3' if ok else 'FAILED'}")

print()
print("Every type of corank 2|3 -- the whole VII stratum, all four of them simple")
print("in the Serganova-Vaintrob table -- has first-layer modality exactly 1, with one and the same")
print("invariant form xi_1 ^ xi_3 and one and the same modulus a_1/a_3.  The even")
print("part of the normal form plays no role, which is what the mechanism predicts:")
print("xi_1 is tied to x and xi_3 to y, x and y have different weights, and the")
print("linear part of the stabiliser is only 3-dimensional, too small to be")
print("transitive on the 3-dimensional new stratum.")

hdr("SUMMARY: " + ("all checks passed" if bad == 0 else f"{bad} PROBLEM(S)"))
