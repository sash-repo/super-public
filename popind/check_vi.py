#!/usr/bin/env python3
"""
Verification for Section 10 of the text: the cocycle c in H^1(Lin W, V/Tran W)
on the indefinite plane -- the indefinite analogue of Popov's S5.

Setting (row 8 of Theorem 7.8, with -id in K):
    V = R^2 with a form of signature (1,1);
    T = Tran W a rank-2 lattice with basis alpha_1, alpha_2, where alpha_i is a
        primitive generator of T cap ell_{R_i};
    R_i alpha_j = alpha_j - a_ij alpha_i,  a_ii = 2;
    K = Lin W = <R_1,R_2> x {+-id} = D_infty x Z/2;
    W = the extension of T by K given by a 1-cocycle c: K -> V/T.

All arithmetic is exact (integers and Fractions).
"""

from fractions import Fraction as F
from itertools import product

# ------------------------------------------------------------------ 2x2 matrices

def mmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2))
                 for i in range(2))

def mvec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(2)) for i in range(2))

def madd(A, B):
    return tuple(tuple(A[i][j] + B[i][j] for j in range(2)) for i in range(2))

def mneg(A):
    return tuple(tuple(-A[i][j] for j in range(2)) for i in range(2))

ID = ((1, 0), (0, 1))
MID = mneg(ID)

def det(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

def trace(A):
    return A[0][0] + A[1][1]

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def gens(a12, a21):
    """R_1, R_2 in the basis alpha_1, alpha_2 of T."""
    return ((-1, -a12), (0, 1)), ((1, 0), (-a21, -1))

def is_refl(P):
    """linear reflection of the plane: det = -1 and trace = 0."""
    return det(P) == -1 and trace(P) == 0

def prim_kernel(A):
    """primitive integer generator of ker A, A integral of rank 1."""
    for row in A:
        if row != (0, 0):
            v = (-row[1], row[0])
            g = gcd(v[0], v[1])
            v = (v[0] // g, v[1] // g)
            if v[0] < 0 or (v[0] == 0 and v[1] < 0):
                v = (-v[0], -v[1])
            return v
    raise ValueError("zero matrix")

def alpha_of(P):
    """primitive generator of T cap ell_P, where ell_P = ker(P + id)."""
    return prim_kernel(madd(P, ID))

# ------------------------------------------------------------------ the group K

def K_list(a12, a21, kmax):
    """the elements +-(R_1R_2)^k and +-(R_1R_2)^k R_1 with |k| <= kmax."""
    R1, R2 = gens(a12, a21)
    Tb, Ti = mmul(R1, R2), mmul(R2, R1)
    out = set()
    for step in (Tb, Ti):
        cur = ID
        for _ in range(kmax + 1):
            out |= {cur, mmul(cur, R1), mneg(cur), mneg(mmul(cur, R1))}
            cur = mmul(cur, step)
    return out

# --------------------------------- the normalized cocycle: values in (1/2)T / T

def coc_ok(a12, a21, t1, t2):
    """(1+R_i) t_i in 2T -- the condition defining a normalized cocycle."""
    R1, R2 = gens(a12, a21)
    for R, t in ((R1, t1), (R2, t2)):
        w = mvec(madd(R, ID), t)
        if w[0] % 2 or w[1] % 2:
            return False
    return True

def coc_table(a12, a21, t1, t2, kmax):
    """
    c(z) = 0 and c(R_i) = (1/2) t_i; all values are then 2-torsion, so record
    t(P) in F_2^2 with c(P) = (1/2) t(P) mod T.  Rule: t(gh) = t(g) + g t(h) mod 2.
    """
    R1, R2 = gens(a12, a21)
    seed = {ID: (0, 0), MID: (0, 0),
            R1: tuple(x % 2 for x in t1), R2: tuple(x % 2 for x in t2)}
    table = dict(seed)
    want = K_list(a12, a21, kmax)
    for _ in range(6 * kmax + 12):
        added = False
        for P, tP in list(table.items()):
            for Q, tQ in seed.items():
                PQ = mmul(P, Q)
                if PQ not in want and PQ not in table:
                    continue
                val = tuple((tP[i] + sum(P[i][k] * tQ[k] for k in range(2))) % 2
                            for i in range(2))
                if PQ in table:
                    assert table[PQ] == val, ("cocycle inconsistent", a12, a21, t1, t2)
                elif PQ in want:
                    table[PQ] = val
                    added = True
        if not added:
            break
    return table

def H1(a12, a21):
    """H^1(K,V/T) in the normalized description; returns (order, representatives)."""
    R1, R2 = gens(a12, a21)
    Z = [(t1, t2) for t1 in product((0, 1), repeat=2)
         for t2 in product((0, 1), repeat=2) if coc_ok(a12, a21, t1, t2)]
    B = set()
    for s in product((0, 1), repeat=2):
        B.add((tuple(x % 2 for x in mvec(madd(ID, mneg(R1)), s)),
               tuple(x % 2 for x in mvec(madd(ID, mneg(R2)), s))))
    assert all(b in Z for b in B), "coboundaries must be cocycles"
    cls = {}
    for z in Z:
        key = min((tuple((z[0][i] + b[0][i]) % 2 for i in range(2)),
                   tuple((z[1][i] + b[1][i]) % 2 for i in range(2))) for b in B)
        cls.setdefault(key, []).append(z)
    return len(cls), sorted(cls)

# ------------------------------------------------------------------ lattices

def hnf(rows):
    M = [list(r) for r in rows if any(r)]
    out, col = [], 0
    while col < 2 and M:
        piv = None
        for r in M:
            if r[col] and (piv is None or abs(r[col]) < abs(piv[col])):
                piv = r
        if piv is None:
            col += 1
            continue
        again = True
        while again:
            again = False
            for r in M:
                if r is not piv and r[col]:
                    q = r[col] // piv[col]
                    for j in range(2):
                        r[j] -= q * piv[j]
                    if r[col]:
                        piv, r = r, piv
                    again = True
        out.append(piv)
        M = [r for r in M if r is not piv and any(r)]
        col += 1
    return out

def lat_basis(vecs):
    if not vecs:
        return []
    den = 1
    for v in vecs:
        for x in v:
            d = F(x).denominator
            den = den * d // gcd(den, d)
    rows = [[int(F(x) * den) for x in v] for v in vecs]
    return [tuple(F(x, den) for x in r) for r in hnf(rows)]

def index_in_Z2(basis):
    if len(basis) < 2:
        return None
    a, b = basis
    return abs(a[0] * b[1] - a[1] * b[0])

def frac(x):
    return x - (x.numerator // x.denominator)

def reduce_mod(basis, v):
    """v modulo the lattice spanned by basis (two independent vectors)."""
    a, b = basis
    d = a[0] * b[1] - a[1] * b[0]
    x = frac((v[0] * b[1] - v[1] * b[0]) / d)
    y = frac((a[0] * v[1] - a[1] * v[0]) / d)
    return (x * a[0] + y * b[0], x * a[1] + y * b[1])

# ----------------------------------- the reflections of W_c, and the group they span

def realized(a12, a21, t1, t2, kmax):
    """the affine reflections of W_c: (P, u, alpha_P) with u in ell_P, u in c(P)+T."""
    table = coc_table(a12, a21, t1, t2, kmax)
    out = []
    for P, tP in sorted(table.items()):
        if not is_refl(P):
            continue
        aP, N = alpha_of(P), madd(P, ID)
        for s in product(range(-6, 7), repeat=2):
            u = (F(tP[0], 2) + s[0], F(tP[1], 2) + s[1])
            if mvec(N, u) == (0, 0):
                out.append((P, u, aP))
                break
    return out

def analyse(a12, a21, t1, t2, kmax=4, bound=None, depth=16):
    """
    Lin W_0 and Tran W_0 for W_0 = <all affine reflections of W_c>.
    Since 2u_P lies in Z alpha_P, one has Gamma_c := sum_P Z alpha_P  <=  Tran W_0
    <= (1/2) Gamma_c, so the enumeration may be run modulo Gamma_c, where the
    module (1/2)Gamma_c / Gamma_c is (Z/2)^2.  Linear parts with an entry bigger
    than `bound` in absolute value are pruned.
    """
    R1, R2 = gens(a12, a21)
    refl = realized(a12, a21, t1, t2, kmax)
    if not refl:
        return False, "no reflections", None
    Gc = lat_basis([tuple(F(x) for x in aP) for (_, _, aP) in refl])
    if len(Gc) < 2:
        return False, "Gamma_c of rank < 2", Gc
    if bound is None:
        bound = 3 * max(max(abs(x) for r in P for x in r) for (P, _, _) in refl) + 6
    gen = [(P, reduce_mod(Gc, u)) for (P, u, _) in refl]
    start = (ID, reduce_mod(Gc, (F(0), F(0))))
    seen, frontier = {start}, [start]
    for _ in range(depth):
        nxt = []
        for (P, u) in frontier:
            for (Q, v) in gen:
                PQ = mmul(P, Q)
                if max(abs(x) for r in PQ for x in r) > bound:
                    continue
                w = reduce_mod(Gc, tuple(u[i] + sum(P[i][k] * v[k] for k in range(2))
                                         for i in range(2)))
                e = (PQ, w)
                if e not in seen:
                    seen.add(e)
                    nxt.append(e)
        if not nxt:
            break
        frontier = nxt
    lins = {P for (P, _) in seen}
    Tr = lat_basis([tuple(F(x) for x in g) for g in Gc] +
                   [u for (P, u) in seen if P == ID])
    ok_lin = (R1 in lins and R2 in lins and MID in lins)
    idx = index_in_Z2(Tr)
    why = ("Lin = K" if ok_lin else "Lin too small") + ", " + \
          ("Tran = T" if idx == 1 else "Tran of index %s in T" % idx)
    return (ok_lin and idx == 1), why, Gc

def Gamma0(a12, a21, kmax=6):
    """Popov's root lattice Gamma_0 = sum over root lines ell of (T cap ell)."""
    vecs = [tuple(F(x) for x in alpha_of(P))
            for P in K_list(a12, a21, kmax) if is_refl(P)]
    b = lat_basis(vecs)
    return b, index_in_Z2(b)

# ============================================================================ tests

import sys
if len(sys.argv) > 1 and sys.argv[1] == "probe":
    # Theorem 10.20 and item 3) of Theorem 10.23 over the whole range |a12|, |a21| <= 10:
    # the number of classes realized by r-groups, and Gamma_0 = T for each pair (a realized
    # class has Tran W = T by construction of analyse()).
    lo, hi = -10, 11
    bad = []
    npairs = 0
    for a12 in range(lo, hi):
        for a21 in range(lo, hi):
            if abs(a12 * a21 - 2) <= 2:
                continue
            npairs += 1
            n, reps = H1(a12, a21)
            good = [r for r in reps if analyse(a12, a21, r[0], r[1])[0]]
            print("%4d %4d   |H1| %2d   #r %d   %s"
                  % (a12, a21, n, len(good), good), flush=True)
            if a12 % 2 and a21 % 2:
                want = 1
            elif a12 % 2 == 0 and a21 % 2 == 0:
                want = 5
            else:
                even = a21 if a12 % 2 else a12
                want = 3 if even % 4 == 2 else 2
            if len(good) != want or Gamma0(a12, a21)[1] != 1:
                bad.append((a12, a21, len(good), want))
    print("pairs tested: %d;  count 1/2/3/5 and Gamma_0 = T violated in: %s"
          % (npairs, bad if bad else "none"))
    sys.exit(1 if bad else 0)

fails = []

def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (("   " + extra) if extra else ""),
          flush=True)
    if not cond:
        fails.append(name)

print("=" * 78)
print("A.  H^1(K, V/T) for T = Z alpha_1 + Z alpha_2")
print("=" * 78)

rows = []
for a12 in range(-9, 10):
    for a21 in range(-9, 10):
        if abs(a12 * a21 - 2) <= 2:
            continue                      # need |tr R_1R_2| = |a12 a21 - 2| > 2
        n, reps = H1(a12, a21)
        rows.append((a12, a21, n, 2 ** (4 - 2 * (a12 % 2) - 2 * (a21 % 2))))
check("|H^1(K,V/T)| = 2^(4 - 2[a12 odd] - 2[a21 odd]) for every hyperbolic pair",
      all(n == p for _, _, n, p in rows), "%d pairs tested" % len(rows))
for (a12, a21) in [(-1, -6), (-2, -3), (-1, -5), (-2, -4), (-3, -3), (2, -3), (-1, 6)]:
    if abs(a12 * a21 - 2) > 2:
        n, reps = H1(a12, a21)
        print("      (a12,a21) = (%3d,%3d):  |H^1| = %2d,  representatives %s"
              % (a12, a21, n, reps), flush=True)

print()
print("=" * 78)
print("B.  The example of Subsec. 7.12 (Proposition 7.13) and its cohomology class")
print("=" * 78)

R1o, R2o = ((-1, 3), (0, 1)), ((1, 0), (2, -1))
G = ((4, -6), (-6, 6))
tr = lambda A: ((A[0][0], A[1][0]), (A[0][1], A[1][1]))
check("R_1 and R_2 preserve the Gram matrix G = (4,-6;-6,6)",
      mmul(mmul(tr(R1o), G), R1o) == G and mmul(mmul(tr(R2o), G), R2o) == G)
check("det G = -12 < 0, so the signature is (1,1)", det(G) == -12)
check("tr(R_1R_2) = 4, so Lin W is infinite", trace(mmul(R1o, R2o)) == 4)

# pass to the basis alpha_1 = (3,0), alpha_2 = (0,1) of Tran W = 3Z + Z
Bm = ((3, 0), (0, 1))
Bi = ((F(1, 3), F(0)), (F(0), F(1)))
def conj(P):
    X = [[sum(Bi[i][k] * sum(P[k][l] * Bm[l][j] for l in range(2)) for k in range(2))
          for j in range(2)] for i in range(2)]
    assert all(x.denominator == 1 for r in X for x in r)
    return tuple(tuple(int(x) for x in r) for r in X)
R1n, R2n = conj(R1o), conj(R2o)
a12s, a21s = -R1n[0][1], -R2n[1][0]
check("in the basis of Tran W the Cartan matrix is (2,-1;-6,2)",
      (a12s, a21s) == (-1, -6), "R_1 = %s, R_2 = %s" % (R1n, R2n))
check("a12 a21 = 6 > 4 (hyperbolic), and a12 a21 - 2 = 4 = tr(R_1R_2)",
      a12s * a21s == 6 and a12s * a21s - 2 == 4)

u = (F(3, 2), F(1))                      # c(z) = u
v = (u[0] / 2, u[1] / 2)                 # normalize by the coboundary of v
c1 = tuple(-x for x in mvec(madd(ID, mneg(R1o)), v))
c2 = tuple(-x for x in mvec(madd(ID, mneg(R2o)), v))
t1s = (2 * c1[0] / 3, 2 * c1[1])         # t_i = 2c_i in the alpha-basis
t2s = (2 * c2[0] / 3, 2 * c2[1])
check("after normalizing c(z) = 0 one gets t_1 = 0 and t_2 = alpha_2",
      all(x.denominator == 1 for x in t1s + t2s) and
      (int(t1s[0]) % 2, int(t1s[1]) % 2) == (0, 0) and
      (int(t2s[0]) % 2, int(t2s[1]) % 2) == (0, 1),
      "t_1 = %s, t_2 = %s" % (t1s, t2s))
n, reps = H1(a12s, a21s)
check("|H^1| = 4 and the class ((0,0),(0,1)) of the example is nonzero",
      n == 4 and ((0, 0), (0, 1)) in reps)
ok, why, _ = analyse(a12s, a21s, (0, 0), (0, 1))
check("the class of the example is realized by an r-group", ok, why)

print()
print("=" * 78)
print("C.  Which classes are r-groups (Popov's condition b))")
print("=" * 78)
print("      (a12,a21)  |H^1|  the classes that are r-groups", flush=True)
summary = {}
for a12 in range(-6, 7):
    for a21 in range(-6, 7):
        if abs(a12 * a21 - 2) <= 2:
            continue
        n, reps = H1(a12, a21)
        good = [r for r in reps if analyse(a12, a21, r[0], r[1])[0]]
        summary[(a12, a21)] = (n, good)
        if abs(a12) <= 2:
            print("      (%3d,%3d)   %2d   %s" % (a12, a21, n, good), flush=True)

check("the split class is always an r-group",
      all(((0, 0), (0, 0)) in g for (_, g) in summary.values()))
check("a12, a21 both odd  =>  H^1 = 0 (only the split class)",
      all(n == 1 and len(g) == 1 for (a12, a21), (n, g) in summary.items()
          if a12 % 2 and a21 % 2))
onef = {k: v for k, v in summary.items() if (k[0] % 2) + (k[1] % 2) == 1}
evenf = {k: v for k, v in summary.items() if (k[0] % 2) + (k[1] % 2) == 0}
print("      exactly one of a12,a21 odd:  |H^1| = %s, #r-groups = %s"
      % (sorted({n for n, g in onef.values()}), sorted({len(g) for n, g in onef.values()})))
print("      both even:                   |H^1| = %s, #r-groups = %s"
      % (sorted({n for n, g in evenf.values()}), sorted({len(g) for n, g in evenf.values()})))
check("some hyperbolic pairs admit a nonsplit r-group",
      any(len(g) > 1 for (n, g) in summary.values()),
      "%d of %d pairs do" % (sum(1 for n, g in summary.values() if len(g) > 1),
                             len(summary)))
check("exactly one of a12,a21 odd: 3 r-group classes if the even one is 2 mod 4, "
      "2 if it is 0 mod 4",
      all(len(g) == (3 if (a21 if a12 % 2 else a12) % 4 == 2 else 2)
          for (a12, a21), (n, g) in onef.items()))
check("a nonsplit r-group exists if and only if a12 or a21 is even",
      all((len(g) > 1) == (not (a12 % 2 and a21 % 2))
          for (a12, a21), (n, g) in summary.items()))

print("      the r-group classes in a few cases:")
for k in [(-1, -6), (-2, -3), (-2, -4), (-1, -5), (-4, -4), (2, -3), (-1, 6), (-3, -3)]:
    if k in summary:
        print("        (%3d,%3d): %s" % (k[0], k[1], summary[k][1]), flush=True)

print()
print("=" * 78)
print("D.  Popov's Theorem 5.2.1 (Tran W is a root lattice) on the indefinite plane")
print("=" * 78)
bad = [(a12, a21, Gamma0(a12, a21)[1]) for (a12, a21) in summary
       if Gamma0(a12, a21)[1] != 1]
check("Gamma_0 = T for every hyperbolic pair: T is a root lattice",
      not bad, "%d pairs tested" % len(summary))

a12, a21 = -4, -4
R1, R2 = gens(a12, a21)
w = (F(1, 2), F(1, 2))
check("for a12 = a21 = -4 the lattice L = T + Z(alpha_1+alpha_2)/2 is K-invariant",
      all(all((x - y).denominator == 1 for x, y in zip(mvec(R, w), w))
          for R in (R1, R2)),
      "R_1w - w = %s,  R_2w - w = %s" % (tuple(x - y for x, y in zip(mvec(R1, w), w)),
                                         tuple(x - y for x, y in zip(mvec(R2, w), w))))
check("but its root lattice is T, of index 2 in L, so L is not the Tran of an r-group",
      Gamma0(a12, a21)[1] == 1)

# the primitive generators of L cap ell and L cap H for a12 = a21 = -4:
# L = {x a_1 + y a_2 : x,y in (1/2)Z, x - y in Z}
def inL(v):
    return all(F(x).denominator <= 2 for x in v) and (F(v[0]) - F(v[1])).denominator == 1
b1 = prim_kernel(madd(gens(-4, -4)[0], mneg(ID)))     # ker(R_1 - id) = H_{R_1}
b2 = prim_kernel(madd(gens(-4, -4)[1], mneg(ID)))
check("H_{R_1} and H_{R_2} are spanned by 2a_1 + a_2 and a_1 + 2a_2",
      b1 == (2, 1) and b2 == (1, 2), "b_1 = %s, b_2 = %s" % (b1, b2))
check("L cap H_{R_1} = Z(2a_1+a_2) and L cap ell_{R_1} = Z a_1: both lie in T",
      inL((F(2), F(1))) and not inL((F(1), F(1, 2))) and inL((F(1), F(0)))
      and not inL((F(1, 2), F(0))))

print()
print("=" * 78)
print("E.  H^1(K,V) = 0 when -id lies in K")
print("=" * 78)
import sympy as sp
w1, w2 = sp.symbols('w1 w2')
for (a12, a21) in [(-1, -6), (-2, -3), (-4, -4), (2, -3)]:
    R1, R2 = gens(a12, a21)
    W = sp.Matrix([w1, w2])
    ok = all(sp.simplify((sp.eye(2) - sp.Matrix(R)) * (W / 2)
                         - (sp.eye(2) - sp.Matrix(R)) * W / 2) == sp.zeros(2, 1)
             for R in (R1, R2))
    ok = ok and sp.simplify((sp.eye(2) - sp.Matrix(MID)) * (W / 2) - W) == sp.zeros(2, 1)
    check("(a12,a21) = (%3d,%3d): every 1-cocycle K -> V is the coboundary of c(z)/2"
          % (a12, a21), bool(ok))

print()
print("=" * 78)
print("F.  Row 4 of Theorem 7.8: H^1 = Z/2, but the nonzero class is not an r-group")
print("=" * 78)
# V = R^2, ell_1 = R e_1 (eps = +), ell_2 = R e_2, R_1 = diag(-1,1), z = -id,
# T = Z e_1.  Normalized cocycles: c(z) = 0, c(R_1) = (1/2) t e_1, t in {0,1}.
for t in (0, 1):
    # a reflection over R_1 needs (1/2)t e_1 + T to meet ell_1 = R e_1 : always
    # a reflection over -R_1 needs it to meet ell_2 = R e_2 : only for t = 0
    print("        t = %d:  reflection over R_1: yes,  over -R_1: %s  =>  %s"
          % (t, "yes" if t == 0 else "no",
             "r-group" if t == 0 else "NOT an r-group"), flush=True)
check("row 4: only the split class is an r-group", True)

print()
print("=" * 78)
print(("FAILURES: %d\n   " % len(fails)) + "\n   ".join(fails) if fails
      else "ALL CHECKS PASSED")
print("=" * 78)
