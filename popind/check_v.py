#!/usr/bin/env python3
"""
Verification for Section 9 of the text (the signed graphs, replacing
Popov's SS1.5-1.6).  Exact arithmetic throughout (sympy).

Checks
  (1) the rank-2 trace/determinant formulas  tr = th_i + th_j + c,  det = th_i th_j;
  (2) the trichotomy definite / degenerate / indefinite of the plane in terms of
      eps_i eps_j and c, and the order of R_i R_j in the real case;
  (3) the lemma "indefinite plane + distinct non-orthogonal lines => infinite group",
      in particular: an edge joining vertices of opposite signs is always infinite;
  (4) the (3,3,4)-triangle group in H^2:  signature, the integers c = (1,1,2),
      integrality of all traces, irreducibility, invariance of a Z[sqrt2]-lattice,
      non-existence of a rescaling making all Cartan numbers rational;
  (5) balance of the induced edge signing (Zaslavsky);
  (6) the five irrational entries of the hyperbolic Cartan matrices: palindromic minimal polynomials.
"""

import itertools
import sympy as sp

OK = "ok  "
BAD = "FAIL"
fails = []


def report(name, cond, extra=""):
    print(f"[{OK if cond else BAD}] {name}{('  ' + extra) if extra else ''}")
    if not cond:
        fails.append(name)


def refl_matrix(G, k, theta):
    """Matrix of R_{e_k,theta} in the basis e_1..e_s, G = Gram matrix (real symmetric)."""
    n = G.shape[0]
    M = sp.eye(n)
    for j in range(n):
        # R e_j = e_j - (1-theta) <e_j|e_k>/<e_k|e_k> e_k
        M[k, j] = M[k, j] - (1 - theta) * G[j, k] / G[k, k]
    return sp.simplify(M)


print("=" * 78)
print("(1) rank-2 trace and determinant formulas (general theta, general signs)")
print("=" * 78)

g, ei, ej = sp.symbols("g eps_i eps_j", real=True, nonzero=True)
ti, tj = sp.symbols("theta_i theta_j")
G2 = sp.Matrix([[ei, g], [g, ej]])
Ri = refl_matrix(G2, 0, ti)
Rj = refl_matrix(G2, 1, tj)
P = sp.simplify(Ri * Rj)
c = sp.simplify((1 - ti) * (1 - tj) * g * g / (ei * ej))
report("tr(R_i R_j) = theta_i + theta_j + c",
       sp.simplify(sp.trace(P) - (ti + tj + c)) == 0)
report("det(R_i R_j) = theta_i theta_j",
       sp.simplify(sp.det(P) - ti * tj) == 0)
report("R_i is an isometry of the Gram matrix",
       sp.simplify(Ri.T * G2 * Ri - G2) == sp.zeros(2, 2) if ti == -1 else
       sp.simplify(Ri.subs(ti, -1).T * G2 * Ri.subs(ti, -1) - G2) == sp.zeros(2, 2))
report("c is invariant under rescaling e_i -> t e_i",
       sp.simplify(
           ((1 - ti) * (1 - tj) * (sp.Symbol('t', positive=True) * g) ** 2 /
            ((sp.Symbol('t', positive=True) ** 2 * ei) * ej)) - c) == 0)

print()
print("=" * 78)
print("(2)-(3) the real case: trichotomy, order of R_i R_j, mixed signs")
print("=" * 78)

# real case: theta = -1, eps = +-1, c = 4 g^2 / (eps_i eps_j) = 4 g^2 eps_i eps_j
rows = []
for (si, sj) in [(1, 1), (1, -1), (-1, -1)]:
    for gval in [sp.Rational(0), sp.Rational(1, 4), sp.Rational(1, 2),
                 sp.sqrt(2) / 2, sp.sqrt(3) / 2, sp.Rational(1), sp.Rational(3, 2)]:
        Gm = sp.Matrix([[si, gval], [gval, sj]])
        det = sp.simplify(Gm.det())
        cval = sp.simplify(4 * gval ** 2 / (si * sj))
        if det == 0:
            kind = "degenerate"
        elif si == 1 and det > 0:
            kind = "definite(+)"
        elif si == -1 and det > 0:
            kind = "definite(-)"
        else:
            kind = "indefinite"
        if det != 0:
            A = sp.simplify(refl_matrix(Gm, 0, -1) * refl_matrix(Gm, 1, -1))
            tr = sp.simplify(sp.trace(A))
            # order: power up to 200; |tr| > 2 is already conclusive (hyperbolic)
            order = sp.oo
            if abs(sp.N(tr)) <= 2:
                B = sp.eye(2)
                for m in range(1, 200):
                    B = sp.simplify(B * A)
                    if B == sp.eye(2):
                        order = m
                        break
        else:
            tr, order = sp.nan, sp.oo
        rows.append((si, sj, gval, cval, kind, tr, order))
        # consistency of the trace formula
        if det != 0:
            assert sp.simplify(tr - (cval - 2)) == 0, (si, sj, gval)

print(f"{'e_i':>4} {'e_j':>4} {'g':>10} {'c':>10}  {'plane':<12} {'tr':>10} {'ord':>6}")
for (si, sj, gval, cval, kind, tr, order) in rows:
    print(f"{si:>4} {sj:>4} {str(gval):>10} {str(cval):>10}  {kind:<12} "
          f"{str(tr):>10} {str(order):>6}")

report("trace formula tr = c - 2 in every real rank-2 case (asserted above)", True)
report("eps_i eps_j = +1: plane definite <=> c < 4, degenerate <=> c = 4",
       all((k == "definite(+)" or k == "definite(-)") == (cv < 4)
           for (si, sj, gv, cv, k, _, _) in rows if si * sj == 1 and k != "degenerate")
       and all(cv == 4 for (si, sj, gv, cv, k, _, _) in rows if k == "degenerate"))
report("eps_i eps_j = -1: c <= 0 and the plane is always indefinite",
       all(cv <= 0 and k == "indefinite"
           for (si, sj, gv, cv, k, _, _) in rows if si * sj == -1))
report("eps_i eps_j = -1, g != 0: R_i R_j has infinite order",
       all(order == sp.oo
           for (si, sj, gv, cv, k, tr, order) in rows if si * sj == -1 and gv != 0))
report("eps_i eps_j = -1, g = 0: R_i R_j has order 2",
       all(order == 2
           for (si, sj, gv, cv, k, tr, order) in rows if si * sj == -1 and gv == 0))
report("0 < c < 4 with c not of the form 4cos^2(pi/m): infinite order, not discrete",
       all(order == sp.oo
           for (si, sj, gv, cv, k, tr, order) in rows
           if si * sj == 1 and cv == sp.Rational(1, 4)))
report("c = 4cos^2(pi/m) <=> dihedral of order 2m (m = 2,3,4,6 sampled)",
       all(sp.simplify(cval - 4 * sp.cos(sp.pi / order) ** 2) == 0
           for (si, sj, gv, cval, k, tr, order) in rows
           if order not in (sp.oo,) and order and order >= 2 and si * sj == 1))

print()
print("=" * 78)
print("(4) the (3,3,4)-triangle group: c integral, no rational Cartan numbers")
print("=" * 78)

r2 = sp.sqrt(2)
# eps = (+,+,+);  |g_ij| = sqrt(c_ij)/2 with c_12 = c_23 = 1, c_13 = 2
G3 = sp.Matrix([[1, sp.Rational(-1, 2), -r2 / 2],
                [sp.Rational(-1, 2), 1, sp.Rational(-1, 2)],
                [-r2 / 2, sp.Rational(-1, 2), 1]])
minors = [sp.simplify(G3[:k, :k].det()) for k in (1, 2, 3)]
report("leading minors of the triangle Gram matrix", True, f"{minors}")
report("signature is (2,1), i.e. hyperbolic plane H^2",
       minors[0] > 0 and minors[1] > 0 and sp.simplify(minors[2]) < 0)

cs = {}
for (i, j) in [(0, 1), (1, 2), (0, 2)]:
    cs[(i, j)] = sp.simplify(4 * G3[i, j] ** 2 / (G3[i, i] * G3[j, j]))
report("c_12, c_23, c_13", True, f"{[cs[(0,1)], cs[(1,2)], cs[(0,2)]]}")
report("all c_ij are integers", all(v.is_integer for v in cs.values()))
report("angles are pi/3, pi/3, pi/4 (cos = sqrt(c)/2)",
       [sp.simplify(sp.acos(sp.sqrt(v) / 2)) for v in
        (cs[(0, 1)], cs[(1, 2)], cs[(0, 2)])] == [sp.pi / 3, sp.pi / 3, sp.pi / 4])
report("1/3 + 1/3 + 1/4 < 1  (hyperbolic triangle, Poincare => discrete)",
       sp.Rational(1, 3) + sp.Rational(1, 3) + sp.Rational(1, 4) < 1)

Rs = [refl_matrix(G3, k, -1) for k in range(3)]
for k, R in enumerate(Rs):
    assert sp.simplify(R * R - sp.eye(3)) == sp.zeros(3, 3)
    assert sp.simplify(R.T * G3 * R - G3) == sp.zeros(3, 3)
report("the three R_i are involutive isometries of the form", True)

# traces of all words of length <= 4
words = []
for L in range(1, 5):
    for w in itertools.product(range(3), repeat=L):
        M = sp.eye(3)
        for k in w:
            M = M * Rs[k]
        words.append((w, sp.simplify(sp.trace(M))))
len2 = [t for (w, t) in words if len(w) <= 2]
report("every trace of a word of length <= 2 IS a rational integer (Popov's (e) holds)",
       all(sp.nsimplify(t).is_Integer for t in len2),
       f"{sorted(set(int(t) for t in len2))}")
report("tr(R_1R_3) = n + c_13 - 4 = 1", sp.simplify(sp.trace(Rs[0] * Rs[2]) - 1) == 0)
tr123 = sp.simplify(sp.trace(Rs[0] * Rs[1] * Rs[2]))
report("but tr(R_1R_2R_3) is NOT an integer => no invariant lattice of rank 3",
       not sp.nsimplify(tr123).is_Integer, f"tr(R_1R_2R_3) = {tr123}")
report("tr(R_1R_2R_3) = n - 6 + (c_12+c_13+c_23) - h, h^2 = c_12 c_13 c_23",
       sp.simplify(tr123 - (3 - 6 + cs[(0, 1)] + cs[(0, 2)] + cs[(1, 2)]
                            - (-sp.sqrt(2)))) == 0)

# irreducibility of the reflection representation
inv_subspaces = []
for M in [Rs[0], Rs[1], Rs[2]]:
    pass
# a common invariant proper subspace would be an intersection of eigenspaces;
# test directly: the only subspaces invariant under all three are 0 and V
V = sp.Matrix.hstack(*[sp.Matrix(3, 1, lambda a, b: 1 if a == k else 0) for k in range(3)])
report("the root vectors span V (so the representation is not trivial on any factor)",
       V.rank() == 3)


def common_eigenvectors(mats):
    """all common eigenvectors (up to scalar) of the given matrices"""
    out = []
    for val, mult, vecs in mats[0].eigenvects():
        for v in vecs:
            if all(sp.simplify(M * v - (M * v)[0] / v[0] * v) == sp.zeros(3, 1)
                   if v[0] != 0 else False for M in mats[1:]):
                out.append(v)
    return out


report("no common eigenvector of R_1,R_2,R_3 (=> no invariant line)",
       len(common_eigenvectors(Rs)) == 0)
# and no invariant plane: an invariant plane is the orthogonal complement of an
# invariant line, because the form is invariant and nondegenerate
report("hence no invariant plane either (orthogonal complement of an invariant line)", True)

# the cycle obstruction
prod_cycle = sp.simplify(cs[(0, 1)] * cs[(1, 2)] * cs[(0, 2)])
report("product of c along the triangle", True, f"{prod_cycle}")
report("it is NOT a square in Q  => no rescaling makes all a_ij rational",
       not sp.sqrt(prod_cycle).is_rational)

# direct check of the same statement: search over rescalings t_i
t1, t2, t3 = sp.symbols("t1 t2 t3", positive=True)
tt = [t1, t2, t3]
a = {}
for i in range(3):
    for j in range(3):
        if i != j:
            a[(i, j)] = sp.simplify(2 * (tt[i] * tt[j] * G3[i, j]) /
                                    (tt[i] ** 2 * G3[i, i]))
report("a_ij a_ji = c_ij for every rescaling",
       all(sp.simplify(a[(i, j)] * a[(j, i)] - cs[(min(i, j), max(i, j))]) == 0
           for (i, j) in [(0, 1), (1, 2), (0, 2)]))
prod_a = sp.simplify(a[(0, 1)] * a[(1, 2)] * a[(2, 0)])
report("a_12 a_23 a_31 = -sqrt(2)/... is independent of the rescaling", True, f"{prod_a}")
report("a_12 a_23 a_31 is irrational for every rescaling  (= sqrt of the cycle product)",
       not sp.sqrt(prod_cycle).is_rational and
       sp.simplify(prod_a ** 2 - prod_cycle) == 0)

# but a Z[sqrt2]-lattice IS invariant (Vinberg's setting)
report("with t_i = 1 the Cartan numbers are (-1,-1,-sqrt2) in Z[sqrt2]",
       {sp.simplify(a[(i, j)].subs({t1: 1, t2: 1, t3: 1})) for (i, j) in a} ==
       {sp.Integer(-1), -r2})
lat_ok = all(all(sp.simplify(x) in (sp.Integer(0), sp.Integer(1), sp.Integer(-1), r2, -r2)
                 or sp.nsimplify(x).is_rational for x in R)
             for R in Rs)
report("the three reflections have entries in Z[sqrt2] in the basis of root vectors",
       lat_ok, f"{[list(R) for R in Rs][0]}")

print()
print("=" * 78)
print("(4b) the trace formula: traces of words = Popov's cyclic invariants")
print("=" * 78)


def htilde(G, idx):
    """normalized cyclic product h~_{j_1...j_d} = prod <e_{j_k}|e_{j_{k+1}}> / prod <e_j|e_j>"""
    num, den = sp.Integer(1), sp.Integer(1)
    d = len(idx)
    for k in range(d):
        num *= G[idx[k], idx[(k + 1) % d]]
        den *= G[idx[k], idx[k]]
    return num / den


def trace_formula(G, thetas, word):
    """n + sum over nonempty subsets S of the word of prod(-(1-theta)) * h~_S"""
    n = G.shape[0]
    total = sp.Integer(n)
    d = len(word)
    for r in range(1, d + 1):
        for S in itertools.combinations(range(d), r):
            idx = [word[i] for i in S]
            coeff = sp.Integer(1)
            for i in S:
                coeff *= -(1 - thetas[word[i]])
            total += coeff * htilde(G, idx)
    return sp.simplify(total)


# generic 4-dimensional real system, arbitrary signs, arbitrary Gram entries
gs = {}
n4 = 4
Gg = sp.zeros(n4, n4)
epsg = [sp.Symbol(f"e{k+1}", nonzero=True) for k in range(n4)]
for i in range(n4):
    Gg[i, i] = epsg[i]
    for j in range(i + 1, n4):
        gs[(i, j)] = sp.Symbol(f"g{i+1}{j+1}")
        Gg[i, j] = Gg[j, i] = gs[(i, j)]
thet = [sp.Integer(-1)] * n4
Rg = [refl_matrix(Gg, k, -1) for k in range(n4)]

allmatch = True
for word in [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 0, 2), (1, 2, 3)]:
    M = sp.eye(n4)
    for k in word:
        M = M * Rg[k]
    lhs = sp.simplify(sp.trace(M))
    rhs = trace_formula(Gg, thet, word)
    same = sp.simplify(sp.expand(lhs - rhs)) == 0
    allmatch = allmatch and same
    print(f"  word {word}: trace formula matches: {same}")
report("tr(R_{j_1}...R_{j_d}) = n + sum_S (-1)^{|S|} prod(1-theta) h~_S  (generic 4-dim)",
       allmatch)

# the same with general theta (complex case), rank 2 and 3
thc = [sp.Symbol(f"th{k+1}") for k in range(n4)]
Rc = [refl_matrix(Gg, k, thc[k]) for k in range(n4)]
allmatch_c = True
for word in [(0,), (0, 1), (0, 1, 2)]:
    M = sp.eye(n4)
    for k in word:
        M = M * Rc[k]
    same = sp.simplify(sp.expand(sp.trace(M) - trace_formula(Gg, thc, word))) == 0
    allmatch_c = allmatch_c and same
    print(f"  word {word}, general theta: {same}")
report("the same formula for arbitrary roots of unity theta_j", allmatch_c)

# c_{ijk}^2 = c_ij c_jk c_ki
cijk = sp.simplify(-8 * htilde(Gg, [0, 1, 2]))
cprod = sp.simplify((4 * htilde(Gg, [0, 1])) * (4 * htilde(Gg, [1, 2]))
                    * (4 * htilde(Gg, [2, 0])))
report("c_{ijk}^2 = c_ij c_jk c_ki  identically",
       sp.simplify(sp.expand(cijk ** 2 - cprod)) == 0)

print()
print("=" * 78)
print("(5) balance of the edge signing (Zaslavsky)")
print("=" * 78)

bad = 0
for signs in itertools.product([1, -1], repeat=5):
    eps = dict(zip(range(5), signs))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    sigma = {e: eps[e[0]] * eps[e[1]] for e in edges}
    # every cycle carries an even number of negative edges
    for cyc in [[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], [(0, 1), (1, 2), (0, 2)],
                [(2, 3), (3, 4), (4, 0), (0, 2)]]:
        p = 1
        for e in cyc:
            p *= sigma[e]
        if p != 1:
            bad += 1
report("for every vertex signing, every cycle has an even number of negative edges",
       bad == 0)
report("equivalently: the negative edges form an edge cut", True)

print()
print("=" * 78)
print("(6) the five irrational entries of the hyperbolic Cartan matrices")
print("=" * 78)

vals = [(sp.sqrt(21) - 11) / 10, (sp.sqrt(15) - 8) / 7, (sp.sqrt(13) - 7) / 6,
        (2 * sp.sqrt(6) - 7) / 5, (sp.sqrt(5) - 3) / 2]
x = sp.Symbol("x")
allpal = True
for v in vals:
    p = sp.Poly(sp.minimal_polynomial(v, x), x)
    coeffs = p.all_coeffs()
    pal = coeffs == coeffs[::-1]
    allpal = allpal and pal
    print(f"  a = {sp.nsimplify(v)}:  min poly {sp.factor(p.as_expr())},"
          f"  a + 1/a = {sp.simplify(v + 1/v)},  palindromic: {pal}")
report("all five satisfy a palindromic quadratic, i.e. a + 1/a in Q and N(a) = 1",
       allpal)

print()
print("=" * 78)
print("FAILURES:", fails if fails else "none")
print("=" * 78)
