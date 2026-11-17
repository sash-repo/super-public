"""The metric lift of Problem 3.9, checked: what is closed, and what is left.

Answers, in exact arithmetic, four questions (raised by H. Andersen) about the
metric-lift Problem of the imaging text:

  1) Is a lift of surface data to PT^*Sigma contact-natural?  NO, and the simplest
     witness is the generating function f = p^2, exactly as she says: the contact
     field X_{p^2} does not preserve the fibration PT^*Sigma -> Sigma, hence
     preserves neither the vertical line nor any horizontal complement of it.
     Therefore the classification of Subsec. 3.10 -- which classifies the operators
     invariant under the WHOLE contact algebra -- cannot decide Problem 3.9.

  2) How big is the answer then?  Treated as a Riemannian natural-operator problem,
     the bilinear scalar expressions of total weight 2 in (A,B) form a space of
     dimension 15 = 10 (symmetric) + 5 (antisymmetric).  Her count, reproduced here
     as an exact invariant-theory computation, not quoted.

  3) Statement 3.11 says the bracket has the harmonics 1, 3, 5 and no others.  Still
     true; verified symbolically for a generic 1-jet.

  4) Her corrected remark that harmonics up to 6 occur with both parities:  true for
     the general weight-2 pairing, and the rule is exact -- the parity of a harmonic
     is the parity of the number of HORIZONTAL derivatives in the term.  Statement
     3.11 is the one-horizontal-derivative case, whence its odd harmonics.

Conventions.  Sigma is a surface, g a metric on it, A and B covariant symmetric
2-tensor fields.  "Weight" counts derivatives: each nabla is 1, the Gauss curvature K
is 2.  PT^*Sigma carries the coordinates (x,y,theta), the direction being
v = (cos theta, sin theta), the contact form alpha = -sin theta dx + cos theta dy,
the frame e_1 = d_theta (vertical), e_2 = cos theta d_x + sin theta d_y (horizontal,
Levi-Civita) and the Reeb field R = -sin theta d_x + cos theta d_y, with
[e_1,e_2] = R -- the same structure the engine of the contact paper uses.

Nothing is asserted below that the run does not check.
"""

import os
import sys
from fractions import Fraction as Fr
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exactlin import nullspace, rank                       # noqa: E402

import sympy as sp                                         # noqa: E402
from sympy.simplify import fu                              # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


CHECKS = [0, 0]


def check(name, ok):
    CHECKS[0] += 1
    CHECKS[1] += bool(ok)
    say(("  PASS  " if ok else "  FAIL  ") + name)
    return bool(ok)


# ======================================================================= part 1
# The contact fields that do not see the fibration.
#
# Chart of PT^*R^2: (x,y,m), the contact element at (x,y) of slope m, alpha = dy - m dx.
# The contact field of the generating function f(x,y,m) is
#     X_f = -f_m d_x + (f - m f_m) d_y + (f_x + m f_y) d_m,
# which is what the code derives rather than quotes.

def part1():
    say("\n=== 1. Which contact fields preserve the fibration PT^*Sigma -> Sigma")
    x, y, m = sp.symbols("x y m")
    f = sp.Function("f")(x, y, m)

    a, b, c = sp.symbols("a b c", cls=sp.Function)
    A_, B_, C_ = a(x, y, m), b(x, y, m), c(x, y, m)
    # alpha = dy - m dx.  Solve  alpha(X) = f  and  L_X alpha = lambda alpha.
    # L_X alpha = d(alpha(X)) + i_X d alpha ;  d alpha = -dm ^ dx = dx ^ dm.
    # i_X (dx ^ dm) = A_ dm - C_ dx.
    # So L_X alpha = (f_x - C_) dx + f_y dy + (f_m + A_) dm, and it must be
    # proportional to dy - m dx.
    fx, fy, fm = sp.diff(f, x), sp.diff(f, y), sp.diff(f, m)
    lam = fy                                    # the dy-component
    eqs = [sp.Eq(fx - C_, -m * lam), sp.Eq(fm + A_, 0), sp.Eq(B_ - m * A_, f)]
    sol = sp.solve(eqs, [A_, B_, C_], dict=True)[0]
    Xa, Xb, Xc = sol[A_], sol[B_], sol[C_]
    say("  X_f = (%s) d_x + (%s) d_y + (%s) d_m" % (Xa, Xb, Xc))
    check("X_f is derived, not quoted: alpha(X_f) = f",
          sp.simplify(Xb - m * Xa - f) == 0)
    check("L_{X_f} alpha is proportional to alpha",
          sp.simplify(fx - Xc + m * fy) == 0 and sp.simplify(fm + Xa) == 0)

    # Projectable  <=>  the d_x, d_y components do not depend on m.
    def project_defect(fun):
        Xa_ = Xa.subs(f, fun).doit()
        Xb_ = Xb.subs(f, fun).doit()
        Xa_ = sp.diff(sp.simplify(-sp.diff(fun, m)), m)
        Xb_ = sp.diff(sp.simplify(fun - m * sp.diff(fun, m)), m)
        return sp.simplify(Xa_), sp.simplify(Xb_)

    for name, fun in [("f = 1", sp.Integer(1)),
                      ("f = y (a translation)", y),
                      ("f = x*? -- a lift: f linear in m", x * m + y),
                      ("f = m^2  (the p^2 generator)", m ** 2),
                      ("f = m^3", m ** 3)]:
        da, db = project_defect(fun)
        say("    %-28s  d_m(X^x) = %-8s  d_m(X^y) = %-8s  %s"
            % (name, da, db, "projectable" if (da == 0 and db == 0) else "NOT projectable"))

    da, db = project_defect(m ** 2)
    check("f = m^2 gives a NON-projectable contact field (her p^2 generator)",
          not (da == 0 and db == 0))
    check("f linear in m is projectable (the lifts of vector fields on Sigma)",
          project_defect(x * m + y) == (0, 0))

    # The vertical line is span(d_m).  Its Lie derivative along X_{m^2}:
    Xm2 = [(-2 * m), (m ** 2 - 2 * m ** 2), sp.Integer(0)]
    brk = [-sp.diff(Xm2[0], m), -sp.diff(Xm2[1], m), -sp.diff(Xm2[2], m)]
    say("  [X_{m^2}, d_m] = (%s) d_x + (%s) d_y + (%s) d_m" % tuple(brk))
    check("[X_{m^2}, d_m] is not proportional to d_m -- the vertical line moves",
          not (sp.simplify(brk[0]) == 0 and sp.simplify(brk[1]) == 0))
    say("  Whence: no lift of data from Sigma to PT^*Sigma is k(3)-equivariant, and the")
    say("  classification of Subsec. 3.10 -- of the operators invariant under all of k(3) --")
    say("  cannot decide Problem 3.9.  It decides the pairings of the FUNCTIONS kappa_A.")


# ======================================================================= part 2
# The Riemannian natural-operator count.
#
# The bilinear scalar expressions of weight 2 in (A,B) on a Riemannian surface are
#   (a,b) = (2,0): nabla^2 A . B      (b) = (1,1): nabla A . nabla B
#   (a,b) = (0,2): A . nabla^2 B      plus the curvature terms  K * (A,B).
# In the flat model the first three are the O(2)-invariant bilinear forms on the
# jet spaces; the curvature terms are invisible there and are added by hand (there
# are exactly two of them, K tr A tr B and K <A,B>, and both are symmetric).

DIM = 2
J = [[0, -1], [1, 0]]                 # the so(2) generator
REFL = [[1, 0], [0, -1]]              # a reflection


def _sym_positions(n_pairs):
    """Index tuples of length n over {0,1}."""
    return list(product(range(DIM), repeat=n_pairs))


def invariant_space(a, b, extra_rows=None):
    """O(2)-invariant bilinear forms on (S^2 (x) S^a) x (S^2 (x) S^b), as vectors
    indexed by tuples of length 4+a+b; returns (basis, index list)."""
    n = 4 + a + b
    idx = _sym_positions(n)
    pos = {t: i for i, t in enumerate(idx)}
    rows = []

    def add_sym(k, l):
        """impose c[... i_k ... i_l ...] = c[... i_l ... i_k ...]"""
        for t in idx:
            if t[k] <= t[l]:
                continue
            s = list(t)
            s[k], s[l] = s[l], s[k]
            r = [0] * len(idx)
            r[pos[t]] += 1
            r[pos[tuple(s)]] -= 1
            rows.append(r)

    # A-side indices: 0,1 = the tensor pair; 2..a+1 = derivatives.
    # B-side: a+2,a+3 = pair; a+4..  = derivatives.
    Aid, Ader = (0, 1), tuple(range(2, 2 + a))
    Bid, Bder = (2 + a, 3 + a), tuple(range(4 + a, 4 + a + b))
    add_sym(*Aid)
    add_sym(*Bid)
    for grp in (Ader, Bder):
        for i in range(len(grp) - 1):
            add_sym(grp[i], grp[i + 1])

    # so(2)-invariance
    for t in idx:
        r = [0] * len(idx)
        for k in range(n):
            for mm in range(DIM):
                if J[mm][t[k]]:
                    s = list(t)
                    s[k] = mm
                    r[pos[tuple(s)]] += J[mm][t[k]]
        if any(r):
            rows.append(r)

    # reflection invariance
    for t in idx:
        sign = 1
        for k in range(n):
            sign *= REFL[t[k]][t[k]]
        if sign == -1:
            r = [0] * len(idx)
            r[pos[t]] = 1
            rows.append(r)

    if extra_rows:
        rows.extend(extra_rows)
    if not rows:
        return [[1 if i == j else 0 for i in range(len(idx))] for j in range(len(idx))], idx
    return nullspace(rows), idx


def contraction_vector(pairing, a, b):
    """The invariant given by a complete contraction: `pairing` is a list of index
    pairs (positions) contracted with delta.  The result is symmetrized over the two
    tensor pairs and over each group of derivative indices -- the arguments being
    symmetric, only that part of it is seen."""
    from itertools import permutations
    n = 4 + a + b
    idx = _sym_positions(n)
    pos = {t: i for i, t in enumerate(idx)}
    v = [Fr(0)] * len(idx)
    for t in idx:
        if all(t[i] == t[j] for i, j in pairing):
            v[pos[t]] = Fr(1)

    groups = [(0, 1), (2 + a, 3 + a)]
    if a > 1:
        groups.append(tuple(range(2, 2 + a)))
    if b > 1:
        groups.append(tuple(range(4 + a, 4 + a + b)))

    for grp in groups:
        w = [Fr(0)] * len(idx)
        perms = list(permutations(range(len(grp))))
        for t in idx:
            for pm in perms:
                s = list(t)
                for k, g in enumerate(grp):
                    s[g] = t[grp[pm[k]]]
                w[pos[tuple(s)]] += v[pos[t]] / len(perms)
        v = w
    return v, idx


def in_span(basis, v):
    if not basis:
        return all(x == 0 for x in v)
    M = [list(map(Fr, r)) for r in basis]
    r0 = rank(M)
    r1 = rank(M + [list(map(Fr, v))])
    return r0 == r1


def part2():
    say("\n=== 2. The bilinear scalar natural operators of weight 2 on a Riemannian surface")
    dims = {}
    for (a, b) in [(2, 0), (1, 1), (0, 2)]:
        basis, idx = invariant_space(a, b)
        dims[(a, b)] = len(basis)
        say("  (nabla^%d A) . (nabla^%d B):  dim of O(2)-invariants = %d" % (a, b, len(basis)))
    flat = dims[(2, 0)] + dims[(1, 1)] + dims[(0, 2)]
    say("  flat-model total = %d" % flat)

    # The swap A <-> B.  It exchanges the (2,0) and (0,2) blocks, so those contribute
    # dims[(2,0)] symmetric and dims[(2,0)] antisymmetric combinations; inside the
    # (1,1) block it is an involution whose eigenspaces have to be computed.
    a = b = 1
    n = 6
    idx = _sym_positions(n)
    pos = {t: i for i, t in enumerate(idx)}
    basis, _ = invariant_space(1, 1)

    def swap_vec(v):
        """A-block (0,1,2) <-> B-block (3,4,5)."""
        w = [Fr(0)] * len(idx)
        for t in idx:
            s = (t[3], t[4], t[5], t[0], t[1], t[2])
            w[pos[s]] = Fr(v[pos[t]])
        return w

    sym, anti = [], []
    for v in basis:
        v = [Fr(z) for z in v]
        sv = swap_vec(v)
        sym.append([p + q for p, q in zip(v, sv)])
        anti.append([p - q for p, q in zip(v, sv)])
    ds, da = rank(sym), rank(anti)
    say("  inside (nabla A).(nabla B):  symmetric %d, antisymmetric %d" % (ds, da))
    check("the (1,1) block splits %d = %d + %d" % (dims[(1, 1)], ds, da),
          ds + da == dims[(1, 1)])

    tot_sym = dims[(2, 0)] + ds + 2          # +2: the curvature terms, both symmetric
    tot_anti = dims[(2, 0)] + da
    say("  curvature terms (invisible in the flat model): K tr(A) tr(B) and K <A,B>,"
        " both symmetric: +2")
    say("  TOTAL  %d = %d symmetric + %d antisymmetric"
        % (tot_sym + tot_anti, tot_sym, tot_anti))
    check("the total is 15", tot_sym + tot_anti == 15)
    check("the split is 10 symmetric and 5 antisymmetric", (tot_sym, tot_anti) == (10, 5))

    # Weight 0, for completeness: the order-zero terms an answer may also carry.
    basis00, _ = invariant_space(0, 0)
    say("  weight 0 (no derivative at all): dim = %d -- <A,B> and tr(A) tr(B), both symmetric"
        % len(basis00))
    check("there are exactly 2 bilinear scalars of weight 0", len(basis00) == 2)
    say("  so the scalars of weight <= 2 make %d = %d symmetric + %d antisymmetric"
        % (tot_sym + tot_anti + 2, tot_sym + 2, tot_anti))

    # The named contractions, and the relations dimension 2 imposes.
    say("\n  The named contractions, and what dimension 2 identifies:")
    # (1,1): indices  A: (0,1)=ij, (2)=derivative a ;  B: (3,4)=kl, (5)=derivative b
    named11 = {
        "<nabla A, nabla B>       = A_{ij,a} B_{ij,a}": [(0, 3), (1, 4), (2, 5)],
        "A_{ij,a} B_{aj,i}                            ": [(2, 3), (0, 5), (1, 4)],
        "(div A).(div B)          = A_{aj,a} B_{bj,b} ": [(0, 2), (3, 5), (1, 4)],
        "(div A).(grad tr B)      = A_{aj,a} B_{kk,j} ": [(0, 2), (3, 4), (1, 5)],
        "(grad tr A).(div B)      = A_{ii,b} B_{bl,l} ": [(0, 1), (2, 3), (4, 5)],
        "(grad tr A).(grad tr B)                      ": [(0, 1), (3, 4), (2, 5)],
    }
    vecs, names = [], []
    for nm, pr in named11.items():
        v, _ = contraction_vector(pr, 1, 1)
        vecs.append(v)
        names.append(nm)
        check("   %s is O(2)-invariant" % nm.strip(), in_span(basis, v))
    r = rank(vecs)
    say("   the 6 contractions span a space of dimension %d -- so there is %d relation"
        % (r, len(vecs) - r))
    check("the six contractions of (nabla A).(nabla B) satisfy exactly one relation in dim 2",
          len(vecs) - r == 1 and r == dims[(1, 1)])
    rel = nullspace([[vecs[i][j] for i in range(len(vecs))] for j in range(len(vecs[0]))])
    if rel:
        say("   the relation, in the order listed:  " +
            "  ".join("%+s" % str(Fr(c)) for c in rel[0]))

    # (0,2): A_{ij} nabla_a nabla_b B_{kl}; indices A:(0,1); B pair (2,3); derivs (4,5)
    named02 = {
        "A_{ij} nabla_i nabla_j tr B ": [(0, 4), (1, 5), (2, 3)],
        "A_{ij} Laplace B_{ij}       ": [(0, 2), (1, 3), (4, 5)],
        "A_{ij} nabla_i nabla_a B_{aj}": [(0, 4), (2, 5), (1, 3)],
        "tr A * Laplace tr B         ": [(0, 1), (2, 3), (4, 5)],
        "tr A * nabla_a nabla_b B_{ab}": [(0, 1), (2, 4), (3, 5)],
    }
    basis02, _ = invariant_space(0, 2)
    vecs2 = []
    for nm, pr in named02.items():
        v, _ = contraction_vector(pr, 0, 2)
        vecs2.append(v)
        check("   %s is O(2)-invariant" % nm.strip(), in_span(basis02, v))
    r2 = rank(vecs2)
    say("   the 5 contractions span a space of dimension %d -- so there is %d relation"
        % (r2, len(vecs2) - r2))
    check("the five contractions of A.(nabla^2 B) satisfy exactly one relation in dim 2",
          r2 == dims[(0, 2)])
    return tot_sym, tot_anti


# ======================================================================= part 3
# The harmonics.  kappa_A(x,theta) = A(v,v) with v = (cos theta, sin theta), and
#   e_1 = d_theta,  e_2 = v^k d_k,  R = v_perp^k d_k.

def part3():
    say("\n=== 3. The harmonics of the bracket (Statement 3.11) and of a general weight-2 pairing")
    th = sp.symbols("theta", real=True)
    v = sp.Matrix([sp.cos(th), sp.sin(th)])

    # generic 2-jets at the origin
    A0 = {(i, j): sp.Symbol("a%d%d" % (min(i, j), max(i, j))) for i in range(2) for j in range(2)}
    B0 = {(i, j): sp.Symbol("b%d%d" % (min(i, j), max(i, j))) for i in range(2) for j in range(2)}
    A1 = {(i, j, k): sp.Symbol("a%d%d_%d" % (min(i, j), max(i, j), k))
          for i in range(2) for j in range(2) for k in range(2)}
    B1 = {(i, j, k): sp.Symbol("b%d%d_%d" % (min(i, j), max(i, j), k))
          for i in range(2) for j in range(2) for k in range(2)}
    A2 = {(i, j, k, l): sp.Symbol("a%d%d_%d%d" % (min(i, j), max(i, j), min(k, l), max(k, l)))
          for i in range(2) for j in range(2) for k in range(2) for l in range(2)}
    B2 = {(i, j, k, l): sp.Symbol("b%d%d_%d%d" % (min(i, j), max(i, j), min(k, l), max(k, l)))
          for i in range(2) for j in range(2) for k in range(2) for l in range(2)}

    def kappa(T):
        return sp.expand_trig(sp.expand(sum(T[(i, j)] * v[i] * v[j]
                                            for i in range(2) for j in range(2))))

    def dtheta(F):
        return sp.diff(F, th)

    def horiz(T1, T0order):
        """e_2 kappa = v^k d_k kappa: the k-th derivative of the coefficients."""
        return sp.expand(sum(T1[(i, j, k)] * v[i] * v[j] * v[k]
                             for i in range(2) for j in range(2) for k in range(2)))

    def horiz2(T2):
        return sp.expand(sum(T2[(i, j, k, l)] * v[i] * v[j] * v[k] * v[l]
                             for i in range(2) for j in range(2)
                             for k in range(2) for l in range(2)))

    z = sp.Symbol("z")

    def harmonics(expr, maxn=12):
        """The Fourier harmonics actually present.  Substituting
        cos th = (z + 1/z)/2, sin th = (z - 1/z)/(2i) turns the expression into
        a Laurent polynomial in z whose exponents ARE the harmonics -- exact, and
        far cheaper than integrating."""
        e = sp.expand(expr.subs({sp.cos(th): (z + 1 / z) / 2,
                                 sp.sin(th): (z - 1 / z) / (2 * sp.I)}))
        e = sp.expand(sp.together(e) * z ** (2 * maxn))
        p = sp.Poly(sp.expand(e), z)
        out = set()
        for (k,), c in p.terms():
            if sp.simplify(c) != 0:
                out.add(abs(k - 2 * maxn))
        return sorted(out)

    kA, kB = kappa(A0), kappa(B0)
    dA, dB = horiz(A1, None), horiz(B1, None)
    brk = sp.expand(dtheta(kA) * dB - dA * dtheta(kB))
    hs = harmonics(brk)
    say("  {kappa_A, kappa_B} has the harmonics %s" % hs)
    check("Statement 3.11: the harmonics are 1, 3, 5 and no others", hs == [1, 3, 5])
    check("Statement 3.11: the mean over the directions is 0", 0 not in hs)
    check("Statement 3.11: it is odd under theta -> theta + pi", all(n % 2 == 1 for n in hs))
    check("Statement 3.11: it is antisymmetric in (A,B)",
          sp.simplify(brk + brk.subs(list(zip(
              [A0[(0, 0)], A0[(0, 1)], A0[(1, 1)], A1[(0, 0, 0)], A1[(0, 0, 1)],
               A1[(0, 1, 0)], A1[(0, 1, 1)], A1[(1, 1, 0)], A1[(1, 1, 1)]],
              [B0[(0, 0)], B0[(0, 1)], B0[(1, 1)], B1[(0, 0, 0)], B1[(0, 0, 1)],
               B1[(0, 1, 0)], B1[(0, 1, 1)], B1[(1, 1, 0)], B1[(1, 1, 1)]])) +
              list(zip(
              [B0[(0, 0)], B0[(0, 1)], B0[(1, 1)], B1[(0, 0, 0)], B1[(0, 0, 1)],
               B1[(0, 1, 0)], B1[(0, 1, 1)], B1[(1, 1, 0)], B1[(1, 1, 1)]],
              [A0[(0, 0)], A0[(0, 1)], A0[(1, 1)], A1[(0, 0, 0)], A1[(0, 0, 1)],
               A1[(0, 1, 0)], A1[(0, 1, 1)], A1[(1, 1, 0)], A1[(1, 1, 1)]])),
              simultaneous=True)) == 0)
    check("it vanishes for B = A",
          sp.simplify(brk.subs(list(zip(
              [B0[(0, 0)], B0[(0, 1)], B0[(1, 1)], B1[(0, 0, 0)], B1[(0, 0, 1)],
               B1[(0, 1, 0)], B1[(0, 1, 1)], B1[(1, 1, 0)], B1[(1, 1, 1)]],
              [A0[(0, 0)], A0[(0, 1)], A0[(1, 1)], A1[(0, 0, 0)], A1[(0, 0, 1)],
               A1[(0, 1, 0)], A1[(0, 1, 1)], A1[(1, 1, 0)], A1[(1, 1, 1)]])),
              simultaneous=True)) == 0)
    check("it vanishes when both arguments are multiples of g (pull-backs)",
          sp.simplify(brk.subs([(A0[(0, 1)], 0), (B0[(0, 1)], 0),
                                (A0[(1, 1)], A0[(0, 0)]), (B0[(1, 1)], B0[(0, 0)]),
                                (A1[(0, 1, 0)], 0), (A1[(0, 1, 1)], 0),
                                (B1[(0, 1, 0)], 0), (B1[(0, 1, 1)], 0),
                                (A1[(1, 1, 0)], A1[(0, 0, 0)]),
                                (A1[(1, 1, 1)], A1[(0, 0, 1)]),
                                (B1[(1, 1, 0)], B1[(0, 0, 0)]),
                                (B1[(1, 1, 1)], B1[(0, 0, 1)])])) == 0)

    say("\n  The other weight-2 terms, and the parity rule:")
    terms = {
        "(d_theta^2 kappa_A) kappa_B          [0 horizontal]": dtheta(dtheta(kA)) * kB,
        "(d_theta kappa_A)(d_theta kappa_B)   [0 horizontal]": dtheta(kA) * dtheta(kB),
        "(d_theta kappa_A)(e_2 kappa_B)       [1 horizontal]": dtheta(kA) * dB,
        "(e_2 kappa_A)(e_2 kappa_B)           [2 horizontal]": dA * dB,
        "(e_2 e_2 kappa_A) kappa_B            [2 horizontal]": horiz2(A2) * kB,
        "(e_2 d_theta kappa_A) kappa_B        [1 horizontal]": sp.diff(dA, th) * kB,
    }
    worst = 0
    for nm, e in terms.items():
        hs = harmonics(e)
        worst = max(worst, max(hs) if hs else 0)
        nhor = int(nm.split("[")[1][0])
        par_ok = all(n % 2 == nhor % 2 for n in hs)
        say("    %-52s harmonics %-18s parity %s" % (nm, hs, "OK" if par_ok else "MISMATCH"))
        check("   parity = number of horizontal derivatives for %s" % nm.split("[")[0].strip(),
              par_ok)
    check("the largest harmonic occurring at weight 2 is 6 (her corrected remark)", worst == 6)


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metriclift.log"), "w")
    say("metriclift.py -- the metric-lift Problem")
    part1()
    part2()
    part3()
    say("\n%d checks, %d passed, %d failed" % (CHECKS[0], CHECKS[1], CHECKS[0] - CHECKS[1]))
    LOG.close()


if __name__ == "__main__":
    main()
