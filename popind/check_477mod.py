"""Sec. 4.7.7 of [P], part 2: the lattice pairs and the modular bookkeeping.

The data of a K-invariant lattice for the four palindromic groups is a pair
(Delta_1, Delta_2) of lattices in C with Delta_1 <= Delta_2 <= c^{-1} Delta_1,
taken up to a common scalar.  The reversal Q of check_477.py acts by

        D(Delta_1, Delta_2) = (Delta_2, c^{-1} Delta_1)     (up to a scalar),

which is the Fricke involution W_c of the modular curve Y_0(c).  Here we verify,
with a symbolic modulus t, that D identifies

  * the two additional lattices of Sec. 4.7.7 with rows of Table 2 of [P], and
  * exactly the pairs of rows listed in Table 3 of [P] (Theorem 2.7.1),

and that the sporadic identifications of Table 3 at t = i and t = 1+omega come
from the extra units of Z[i] and Z[omega].
"""
import sympy as sp

t = sp.symbols('t')
i = sp.I
om = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2          # omega = e^{2 pi i/3}

OUT, CH = [], [0, 0]


def say(s=""):
    OUT.append(s)
    print(s)


def check(ok, what):
    CH[0] += 1
    CH[1] += 1 if ok else 0
    say("   [%s] %s" % ("ok " if ok else "FAIL", what))


# --------------------------------------------------- lattices in Q + Q*param
def coeffs(x, par):
    """x = a + b*par with a,b rational; returns (a,b) or None."""
    x = sp.simplify(sp.expand(x))
    p = sp.Poly(sp.together(x), par) if par.is_Symbol else None
    if p is not None:
        if p.degree() > 1:
            return None
        c = p.all_coeffs()
        c = [sp.nsimplify(z) for z in ([0] * (2 - len(c)) + c)]
        b, a = c[0], c[1]
    else:
        a, b = x, 0
    if not (sp.Rational(a) is not None):
        return None
    try:
        a, b = sp.Rational(a), sp.Rational(b)
    except TypeError:
        return None
    return (a, b)


def hnf(gens, par):
    """Hermite normal form of the lattice Z g_1 + Z g_2, g_j in Q + Q par."""
    rows = []
    for g in gens:
        c = coeffs(sp.simplify(g), par)
        if c is None:
            raise ValueError("not in Q + Q*par: %s" % g)
        rows.append([sp.Rational(c[0]), sp.Rational(c[1])])
    den = sp.lcm([sp.Rational(x).q for r in rows for x in r])
    M = [[int(x * den) for x in r] for r in rows]
    a, b = M[0]
    c_, d = M[1]
    while c_ != 0:
        q = a // c_
        a, b, c_, d = c_, d, a - q * c_, b - q * d
    if a < 0:
        a, b = -a, -b
    if d < 0:
        d = -d
    if d:
        b %= d
    return (sp.Rational(a, den), sp.Rational(b, den), sp.Rational(d, den))


def lat_eq(g1, g2, par):
    return hnf(g1, par) == hnf(g2, par)


def scale(gens, s):
    return [sp.simplify(sp.expand(s * g)) for g in gens]


# ----------------------------------------------------- the pairs of Table 2
def pairs(c, par):
    """the admissible (Delta_1, Delta_2) with Delta_1 = [1,par]: the index-1
    one, the c+1 of index c, and the index-c^2 one."""
    P = {}
    P["1"] = ([1, par], [1, par])
    for a in range(c):
        P["c:%d" % a] = ([1, par], [1, sp.Rational(a, c) + par / c])
    P["c:inf"] = ([1, par], [sp.Rational(1, c), par])
    P["c2"] = ([1, par], [sp.Rational(1, c), par / c])
    return P


def D(pair, c):
    """the reversal: (Delta_1,Delta_2) -> (Delta_2, c^{-1} Delta_1)."""
    d1, d2 = pair
    return (list(d2), [g / c for g in d1])


def same_pair(X, Y, sX, sY, par, what):
    ok = (lat_eq(scale(X[0], sX), scale(Y[0], sY), par) and
          lat_eq(scale(X[1], sX), scale(Y[1], sY), par))
    check(ok, what)
    return ok


say("=" * 78)
say("THE ADDITIONAL LATTICES OF Sec. 4.7.7 AND THE ROWS OF TABLE 2 OF [P]")
say("=" * 78)

for c, groups in ((2, "G(2,1,2)  and  K_28 = F_4"), (3, "G(6,6,2)")):
    say("")
    say("--- c = %d :  %s" % (c, groups))
    P = pairs(c, t)
    # the top lattice Delta_2 = c^{-1}Delta_1 is the image of Delta_2 = Delta_1
    same_pair(D(P["1"], c), P["c2"], 1, 1, t,
              "D(index 1, t) = (index c^2, t): the additional lattice "
              "Delta_2 = c^{-1}Delta_1 gives the r-group of row 1 with the same "
              "modulus")
    # the omitted index-c type [1/c, t] goes to the row [1, t/c] with t -> c t
    P2ct = ([1, c * t], [1, t])                    # = row "[1,beta/c]" with beta = c t
    same_pair(D(P["c:inf"], c), P2ct, c, 1, t,
              "D([1/c,t]) = ([1,ct],[1,t]): the omitted index-c lattice gives "
              "the r-group of the row [1,beta/c] with beta = %d t" % c)

say("")
say("=" * 78)
say("TABLE 3 OF [P] (Theorem 2.7.1), THE GENERIC ROWS")
say("=" * 78)

# c = 2: rows [.]_2 = ([1,b],[1,b/2]), [.]_3 = ([1,g],[1,(1+g)/2])
r2 = lambda b: ([1, b], [1, b / 2])
r3 = lambda g: ([1, g], [1, (1 + g) / 2])
say("")
say("--- [G(2,1,2)] and [F_4]  (c = 2)")
same_pair(D(r2(t), 2), r2(-2 / t), 2, t, t, "[.]^b_2 ~ [.]^{-2/b}_2")
same_pair(D(r2(t), 2), r3(1 - 2 / t), 2, t, t, "[.]^b_2 ~ [.]^{1-2/b}_3")
same_pair(D(r2(t), 2), r3(-1 - 2 / t), 2, t, t, "[.]^b_2 ~ [.]^{-1-2/b}_3")
same_pair(D(r3(t), 2), r3((t - 1) / (t + 1)), 2, t + 1, t, "[.]^g_3 ~ [.]^{(g-1)/(g+1)}_3")

# c = 3: [.]_2 = ([1,b],[1,b/3]), [.]_3 = ([1,g],[1,(1+g)/3]), [.]_4 = ([1,d],[1,(2+d)/3])
q2 = lambda b: ([1, b], [1, b / 3])
q3 = lambda g: ([1, g], [1, (1 + g) / 3])
q4 = lambda d: ([1, d], [1, (2 + d) / 3])
say("")
say("--- [G(6,6,2)]  (c = 3)")
same_pair(D(q2(t), 3), q2(-3 / t), 3, t, t, "[.]^b_2 ~ [.]^{-3/b}_2")
same_pair(D(q3(t), 3), q3((2 * t - 1) / (t + 1)), 3, t + 1, t,
          "[.]^g_3 ~ [.]^{(2g-1)/(g+1)}_3")
same_pair(D(q2(t), 3), q3(-1 + 3 / t), 3, t, t, "[.]^b_2 ~ [.]^{-1+3/b}_3")
same_pair(D(q2(t), 3), q3(2 - 3 / t), 3, t, t, "[.]^b_2 ~ [.]^{2-3/b}_3")
# and the fourth index-3 type is Popov's row 2 with the modulus shifted by 2
same_pair(D(q4(t), 3), ([1, (t + 2) / 3], [sp.Rational(1, 3), (t + 2) / 3]), 1, 1, t,
          "D([1,(2+d)/3]) = the omitted type [1/3,.] at (d+2)/3")

say("")
say("=" * 78)
say("TABLE 3 OF [P], THE SPORADIC ROWS AT THE ELLIPTIC POINTS (s >= 3)")
say("=" * 78)
say("   for s >= 3 the chain of G(2,1,s) is not palindromic: no reversal Q.")
say("   The identifications come from the units of End(Delta_1) instead:")
say("   Popov's rows are 2 = [1,(1+b)/2], 3 = [1/2,g], 4 = [1,d/2], 5 = (1/2)[1,l].")

# at t = i and t = 1+omega: the units of End(Delta_1) permute the three
# index-2 overlattices of Delta_1 = [1,t]
row2 = lambda x: [1, (1 + x) / 2]
row3 = lambda x: [sp.Rational(1, 2), x]
row4 = lambda x: [1, x / 2]


def mat_of(gens):
    return sp.Matrix([[sp.nsimplify(sp.re(g)), sp.nsimplify(sp.im(g))] for g in gens])


def eq_lat(g1, g2):
    T = sp.simplify(mat_of(g1) * mat_of(g2).inv())
    return all(sp.nsimplify(z).q == 1 for z in T) and abs(sp.simplify(sp.det(T))) == 1


for (u, x, A, B, nm) in ((i, i, row3, row4, "[G(2,1,s)]^i_3 ~ [G(2,1,s)]^i_4"),
                         (1 + om, 1 + om, row2, row3,
                          "[G(2,1,s)]^{1+w}_2 ~ [G(2,1,s)]^{1+w}_3"),
                         (1 + om, 1 + om, row2, row4,
                          "[G(2,1,s)]^{1+w}_2 ~ [G(2,1,s)]^{1+w}_4")):
    ok1 = eq_lat(scale([1, x], u), [1, x])
    found = None
    for k in range(1, 7):
        if eq_lat(scale(A(x), u ** k), B(x)):
            found = k
            break
    check(ok1 and found is not None,
          "%s : Delta_1 is stable under u = %s, and u^%s carries one "
          "index-2 overlattice to the other" % (nm, sp.simplify(u), found))

say("")
say("=" * 78)
say("checks passed: %d / %d" % (CH[1], CH[0]))
open("check_477mod.out", "w").write("\n".join(OUT) + "\n")


# ---------------------------------------------------------------------------
# The disputed row of Table 3 of [P]: for [G(2,1,2)] it is printed
#       [G(2,1,2)]^g_2 ~ [G(2,1,2)]^{(g-1)/(g+1)}_3
# while its F_4 twin, same c = 2, is printed
#       [F_4]^g_3 ~ [F_4]^{(g-1)/(g+1)}_3 .
# Which subscript on the left is right?  Decide numerically-decisively: two
# pairs are equivalent as r-groups iff they differ by a scalar, possibly after
# applying the reversal D; the scalar is bounded by the covolumes, so a finite
# search settles it.
say("")
say("=" * 78)
say("THE LEFT SUBSCRIPT OF THE ROW (g-1)/(g+1) OF TABLE 3 OF [P]")
say("=" * 78)

import itertools


def num(pair, g):
    return [[complex(sp.N(sp.sympify(x).subs(t, g))) for x in L] for L in pair]


def covol(L):
    return abs((L[0].conjugate() * L[1]).imag)


def equiv(P, Q_, box=8, eps=1e-9):
    """is there mu with mu*P = Q_ (as pairs of lattices in C)?"""
    u = P[0][0]
    for a, b in itertools.product(range(-box, box + 1), repeat=2):
        w = a * Q_[0][0] + b * Q_[0][1]
        if abs(w) < eps:
            continue
        mu = w / u
        ok = True
        for L, M in zip(P, Q_):
            sc = [mu * x for x in L]
            # transition matrix from sc to M must be integral with det +-1
            det = (M[0] * M[1].conjugate()).imag
            if abs(det) < eps:
                ok = False
                break
            coef = []
            for x in sc:
                p = (x * M[1].conjugate()).imag / det
                q = -(x * M[0].conjugate()).imag / det
                coef += [p, q]
            if any(abs(z - round(z)) > 1e-7 for z in coef):
                ok = False
                break
            d = round(coef[0]) * round(coef[3]) - round(coef[1]) * round(coef[2])
            if abs(d) != 1:
                ok = False
                break
        if ok:
            return mu
    return None


g0 = sp.Rational(3, 10) + sp.Rational(17, 10) * sp.I      # a generic modulus
Dm = lambda P: (list(P[1]), [x / 2 for x in P[0]])
lhs2, lhs3 = num(r2(t), g0), num(r3(t), g0)
rhs = num(r3((t - 1) / (t + 1)), g0)
for nm, L in (("[.]^g_2", lhs2), ("[.]^g_3", lhs3)):
    direct = equiv(L, rhs)
    viaD = equiv(Dm(L), rhs)
    say("   %s ~ [.]^{(g-1)/(g+1)}_3 : direct %s, after the reversal %s"
        % (nm, "yes" if direct else "no", "yes" if viaD else "no"))
check(equiv(lhs2, rhs) is None and equiv(Dm(lhs2), rhs) is None,
      "the printed left entry [G(2,1,2)]^g_2 is NOT equivalent to "
      "[G(2,1,2)]^{(g-1)/(g+1)}_3")
check(equiv(Dm(lhs3), rhs) is not None,
      "the left entry [.]^g_3 IS equivalent to [.]^{(g-1)/(g+1)}_3, as in the "
      "F_4 row of the same table")

say("")
say("checks passed: %d / %d" % (CH[1], CH[0]))
open("check_477mod.out", "w").write("\n".join(OUT) + "\n")
