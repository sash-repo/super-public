"""Self-tests for skew.py: the engine is calibrated against facts nobody disputes
(Specht module dimensions, the Littlewood-Richardson rule, Kostka numbers) before
it is used on the questions in the text."""

from itertools import combinations
from skew import (Diagram, dim_T, dim_S_B_of_A, dim_S_pair, dim_intersection_S,
                  nu, semi_ordered_of_type, tableaux_of_type, is_semi_ordered,
                  has_cycles_for)

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    ok = got == want
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(("  ok   " if ok else "  FAIL ") + f"{name}: got {got}, want {want}")
    return ok


def young(lam, name=None):
    return Diagram.skew(lam, (), name or str(lam))


# --------------------------------------------------------------------------- #
print("1. the groups and the diagrams")
A = young((3, 2))
check("|A|", len(A), 5)
check("row sizes", A.row_sizes, (3, 2))
check("|R_A|", len(A.row_stabilizer()), 6 * 2)
check("|C_A|", len(A.column_stabilizer()), 2 * 2 * 1)
check("A'' = A", young((3, 2)).transpose().transpose().cells, A.cells)
check("(3,2)' = (2,2,1)", young((3, 2)).transpose().cells, young((2, 2, 1)).cells)
check("skew (3,2,1)/(1) size", len(Diagram.skew((3, 2, 1), (1,))), 5)

# --------------------------------------------------------------------------- #
print("\n2. dim T_B(A) = number of contingency matrices (Kostka-type count)")
# rows sums (2,1), col sums (2,1): matrices [[a,b],[c,d]], a+b=2, c+d=1, a+c=2 -> 2
check("T for (2,1),(2,1)", dim_T(young((2, 1)), young((2, 1))), 2)
check("T for row_3, col_3", dim_T(Diagram.row(3), Diagram.column(3)), 1)
check("T for col_3, row_3", dim_T(Diagram.column(3), Diagram.row(3)), 1)

# --------------------------------------------------------------------------- #
print("\n3. S_B(A) is the Specht-module count (Prop. 3.1): dim = # semi-ordered")
for lam in [(2,), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (2, 1, 1)]:
    for mu in [(2,), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (2, 1, 1)]:
        A, B = young(lam), young(mu)
        if len(A) != len(B):
            continue
        want = len([t for t in semi_ordered_of_type(A, B)])
        check(f"dim S_{mu}({lam})", dim_S_B_of_A(A, B), want)

# --------------------------------------------------------------------------- #
print("\n4. type = one row: content (q), so all entries are 1 and a second row kills it")
for lam in [(3,), (2, 1), (3, 1), (2, 2)]:
    A = young(lam)
    check(f"semi-ordered {lam}-tableaux of type (row)",
          len(semi_ordered_of_type(A, Diagram.row(len(A)))), 1 if len(A.rows) == 1 else 0)

# the honest Kostka check: type = column of q cells means content (1,1,...,1),
# so the count is the number of standard (ordered) tableaux -- the hook length formula
def hook_count(lam):
    cells = [(i, j) for i, li in enumerate(lam) for j in range(li)]
    n = len(cells)
    conj = [sum(1 for li in lam if li > j) for j in range(max(lam))]
    num, den = 1, 1
    for k in range(1, n + 1):
        num *= k
    for (i, j) in cells:
        den *= (lam[i] - j) + (conj[j] - i) - 1
    return num // den


print("\n5. content (1^q): the count is the number of standard tableaux (hook formula)")
for lam in [(2, 1), (3, 1), (2, 2), (3, 2), (2, 2, 1)]:
    A = young(lam)
    check(f"std tableaux of {lam}",
          len(semi_ordered_of_type(A, Diagram.column(len(A)))), hook_count(lam))

# --------------------------------------------------------------------------- #
print("\n6. Remark 3.4.3 d): nu(A,B) = delta_{AB} for Young diagrams A, B")
shapes = [(2,), (1, 1), (3,), (2, 1), (1, 1, 1), (2, 2), (3, 1), (2, 1, 1)]
for lam in shapes:
    for mu in shapes:
        if sum(lam) != sum(mu):
            continue
        check(f"nu({lam},{mu})", nu(young(lam), young(mu)), 1 if lam == mu else 0)

# --------------------------------------------------------------------------- #
print("\n7. Theorem 3.4: dim S(A,B) = nu(A,B), and S(A,B) = S_B(A) cap S_A^{-1}(B)")
tests = []
for lam, mu in [((2,), (2,)), ((2,), (1, 1)), ((1, 1), (1, 1)),
                ((2, 1), (2, 1)), ((2, 1), (3,)), ((3,), (2, 1)),
                ((2, 1), (1, 1, 1)), ((2, 2), (2, 2)), ((3, 1), (2, 2))]:
    tests.append((young(lam), young(mu)))
tests.append((Diagram.skew((3, 2, 1), (1,), "(321)/(1)"), Diagram.skew((3, 2, 1), (1,), "(321)/(1)")))
tests.append((Diagram.skew((2, 2), (1,), "(22)/(1)"), young((2, 1))))
tests.append((Diagram.skew((2, 2), (1,), "(22)/(1)"), Diagram.skew((2, 2), (1,), "(22)/(1)")))
tests.append((Diagram.skew((3, 1), (1,), "(31)/(1)"), young((2, 1))))

for A, B in tests:
    n, d = nu(A, B), dim_S_pair(A, B)
    check(f"dim S({A},{B}) = nu", d, n)
    check(f"  intersection for ({A},{B})", dim_intersection_S(A, B), d)

# --------------------------------------------------------------------------- #
print("\n8. Remark 3.4.3 a)-c): the symmetries of nu")
sym = [(young((2, 1)), young((2, 1))),
       (Diagram.skew((3, 2, 1), (1,)), Diagram.skew((3, 2, 1), (1,))),
       (Diagram.skew((2, 2), (1,)), young((2, 1))),
       (Diagram.skew((3, 1), (1,)), Diagram.skew((2, 2), (1,))),
       (Diagram.skew((3, 2), (1,)), young((2, 2))),
       (young((3, 1)), Diagram.skew((2, 2, 1), (1,)))]
for A, B in sym:
    if len(A) != len(B):
        continue
    check(f"a) nu({A},{B})=nu({B},{A})", nu(A, B), nu(B, A))
    check(f"b) nu(A',B)=nu(A,B') for ({A},{B})", nu(A.transpose(), B), nu(A, B.transpose()))
    check(f"c) nu(A,B)=nu(A',B') for ({A},{B})", nu(A, B), nu(A.transpose(), B.transpose()))

# --------------------------------------------------------------------------- #
print("\n9. Remark 3.4.3 e): B a Young diagram -> nu(A,B) is the LR coefficient")
# For A = lam/mu a skew Young diagram and B a Young diagram of the same size,
# nu(A,B) must be the LR coefficient c^{lam}_{mu,B}.  Compute the latter
# independently, by the classical LR rule (lattice reverse reading word).
def lr_coefficient(lam, mu, nuu):
    """c^lam_{mu,nuu}: semistandard fillings of lam/mu of content nuu whose reverse
    reading word is a lattice word."""
    A = Diagram.skew(lam, mu)
    B = Diagram.skew(nuu)
    cnt = 0
    for t in semi_ordered_of_type(A, B):
        word = []
        for (i, j) in sorted(A.cells, key=lambda c: (c[0], -c[1])):
            word.append(t[(i, j)])
        run = {}
        good = True
        for x in word:
            run[x] = run.get(x, 0) + 1
            if x > 1 and run[x] > run.get(x - 1, 0):
                good = False
                break
        if good:
            cnt += 1
    return cnt


lr_cases = [((3, 2, 1), (1,), (3, 2)), ((3, 2, 1), (1,), (2, 2, 1)),
            ((3, 2, 1), (1,), (4, 1)), ((2, 2), (1,), (2, 1)),
            ((3, 2), (1,), (2, 2)), ((3, 3, 1), (2, 1), (2, 2)),
            ((4, 2), (2,), (2, 2)), ((3, 2, 1), (1, 1), (2, 2))]
for lam, mu, nuu in lr_cases:
    A = Diagram.skew(lam, mu)
    B = Diagram.skew(nuu)
    if len(A) != len(B):
        continue
    check(f"nu({lam}/{mu}, {nuu}) = c^lam_mu,nu", nu(A, B), lr_coefficient(lam, mu, nuu))

# --------------------------------------------------------------------------- #
print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
