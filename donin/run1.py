"""Single cases of Sections 2-3 of the text: the Example after Remark 3.4.4, Lemma 3.3
in its smallest instance, and the vanishing in the proof of Proposition 2.1."""

from itertools import permutations, combinations
from skew import (Diagram, dim_T, dim_S_B_of_A, dim_S_pair, dim_intersection_S,
                  nu, semi_ordered_of_type, tableaux_of_type, is_semi_ordered,
                  has_cycles_for, show, hom, _mapping_tabloid, e_of_mapping,
                  _span_dim, tabloid, _sign)

line = "-" * 70


# =========================================================================== #
print(line)
print('Q "why is it 3?"  (the Example after Remark 3.4.4)')
print(line)
A = Diagram.skew((3, 2, 1), (1,), "A")
print("A = (3,2,1)/(1), semi-standard numbering:")
print(show(A, {c: A.row_number[c] for c in A.cells}))
print("\ncolumns of A (hence the columns whose images must be cycles):")
for col in A.columns:
    print("   ", col, "-> rows", [c[0] for c in col])

allt = semi_ordered_of_type(A, A)
print(f"\nsemi-ordered A-tableaux of type A: {len(allt)}")
good = 0
for t in allt:
    ok = has_cycles_for(A, A, t)
    good += ok
    print(("  [cycles]    " if ok else "  [no cycles]") + "  " + show(A, t).replace("\n", " | "))
print(f"\nnu(A,A) = {good}     dim S(A,A) = {dim_S_pair(A, A)}")


# =========================================================================== #
print(line)
print("Q on Lemma 3.3: A = one row of 2 cells, B = one column of 2 cells")
print(line)
A = Diagram.row(2, "row_2")
B = Diagram.column(2, "col_2")
print(f"dim T_B(A) = {dim_T(A, B)}      dim T_A(B) = {dim_T(B, A)}")
print(f"dim S_B(A) = {dim_S_B_of_A(A, B)}   (so dim S_B^-1(A) is the same)")
print(f"dim S_A(B) = {dim_S_B_of_A(B, A)}   (so dim S_A^-1(B) is the same)")
print(f"dim (S_B(A) cap S_A^-1(B)) = {dim_intersection_S(A, B)}")
print(f"dim S(A,B) = {dim_S_pair(A, B)}      nu(A,B) = {nu(A, B)}")
print()
print("the point is about Lemma 3.3, i.e. about S_B^-1(A) subset T_A(B):")
print(f"  columns of A with more than one cell: "
      f"{[c for c in A.columns if len(c) > 1]}  <- none")
print("  so 'cycles corresponding to A' is a condition on the EMPTY set of columns,")
print("  hence every B-tableau of type A satisfies it vacuously.")
bt = tableaux_of_type(B, A)
print(f"  B-tableaux of type A: {len(bt)}; those with cycles corresponding to A: "
      f"{sum(has_cycles_for(B, A, t) for t in bt)}")
print(f"  Lemma 3.3 therefore asks for k = dim S_B^-1(A) = {dim_S_B_of_A(A, B)} of them.  OK.")
print()
print("and Theorem 3.4 is not touched, because the intersection involves S_A^-1(B),")
print("which is 0 here (C_B is the full S_2 on the single column and the two")
print("tabloids it produces coincide, so every e_g cancels).")


# =========================================================================== #
print(line)
print("Q on the Garnir relation: does  sum_{mu in sigma_{piX}} (-1)^mu [mu pi t] = 0 ?")
print(line)


def garnir_check(lam, verbose=True):
    """For every A-tableau t that is column-increasing but not ordered, build the
    Garnir set X of the first offending pair of neighbouring columns and test the
    vanishing of the inner sum, for every pi in C_A."""
    A = Diagram.skew(lam)
    q = len(A)
    cols = A.columns
    tested = vanished = 0
    for perm in permutations(range(1, q + 1)):
        t = dict(zip(A.cells, perm))
        # column-increasing?
        if any(t[c] > t[d] for col in cols for c, d in zip(col, col[1:])):
            continue
        # find neighbouring columns j, j+1 and a row where the left entry exceeds
        # the right one: the classical Garnir situation
        found = None
        for jj in range(len(cols) - 1):
            cj, cj1 = cols[jj], cols[jj + 1]
            for r in range(len(cj1)):
                if t[cj[r]] > t[cj1[r]]:
                    found = (jj, r)
                    break
            if found:
                break
        if not found:
            continue
        jj, r = found
        cj, cj1 = cols[jj], cols[jj + 1]
        X = cj[r:] + cj1[:r + 1]          # a_q..a_r and b_1..b_q
        tested += 1
        ok = True
        for pi, spi in A.column_stabilizer():
            piX = [pi[c] for c in X]
            acc = {}
            for img in permutations(piX):
                mu = {c: d for c, d in zip(piX, img)}
                s = _sign({**{c: c for c in A.cells}, **mu}, A.cells)
                tt = {c: t[c] for c in A.cells}
                # (mu pi) t : the tableau whose cell mu(pi(c)) carries t(c)
                base = {pi[c]: t[c] for c in A.cells}
                newt = {mu.get(c, c): v for c, v in base.items()}
                key = tabloid(A, newt)
                acc[key] = acc.get(key, 0) + s
            acc = {k: v for k, v in acc.items() if v}
            if acc:
                ok = False
                if verbose:
                    print(f"    NONZERO for lam={lam}, t={perm}, X={X}")
                break
        vanished += ok
    return tested, vanished


for lam in [(2,), (1, 1), (2, 1), (2, 2), (3, 1), (2, 2, 1), (3, 2)]:
    tested, vanished = garnir_check(lam)
    print(f"  lam={str(lam):10s}  Garnir situations tested: {tested:4d}   inner sum "
          f"vanished in {vanished:4d}")
print()
print("For lam=(2,) -- one row of two cells -- there is one Garnir situation: X is")
print("the whole row, sigma_X = S_2, and the two summands are one tabloid taken with")
print("opposite signs.")
