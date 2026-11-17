"""Statement 7.1, items 1)-3), over ALL pairs (A, B) of equal order taken from the
stock of skew diagrams with at most five cells (the same stock as in run2.py):

  1) Theorem 3.4: dim S(A,B) = nu(A,B) and dim (S_B(A) cap S_A^{-1}(B)) = nu(A,B);
  2) Remark 3.4.3 a)-e): the symmetries of nu, nu(A,B) = delta_{AB} for Young A, B,
     and, for a Young diagram B, nu(A,B) = the number of semi-ordered A-tableaux of
     type B whose reading from right to left and downwards is a lattice word;
     and, for every B, the same count with the B-nice words of Remark 3.4.3;
  3) Theorem 3.4.2: nu'(A,B) = dim S'(A,B) = nu(A',B) = nu(A,B').
"""

import contextlib
import io

from skew import Diagram, nu, dim_S_pair, dim_intersection_S, semi_ordered_of_type

with contextlib.redirect_stdout(io.StringIO()):      # run2 prints its own report on import
    from run2 import diags, dim_S_prime

PASS = FAIL = 0


def check(name, ok):
    global PASS, FAIL
    PASS, FAIL = PASS + bool(ok), FAIL + (not ok)
    if not ok:
        print("  FAIL " + name)


def reading_word(A, t):
    """right to left inside each row, rows from the top down."""
    return [t[c] for c in sorted(A.cells, key=lambda c: (c[0], -c[1]))]


def d_numbers(B):
    """d_i (i >= 2): cells of the i-th row of B to the left of the first cell of row i-1."""
    rows = B.rows
    d = {}
    for i in range(1, len(rows)):
        first_above = min(c[1] for c in rows[i - 1])
        d[i + 1] = sum(1 for c in rows[i] if c[1] < first_above)
    return d


def is_B_lattice(word, d):
    """every letter B-nice: 1 always; k if #(nice k before it) < d_k + #(nice k-1 before it)."""
    cnt = {}
    for x in word:
        if x > 1 and not cnt.get(x, 0) < d.get(x, 0) + cnt.get(x - 1, 0):
            return False
        cnt[x] = cnt.get(x, 0) + 1
    return True


stock = diags(5)
pairs = [(A, B) for A in stock for B in stock if len(A) == len(B)]
print(f"{len(stock)} skew diagrams with at most five cells, {len(pairs)} ordered pairs of equal order")

for A, B in pairs:
    n_ab = nu(A, B)
    At, Bt = A.transpose(), B.transpose()
    check(f"Theorem 3.4 dim S({A},{B})", dim_S_pair(A, B) == n_ab)
    check(f"Theorem 3.4 intersection ({A},{B})", dim_intersection_S(A, B) == n_ab)
    check(f"a) ({A},{B})", n_ab == nu(B, A))
    check(f"b) ({A},{B})", nu(At, B) == nu(A, Bt))
    check(f"c) ({A},{B})", n_ab == nu(At, Bt))
    if A.is_young() and B.is_young():
        check(f"d) ({A},{B})", n_ab == (1 if A.cells == B.cells else 0))
    tabs = semi_ordered_of_type(A, B)
    if B.is_young():
        lat = sum(is_B_lattice(reading_word(A, t), {}) for t in tabs)
        check(f"e) ({A},{B})", n_ab == lat)
    nice = sum(is_B_lattice(reading_word(A, t), d_numbers(B)) for t in tabs)
    check(f"B-nice rule ({A},{B})", n_ab == nice)
    check(f"Theorem 3.4.2 ({A},{B})", dim_S_prime(A, B) == nu(At, B) == nu(A, Bt))

print(f"{PASS} passed, {FAIL} failed")
