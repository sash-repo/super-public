"""Theorem 5.2 against the honest decomposition of S^k(gl(n)) and E^k(gl(n)).

The essential multiplicity is extracted from the total one by Kostant's theorem
S(g) = H (x) I with I = C[tr X, tr X^2, ..., tr X^n]:
      M(t) = E(t) / prod_{d=1..n} (1-t^d),   so   E(t) = M(t) * prod (1-t^d).
For the exterior powers the invariants form the exterior algebra on primitives of
degrees 1,3,...,2n-1, and E(t) = M(t) / prod (1+t^{2i-1}) IF the analogous
splitting holds -- which is exactly what the text says fails in general, so we
test it rather than assume it.
"""

from itertools import product as iproduct
from skew import Diagram, nu
from run2 import nu_triple_formula
from gln import decompose

PASS = FAIL = 0
line = "-" * 70


def check(name, got, want):
    global PASS, FAIL
    ok = got == want
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(("  ok   " if ok else "  FAIL ") + f"{name}: got {got}, want {want}")


# --------------------------------------------------------------------------- #
def partitions_into(k, q, maxpart):
    """partitions of k into exactly q parts, each between 1 and maxpart."""
    out = []

    def rec(left, parts, hi):
        if len(parts) == q:
            if left == 0:
                out.append(tuple(parts))
            return
        room = q - len(parts)
        for p in range(min(hi, left - (room - 1)), 0, -1):
            if p * room < left:
                break
            rec(left - p, parts + [p], p)

    rec(k, [], maxpart)
    return out


def C_data(mu, exterior):
    """mu is a partition of k into q parts.  Group the equal parts: the distinct
    values k_1 > ... > k_p occur mu_1,...,mu_p times.  C^0_mu is the disjoint
    union of rows of sizes mu_i; in C^1_mu the i-th component is a row if k_i is
    even and a column if k_i is odd.  Returns (kinds, sizes)."""
    vals, sizes = [], []
    for v in mu:
        if vals and vals[-1] == v:
            sizes[-1] += 1
        else:
            vals.append(v); sizes.append(1)
    kinds = [(0 if not exterior else (0 if v % 2 == 0 else 1)) for v in vals]
    return kinds, sizes


def essential_by_theorem(f, k, n, exterior=False):
    """formulas (5.7) / (5.8)."""
    alpha = tuple(sorted([x for x in f if x > 0], reverse=True))
    beta = tuple(sorted([-x for x in f if x < 0], reverse=True))
    if sum(alpha) != sum(beta):
        return 0
    q = sum(alpha)
    if q == 0:
        return 1 if k == 0 else 0
    A, B = Diagram.skew(alpha), Diagram.skew(beta)
    total = 0
    parts = partitions_into(k, q, n - 1 if not exterior else k)
    for mu in parts:
        kinds, sizes = C_data(mu, exterior)
        total += nu_triple_formula(kinds, sizes, A, B)
    return total


def essential_by_kostant(dec, n, kmax):
    """E(t) = M(t) * prod_{d=1..n}(1-t^d), coefficientwise per weight."""
    weights = set()
    for k in range(kmax + 1):
        weights |= set(dec[k])
    ess = {}
    for w in weights:
        M = [dec[k].get(w, 0) for k in range(kmax + 1)]
        E = list(M)
        for d in range(1, n + 1):
            new = list(E)
            for j in range(kmax, d - 1, -1):
                new[j] = E[j] - E[j - d]
            E = new
        ess[w] = E
    return ess


def essential_by_kostant_ext(dec, n, kmax):
    """E(t) = M(t) / prod_{i=1..n}(1+t^{2i-1})."""
    weights = set()
    for k in range(kmax + 1):
        weights |= set(dec[k])
    ess = {}
    for w in weights:
        E = [dec[k].get(w, 0) for k in range(kmax + 1)]
        for i in range(1, n + 1):
            d = 2 * i - 1
            if d > kmax:
                break
            new = list(E)
            for j in range(d, kmax + 1):
                new[j] = E[j] - new[j - d]
            E = new
        ess[w] = E
    return ess


# =========================================================================== #
print(line)
print("SYMMETRIC POWERS:  formula (5.7) against S^k(gl(n)) / Kostant")
print(line)
for n, kmax in [(3, 6), (4, 5)]:
    dec = decompose(n, kmax)
    ess = essential_by_kostant(dec, n, kmax)
    print(f"\n  n = {n}, k up to {kmax}")
    for k in range(1, kmax + 1):
        rows = []
        for w in sorted(ess, reverse=True):
            a, b = ess[w][k], essential_by_theorem(w, k, n)
            if a or b:
                rows.append((w, a, b))
        bad = [r for r in rows if r[1] != r[2]]
        check(f"    k={k}: {len(rows)} weights with a nonzero essential multiplicity",
              [r[1] for r in rows], [r[2] for r in rows])
        for w, a, b in rows[:4]:
            print(f"        f={w}  Kostant {a}   formula (5.7) {b}")
        if bad:
            for w, a, b in bad:
                print(f"        MISMATCH f={w}: Kostant {a}, formula {b}")

# =========================================================================== #
print()
print(line)
print("EXTERIOR POWERS:  formula (5.8) against E^k(gl(n)), k < n")
print(line)
for n, kmax in [(4, 3), (5, 4)]:
    dec = decompose(n, kmax, exterior=True)
    ess = essential_by_kostant_ext(dec, n, kmax)
    print(f"\n  n = {n}, k up to {kmax}  (the theorem claims k < n)")
    for k in range(1, kmax + 1):
        rows = []
        for w in sorted(ess, reverse=True):
            a, b = ess[w][k], essential_by_theorem(w, k, n, exterior=True)
            if a or b:
                rows.append((w, a, b))
        check(f"    k={k}", [r[1] for r in rows], [r[2] for r in rows])
        for w, a, b in rows[:4]:
            print(f"        f={w}  invariants {a}   formula (5.8) {b}")

print(f"\n{PASS} passed, {FAIL} failed")
