"""The census: which invariant operators on the line are indecomposable.

Malyshev's (n,k)-operators are the invariant n-linear operators
cF_0^{(x)n} -> cF_k all of whose terms differentiate every argument.  Since
every argument is differentiated at least once, such an operator is
D o (d (x) ... (x) d) with D invariant on cF_1^{(x)n} of order s = k-n; that
reduction is checked here (STEP 1) and the census is then run on cF_1
(STEP 2), where "decomposable" has its Grozman meaning.
"""

import sys
from fractions import Fraction as F
from line import inv_space, inv_dim, midx, decomposable, atoms
from exactlin import rank

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def mahonian(n):
    poly = [1]
    for r in range(2, n + 1):
        new = [0] * (len(poly) + r - 1)
        for i, c in enumerate(poly):
            for j in range(r):
                new[i + j] += c
        poly = new
    return poly


print("=" * 72)
print("STEP 1.  W_n^{(k)} (Malyshev) = invariants on cF_1 of order k-n")
print("=" * 72)
for n in range(1, min(NMAX, 5) + 1):
    for k in range(n, n * (n + 1) // 2 + 1):
        # the subspace of inv((0,..,0),k) with every z_i >= 1
        full = inv_space(tuple([F(0)] * n), k)
        cols = midx(n, k)
        keep = [j for j, z in enumerate(cols) if all(zi >= 1 for zi in z)]
        proj = [[v[j] for j in keep] for v in full]
        dW = rank(proj) if proj else 0
        dU = inv_dim(tuple([F(1)] * n), k - n)
        flag = "ok" if dW == dU else "MISMATCH"
        print(f"  n={n} k={k}: dim W_n^(k) = {dW}, dim inv(cF_1,s={k-n}) = {dU}  {flag}")

print()
print("=" * 72)
print("STEP 2.  the census on cF_1^{(x)n}:  full / decomposable / atoms")
print("=" * 72)
for n in range(2, NMAX + 1):
    w = tuple([F(1)] * n)
    mah = mahonian(n)
    smax = n * (n - 1) // 2
    print(f"\n  n = {n}   (Malyshev's a_(n-1,s): {mah})")
    tot_a = 0
    for s in range(0, smax + 1):
        d = inv_dim(w, s)
        dec, wit = decomposable(w, s)
        dd = rank([list(v) for v in dec]) if dec else 0
        a = d - dd
        tot_a += a
        conj = mah[s] if s < len(mah) else 0
        mark = "" if d == conj else "   <-- differs from a_(n-1,s)"
        print(f"    s={s:2d}  k={n+s:2d}   dim={d:3d}  decomposable={dd:3d}"
              f"  INDECOMPOSABLE={a:3d}{mark}")
        sys.stdout.flush()
    print(f"    total dim = {sum(inv_dim(w, s) for s in range(smax+1))}"
          f"  (n! = {__import__('math').factorial(n)}),"
          f"  total indecomposable = {tot_a}")
