"""Does the answer depend on the weights?  (It must not, for the bound to work.)

The composites that could reach a high order pass through intermediate
weights, so the statement "no invariant operator of arity m has order above
m(m-1)/2" is needed for ALL tuples of nonzero weights, not only for (1,...,1).
Tested here for arity 2, 3, 4 (and 5 on a few tuples), over Q.
"""

import sys
from fractions import Fraction as F
from line import inv_dim


def mahonian(n):
    poly = [1]
    for r in range(2, n + 1):
        new = [0] * (len(poly) + r - 1)
        for i, c in enumerate(poly):
            for j in range(r):
                new[i + j] += c
        poly = new
    return poly


ok, bad = 0, []


def chk(name, cond):
    global ok
    if cond:
        ok += 1
    else:
        bad.append(name)


TUPLES = {
    2: [(1, 1), (1, 2), (2, 3), (5, 1), (3, 3), (F(1, 2), F(1, 2)),
        (F(-2, 3), F(-2, 3)), (-1, -1), (7, 4)],
    3: [(1, 1, 1), (2, 1, 1), (3, 1, 1), (1, 2, 3), (4, 4, 4), (2, 2, 3),
        (6, 1, 1), (F(1, 2), 1, 2), (-1, -1, -1), (F(-2, 3), 1, 1)],
    4: [(1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1), (2, 2, 1, 1), (4, 1, 1, 1),
        (1, 2, 3, 4), (7, 1, 1, 1), (-1, -1, -1, -1)],
    5: [(1, 1, 1, 1, 1), (2, 1, 1, 1, 1), (3, 1, 1, 1, 1)],
}

for n, tuples in TUPLES.items():
    mah = mahonian(n)
    top = n * (n - 1) // 2
    print(f"arity {n}: Mahonian {mah}")
    for w in tuples:
        w = tuple(F(a) for a in w)
        dims = [inv_dim(w, s) for s in range(top + 3)]
        same = dims[:top + 1] == mah and dims[top + 1:] == [0, 0]
        print(f"   {tuple(str(a) for a in w)}: {dims}"
              + ("" if same else "   <-- differs"))
        chk(f"arity {n} weights {w}", same)
        sys.stdout.flush()

print()
print(f"{ok} checks passed, {len(bad)} failed")
for b in bad:
    print("  FAIL", b)

# the bound the census rests on
print()
print("max order reachable by a composite of arity n (blocks of sizes n_1..n_r,")
print("r < n blocks, top operator of arity r):")
from itertools import product


def partitions(n, maxpart=None):
    if maxpart is None:
        maxpart = n
    if n == 0:
        yield []
        return
    for p in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - p, p):
            yield [p] + rest


for n in range(2, 9):
    best, arg = -1, None
    for part in partitions(n):
        r = len(part)
        if r == n or r == 1:
            continue
        val = sum(p * (p - 1) // 2 for p in part) + r * (r - 1) // 2
        if val > best:
            best, arg = val, part
    top = n * (n - 1) // 2
    print(f"   n={n}: max composite order {best} (blocks {arg}),"
          f" top order {top},  forced indecomposable in orders"
          f" {best+1}..{top}")
