"""dim HS_f^k (the honest image of h^0) against formula (5.7) and against the
multiplicity in Kostant's harmonics."""

from itertools import permutations, product
from fractions import Fraction
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank

from prim import delta, mul, add, smul, is_primitive, trace, e, mono
from gln import decompose
from run3 import essential_by_kostant, essential_by_theorem, partitions_into
from skew import Diagram

line = "-" * 70


def young_cells(lam):
    return [(i + 1, j + 1) for i, li in enumerate(lam) for j in range(li)]


def col_stab(cells):
    cols = {}
    for c in cells:
        cols.setdefault(c[1], []).append(c)
    blocks = [sorted(v) for v in cols.values()]
    out = []
    for choice in product(*[permutations(b) for b in blocks]):
        perm, sgn = {}, 1
        for b, img in zip(blocks, choice):
            for c, d in zip(b, img):
                perm[c] = d
        # sign
        seen = set()
        for c in cells:
            if c in seen:
                continue
            L, d = 0, c
            while d not in seen:
                seen.add(d); d = perm[d]; L += 1
            if L % 2 == 0:
                sgn = -sgn
        out.append((perm, sgn))
    return out


def HS_dim(f, k, n, verbose=False):
    """dimension of the span of all Donin vectors (3) of weight f in S^k(gl(n))."""
    alpha = tuple(sorted([x for x in f if x > 0], reverse=True))
    beta = tuple(sorted([-x for x in f if x < 0], reverse=True))
    if not alpha or sum(alpha) != sum(beta):
        return 0
    q = sum(alpha)
    A, B = young_cells(alpha), young_cells(beta)
    iA = {c: c[0] for c in A}                       # row of A  -> index 1,2,...
    iB = {c: n + 1 - c[0] for c in B}               # row of B  -> index n, n-1, ...
    CA, CB = col_stab(A), col_stab(B)

    vectors = []
    for kappa in partitions_into(k, q, n - 1):
        # a numbering of A and of B by the multiset kappa, and a compatible bijection
        for permA in set(permutations(kappa)):
            numA = dict(zip(A, permA))
            for bij in permutations(B):
                fmap = dict(zip(A, bij))
                numB = {fmap[a]: numA[a] for a in A}
                # (the bijection is compatible by construction; numB must be a
                #  numbering of B by the same multiset, which it is)
                total = {}
                for pi, sp in CA:
                    for tau, st in CB:
                        term = {(): 1}
                        for a in A:
                            term = mul(term, delta(numA[a], iA[pi[a]],
                                                   iB[tau[fmap[a]]], n))
                        total = add(total, smul(sp * st, term))
                if total:
                    vectors.append(total)
    if not vectors:
        return 0
    keys = sorted({m for v in vectors for m in v})
    rows = [[Fraction(v.get(m, 0)) for m in keys] for v in vectors]
    return rank(rows)


print(line)
print("dim HS_f^k  vs  formula (5.7)  vs  multiplicity in Kostant's harmonics")
print(line)
print(f"{'n':>2} {'k':>2}  {'f':<18} {'dim HS':>7} {'(5.7)':>7} {'harmonic':>9}  verdict")

bad48 = badharm = 0
for n, kmax in [(3, 4), (4, 4)]:
    dec = decompose(n, kmax)
    ess = essential_by_kostant(dec, n, kmax)
    for k in range(1, kmax + 1):
        seen = set()
        for w in sorted(ess, reverse=True):
            if w in seen:
                continue
            seen.add(w)
            h = ess[w][k]
            t = essential_by_theorem(w, k, n)
            if not (h or t):
                continue
            d = HS_dim(w, k, n)
            v = []
            if d != t:
                v.append("(5.7) != dim HS"); bad48 += 1
            if d != h:
                v.append("dim HS != harmonic"); badharm += 1
            print(f"{n:>2} {k:>2}  {str(w):<18} {d:>7} {t:>7} {h:>9}  "
                  + ("; ".join(v) if v else "ok"))

print()
print(f"cases where (5.7) differs from dim HS:              {bad48}")
print(f"cases where dim HS differs from the harmonic count: {badharm}")
