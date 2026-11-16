"""
Find the correction to the alpha-part of equation 273: solve for a genuine
cocycle that agrees with the printed table on all entries except a chosen set,
and print what the entries in that set must be.
"""

from fractions import Fraction as Q
from itertools import product

from stringy import mono
from kL4 import (bracket, monomials, weight, by_weight, name_of, c_mono)
from solveL import Unknowns, reduce_rows, nullspace, eqs

B = 4
A = 2 * B + 1
S0 = (-1, (1, 2, 3, 4))
keysA = [k for k in monomials(range(-A, A + 1)) if k != S0]
keysB = [k for k in monomials(range(-B, B + 1)) if k != S0]

U = Unknowns()
WA = by_weight(keysA)
for k1 in keysB:
    for k2 in WA.get(-weight(k1), ()):
        U.get(k1, k2)
nB = len(U.keys)
rows = list(eqs(U, keysB))
piv = reduce_rows(rows)
free, basis = nullspace(piv, len(U.keys))
bas = [tuple(v[:nB]) for v in basis]
piv2 = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in bas])
BAS = []
for p in sorted(piv2):
    v = [Q(0)] * nB
    for i, x in piv2[p].items():
        v[i] = x
    BAS.append(v)
print("B=%d, dim of the cocycle space in the window = %d" % (B, len(BAS)))

alpha = [c_mono(k1, k2)[0] for (k1, k2) in U.keys]

def types(i):
    n1, n2 = name_of(U.keys[i][0]), name_of(U.keys[i][1])
    return (n1[0], n2[0])

def solve(free_types):
    """cocycle equal to the table's alpha outside free_types."""
    fixed = [i for i in range(nB) if types(i) not in free_types]
    m = len(BAS)
    aug = [[v[i] for v in BAS] + [alpha[i]] for i in fixed]
    pivc, r = {}, 0
    for c in range(m):
        pr = next((i for i in range(r, len(aug)) if aug[i][c]), None)
        if pr is None:
            continue
        aug[r], aug[pr] = aug[pr], aug[r]
        pv = aug[r][c]
        aug[r] = [x / pv for x in aug[r]]
        for i in range(len(aug)):
            if i != r and aug[i][c]:
                f = aug[i][c]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[r])]
        pivc[c] = r
        r += 1
    for row in aug:
        if all(x == 0 for x in row[:m]) and row[m]:
            return None
    x = [Q(0)] * m
    for c, rr in pivc.items():
        x[c] = aug[rr][m]
    return [sum(x[j] * BAS[j][i] for j in range(m)) for i in range(nB)]


GG = {("G1", "Gb1"), ("Gb1", "G1"), ("G2", "Gb2"), ("Gb2", "G2")}
FF = {("F1", "Fb1"), ("Fb1", "F1"), ("F2", "Fb2"), ("Fb2", "F2")}

for label, ft in (("only c(G^i_m, Gbar^i_n) free", GG),
                  ("c(G,Gbar) and c(F,Fbar) free", GG | FF)):
    sol = solve(ft)
    print("\n=== %s ===" % label)
    if sol is None:
        print("   no cocycle of this shape")
        continue
    print("   solution found; the free entries must be:")
    seen = {}
    for i in range(nB):
        if types(i) in ft:
            n1, n2 = name_of(U.keys[i][0]), name_of(U.keys[i][1])
            seen.setdefault((n1[0], n2[0]), []).append((n1[1], n2[1], sol[i]))
    for t in sorted(seen):
        ents = sorted(seen[t])
        print("     c(%s_m, %s_n), m+n+1=0:" % t,
              ", ".join("m=%s: %s" % (a, c) for a, b, c in ents))
    # check the rest really equals the table
    diff = [i for i in range(nB) if types(i) not in ft and sol[i] != alpha[i]]
    print("   entries outside the free set that differ from the table:", len(diff))

# uniqueness: is there a nonzero cocycle supported only on the free entries?
def homog(free_types):
    fixed = [i for i in range(nB) if types(i) not in free_types]
    m = len(BAS)
    rows = [{j: BAS[j][i] for j in range(m) if BAS[j][i]} for i in fixed]
    p = reduce_rows(rows)
    return m - len(p)

print("\nhomogeneous solutions supported on the G-entries only:", homog(GG))
print("homogeneous solutions supported on the G- and F-entries only:", homog(GG | FF))
