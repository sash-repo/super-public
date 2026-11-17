"""What the indecomposable operators are: explicit formulas and S_n-structure.

The symmetric group permutes the arguments, so every graded piece is an
S_n-module and so is the space of indecomposables (the decomposable subspace
is S_n-stable, being defined by a construction that does not see the order of
the arguments).  Printed here: the character, the decomposition into
irreducibles, and an explicit basis of the top pieces.
"""

from fractions import Fraction as F
from itertools import permutations
import sys
from line import inv_space, inv_dim, midx, decomposable
from exactlin import rank


def perm_action(vec, n, k, sigma):
    """(sigma.D)(u_1,...,u_n) = D(u_{sigma(1)},...,u_{sigma(n)})."""
    cols = midx(n, k)
    pos = {z: j for j, z in enumerate(cols)}
    out = [F(0)] * len(cols)
    for z, c in zip(cols, vec):
        if c:
            out[pos[tuple(z[sigma[i]] for i in range(n))]] += c
    return out


def character(basis, n, k):
    """Trace of each permutation on the span of `basis` (assumed stable).

    The span is put in reduced row echelon form; then the coordinates of any
    vector of the span in that basis are simply its entries at the pivot
    columns, so a trace costs one matrix-vector pass instead of a solve.
    """
    if not basis:
        return {sigma: F(0) for sigma in permutations(range(n))}
    from exactlin import rref
    R, piv = rref([list(v) for v in basis])
    R = [r for r in R if any(r)]
    assert len(R) == len(piv)
    chars = {}
    for sigma in permutations(range(n)):
        tr = F(0)
        for j, row in enumerate(R):
            img = perm_action(row, n, k, sigma)
            tr += F(img[piv[j]])
        chars[sigma] = tr
    return chars


def cycle_type(sigma):
    n = len(sigma)
    seen = [False] * n
    t = []
    for i in range(n):
        if not seen[i]:
            ln = 0
            j = i
            while not seen[j]:
                seen[j] = True
                j = sigma[j]
                ln += 1
            t.append(ln)
    return tuple(sorted(t, reverse=True))


# character tables, S_2..S_5, indexed by partition
IRR = {
    2: {(2,): {(1, 1): 1, (2,): 1},
        (1, 1): {(1, 1): 1, (2,): -1}},
    3: {(3,): {(1, 1, 1): 1, (2, 1): 1, (3,): 1},
        (2, 1): {(1, 1, 1): 2, (2, 1): 0, (3,): -1},
        (1, 1, 1): {(1, 1, 1): 1, (2, 1): -1, (3,): 1}},
    4: {(4,): {(1, 1, 1, 1): 1, (2, 1, 1): 1, (2, 2): 1, (3, 1): 1, (4,): 1},
        (3, 1): {(1, 1, 1, 1): 3, (2, 1, 1): 1, (2, 2): -1, (3, 1): 0, (4,): -1},
        (2, 2): {(1, 1, 1, 1): 2, (2, 1, 1): 0, (2, 2): 2, (3, 1): -1, (4,): 0},
        (2, 1, 1): {(1, 1, 1, 1): 3, (2, 1, 1): -1, (2, 2): -1, (3, 1): 0, (4,): 1},
        (1, 1, 1, 1): {(1, 1, 1, 1): 1, (2, 1, 1): -1, (2, 2): 1, (3, 1): 1, (4,): -1}},
    5: {(5,): {(1,)*5: 1, (2, 1, 1, 1): 1, (2, 2, 1): 1, (3, 1, 1): 1,
               (3, 2): 1, (4, 1): 1, (5,): 1},
        (4, 1): {(1,)*5: 4, (2, 1, 1, 1): 2, (2, 2, 1): 0, (3, 1, 1): 1,
                 (3, 2): -1, (4, 1): 0, (5,): -1},
        (3, 2): {(1,)*5: 5, (2, 1, 1, 1): 1, (2, 2, 1): 1, (3, 1, 1): -1,
                 (3, 2): 1, (4, 1): -1, (5,): 0},
        (3, 1, 1): {(1,)*5: 6, (2, 1, 1, 1): 0, (2, 2, 1): -2, (3, 1, 1): 0,
                    (3, 2): 0, (4, 1): 0, (5,): 1},
        (2, 2, 1): {(1,)*5: 5, (2, 1, 1, 1): -1, (2, 2, 1): 1, (3, 1, 1): -1,
                    (3, 2): -1, (4, 1): 1, (5,): 0},
        (2, 1, 1, 1): {(1,)*5: 4, (2, 1, 1, 1): -2, (2, 2, 1): 0, (3, 1, 1): 1,
                       (3, 2): 1, (4, 1): 0, (5,): -1},
        (1,)*5: {(1,)*5: 1, (2, 1, 1, 1): -1, (2, 2, 1): 1, (3, 1, 1): 1,
                 (3, 2): -1, (4, 1): -1, (5,): 1}},
}


def decompose(chars, n):
    from math import factorial
    out = []
    for lam, ch in IRR[n].items():
        m = F(0)
        for sigma, val in chars.items():
            m += val * ch[cycle_type(sigma)]
        m /= factorial(n)
        if m:
            out.append((lam, m))
    return out


def fmt(lam):
    return "[" + ",".join(map(str, lam)) + "]"


def basis_of_atoms(n, s):
    """A basis of a complement of the decomposables (here: the whole space,
    for the top pieces where nothing is decomposable)."""
    full = inv_space(tuple([F(1)] * n), s)
    dec, _ = decomposable(tuple([F(1)] * n), s)
    ddec = rank([list(v) for v in dec]) if dec else 0
    return full, ddec


def show(vec, n, k):
    names = "abcdefg"[:n]
    terms = []
    for z, c in zip(midx(n, k), vec):
        if not c:
            continue
        mon = " ".join(names[i] + ("" if z[i] == 0 else
                                   ("'" * z[i] if z[i] <= 3 else f"^({z[i]})"))
                       for i in range(n))
        sign = "+" if c > 0 else "-"
        cc = abs(c)
        terms.append(f"{sign} {cc if cc != 1 else ''}{mon}")
    return " ".join(terms).lstrip("+ ") if terms else "0"


import sys as _sys
NMAX = int(_sys.argv[1]) if len(_sys.argv) > 1 else 4

for n in range(2, NMAX + 1):
    print("=" * 70)
    print(f"n = {n}")
    for s in range(0, n * (n - 1) // 2 + 1):
        full = inv_space(tuple([F(1)] * n), s)
        if not full:
            continue
        dec, _ = decomposable(tuple([F(1)] * n), s)
        ddec = rank([list(v) for v in dec]) if dec else 0
        ch = character(full, n, s)
        dec_full = decompose(ch, n)
        line = f"  s={s}: dim {len(full)} = " + " + ".join(
            (f"{m}*" if m != 1 else "") + fmt(lam) for lam, m in dec_full)
        line += f"   | decomposable {ddec}"
        if dec:
            chd = character(dec, n, s)
            line += " = " + (" + ".join((f"{m}*" if m != 1 else "") + fmt(lam)
                                        for lam, m in decompose(chd, n))
                             or "0")
        line += f", INDECOMPOSABLE {len(full)-ddec}"
        print(line)
        _sys.stdout.flush()
    print()

print("=" * 70)
print("the indecomposables, explicitly")
print("=" * 70)
for n, ss in [(2, [0, 1]), (3, [3]), (4, [5, 6])]:
    for s in ss:
        full, ddec = basis_of_atoms(n, s)
        print(f"\n n={n}, order s={s} (k={n+s}), dim {len(full)}:")
        for v in full:
            print("   ", show(v, n, s))
