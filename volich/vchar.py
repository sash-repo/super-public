"""S_n-module structure of the identities of Volichenko algebras.

S_n permutes the variables, hence permutes the tree basis of M_n, and both
I_n (the identities) and C_n (the ones that follow from lower degrees) are
invariant.  The trace of a permutation on a subspace given in row-reduced
echelon form is read off the pivot columns, so the characters are cheap.
"""

import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rref as ex_rref  # noqa: E402

from videal import MSpace, identities, consequences, relabel_tree

# ---- character tables ------------------------------------------------------
# classes given as cycle types; sizes and irreducible characters
CHARS = {
    4: dict(
        classes=[(1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,)],
        sizes=[1, 6, 3, 8, 6],
        irr={"[4]":      [1, 1, 1, 1, 1],
             "[3,1]":    [3, 1, -1, 0, -1],
             "[2,2]":    [2, 0, 2, -1, 0],
             "[2,1,1]":  [3, -1, -1, 0, 1],
             "[1^4]":    [1, -1, 1, 1, -1]},
    ),
    5: dict(
        classes=[(1, 1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1), (3, 1, 1),
                 (3, 2), (4, 1), (5,)],
        sizes=[1, 10, 15, 20, 20, 30, 24],
        irr={"[5]":       [1, 1, 1, 1, 1, 1, 1],
             "[4,1]":     [4, 2, 0, 1, -1, 0, -1],
             "[3,2]":     [5, 1, 1, -1, 1, -1, 0],
             "[3,1,1]":   [6, 0, -2, 0, 0, 0, 1],
             "[2,2,1]":   [5, -1, 1, -1, -1, 1, 0],
             "[2,1,1,1]": [4, -2, 0, 1, 1, 0, -1],
             "[1^5]":     [1, -1, 1, 1, -1, -1, 1]},
    ),
}


def perm_from_cycletype(ct, n):
    """A representative permutation of the given cycle type, as a dict."""
    p, nxt = {}, 1
    for c in ct:
        cyc = list(range(nxt, nxt + c))
        for i, a in enumerate(cyc):
            p[a] = cyc[(i + 1) % c]
        nxt += c
    return p


def perm_index(ms, perm):
    """pi with  sigma . e_j = e_{pi(j)}  on the tree basis, and its inverse."""
    pi = [ms.pos[relabel_tree(t, perm)] for t in ms.trees]
    inv = [0] * len(pi)
    for j, k in enumerate(pi):
        inv[k] = j
    return pi, inv


def trace_on(basis, pivots, ms, perm, p):
    """Trace of the permutation on the subspace spanned by the rref rows.

    The rows are in rref with pivot columns `pivots`, so the coordinate of any
    v in the subspace along the i-th row is simply v[pivots[i]].
    """
    _, inv = perm_index(ms, perm)
    tr = 0
    for i, row in enumerate(basis):
        tr += row[inv[pivots[i]]]
    return tr % p


def decompose(char, n, p):
    """Multiplicities of the irreducibles in a character given by values."""
    tab = CHARS[n]
    order = 1
    for i in range(1, n + 1):
        order *= i
    out = {}
    for name, vals in tab["irr"].items():
        s = sum(sz * cv * ch for sz, cv, ch in zip(tab["sizes"], vals, char))
        assert s % order == 0, (name, s, order)
        m = s // order
        if m:
            out[name] = m
    return out


def lift(x, p):
    return x - p if x > p // 2 else x


def module_of(rows, ms, n, p, label=""):
    """rref the rows, then give the character and its decomposition."""
    if not rows:
        return {}, [0] * len(CHARS[n]["classes"])
    R, piv = ex_rref(rows, p=p)
    R = [r for r in R if any(r)]
    piv = list(piv)[:len(R)]
    char = []
    for ct in CHARS[n]["classes"]:
        perm = perm_from_cycletype(ct, n)
        char.append(lift(trace_on(R, piv, ms, perm, p), p))
    dec = decompose(char, n, p)
    if label:
        print("  %-28s dim %-5d  %s   %s"
              % (label, len(R), char,
                 " + ".join("%d.%s" % (m, k) if m > 1 else k
                            for k, m in sorted(dec.items()))))
    return dec, char


def run(n, p=2147483647):
    ms = MSpace(n)
    lower = {}
    for k in range(3, n):
        lower[k] = identities(k, MSpace(k))
    t0 = time.time()
    I = identities(n, ms)
    Irows = [[int(c) % p for c in ms.vec(f)] for f in I]
    Crows, seen = [], set()
    for f in consequences(lower, n):
        v = ms.vec(f)
        if not any(v):
            continue
        key = tuple(v)
        if key in seen:
            continue
        seen.add(key)
        Crows.append([int(c) % p for c in v])
        if len(Crows) > 4000:                     # keep the rref cheap
            R, _ = ex_rref(Crows, p=p)
            Crows = [r for r in R if any(r)]
    print("degree %d   (mod p = %d)" % (n, p))
    dI, cI = module_of(Irows, ms, n, p, "identities I_n")
    dC, cC = module_of(Crows, ms, n, p, "consequences C_n")
    newchar = [a - b for a, b in zip(cI, cC)]
    print("  %-28s dim %-5d  %s   %s"
          % ("NEW  I_n / C_n", newchar[0], newchar,
             " + ".join("%d.%s" % (m, k) if m > 1 else k
                        for k, m in sorted(decompose(newchar, n, p).items()))))
    print("  [%.1fs]" % (time.time() - t0))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
