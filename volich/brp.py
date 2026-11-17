"""Bremner--Peresi, J. Algebra 322 (2009) 2000--2026, checked against our own
degree-<=5 computation, and their degree-5 identity verified from scratch.

Four parts.

(A)  Their Lemmas 10--14 publish, for each partition lambda of n, the number
     e-rank(lambda) of irreducible identities and the number l-rank(lambda) of
     those following from lower degrees.  Multiplying by the dimension d_lambda
     and summing must reproduce our dim I_n, dim C_n, dim I_n/C_n.

(B)  Their Theorem 7 is a NONLINEAR identity in three variables, 13 terms.
     Evaluate it on the generic elements x_1, x_2, x_3 of the free Volichenko
     algebra: it must vanish identically.

(C)  Its linearization must have 52 terms (their Remark 5), must lie in I_5,
     and together with C_5 must span the whole of I_5 -- that is their claim
     that the four new irreducible identities of degree 5 follow from it alone.

(D)  Their Theorem 6: identity (5) of their Theorem 1 (= Baranov's (v)) implies
     the new irreducibles for lambda = 5, 41, 32 but NOT the one for 2^2 1.
     So C_5 + <orbit of (5)> must have dim 1311 - 5 = 1306.
"""

import itertools
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rref as ex_rref                     # noqa: E402

import volid                                             # noqa: E402
from videal import (MSpace, identities, consequences, madd, msmul,      # noqa
                    mcomm, mcirc, mjbar, var, mevaluate, relabel_tree)

P = 2147483647            # a prime > n, so ranks agree with those over Q
                          # (this is their Lemma 8)


# --------------------------------------------------------------- (A) tables ---

# dimensions of the irreducible representations of S_n
DIM = {
    3: [("3", 1), ("21", 2), ("1^3", 1)],
    4: [("4", 1), ("31", 3), ("2^2", 2), ("21^2", 3), ("1^4", 1)],
    5: [("5", 1), ("41", 4), ("32", 5), ("31^2", 6),
        ("2^2 1", 5), ("21^3", 4), ("1^5", 1)],
}

# their published ranks
ERANK = {3: [1, 1, 1],                       # their Theorem 4
         4: [4, 9, 6, 9, 3],                 # their Lemma 10
         5: [12, 45, 54, 66, 54, 43, 11]}    # their Lemma 12
LRANK = {3: [0, 0, 0],
         4: [3, 7, 5, 9, 3],                 # their Lemma 11
         5: [11, 44, 53, 66, 53, 43, 11]}    # their Lemma 13

# what our own engine found (the output of vnew.py 5)
OURS = {3: (4, 0, 4), 4: (73, 64, 9), 5: (1311, 1296, 15)}

# our decomposition of the NEW identities, Statement 2.5.1 of the
# text: multiplicity of each partition
OURS_NEW = {3: [1, 1, 1], 4: [1, 2, 1, 0, 0], 5: [1, 1, 1, 0, 1, 0, 0]}


def part_a():
    print("(A)  their rank tables against our dimensions")
    ok = True
    for n in (3, 4, 5):
        dims = [d for _, d in DIM[n]]
        di = sum(e * d for e, d in zip(ERANK[n], dims))
        dc = sum(l * d for l, d in zip(LRANK[n], dims))
        new = [e - l for e, l in zip(ERANK[n], LRANK[n])]
        dnew = sum(m * d for m, d in zip(new, dims))
        oi, oc, onew = OURS[n]
        good = (di, dc, dnew) == (oi, oc, onew) and new == OURS_NEW[n]
        ok &= good
        print("     n=%d  dim I_n = %-5d (ours %-5d)  dim C_n = %-5d (ours %-5d)"
              "  new = %-3d (ours %-3d)   %s"
              % (n, di, oi, dc, oc, dnew, onew, "OK" if good else "MISMATCH"))
        print("           new multiplicities  %s" % dict(
            (name, m) for (name, _), m in zip(DIM[n], new) if m))
    print("     total irreducible identities in degrees 3,4,5: %d"
          % sum(sum(e - l for e, l in zip(ERANK[n], LRANK[n])) for n in (3, 4, 5)))
    return ok


# --------------------------------------------- (B) their Theorem 7 identity ---

# Juxtaposition is the Volichenko product.  Leaves are 'a', 'b', 'c'; the
# thirteen terms are transcribed from p. 2012 of the paper.
T7 = [
    (+1, (((('c', 'a'), 'b'), 'a'), 'b')),
    (-1, (((('c', 'a'), 'b'), 'b'), 'a')),
    (-1, (((('c', 'b'), 'a'), 'a'), 'b')),
    (+1, (((('c', 'b'), 'a'), 'b'), 'a')),
    (+1, ((('c', ('b', 'a')), 'a'), 'b')),
    (-1, ((('c', ('b', 'a')), 'b'), 'a')),
    (+1, (('a', ('a', ('b', 'b'))), 'c')),
    (-1, ((('a', 'a'), 'c'), ('b', 'b'))),
    (+1, ((('a', 'b'), 'c'), ('a', 'b'))),
    (-1, ((('b', 'b'), 'a'), ('a', 'c'))),
    (-1, (('a', ('b', 'c')), ('a', 'b'))),
    (+1, (('b', ('a', 'c')), ('a', 'b'))),
    (+1, ('a', ((('b', 'b'), 'a'), 'c'))),
]


def ev(tree, val):
    """Evaluate a symbol tree in the free Volichenko algebra."""
    if isinstance(tree, str):
        return val[tree]
    return volid.mul(ev(tree[0], val), ev(tree[1], val))


def part_b():
    print("(B)  their Theorem 7 identity, evaluated on generic x_1, x_2, x_3")
    val = {'a': volid.gen(1), 'b': volid.gen(2), 'c': volid.gen(3)}
    tot = {}
    for s, t in T7:
        tot = volid.add(tot, volid.smul(Fr(s), ev(t, val)))
    print("     %d terms; value in the free algebra: %s"
          % (len(T7), "0  -- it IS an identity" if not tot
             else "NONZERO (%d words)" % len(tot)))
    # a control: the same expression with one sign flipped must NOT vanish
    ctl = {}
    for i, (s, t) in enumerate(T7):
        ctl = volid.add(ctl, volid.smul(Fr(-s if i == 0 else s), ev(t, val)))
    print("     control (first sign flipped): %s"
          % ("nonzero, as it must be" if ctl else "ZERO -- test is vacuous!"))
    return (not tot) and bool(ctl)


# ------------------------------------------------- (C) its linearization ---

def leaves_in_order(t, out):
    if isinstance(t, str):
        out.append(t)
        return
    leaves_in_order(t[0], out)
    leaves_in_order(t[1], out)


def label(t, assign, ctr):
    """Rebuild the tree with integer labels taken in traversal order."""
    if isinstance(t, str):
        i = ctr[t]
        ctr[t] += 1
        return assign[t][i]
    return (label(t[0], assign, ctr), label(t[1], assign, ctr))


def linearize():
    """Sum over a -> {1,2}, b -> {3,4} in both orders, c -> 5."""
    f = {}
    for pa in itertools.permutations((1, 2)):
        for pb in itertools.permutations((3, 4)):
            assign = {'a': pa, 'b': pb, 'c': (5,)}
            for s, t in T7:
                tr = label(t, assign, {'a': 0, 'b': 0, 'c': 0})
                f[tr] = f.get(tr, 0) + Fr(s)
    return {t: c for t, c in f.items() if c}


def orbit(f):
    """The S_5-orbit of a multilinear element of M_5."""
    out = []
    for p in itertools.permutations(range(1, 6)):
        sub = {i + 1: p[i] for i in range(5)}
        g = {}
        for t, c in f.items():
            k = relabel_tree(t, sub)
            g[k] = g.get(k, 0) + c
        out.append({t: c for t, c in g.items() if c})
    return out


def echelon(rows, basis=None):
    rows = (basis or []) + [r for r in rows if any(r)]
    if not rows:
        return []
    R, _ = ex_rref(rows, p=P)
    return [r for r in R if any(r)]


def build_c5(ms):
    """C_5, the degree-5 consequences of the identities of degree < 5."""
    lower = {3: identities(3), 4: identities(4)}
    basis, chunk = [], []
    for g in consequences(lower, 5):
        chunk.append(ms.vec(g))
        if len(chunk) >= 400:
            basis = echelon(chunk, basis)
            chunk = []
    return echelon(chunk, basis)


def part_cd(ms, c5):
    print("(C)  the linearization of their Theorem 7 identity")
    f = linearize()
    print("     %d terms (their Remark 5 says 52)" % len(f))
    val = mevaluate(f)
    print("     lies in I_5: %s" % ("yes" if not val else "NO"))
    span = echelon([ms.vec(g) for g in orbit(f)], list(c5))
    print("     dim (C_5 + its S_5-orbit) = %d   (dim I_5 = 1311)" % len(span))
    okc = len(f) == 52 and not val and len(span) == 1311

    print("(D)  their Theorem 6: identity (5) alone")
    a, b, c, d, e = (var(i) for i in range(1, 6))
    J = mjbar(a, b, c)
    i5 = madd(mcomm(J, mcirc(d, e)),
              msmul(Fr(-1), mcirc(mcomm(J, d), e)),
              msmul(Fr(-1), mcirc(mcomm(J, e), d)))
    print("     identity (5) has %d terms; lies in I_5: %s"
          % (len(i5), "yes" if not mevaluate(i5) else "NO"))
    sp5 = echelon([ms.vec(g) for g in orbit(i5)], list(c5))
    print("     dim (C_5 + its S_5-orbit) = %d   (1311 - 5 = 1306 would leave"
          " exactly [2,2,1])" % len(sp5))
    okd = len(sp5) == 1306
    return okc, okd


# ---------------------------- (E) why their identity has three variables ---

def ssyt_count(shape, content):
    """Kostka number K_{shape, content}: semistandard tableaux, columns strict."""
    cells = [(i, j) for i, r in enumerate(shape) for j in range(r)]

    def fill(k, tab):
        if k == len(cells):
            return 1
        i, j = cells[k]
        tot = 0
        for v in range(1, len(content) + 1):
            if sum(1 for c in tab.values() if c == v) >= content[v - 1]:
                continue
            if j > 0 and tab[(i, j - 1)] > v:          # rows weakly increasing
                continue
            if i > 0 and tab[(i - 1, j)] >= v:         # columns strictly
                continue
            tab[(i, j)] = v
            tot += fill(k + 1, tab)
            del tab[(i, j)]
        return tot

    return fill(0, {})


def partitions(n, mx=None):
    mx = mx or n
    if n == 0:
        yield ()
        return
    for k in range(min(n, mx), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def part_e():
    print("(E)  in how few variables can [2,2,1] be written")
    lam = (2, 2, 1)
    rows = []
    for mu in partitions(5):
        rows.append((mu, len(mu), ssyt_count(lam, mu)))
    for mu, nv, k in rows:
        print("     multidegree %-12s %d variable(s)   K = %d"
              % (str(mu), nv, k))
    few = [k for mu, nv, k in rows if nv <= 2]
    here = [k for mu, nv, k in rows if mu == lam]
    ok = all(k == 0 for k in few) and here == [1]
    print("     => [2,2,1] is invisible in one and two variables, and first"
          " appears,\n        with multiplicity one, in multidegree (2,2,1)."
          "   %s" % ("OK" if ok else "FAIL"))
    return ok


def main():
    t0 = time.time()
    oka = part_a()
    print()
    okb = part_b()
    print()
    ms = MSpace(5)
    print("     building C_5 ...")
    c5 = build_c5(ms)
    print("     dim C_5 = %d" % len(c5))
    okc, okd = part_cd(ms, c5)
    print()
    oke = part_e()
    print("\n[%.0fs]  A %s   B %s   C %s   D %s   E %s"
          % (time.time() - t0,
             *("OK" if x else "FAIL" for x in (oka, okb, okc, okd, oke))))


if __name__ == "__main__":
    main()
