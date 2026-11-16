#!/usr/bin/env python3
r"""Independent controls on the two invariants of `leiso.py`.

1) THE SPAN of the [x,x], x odd — the invariant used in the text (Theorem 6.3,
   Remark 6.4).  Since p =/= 2, that span is the span of all [e_i,e_j] with
   i, j odd (polarization), so it is computed without any enumeration; it must
   reproduce the numbers of `peribr.py`: 0 against p^{u_1} for n = 1, and 18
   against 18 for n = 2 with u = One.  This is a cross-engine control: this
   file's brackets are built from the closed formulas of Theorem 6.1 alone,
   `peribr.py` builds them from the form omega by inverting its Gram matrix.

2) THE COUNT #{x odd | [x,x] = 0} for the undeformed shape, computed in the
   OTHER direction: for fixed c in C the map a -> [a+c, a+c] = 2B(a,c) is
   linear in a, so the count is a sum over the 3^9 values of c.  It must give
   the same number as the fiber-over-a method of `leiso.py`.

3) A MONTE-CARLO control on both counts, which uses neither splitting: sample
   random odd elements and count the square-zero ones.

Usage:  python3 lecount.py
"""
import sys, os, itertools, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leiso import LeAlg, odd_split, span_dim, count_sqzero_affine     # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import rank as _rank                                   # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def span_of_squares(L):
    """dim span{[x,x] | x in g_od} = dim span{[e_i,e_j] | i,j in od} for p > 2."""
    T = L.table()
    vs = []
    for i in L.od:
        for j in L.od:
            vs.append(T[i][j])
    return span_dim(vs, L.p)


def count_sqzero_over_c(L, A, C):
    """for the undeformed shape: fiber over c, linear in a."""
    p, d = L.p, L.dim
    T = L.table()
    for i in A:
        for j in A:
            if any(x % p for x in T[i][j]):
                raise AssertionError("[A,A] != 0: this direction does not apply")
    BAC = {(i, j): [(2 * T[i][j][k]) % p for k in range(d)] for i in A for j in C}
    total = 0
    for co in itertools.product(range(p), repeat=len(C)):
        colsT = []
        for i in A:
            colv = [0] * d
            for b, j in enumerate(C):
                cb = co[b]
                if not cb:
                    continue
                for k, v in enumerate(BAC[(i, j)]):
                    if v:
                        colv[k] = (colv[k] + cb * v) % p
            colsT.append(colv)
        M = [[colsT[a][k] for a in range(len(A))] for k in range(d)]
        r = _rank(M, p) if any(any(row) for row in M) else 0
        total += p ** (len(A) - r)
    return total


def monte_carlo(L, nsample=1500000, seed=7):
    p, d = L.p, L.dim
    T = L.table()
    od = L.od
    rnd = random.Random(seed)
    Q = {i: T[i][i] for i in od}
    B = {}
    for a in range(len(od)):
        for b in range(a + 1, len(od)):
            i, j = od[a], od[b]
            B[(a, b)] = [(2 * T[i][j][k]) % p for k in range(d)]
    hits = 0
    for _ in range(nsample):
        co = [rnd.randrange(p) for _ in od]
        val = [0] * d
        for a in range(len(od)):
            ca = co[a]
            if not ca:
                continue
            c2 = (ca * ca) % p
            for k, v in enumerate(Q[od[a]]):
                if v:
                    val[k] = (val[k] + c2 * v) % p
            for b in range(a + 1, len(od)):
                cb = co[b]
                if not cb:
                    continue
                cc = (ca * cb) % p
                for k, v in enumerate(B[(a, b)]):
                    if v:
                        val[k] = (val[k] + cc * v) % p
        if not any(val):
            hits += 1
    return hits, nsample


def main():
    global LOG
    here = os.path.dirname(os.path.abspath(__file__))
    LOG = open(os.path.join(here, "lecount.log"), "w")

    say("=== 1) the span of the [x,x] (cross-engine control against peribr.py) ===")
    for (n, u) in [(1, (1,)), (1, (2,)), (2, (1, 1)), (3, (1, 1, 1))]:
        for sh in ("0", "Xi"):
            L = LeAlg(n, 3, u, sh)
            say(f"n={n} u={u} shape={sh}: sdim = {L.sdim()}, "
                f"dim span[x,x] = {span_of_squares(L)}")

    say("")
    say("=== 2) the count for shape 0, fibered over c instead of over a ===")
    L0 = LeAlg(2, 3, (1, 1), "0")
    A, C = odd_split(L0)
    n_a = count_sqzero_affine(L0, A, C)
    n_c = count_sqzero_over_c(L0, A, C)
    say(f"n=2 u=One shape=0: over a {n_a}, over c {n_c}, agree = {n_a == n_c}")

    say("")
    say("=== 3) Monte-Carlo control (no splitting used) ===")
    for sh in ("0", "Xi"):
        L = LeAlg(2, 3, (1, 1), sh)
        A, C = odd_split(L)
        exact = count_sqzero_affine(L, A, C)
        hits, ns = monte_carlo(L)
        tot = 3 ** (len(A) + len(C))
        say(f"n=2 u=One shape={sh}: exact {exact} of {tot}; "
            f"sampled {hits} of {ns}, expected {ns * exact / tot:.1f}")
    LOG.close()


if __name__ == "__main__":
    main()
