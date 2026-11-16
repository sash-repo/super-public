"""The classical side of GKF: the Lie algebra h(2) of Hamiltonian formal vector fields
on the plane, its cochain complex relative to sp(2)=sl(2), and the bigraded pieces
C~_1^{q,N} whose dimensions the 1972 note [GKF] printed as the intermediate result of
its machine computation.

Basis of V_1 = h(2): the monomials e_{a,b} <-> x^a y^b, a,b >= 0, a+b >= 1 (Hamiltonians
modulo constants).  Poisson bracket

    {x^a y^b, x^c y^d} = (ad - bc) x^{a+c-1} y^{b+d-1},

the result being 0 when it is a constant.  A q-cochain is an element of Lambda^q(V_1^*);
in the polynomial language of [GKF] the monomial prod_i alpha_i^{a_i} beta_i^{b_i}
corresponds to the wedge of the duals of e_{a_i,b_i}, so the basis of C^q is the set of
q-element subsets of the monomials.  The bigrading is

    N_1 = sum_i a_i - q,   N_2 = sum_i b_i - q,

preserved by d; only N_1 = N_2 = N carries cohomology ([GKF], Sec. 2.1), and C^{q,N}
denotes that diagonal piece.

The subalgebra sp(1) = sl(2) is spanned by the three monomials of degree 2.  [GKF],
Sec. 1.4: a cochain p lies in the relative complex C~ iff

    1) no monomial of degree 2 occurs among its arguments (iota_g p = 0), and
    2) the same holds for dp (iota_g dp = 0, i.e. p is sp(1)-invariant).

Everything is exact arithmetic over Q.
"""

import sys
import os
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank, nullspace  # noqa: E402


# ---------------------------------------------------------------- the algebra

def bracket(m1, m2):
    """{x^a y^b, x^c y^d} = (ad-bc) x^{a+c-1} y^{b+d-1}; returns (coeff, monomial)."""
    a, b = m1
    c, d = m2
    coeff = a * d - b * c
    if coeff == 0:
        return 0, None
    p, q = a + c - 1, b + d - 1
    if p < 0 or q < 0 or (p == 0 and q == 0):
        return 0, None          # a constant Hamiltonian is 0 in V_1
    return coeff, (p, q)


def keyof(m):
    """The global order on the monomials: by degree first.  Every basis tuple, every
    sorting and every sign below uses this one order."""
    return (m[0] + m[1], m[0], m[1])


def monomials(maxa, maxb, relative):
    """All e_{a,b} with a<=maxa, b<=maxb, a+b>=1; degree 2 dropped if relative."""
    out = []
    for a in range(maxa + 1):
        for b in range(maxb + 1):
            if a + b < 1:
                continue
            if relative and a + b == 2:
                continue
            out.append((a, b))
    out.sort(key=keyof)
    return out


# ------------------------------------------------------- the cochain bases

def basis(q, N, relative):
    """The q-subsets S of monomials with sum a_i = sum b_i = N+q, sorted."""
    if q == 0:
        return [()] if N == 0 else []
    tot = N + q
    mons = monomials(tot, tot, relative)
    n = len(mons)
    deg = [m[0] + m[1] for m in mons]
    # suffix sums of the degrees, for the pruning below
    sufmin = [0] * (n + 1)          # smallest achievable degree sum, k items from index i
    out = []

    def mindeg(i, k):
        # the list is sorted by degree, so the k cheapest at index >= i are the next k
        if k == 0:
            return 0
        if i + k > n:
            return None
        return sum(deg[i:i + k])

    maxdeg_suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        maxdeg_suffix[i] = maxdeg_suffix[i + 1] + deg[i]

    def rec(i, k, ra, rb, acc):
        if k == 0:
            if ra == 0 and rb == 0:
                out.append(tuple(acc))
            return
        if i >= n:
            return
        need = ra + rb
        lo = mindeg(i, k)
        if lo is None or lo > need:
            return
        if maxdeg_suffix[i] - maxdeg_suffix[min(n, i + 0)] >= 0:  # no-op, kept for clarity
            pass
        # the k most expensive at index >= i
        hi = sum(deg[max(i, n - k):n]) if n - k >= i else maxdeg_suffix[i]
        if hi < need:
            return
        for j in range(i, n):
            a, b = mons[j]
            if a > ra or b > rb:
                continue
            acc.append(mons[j])
            rec(j + 1, k - 1, ra - a, rb - b, acc)
            acc.pop()

    rec(0, q, tot, tot, [])
    out.sort(key=lambda S: [keyof(m) for m in S])
    _ = sufmin
    return out


def perm_sign(seq):
    """Sign of the permutation putting seq into the order of keyof; None if repeated."""
    s = [keyof(m) for m in seq]
    if len(set(s)) != len(s):
        return None
    sign = 1
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[i] > s[j]:
                sign = -sign
    return sign


def differential(q, N, relative_src, relative_tgt):
    """Matrix of d: C^{q,N} -> C^{q+1,N}, rows = targets, cols = sources.

    relative_src / relative_tgt say whether the degree-2 monomials are dropped from the
    respective basis (condition 1 of [GKF], Sec. 1.4).
    """
    src = basis(q, N, relative_src)
    tgt = basis(q + 1, N, relative_tgt)
    sidx = {S: i for i, S in enumerate(src)}
    M = [[0] * len(src) for _ in range(len(tgt))]
    for r, T in enumerate(tgt):
        for u in range(len(T)):
            for v in range(u + 1, len(T)):
                c, w = bracket(T[u], T[v])
                if c == 0:
                    continue
                rest = [T[i] for i in range(len(T)) if i != u and i != v]
                cand = [w] + rest
                sign = perm_sign(cand)
                if sign is None:
                    continue
                S = tuple(sorted(cand, key=keyof))
                j = sidx.get(S)
                if j is None:
                    continue
                M[r][j] += ((-1) ** (u + v)) * sign * c
    return M, src, tgt


# --------------------------------------------------- the relative complex

def relative_space(q, N):
    """A basis of C~^{q,N} inside the condition-1 space, as coordinate vectors.

    Returns (vectors, cond1_basis).  A vector is the coordinate row of an element in the
    condition-1 basis; the space is the kernel of "the bad part of d".
    """
    cond1 = basis(q, N, True)
    if not cond1:
        return [], cond1
    full_next = basis(q + 1, N, False)
    bad = [i for i, T in enumerate(full_next) if any(m[0] + m[1] == 2 for m in T)]
    if not bad:
        return [[1 if i == j else 0 for i in range(len(cond1))] for j in range(len(cond1))], cond1
    M, src, tgt = differential(q, N, True, False)
    assert src == cond1 and tgt == full_next
    B = [M[r] for r in bad]
    ker = nullspace(B)                      # rows = basis vectors of the kernel
    return [list(v) for v in ker], cond1


def relative_table(Nmax=4, qmax=9):
    """dim C~_1^{q,N}, the table printed on p. 29 of [GKF]."""
    tab = {}
    for N in range(1, Nmax + 1):
        for q in range(1, qmax + 1):
            vecs, _ = relative_space(q, N)
            tab[(q, N)] = len(vecs)
    return tab


def relative_cohomology(N, qmax=9):
    """dim H^{q,N}(V_1, sp(1); R) for q = 0..qmax."""
    # the relative differential, written in the bases produced by relative_space
    spaces = {}
    for q in range(0, qmax + 2):
        spaces[q] = relative_space(q, N) if q >= 1 else ([], [])
    ranks = {}
    for q in range(1, qmax + 1):
        vsrc, bsrc = spaces[q]
        vtgt, btgt = spaces[q + 1]
        if not vsrc:
            ranks[q] = 0
            continue
        if not vtgt:
            # d must vanish
            M, s, t = differential(q, N, True, True)
            if t:
                img = [[sum(M[r][j] * vsrc[k][j] for j in range(len(vsrc[k])))
                        for r in range(len(t))] for k in range(len(vsrc))]
                assert rank(img) == 0, "d does not land in the relative complex"
            ranks[q] = 0
            continue
        M, s, t = differential(q, N, True, True)
        assert s == bsrc and t == btgt
        # image of each basis vector of C~^q, in the coordinates of the condition-1
        # basis of degree q+1
        img = []
        for vec in vsrc:
            col = [sum(M[r][j] * vec[j] for j in range(len(vec))) for r in range(len(t))]
            img.append(col)
        ranks[q] = rank(img) if img else 0
        # the image must lie in C~^{q+1}: check that each img row is in span(vtgt)
        if img:
            r0 = rank(vtgt)
            r1 = rank(list(vtgt) + img)
            assert r0 == r1, "d does not preserve the relative complex (q=%d, N=%d)" % (q, N)
    dims = {q: len(spaces[q][0]) for q in range(1, qmax + 1)}
    coh = {}
    for q in range(1, qmax + 1):
        coh[q] = dims[q] - ranks.get(q, 0) - ranks.get(q - 1, 0)
    return dims, ranks, coh


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "table"
    if what == "table":
        Nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        qmax = int(sys.argv[3]) if len(sys.argv) > 3 else 9
        tab = relative_table(Nmax, qmax)
        print("dim C~_1^{q,N}   (the table on p. 29 of [GKF])")
        print("q      " + "".join("%4d" % q for q in range(1, qmax + 1)))
        for N in range(1, Nmax + 1):
            print("N=%-4d " % N + "".join("%4d" % tab[(q, N)] for q in range(1, qmax + 1)))
    elif what == "coh":
        N = int(sys.argv[2])
        qmax = int(sys.argv[3]) if len(sys.argv) > 3 else 9
        dims, ranks, coh = relative_cohomology(N, qmax)
        print("N =", N)
        print("q        " + "".join("%4d" % q for q in range(1, qmax + 1)))
        print("dim C~   " + "".join("%4d" % dims[q] for q in range(1, qmax + 1)))
        print("rank d   " + "".join("%4d" % ranks[q] for q in range(1, qmax + 1)))
        print("dim H    " + "".join("%4d" % coh[q] for q in range(1, qmax + 1)))
