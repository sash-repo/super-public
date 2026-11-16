"""The sl(2)-weight reduction of the relative complex of h(2) -- the wall of runs/.

`h2.relative_space` cuts C~^{q,N} out of the condition-1 space as the kernel of "the bad
part of d": a nullspace whose matrix has one row per basis element of the FULL (condition-1
violating) space of degree q+1.  Its cost grew by a factor of ~16 per internal degree
(N=6: 762 s, N=7: 13 400 s), which put N=8 at ~50 h and stopped the classical side.

None of that work is needed.  Write g = h(2), s = sl(2) = <x^2, xy, y^2>.  On the
condition-1 space W^q = Lambda^q(g/s)^* the bigrading of [GKF] IS the weight decomposition
of s: ad_{xy} acts on x^a y^b by (b-a), so a q-cochain with sum a_i = N_1+q, sum b_i = N_2+q
has h-weight +-(N_1-N_2), and e = x^2 raises a and lowers b, i.e.

    e : W^q_{(N_1, N_2)} --> W^q_{(N_1+1, N_2-1)},

the total degree N_1+N_2 = 2N being preserved.  Hence, with
M_0 = W^q_{(N,N)} and M_2 = W^q_{(N+1,N-1)}, fixing q and the total degree makes
M = sum_k W^q_{(N+k, N-k)} a finite-dimensional s-module, and

    1) C~^{q,N} = (M)^s = ker(e : M_0 --> M_2)
       -- a weight-0 vector killed by e is a highest weight vector of weight 0, so in
       char 0 it generates the trivial module and is invariant;
    2) e : M_0 --> M_2 is SURJECTIVE (in every irreducible with M_2 != 0 it is an
       isomorphism M_0 --> M_2), whence

           dim C~^{q,N} = dim M_0 - dim M_2                                      (*)

       -- two counts of subsets, and no linear algebra at all.

So the dimensions of the relative complex are a combinatorial count, and its BASIS is the
nullspace of the sparse |M_2| x |M_0| matrix of e (against h2's much larger bad-row
matrix), whose rank is known in advance to be |M_2| by 2) -- a free check on every run.

Everything is exact over Q.  Validated against h2.py and against the table of [GKF] by
`python3 wt.py check`.

    python3 wt.py check            (*) and the invariant bases against h2.py, N <= 7
    python3 wt.py dims  <Nmax> [qmax]     the dimension tables and Euler characteristics
"""

import os
import sys
import time
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2                                                          # noqa: E402
from h2 import bracket, keyof, monomials                           # noqa: E402
from exactlin import rank, nullspace                               # noqa: E402


# ------------------------------------------------------- the two-weight cochain bases

def basis2(q, ta, tb, relative=True):
    """The q-subsets S of monomials with sum a_i = ta, sum b_i = tb, in the keyof order.

    `relative` drops the three monomials of degree 2, i.e. gives a basis of
    Lambda^q(g/s)^* -- condition 1) of [GKF], Subsec. 1.4.
    """
    if q == 0:
        return [()] if (ta == 0 and tb == 0) else []
    if ta < 0 or tb < 0:
        return []
    mons = monomials(ta, tb, relative)
    n = len(mons)
    deg = [m[0] + m[1] for m in mons]
    # the k cheapest / k most expensive degrees available from index i on, for pruning
    suf = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + deg[i]
    out = []

    def rec(i, k, ra, rb, acc):
        if k == 0:
            if ra == 0 and rb == 0:
                out.append(tuple(acc))
            return
        if i >= n or n - i < k:
            return
        need = ra + rb
        if sum(deg[i:i + k]) > need:          # sorted by degree: the k cheapest
            return
        if suf[i] - suf[n - k] < need:        # the k most expensive still available
            return
        for j in range(i, n - k + 1):
            a, b = mons[j]
            if a > ra or b > rb:
                continue
            acc.append(mons[j])
            rec(j + 1, k - 1, ra - a, rb - b, acc)
            acc.pop()

    rec(0, q, ta, tb, [])
    out.sort(key=lambda S: [keyof(m) for m in S])
    return out


@lru_cache(maxsize=None)
def _items(ta, tb, relative, mindeg=1):
    """The monomials available as arguments, in the keyof order.

    mindeg = 1 gives h(2) = ham_2 = sum_{l>=1} S^l, the algebra of [GKF] and of [KM];
    mindeg = 2 gives ham^0_2 = sum_{l>=2} S^l, the Hamiltonian fields WITHOUT the
    constant ones, which is what the table of [MKN] is about.  `relative` drops the
    three monomials of degree 2 = sl(2) in either case.
    """
    return tuple(m for m in monomials(ta, tb, relative) if m[0] + m[1] >= mindeg)


def count2(q, ta, tb, relative=True, mindeg=1):
    """|basis2(q, ta, tb, relative)|, by a subset-sum DP -- no enumeration.

    State: (number of monomials chosen, a still to spend, b still to spend); the
    monomials are run through one at a time, each taken at most once.
    """
    if q == 0:
        return 1 if (ta == 0 and tb == 0) else 0
    if ta < 0 or tb < 0:
        return 0
    mons = _items(ta, tb, relative, mindeg)
    # dp[(k, ra, rb)] = number of ways; ra, rb = amounts already spent
    dp = {(0, 0, 0): 1}
    for (a, b) in mons:
        nxt = dict(dp)
        for (k, ra, rb), c in dp.items():
            if k == q:
                continue
            na, nb = ra + a, rb + b
            if na > ta or nb > tb:
                continue
            key = (k + 1, na, nb)
            nxt[key] = nxt.get(key, 0) + c
        dp = nxt
    return dp.get((q, ta, tb), 0)


def dim_relative(q, N, mindeg=1):
    """dim C~^{q,N} by the formula (*) of the module docstring."""
    t = N + q
    return count2(q, t, t, True, mindeg) - count2(q, t + 1, t - 1, True, mindeg)


# -------------------------------------------------- where the internal degree runs out

def qmax_bound(N, mindeg=1):
    """The largest q with C~^{q,N} possibly non-zero -- so that chi_N is COMPLETE.

    A basis element of C~^{q,N} is a set of q DISTINCT allowed monomials with
    sum_i (deg m_i - 2) = N_1 + N_2 = 2N.  Taking the q cheapest available monomials in
    that cost (the two of degree 1 cost -1 each, the four of degree 3 cost 1 each, the
    five of degree 4 cost 2 each, ...) minimises the sum, so the least q whose cheapest
    q monomials already cost more than 2N bounds the complex.  Returned with a margin.
    """
    costs = []
    d = mindeg
    while len(costs) <= 4 * N + 16:
        if d != 2:                              # degree 2 is sl(2), excluded
            costs.extend([d - 2] * (d + 1))     # the d+1 monomials of degree d
        d += 1
    costs.sort()
    tot, q = 0, 0
    for c in costs:
        tot += c
        if tot > 2 * N:
            break
        q += 1
    return q + 2                                # margin; the tail is asserted to vanish


# ---------------------------------------------------------------- the operator e = x^2

def e_matrix(q, N):
    """L_e : W^q_{(N,N)} --> W^q_{(N-1,N+1)} as (row, col, coeff) triples, plus both bases.

    e = x^2 raises a and lowers b on the ALGEBRA, so on the dual it does the opposite and
    the target is the weight (N-1, N+1) piece -- which has the dimension of the (N+1, N-1)
    one by the automorphism x <-> y, so the count (*) is unaffected.

    The Lie derivative is (L_e p)(v_1, .., v_q) = -sum_i p(v_1, .., [e, v_i], .., v_q), so
    the entry at (T, S) is read off by running over the members of the TARGET subset T:

        A_{T,S} = -sum_i c_i * S^*(t_1, .., w_i, .., t_q),   c_i w_i = {x^2, t_i},

    each term contributing the sign of the permutation that sorts the tuple with t_i
    replaced by w_i.  Running over the members of S instead builds the action on the
    chains Lambda^q(g/s) -- which has the same invariant dimensions, so the error is
    invisible to a dimension check and shows up only against h2.relative_space.
    """
    t = N + q
    src = basis2(q, t, t)
    tgt = basis2(q, t - 1, t + 1)
    sidx = {S: j for j, S in enumerate(src)}
    acc = {}
    for i, T in enumerate(tgt):
        for u, m in enumerate(T):
            c, w = bracket((2, 0), m)             # {x^2, x^a y^b} = 2b x^{a+1} y^{b-1}
            if c == 0 or w[0] + w[1] == 2:        # a degree-2 argument kills S^* on g/s
                continue
            cand = list(T)
            cand[u] = w
            sign = h2.perm_sign(cand)
            if sign is None:                      # w already occurs: the wedge dies
                continue
            S = tuple(sorted(cand, key=keyof))
            j = sidx.get(S)
            if j is None:
                continue
            acc[(i, j)] = acc.get((i, j), 0) - sign * c
    trips = [(i, j, c) for (i, j), c in acc.items() if c]
    return trips, src, tgt


def invariant_space(q, N):
    """A basis of C~^{q,N} = ker(e) inside the condition-1 basis, as coordinate rows.

    Same contract as h2.relative_space -- (vectors, cond1_basis) -- so cohN.py can use
    this in its place.  Asserts rank(e) = |M_2|, which 2) of the docstring guarantees.
    """
    trips, src, tgt = e_matrix(q, N)
    if not src:
        return [], src
    if not tgt:
        return [[1 if i == j else 0 for i in range(len(src))] for j in range(len(src))], src
    M = [[0] * len(src) for _ in range(len(tgt))]
    for i, j, c in trips:
        M[i][j] = c
    ker = nullspace(M)
    r = len(src) - len(ker)
    assert r == len(tgt), \
        "e is not surjective (q=%d, N=%d): rank %d, dim M_2 %d" % (q, N, r, len(tgt))
    return [list(v) for v in ker], src


# ------------------------------------------------------------------------ validation

def check():
    """(*) and the invariant bases against h2.py, on everything already computed."""
    bad = 0
    # 1) the published table of [GKF] and our own N=5,6,7 rows, by the formula (*)
    known = {
        1: [0, 1, 1, 0, 1, 1, 0, 0, 0],
        2: [0, 0, 2, 4, 2, 0, 0, 0, 0],
        3: [0, 1, 2, 4, 8, 6, 1, 0, 0],
        4: [0, 0, 5, 13, 17, 18, 14, 4, 0],
        5: [0, 1, 4, 19, 50, 59, 35, 15, 5],
        6: [0, 0, 9, 38, 80, 126, 134, 74, 16],
        7: [0, 1, 7, 48, 172, 307, 322, 232, 113],
    }
    print("(*)  dim C~^{q,N} = |W^q_{(N,N)}| - |W^q_{(N+1,N-1)}|   against the known rows")
    for N in sorted(known):
        row = []
        for q in range(1, 10):
            row.append(dim_relative(q, N))
        ok = row == known[N]
        bad += 0 if ok else 1
        print("  N=%-2d %-40s %s" % (N, "".join("%5d" % v for v in row),
                                     "ok" if ok else "MISMATCH %s" % known[N]))

    # 2) count2 (DP) against basis2 (enumeration), where the latter is affordable
    print("count2 (DP) against basis2 (enumeration)")
    for (q, ta, tb) in [(3, 8, 8), (4, 9, 9), (5, 10, 10), (4, 10, 8), (6, 11, 11),
                        (5, 12, 10), (7, 13, 13), (2, 9, 9), (6, 13, 11)]:
        c, e = count2(q, ta, tb), len(basis2(q, ta, tb))
        ok = c == e
        bad += 0 if ok else 1
        print("  q=%d (%d,%d)  DP %-7d enum %-7d %s" % (q, ta, tb, c, e, "ok" if ok else "MISMATCH"))

    # 3) the invariant SPACE, not just its dimension, against h2.relative_space
    print("ker(e) against h2.relative_space -- same dimension and same span")
    for (q, N) in [(3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
                   (3, 5), (4, 5), (5, 5), (3, 6), (4, 6), (2, 7), (3, 7), (4, 7)]:
        t0 = time.time()
        v1, b1 = invariant_space(q, N)
        t1 = time.time() - t0
        t0 = time.time()
        v2, b2 = h2.relative_space(q, N)
        t2 = time.time() - t0
        same_basis = b1 == b2
        same_dim = len(v1) == len(v2)
        span = (rank(v1) == rank(v2) == rank(list(v1) + list(v2))) if (v1 and v2) else same_dim
        ok = same_basis and same_dim and span
        bad += 0 if ok else 1
        print("  q=%d N=%d  dim %-4d (h2 %-4d)  %6.2fs vs %7.2fs  x%-6.1f %s"
              % (q, N, len(v1), len(v2), t1, t2, (t2 / t1 if t1 > 1e-9 else 0),
                 "ok" if ok else "MISMATCH"))

    bad += mkn_check()

    print()
    print("defects: %d" % bad)
    return bad


def mkn_check():
    """Our (*) against the published dimensions of [MKN] for ham^0_2 (weights 12..20).

    [MKN] Theorem 7.1 and Remark 7.2 tabulate dim C^k of the sp(2,R)-relative complex of
    ham^0_2 = sum_{l>=2} S^l -- the Hamiltonian fields WITHOUT the constant ones, a
    different (smaller) algebra than the ham_2 of [GKF], [KM] and of this folder.  Their
    weight is 2N, so weight 12..20 is N = 6..10.  An independent published computation,
    by Sp(2,R)-representation theory and Maple, of the same kind of number.
    """
    pub = {                                   # weight -> dims in the degrees 1, 2, 3, ...
        12: [0, 0, 8, 23, 22, 13, 5, 0, 0, 0],
        14: [0, 1, 6, 31, 71, 58, 15, 2, 1, 0],
        16: [0, 0, 12, 61, 126, 147, 95, 24, 0, 0],
        18: [0, 1, 10, 80, 262, 380, 268, 100, 21, 1],
        20: [0, 0, 17, 124, 423, 791, 801, 414, 96, 9, 1],
    }
    print("(*) on ham^0_2 against the published table of [MKN], weights 12 to 20")
    bad = 0
    for w in sorted(pub):
        N = w // 2
        row = [dim_relative(q, N, mindeg=3) for q in range(1, len(pub[w]) + 1)]
        ok = row == pub[w]
        bad += 0 if ok else 1
        print("  w=%-3d N=%-3d %s %s" % (w, N, "".join("%6d" % v for v in row),
                                        "ok" if ok else "MISMATCH\n      published %s"
                                        % "".join("%6d" % v for v in pub[w])))
    return bad


def dims(Nmax, qmax=None):
    """dim C~^{q,N} and the Euler characteristic, by (*) alone.

    The range of q is taken from qmax_bound, so the alternating sum really is the Euler
    characteristic of the whole internal degree; the vanishing of the tail is asserted.
    """
    print("dim C~^{q,N} by (*), and the Euler characteristic of each internal degree")
    print("  chi != 0 PROVES a class: chi = sum (-1)^q dim H^{q,N}.")
    width = qmax or qmax_bound(Nmax)
    hdr = "  N |" + "".join("%7d" % q for q in range(1, width + 1)) + " |    chi"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for N in range(1, Nmax + 1):
        t0 = time.time()
        qb = qmax_bound(N)
        row = [dim_relative(q, N) for q in range(1, qb + 1)]
        assert row[-1] == 0 and row[-2] == 0, \
            "no vanishing tail at N=%d: the q-bound %d is too small" % (N, qb)
        eul = sum((-1) ** q * v for q, v in enumerate(row, start=1))
        row += [0] * max(0, width - len(row))
        print("  %2d |" % N + "".join("%7d" % v for v in row[:width])
              + " |%7d  %s  %.1fs"
              % (eul, "CLASS" if eul else "     ", time.time() - t0), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "check"
    if what == "check":
        sys.exit(1 if check() else 0)
    elif what == "dims":
        dims(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
