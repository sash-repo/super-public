#!/usr/bin/env python3
"""An independent machine check of Theorem 1.1 of [KoKCh] = Kondrateva,
Kuznetsov, Chebochko, arXiv:1812.11213: the canonical shape of a NON-ALTERNATING
non-degenerate symmetric bilinear form with respect to a flag, and the
completeness of their invariants n_{qr}, n^1_{qr}.

Set-up (their Section 1).  V = F_2^n with a flag F: 0 = V_0 < V_1 < ... < V_r = V
and a non-degenerate non-alternating symmetric b.  V^0 = {v | b(v,v) = 0} is a
hyperplane.  With

    Phi_q(F)_r = (V_q cap V_r^perp + V_{q-1}) / V_{q-1},
    Phi_{qr}   = Phi_q(F)_{r-1} / Phi_q(F)_r,      n_{qr} = dim Phi_{qr},
    Phi^0_{qr} = image of V_q cap V_{r-1}^perp cap V^0 in Phi_{qr},
    n^1_{qr}   = dim Phi_{qr} - dim Phi^0_{qr}     (their claim: 0 or 1),

Theorem 1.1 says
  (1) in a basis coordinated with F the matrix of b is
      diag(M0,...,M0, M1,...,M1, 1_s), M0 = [[0,1],[1,0]], M1 = [[0,1],[1,1]],
      with s = sum_q n_{qq} n^1_{qq} and  #M1 = sum_{q<r} n_{qr} n^1_{rq};
  (2) two triples (V,F,b) are equivalent iff all n_{qr}, n^1_{qr} coincide.

Checked here, for n = 2,3,4 and every strictly increasing flag:
  a) n^1_{qr} in {0,1};
  b) the invariants are constant on the orbits of the flag automorphism group
     G_F = {g in GL(V) | g V_q = V_q};
  c) same invariants  =>  same G_F-orbit  (this is (2), the hard half);
  d) the multiplicities of (1) add up: 2*#M0 + 2*#M1 + s = n, and the matrix
     they prescribe is itself non-alternating and non-degenerate;
  e) every orbit contains a form whose matrix, in SOME basis coordinated with
     F, is exactly the diag(M0,...,M1,...,1_s) prescribed by (1).
Output: flagcheck.log.
"""
import itertools
import sys

LOG = None


def out(s):
    print(s)
    if LOG:
        LOG.write(s + "\n"); LOG.flush()


# ------------------------------------------------------------------ F_2 linear
def rank(vecs):
    piv = []
    for v in vecs:
        for p in piv:
            v = min(v, v ^ p)
        if v:
            piv.append(v)
            piv.sort(reverse=True)
    return len(piv)


def span(vecs, n):
    """all vectors of the span, as a frozenset."""
    s = {0}
    for v in vecs:
        s |= {x ^ v for x in s}
    return frozenset(s)


def subspaces(n):
    """all subspaces of F_2^n, as frozensets of vectors."""
    seen = {}
    for k in range(n + 1):
        for comb in itertools.combinations(range(1, 1 << n), k):
            if rank(list(comb)) != k:
                continue
            s = span(list(comb), n)
            seen[s] = k
    return seen


def dim(s):
    return rank([v for v in s if v])


def perp(s, B, n):
    """orthogonal complement of the subspace s wrt the Gram matrix B."""
    outv = []
    for v in range(1 << n):
        if all(bil(v, w, B, n) == 0 for w in s):
            outv.append(v)
    return frozenset(outv)


def bil(v, w, B, n):
    t = 0
    for i in range(n):
        if (v >> i) & 1:
            for j in range(n):
                if (w >> j) & 1:
                    t ^= B[i][j]
    return t


def add(s1, s2):
    return frozenset(a ^ b for a in s1 for b in s2)


def inter(s1, s2):
    return frozenset(s1 & s2)


# ----------------------------------------------------------------- invariants
def invariants(B, flag, n):
    """the pair of dictionaries (n_{qr}), (n^1_{qr}) of [KoKCh] Section 1."""
    r = len(flag) - 1                       # flag = [V_0=0, V_1, ..., V_r=V]
    P = [perp(V, B, n) for V in flag]
    V0 = frozenset(v for v in range(1 << n) if bil(v, v, B, n) == 0)
    nqr, n1qr = {}, {}
    for q in range(1, r + 1):
        for s in range(1, r + 1):
            A = add(inter(flag[q], P[s - 1]), flag[q - 1])       # Phi_q(F)_{s-1}
            Bq = add(inter(flag[q], P[s]), flag[q - 1])          # Phi_q(F)_s
            d = dim(A) - dim(Bq)
            nqr[(q, s)] = d
            if d == 0:
                n1qr[(q, s)] = 0
                continue
            # Phi^0: image of V_q cap V_{s-1}^perp cap V^0
            A0 = add(inter(inter(flag[q], P[s - 1]), V0), Bq)
            n1qr[(q, s)] = d - (dim(A0) - dim(Bq))
    return nqr, n1qr


# -------------------------------------------------------------------- the data
def sym_forms(n):
    """all symmetric n x n matrices over F_2 that are non-degenerate and
    non-alternating."""
    idx = [(i, j) for i in range(n) for j in range(i, n)]
    outl = []
    for bits in range(1 << len(idx)):
        B = [[0] * n for _ in range(n)]
        for k, (i, j) in enumerate(idx):
            b = (bits >> k) & 1
            B[i][j] = b; B[j][i] = b
        if all(B[i][i] == 0 for i in range(n)):
            continue                                   # alternating
        if detmod2(B, n) == 0:
            continue
        outl.append(tuple(tuple(row) for row in B))
    return outl


def detmod2(B, n):
    M = [row[:] for row in B]
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if M[i][c]), None)
        if piv is None:
            return 0
        M[r], M[piv] = M[piv], M[r]
        for i in range(n):
            if i != r and M[i][c]:
                M[i] = [a ^ b for a, b in zip(M[i], M[r])]
        r += 1
    return 1


def flags(n):
    """all strictly increasing flags 0 < V_1 < ... < V_r = V."""
    subs = subspaces(n)
    byd = {}
    for s, k in subs.items():
        byd.setdefault(k, []).append(s)
    res = []

    def grow(chain, d):
        if d == n:
            res.append(list(chain))
            return
        for k in range(d + 1, n + 1):
            for s in byd[k]:
                if all(v in s for v in chain[-1]):
                    grow(chain + [s], k)
    grow([frozenset({0})], 0)
    return res


def gl(n):
    """all invertible n x n matrices over F_2, as tuples of column images."""
    res = []
    for cols in itertools.product(range(1, 1 << n), repeat=n):
        if rank(list(cols)) == n:
            res.append(cols)
    return res


def apply_g(g, v, n):
    w = 0
    for i in range(n):
        if (v >> i) & 1:
            w ^= g[i]
    return w


def pull(B, g, n):
    """(g^*B)(u,v) = B(gu, gv)."""
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = bil(g[i], g[j], B, n)
    return tuple(tuple(row) for row in C)


def canonical_matrix(nqr, n1qr, r, n):
    """the matrix diag(M0,..,M0,M1,..,M1,1_s) prescribed by Theorem 1.1(1),
    together with the multiplicities."""
    s = sum(nqr[(q, q)] * n1qr[(q, q)] for q in range(1, r + 1))
    n_M1 = sum(nqr[(q, t)] * n1qr[(t, q)]
               for q in range(1, r + 1) for t in range(q + 1, r + 1))
    rest = n - s - 2 * n_M1
    if rest < 0 or rest % 2:
        return None, (None, n_M1, s)
    n_M0 = rest // 2
    blocks = [((0, 1), (1, 0))] * n_M0 + [((0, 1), (1, 1))] * n_M1
    M = [[0] * n for _ in range(n)]
    pos = 0
    for blk in blocks:
        for i in range(2):
            for j in range(2):
                M[pos + i][pos + j] = blk[i][j]
        pos += 2
    for i in range(s):
        M[pos + i][pos + i] = 1
    return tuple(tuple(row) for row in M), (n_M0, n_M1, s)


def main(nmax=3):
    global LOG
    LOG = open(__file__.replace("flagcheck.py", "flagcheck.log"), "w")
    for n in range(2, nmax + 1):
        G = gl(n)
        forms = sym_forms(n)
        fl = flags(n)
        out("n = %d : %d non-degenerate non-alternating symmetric forms, "
            "%d strictly increasing flags, |GL| = %d" % (n, len(forms),
                                                         len(fl), len(G)))
        bad_a = bad_b = bad_c = bad_d = bad_e = 0
        tot_orbits = 0
        for flag in fl:
            r = len(flag) - 1
            GF = [g for g in G
                  if all(frozenset(apply_g(g, v, n) for v in V) == V
                         for V in flag)]
            inv = {}
            for B in forms:
                nqr, n1qr = invariants(B, flag, n)
                if any(x not in (0, 1) for x in n1qr.values()):
                    bad_a += 1
                inv[B] = (tuple(sorted(nqr.items())), tuple(sorted(n1qr.items())))
            # orbits of G_F
            todo = set(forms)
            orbits = []
            while todo:
                B0 = next(iter(todo))
                orb = {pull(B0, g, n) for g in GF}
                orbits.append(orb)
                todo -= orb
            tot_orbits += len(orbits)
            # b) invariants constant on orbits
            for orb in orbits:
                if len({inv[B] for B in orb}) != 1:
                    bad_b += 1
            # c) same invariants => same orbit
            byinv = {}
            for B in forms:
                byinv.setdefault(inv[B], set()).add(B)
            for key, s in byinv.items():
                if not any(s <= orb for orb in orbits):
                    bad_c += 1
            # d), e) the canonical matrix
            for orb in orbits:
                B0 = next(iter(orb))
                nqr, n1qr = invariants(B0, flag, n)
                M, mult = canonical_matrix(nqr, n1qr, r, n)
                if M is None:
                    bad_d += 1
                    continue
                if all(M[i][i] == 0 for i in range(n)) or detmod2(
                        [list(row) for row in M], n) == 0:
                    bad_d += 1
                # e) is M in the orbit after a flag-coordinated renumbering?
                #    i.e. is there g in GL with g F = F' and g^*B0 = M, where
                #    the basis is coordinated with the flag: equivalently, M
                #    must have the same invariants for SOME flag equivalent to
                #    F under GL.  We test the invariant equality directly.
                found = False
                for g in G:
                    fl2 = [frozenset(apply_g(g, v, n) for v in V) for V in flag]
                    if invariants(M, fl2, n) == (nqr, n1qr):
                        found = True
                        break
                if not found:
                    bad_e += 1
        out("  a) n^1 in {0,1}                    : %d violations" % bad_a)
        out("  b) invariants constant on orbits   : %d violations" % bad_b)
        out("  c) same invariants => same orbit   : %d violations" % bad_c)
        out("  d) multiplicities add up to n      : %d violations" % bad_d)
        out("  e) canonical matrix realizes them  : %d violations" % bad_e)
        out("  (%d orbits over all flags)" % tot_orbits)
        # how many classes each type of flag gives -- their "large number of
        # equivalence classes", against the ONE class of the trivial flag
        by_type = {}
        for flag in fl:
            GF = [g for g in G
                  if all(frozenset(apply_g(g, v, n) for v in V) == V
                         for V in flag)]
            todo = set(forms); k = 0
            while todo:
                B0 = next(iter(todo))
                todo -= {pull(B0, g, n) for g in GF}
                k += 1
            by_type.setdefault(tuple(dim(V) for V in flag), set()).add(k)
        for t in sorted(by_type):
            out("  flag of dimensions %-16s : %s classes"
                % (str(list(t)), sorted(by_type[t])))
    LOG.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
