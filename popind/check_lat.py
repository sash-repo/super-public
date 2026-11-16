"""The K-invariant full-rank lattices left to the reader at the end of Sec. 4.8 of

  V. L. Popov, Discrete complex reflection groups, arXiv:2304.08941 [= P],

for the three groups of the list  G(4,2,n), G(6,2,n), G(6,3,n), K12, K31  (s = n+1)
that [P] does not carry out: K = G(4,2,2) ("the most complicated case"), K12, K31.

All arithmetic is exact, in Q(zeta_8) (which contains i and sqrt 2 -- every datum of
Table 2 of [P] for these groups lies in it).  Conventions of [P]:
   <u|v> is linear in u;  the e_j are unit vectors;  R_j = R_{e_j,-1};
   (id - R_k) v = 2<v|e_k> e_k;   b_{jk} := 2<e_j|e_k>;   c_{jk} = b_{jk} b_{kj}.
"""
import sys
from fractions import Fraction as Fr
from itertools import product

from lat_core import (F, ZERO, ONE, I, SQ2, Z8, fadd, fsub, fmul, fneg, fscal,
                      fconj, finv, fdiv, fis0, herm, fnum, fstr, Lat, lat_sum,
                      lat_from_vecs, apply_mat, scale_lat, line_part,
                      coords_to_vec, vec_to_coords, hnf_rows, det_rat,
                      coeffs_in, solve_rat)

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


CHECKS = [0, 0]


def check(name, cond):
    CHECKS[0] += 1
    if cond:
        CHECKS[1] += 1
        say(f"  [ok] {name}")
    else:
        say(f"  [FAIL] {name}")


# ------------------------------------------------------------------ utilities
def half(a):
    return fscal(Fr(1, 2), a)


def vscal(a, v):
    return tuple(fmul(a, x) for x in v)


def vadd(u, v):
    return tuple(fadd(a, b) for a, b in zip(u, v))


def vsub(u, v):
    return tuple(fsub(a, b) for a, b in zip(u, v))


def refl(e, n):
    """R_{e,-1}: v -> v - 2<v|e>e ; matrix rows."""
    M = []
    for k in range(n):
        row = []
        for l in range(n):
            d = ONE if k == l else ZERO
            row.append(fsub(d, fscal(2, fmul(e[k], fconj(e[l])))))
        M.append(tuple(row))
    return M


def matmul(A, B, n):
    return [tuple(_sum(fmul(A[r][k], B[k][c]) for k in range(n)) for c in range(n))
            for r in range(n)]


def _sum(gen):
    s = ZERO
    for x in gen:
        s = fadd(s, x)
    return s


def mat_id(n):
    return [tuple(ONE if k == l else ZERO for l in range(n)) for k in range(n)]


def mat_sub(A, B, n):
    return [tuple(fsub(A[r][c], B[r][c]) for c in range(n)) for r in range(n)]


def mat_inv(A, n):
    """inverse over Q(zeta_8) by Gauss-Jordan."""
    M = [list(A[r]) + [ONE if r == c else ZERO for c in range(n)] for r in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if not fis0(M[r][c])), None)
        assert p is not None, "singular"
        M[c], M[p] = M[p], M[c]
        inv = finv(M[c][c])
        M[c] = [fmul(inv, x) for x in M[c]]
        for r in range(n):
            if r != c and not fis0(M[r][c]):
                f = M[r][c]
                M[r] = [fsub(x, fmul(f, y)) for x, y in zip(M[r], M[c])]
    return [tuple(M[r][n:]) for r in range(n)]


def group_order(gens, n, limit=200000):
    """numeric closure, for a sanity check on |K|."""
    import numpy as np
    def tonum(M):
        return np.array([[fnum(x) for x in row] for row in M], dtype=complex)

    def key(M):
        return tuple(np.round(M.flatten(), 7).tolist())

    G = {}
    idm = np.eye(n, dtype=complex)
    G[key(idm)] = idm
    frontier = [idm]
    gs = [tonum(g) for g in gens]
    while frontier:
        new = []
        for M in frontier:
            for g in gs:
                P = g @ M
                k = key(P)
                if k not in G:
                    G[k] = P
                    new.append(P)
                    if len(G) > limit:
                        return None
        frontier = new
    return len(G)


def lat_invariant(L, R, n):
    return apply_mat(R, L, n) == L


def root_lattice(L, es, n):
    """Gamma_0 = Gamma_1 + ... + Gamma_s  (Theorem 4.1.3 of [P])."""
    parts = [line_part(L, e, n) for e in es]
    return lat_sum(*parts)


def coset_reps(sub, big, n):
    """representatives of big/sub (sub <= big, same rank), as vectors in Q^{4n}."""
    B = big.basis()
    A = [[int(x) for x in coeffs_in(big, b)] for b in sub.basis()]
    H = hnf_rows(A)
    k = len(B)
    piv = list(range(k))
    ranges = [range(H[c][c]) for c in range(k)]
    reps = []
    for x in product(*ranges):
        v = [Fr(0)] * len(B[0])
        for c, b in zip(x, B):
            if c:
                v = [vv + c * bb for vv, bb in zip(v, b)]
        reps.append(tuple(v))
    return reps


def intermediate_lattices(sub, big, n):
    """all lattices Gamma with sub <= Gamma <= big."""
    reps = coset_reps(sub, big, n)
    subs = {frozenset([tuple([Fr(0)] * len(reps[0]))]): sub}
    # work with the finite group big/sub represented by its elements: use lattices
    found = {sub}
    frontier = [sub]
    while frontier:
        new = []
        for L in frontier:
            for r in reps:
                M = lat_sum(L, Lat([list(r)]))
                if big.contains(M) and M not in found:
                    found.add(M)
                    new.append(M)
        frontier = new
    return sorted(found, key=lambda L: L.index_in(big))


def show_lat(L, n, name=""):
    say(f"    {name}: den {L.den}, rank {L.rank}")


def sim_scalar(L1, L2, alpha, n):
    return scale_lat(alpha, L1, n) == L2


# ================================================================= the groups
def analyse(tag, es, mu_all_minus1=True, expected_order=None, Kprime_size=None,
            lattice_gens=None, n=None, extra=None):
    say("")
    say("=" * 78)
    say(f"{tag}")
    say("=" * 78)
    s = len(es)
    n = n or (s - 1)
    for j, e in enumerate(es):
        check(f"e_{j+1} is a unit vector", herm(e, e) == ONE)
    Rs = [refl(e, n) for e in es]
    # b_{jk}, c_{jk}
    say("  Gram data:")
    b = {}
    for j in range(s):
        for k in range(s):
            if j != k:
                b[(j, k)] = fscal(2, herm(es[j], es[k]))
    for j in range(s):
        for k in range(j + 1, s):
            c = fmul(b[(j, k)], b[(k, j)])
            say(f"    b_{j+1}{k+1} = {fstr(b[(j,k)])},  b_{k+1}{j+1} = {fstr(b[(k,j)])},"
                f"  c_{j+1}{k+1} = {fstr(c)}")
    if expected_order:
        o = group_order(Rs, n)
        check(f"|K| = {expected_order}", o == expected_order)
    if Kprime_size:
        o = group_order(Rs[:n], n)
        check(f"|K'| = {Kprime_size}", o == Kprime_size)
    # S for K' and its determinant
    idm = mat_id(n)
    S = [tuple(ZERO for _ in range(n)) for _ in range(n)]
    for j in range(n):
        D = mat_sub(idm, Rs[j], n)
        S = [tuple(fadd(S[r][c], D[r][c]) for c in range(n)) for r in range(n)]
    Sinv = mat_inv(S, n)
    say(f"  S = sum_(1<=j<=n) (id - R_j) built for K' = <R_1,...,R_{n}>")

    L = lat_from_vecs(lattice_gens)
    check("Lambda has Z-rank 2n", L.rank == 2 * n)
    for j in range(s):
        check(f"Lambda is R_{j+1}-invariant", lat_invariant(L, Rs[j], n))
    L0 = root_lattice(L, es, n)
    check("Lambda is a root lattice for K (Lambda_0 = Lambda)", L0 == L)
    L0p = root_lattice(L, es[:n], n)
    check("Lambda is a root lattice for K' as well", L0p == L)

    Lstar = apply_mat(Sinv, L, n)
    check("Lambda <= Lambda*(K')", Lstar.contains(L))
    idx = L.index_in(Lstar)
    say(f"  [Lambda*(K') : Lambda] = {idx}")

    inter = intermediate_lattices(L, Lstar, n)
    say(f"  lattices between Lambda and Lambda*(K'): {len(inter)}")
    results = []
    for G in inter:
        invK = all(lat_invariant(G, R, n) for R in Rs)
        invKp = all(lat_invariant(G, R, n) for R in Rs[:n])
        G0 = root_lattice(G, es, n)
        results.append((G, invKp, invK, G0 == G, G.index_in(Lstar), L.index_in(G)))
    say("  index over Lambda | K'-inv | K-inv | root for K")
    for (G, ikp, ik, isroot, i1, i2) in results:
        say(f"    {i2:>3}            {str(ikp):>5}    {str(ik):>5}   {str(isroot):>5}")
    if extra:
        extra(dict(es=es, Rs=Rs, n=n, s=s, L=L, Lstar=Lstar, S=S, Sinv=Sinv,
                   inter=inter, results=results, b=b))
    return dict(es=es, Rs=Rs, n=n, s=s, L=L, Lstar=Lstar, inter=inter,
                results=results, b=b)


# ------------------------------------------------- the root-lattice algorithm
# Theorem 4.7.1 of [P]: Gamma = sum_j Lambda_j (Lambda_j = Delta_j e_j a rank-2
# lattice in the line C e_j) is a K-invariant root lattice with Gamma_j = Lambda_j
# iff  a) Z[Tr K]_j Delta_j <= Delta_j  and  b) b_{jk} Delta_j <= Delta_k for all
# j,k -- plus, when s = n+1, discreteness of the sum.

def lat_C(gens):
    """lattice in C = Q(zeta_8) given by field elements."""
    return Lat([list(g) for g in gens])


def cyc_ring(b, j, s, maxlen=5):
    """the subring Z[Tr K]_j of [P] Thm 4.7.1: generated by the cyclic products
    c_{j...} = b_{j j2} b_{j2 j3} ... b_{jr j}."""
    from itertools import product as iproduct
    gens = [ONE]
    for L in range(1, maxlen):
        for mid in iproduct(range(s), repeat=L):
            path = (j,) + mid + (j,)
            if any(u == v for u, v in zip(path, path[1:])):
                continue
            p = ONE
            ok = True
            for u, v in zip(path, path[1:]):
                if fis0(b[(u, v)]):
                    ok = False
                    break
                p = fmul(p, b[(u, v)])
            if ok:
                gens.append(p)
    R = lat_C(gens)
    # close under multiplication
    for _ in range(6):
        cur = [tuple(x) for x in R.basis()]
        new = list(cur)
        for a in cur:
            for c in cur:
                new.append(fmul(tuple(a), tuple(c)))
        R2 = lat_C(new)
        if R2 == R:
            break
        R = R2
    return R


def stable_under(ring, D):
    """is the lattice D in C stable under multiplication by the ring?"""
    for r in ring.basis():
        for d in D.basis():
            if not D.contains_vec(list(fmul(tuple(r), tuple(d)))):
                return False
    return True


def scaleC(a, D):
    return lat_C([fmul(a, tuple(x)) for x in D.basis()])
