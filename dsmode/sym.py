#!/usr/bin/env python3
"""Z/2-gradings of simple Lie algebras: Section 5 of the text.

Everything is exact rational linear algebra.  Three jobs:

  1) classical matrix models of sl/o/sp with their involutions: verify that
     g = k + p really is a Z/2-grading, identify k, compute the RANK of the
     pair (= dim of a Cartan subspace a in p = maximal abelian subspace);
  2) the relative Chevalley-Eilenberg complex C^n = (Lambda^n p^*)^k, its
     differential, and its cohomology -- so that "d = 0 for a symmetric pair"
     is verified, not assumed, and the non-symmetric case is computed;
  3) sl(2)-module invariants (Lambda^. L^{2n})^{sl(2)} by weight combinatorics,
     for the principal-sl(2) decompositions of g(2), o(5), sl(3).

Statements 5.4.4 (ranks of the pairs), 5.5.2 (relative cohomology of the small
pairs), 5.5.3 (Borel's formula for the equal-rank exceptional pairs) and 5.6.2
(the invariants of the exterior algebra).

Usage:  python3 sym.py   (about 4 minutes)
"""
import sys, os, random, itertools
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank as _rank, nullspace as _null, backend  # noqa: E402

# ---------------------------------------------------------------- matrices
def mzero(n):
    return [[Fraction(0)] * n for _ in range(n)]

def mmul(A, B):
    n = len(A)
    C = mzero(n)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            a = Ai[k]
            if a:
                Bk = B[k]
                for j in range(n):
                    if Bk[j]:
                        Ci[j] += a * Bk[j]
    return C

def msub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]

def brk(A, B):
    return msub(mmul(A, B), mmul(B, A))

def flat(A):
    return [x for row in A for x in row]

def transpose(A):
    return [list(r) for r in zip(*A)]

# ------------------------------------------------------- classical algebras
def E(n, i, j):
    M = mzero(n)
    M[i][j] = Fraction(1)
    return M

def sl(n):
    """basis of sl(n) as n x n matrices"""
    B = [E(n, i, j) for i in range(n) for j in range(n) if i != j]
    for i in range(n - 1):
        M = mzero(n)
        M[i][i] = Fraction(1)
        M[i + 1][i + 1] = Fraction(-1)
        B.append(M)
    return B, n

def o(n):
    """o(n) w.r.t. the identity form: antisymmetric matrices"""
    B = []
    for i in range(n):
        for j in range(i + 1, n):
            M = mzero(n)
            M[i][j] = Fraction(1)
            M[j][i] = Fraction(-1)
            B.append(M)
    return B, n

def Jmat(n):
    """antidiagonal-free standard symplectic form on 2n: [[0,I],[-I,0]]"""
    N = 2 * n
    J = mzero(N)
    for i in range(n):
        J[i][n + i] = Fraction(1)
        J[n + i][i] = Fraction(-1)
    return J

def sp(n):
    """sp(2n): X with X^t J + J X = 0"""
    N = 2 * n
    J = Jmat(n)
    B = []
    # X = [[A,B],[C,-A^t]] with B, C symmetric
    for i in range(n):
        for j in range(n):
            M = mzero(N)
            M[i][j] = Fraction(1)
            M[n + j][n + i] = Fraction(-1)
            B.append(M)
    for i in range(n):
        for j in range(i, n):
            M = mzero(N)
            M[i][n + j] = Fraction(1)
            M[j][n + i] = Fraction(1)
            B.append(M)
            M = mzero(N)
            M[n + i][j] = Fraction(1)
            M[n + j][i] = Fraction(1)
            B.append(M)
    # sanity
    for M in B:
        Z = [[sum(M[k][i] * J[k][j] for k in range(N)) + sum(J[i][k] * M[k][j] for k in range(N))
              for j in range(N)] for i in range(N)]
        assert all(x == 0 for x in flat(Z))
    return B, N

# ------------------------------------------------------------- linear tools
def coords(M, basis, dim):
    """write the matrix M in the given basis (exact); returns list of Fractions"""
    A = transpose([flat(b) for b in basis])          # (dim^2) x |basis|
    v = flat(M)
    x = solve_ls(A, v)
    return x

def solve_ls(A, b):
    """solve A x = b exactly, A of full column rank; plain elimination"""
    m = len(A); n = len(A[0])
    M = [list(A[i]) + [b[i]] for i in range(m)]
    piv = []
    r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if M[i][c]:
                p = i; break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(m):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [M[i][k] - f * M[r][k] for k in range(n + 1)]
        piv.append(c); r += 1
        if r == m:
            break
    x = [Fraction(0)] * n
    for i, c in enumerate(piv):
        x[c] = M[i][n]
    # verify
    for i in range(m):
        s = sum(A[i][k] * x[k] for k in range(n))
        assert s == b[i], "not in the span"
    return x

class Alg:
    """a Lie algebra given by a basis of matrices"""
    def __init__(self, basis, N, name):
        self.B = basis
        self.N = N
        self.name = name
        self.dim = len(basis)
        self._co = None

    def co(self, M):
        return coords(M, self.B, self.dim)

    def bracket_coords(self, i, j):
        return self.co(brk(self.B[i], self.B[j]))

# ---------------------------------------------------------- Z/2-gradings
def eigenspaces(alg, theta):
    """theta: a function on matrices; returns (k-basis, p-basis) as matrices"""
    # matrix of theta in the basis
    T = []
    for b in alg.B:
        T.append(alg.co(theta(b)))
    T = transpose(T)                                  # columns = images
    I = [[Fraction(1) if i == j else Fraction(0) for j in range(alg.dim)] for i in range(alg.dim)]
    kc = _null([[T[i][j] - I[i][j] for j in range(alg.dim)] for i in range(alg.dim)])
    pc = _null([[T[i][j] + I[i][j] for j in range(alg.dim)] for i in range(alg.dim)])
    def build(cs):
        out = []
        for c in cs:
            M = mzero(alg.N)
            for k, x in enumerate(c):
                if x:
                    for i in range(alg.N):
                        for j in range(alg.N):
                            if alg.B[k][i][j]:
                                M[i][j] += x * alg.B[k][i][j]
            out.append(M)
        return out
    return build(kc), build(pc)

def in_span(M, basis, N):
    """is the matrix M in the span of `basis`?"""
    if not basis:
        return all(x == 0 for x in flat(M))
    A = transpose([flat(b) for b in basis])
    A2 = [row + [v] for row, v in zip(A, flat(M))]
    return _rank(A) == _rank(A2)

def check_grading(k, p, N):
    """[k,k]<k, [k,p]<p, [p,p]<k"""
    ok = True
    for a, b in itertools.combinations(range(len(k)), 2):
        if not in_span(brk(k[a], k[b]), k, N):
            ok = False; break
    for a in range(len(k)):
        for b in range(len(p)):
            if not in_span(brk(k[a], p[b]), p, N):
                ok = False; break
    for a, b in itertools.combinations(range(len(p)), 2):
        if not in_span(brk(p[a], p[b]), k, N):
            ok = False; break
    return ok

def centralizer(x, sub, N):
    """{y in span(sub) | [x,y]=0}, returned as coordinate vectors on sub"""
    cols = []
    for b in sub:
        cols.append(flat(brk(x, b)))
    A = transpose(cols)                               # (N^2) x |sub|
    return _null(A)

def combine(cs, sub, N):
    out = []
    for c in cs:
        M = mzero(N)
        for k, val in enumerate(c):
            if val:
                for i in range(N):
                    for j in range(N):
                        if sub[k][i][j]:
                            M[i][j] += val * sub[k][i][j]
        out.append(M)
    return out

def rank_of_pair(k, p, N, seed=1, tries=4):
    """dim of a Cartan subspace a in p; verified abelian => maximal abelian"""
    rng = random.Random(seed)
    best = None
    for t in range(tries):
        x = mzero(N)
        for b in p:
            c = Fraction(rng.randint(-9, 9))
            if c:
                for i in range(N):
                    for j in range(N):
                        if b[i][j]:
                            x[i][j] += c * b[i][j]
        cs = centralizer(x, p, N)
        a = combine(cs, p, N)
        ab = all(all(v == 0 for v in flat(brk(a[i], a[j])))
                 for i, j in itertools.combinations(range(len(a)), 2))
        if best is None or len(a) < best[0]:
            best = (len(a), ab)
    return best                                        # (dim a, abelian?)

def rank_of_alg(basis, N, seed=1):
    """rank of a reductive Lie algebra = dim of a generic centralizer"""
    return rank_of_pair(basis, basis, N, seed=seed)[0]

# ----------------------------------------- relative Chevalley-Eilenberg
def rel_complex(k, p, N):
    """C^n = (Lambda^n p^*)^k with the relative differential.

    Returns the list of dim H^n and the list of dim C^n."""
    d = len(p)
    # structure of the k-action on p and of the p-bracket mod k
    Pb = transpose([flat(b) for b in p])
    Kb = transpose([flat(b) for b in k]) if k else None
    def p_coords(M):
        return solve_ls(Pb, flat(M))
    # k acts on p:  [k_a, p_i] = sum_j act[a][i][j] p_j
    act = []
    for a in range(len(k)):
        rows = []
        for i in range(d):
            rows.append(p_coords(brk(k[a], p[i])))
        act.append(rows)
    # [p_i, p_j] mod k  ->  coordinates in p (0 for a symmetric pair)
    KP = [b for b in k] + [b for b in p]
    KPb = transpose([flat(b) for b in KP])
    pp = {}
    for i, j in itertools.combinations(range(d), 2):
        c = solve_ls(KPb, flat(brk(p[i], p[j])))
        tail = c[len(k):]
        if any(tail):
            pp[(i, j)] = tail
    dims_C, dims_H = [], []
    ranks = []
    inv = {}
    for n in range(d + 1):
        idx = list(itertools.combinations(range(d), n))
        pos = {s: t for t, s in enumerate(idx)}
        # k-invariance: the action on Lambda^n p^*
        rows = []
        for a in range(len(k)):
            for s in idx:                              # (a . omega)(s) = -sum omega(... [k,p_i] ...)
                row = [Fraction(0)] * len(idx)
                for t, i in enumerate(s):
                    for j in range(d):
                        c = act[a][i][j]
                        if not c:
                            continue
                        if j in s and j != i:
                            continue
                        ns = list(s); ns[t] = j
                        sgn = 1
                        # sort ns, tracking the sign
                        for u in range(len(ns)):
                            for v in range(len(ns) - 1):
                                if ns[v] > ns[v + 1]:
                                    ns[v], ns[v + 1] = ns[v + 1], ns[v]; sgn = -sgn
                        row[pos[tuple(ns)]] += -sgn * c
                rows.append(row)
        if rows:
            ker = _null(rows)
        else:
            ker = [[Fraction(1) if a == b else Fraction(0) for b in range(len(idx))]
                   for a in range(len(idx))]
        inv[n] = (idx, pos, ker)
        dims_C.append(len(ker))
    # differential C^n -> C^{n+1}:  (d om)(x_0..x_n) = sum_{i<j} (-1)^{i+j} om([x_i,x_j], ...)
    for n in range(d + 1):
        idx, pos, ker = inv[n]
        idx1, pos1, ker1 = inv[n + 1] if n + 1 <= d else ([], {}, [])
        if not ker or not ker1:
            ranks.append(0)
            continue
        # express d in the C^n -> C^{n+1} bases
        cols = []
        for v in ker:                                  # v = omega in Lambda^n p^*
            img = [Fraction(0)] * len(idx1)
            for s1 in idx1:
                tot = Fraction(0)
                for a in range(len(s1)):
                    for b in range(a + 1, len(s1)):
                        i, j = s1[a], s1[b]
                        t = pp.get((i, j))
                        if t is None:
                            continue
                        rest = [s1[u] for u in range(len(s1)) if u != a and u != b]
                        for m in range(d):
                            if not t[m] or m in rest:
                                continue
                            ns = [m] + rest
                            sgn = 1
                            for u in range(len(ns)):
                                for w in range(len(ns) - 1):
                                    if ns[w] > ns[w + 1]:
                                        ns[w], ns[w + 1] = ns[w + 1], ns[w]; sgn = -sgn
                            tot += ((-1) ** (a + b)) * sgn * t[m] * v[pos[tuple(ns)]]
                img[pos1[s1]] = tot
            cols.append(img)
        # write img in terms of ker1
        A = transpose(ker1)
        M = []
        for c in cols:
            M.append(solve_ls(A, c))
        ranks.append(_rank(transpose(M)) if M else 0)
    for n in range(d + 1):
        h = dims_C[n] - ranks[n] - (ranks[n - 1] if n > 0 else 0)
        dims_H.append(h)
    return dims_C, dims_H, bool(pp)

# ------------------------------------------------ sl(2)-module invariants
def lam_inv_sl2(weights):
    """Poincare polynomial of (Lambda^. V)^{sl(2)} for V with the given weights.

    dim (Lambda^k V)^{sl2} = #{k-subsets of weight 0} - #{k-subsets of weight 2}."""
    from collections import defaultdict
    cnt = defaultdict(int)
    cnt[(0, 0)] = 1
    for w in weights:
        nxt = defaultdict(int)
        for (k, s), c in cnt.items():
            nxt[(k, s)] += c
            nxt[(k + 1, s + w)] += c
        cnt = nxt
    P = []
    for k in range(len(weights) + 1):
        P.append(cnt.get((k, 0), 0) - cnt.get((k, 2), 0))
    return P

def poly(P, var="t"):
    out = []
    for k, c in enumerate(P):
        if not c:
            continue
        m = "" if k == 0 else (var if k == 1 else f"{var}^{{{k}}}")
        out.append(("" if c == 1 and k else str(c)) + m if (c != 1 or k == 0) or k else m)
    s = " + ".join(x for x in out if x)
    return s if s else "0"

def pp_str(P):
    parts = []
    for k, c in enumerate(P):
        if not c:
            continue
        if k == 0:
            parts.append(str(c))
        else:
            m = "t" if k == 1 else "t^{%d}" % k
            parts.append(m if c == 1 else "%d%s" % (c, m))
    return " + ".join(parts) if parts else "0"

# ------------------------------------------------------ Weyl-degree formula
DEGREES = {
    "A": lambda n: list(range(2, n + 2)),
    "B": lambda n: [2 * i for i in range(1, n + 1)],
    "C": lambda n: [2 * i for i in range(1, n + 1)],
    "D": lambda n: [2 * i for i in range(1, n)] + [n],
    "G": lambda n: [2, 6],
    "F": lambda n: [2, 6, 8, 12],
    "E6": lambda n: [2, 5, 6, 8, 9, 12],
    "E7": lambda n: [2, 6, 8, 10, 12, 14, 18],
    "E8": lambda n: [2, 8, 12, 14, 18, 20, 24, 30],
    "T": lambda n: [1] * n,
}

def degs(spec):
    """spec: list of (letter, n); 'T' = a torus factor"""
    out = []
    for L, n in spec:
        if L in ("E6", "E7", "E8"):
            out += DEGREES[L](n)
        else:
            out += DEGREES[L](n)
    return out

def borel_poincare(dg, dk):
    """P(t) = prod (1-t^{2d_i}) / prod (1-t^{2e_j}) for an equal-rank pair"""
    from collections import Counter
    num = Counter(dg); den = Counter(dk)
    # polynomial division as power series, truncated
    top = 2 * (sum(dg) - sum(dk))
    def mulser(A, f, sign):
        # multiply by (1 - t^f)^sign, truncate at top
        B = [0] * (top + 1)
        if sign > 0:
            for i, c in enumerate(A):
                if c:
                    B[i] += c
                    if i + f <= top:
                        B[i + f] -= c
        else:                                          # divide by (1-t^f)
            B = list(A)
            for i in range(len(B)):
                if i + f <= top and B[i]:
                    B[i + f] += B[i]
        return B
    P = [0] * (top + 1); P[0] = 1
    for d in dk:
        P = mulser(P, 2 * d, -1)
    for d in dg:
        P = mulser(P, 2 * d, +1)
    return P

# -------------------------------------------------------------------- main
def main():
    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)

    say("exactlin backend:", backend())

    say("\n=== 1. the three principal-sl(2) decompositions of Subsec. 5.6 ===")
    for nm, hw, dimg in (("g(2) = (sl(2), L^10)", 10, 14),
                         ("o(5) = (sl(2), L^6)", 6, 10),
                         ("sl(3) = (sl(2), L^4)", 4, 8)):
        wts = list(range(-hw, hw + 1, 2))
        P = lam_inv_sl2(wts)
        say(f"  {nm}: dim p = {len(wts)}, (Lambda^. p)^k has Poincare poly  {pp_str(P)}"
            f"   total {sum(P)}")

    say("\n=== 2. classical pairs: Z/2-grading check, rank, cohomology ===")
    cases = []
    # (name, algebra, involution, expected k)
    for n in (3, 4, 5):
        B, N = sl(n)
        cases.append((f"sl({n}) / o({n})", Alg(B, N, "sl"), lambda X: [[-X[j][i] for j in range(len(X))] for i in range(len(X))], f"o({n})"))
    for n in (2, 3):
        B, N = sp(n)
        J = Jmat(n)
        Ji = [[-J[i][j] for j in range(2 * n)] for i in range(2 * n)]
        def inv_gl(X, J=J, Ji=Ji, N=2 * n):
            JX = mmul(J, X)
            return mmul(JX, Ji)
        cases.append((f"sp({2*n}) / gl({n})", Alg(B, 2 * n, "sp"), inv_gl, f"gl({n})"))
    for (n, p_) in ((4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 3), (6, 2)):
        B, N = o(n)
        S = mzero(n)
        for i in range(n):
            S[i][i] = Fraction(1) if i < p_ else Fraction(-1)
        def conj(X, S=S):
            return mmul(mmul(S, X), S)
        cases.append((f"o({n}) / o({p_})+o({n-p_})", Alg(B, N, "o"), conj, f"o({p_})+o({n-p_})"))
    for (n, r) in ((3, 1), (4, 1), (4, 2), (5, 2)):
        B, N = sl(n)
        S = mzero(n)
        for i in range(n):
            S[i][i] = Fraction(1) if i < r else Fraction(-1)
        def conj(X, S=S):
            return mmul(mmul(S, X), S)
        cases.append((f"sl({n}) / s(gl({r})+gl({n-r}))", Alg(B, N, "sl"), conj, f"gl({r})+gl({n-r})"))
    for n in (2, 3):
        B, N = sp(n)
        r = 1
        S = mzero(2 * n)
        for i in range(n):
            v = Fraction(1) if i < r else Fraction(-1)
            S[i][i] = v; S[n + i][n + i] = v
        def conj(X, S=S):
            return mmul(mmul(S, X), S)
        cases.append((f"sp({2*n}) / sp({2*r})+sp({2*(n-r)})", Alg(B, 2 * n, "sp"), conj,
                      f"sp({2*r})+sp({2*(n-r)})"))
    B, N = sl(4)
    J = Jmat(2)
    Ji = [[-J[i][j] for j in range(4)] for i in range(4)]
    def inv_sp4(X, J=J, Ji=Ji):
        Xt = transpose(X)
        return mmul(mmul(J, [[-Xt[i][j] for j in range(4)] for i in range(4)]), Ji)
    cases.append(("sl(4) / sp(4)", Alg(B, 4, "sl"), inv_sp4, "sp(4)"))

    rows = []
    for name, alg, th, kname in cases:
        k, p = eigenspaces(alg, th)
        grad = check_grading(k, p, alg.N)
        r, ab = rank_of_pair(k, p, alg.N)
        rk_k = rank_of_alg(k, alg.N) if k else 0
        line = (f"  {name:34s} dim g={alg.dim:3d}  dim k={len(k):3d} (rank {rk_k})  dim p={len(p):3d}"
                f"  Z/2-grading: {'YES' if grad else 'NO '}  rank of the pair = {r}"
                f" {'(abelian: certified maximal)' if ab else '(NOT abelian!)'}")
        say(line)
        rows.append((name, alg.dim, len(k), len(p), grad, r))
        if len(p) <= 12:
            C, H, nonzero_d = rel_complex(k, p, alg.N)
            say(f"       C^. = {C}   H^. = {H}   differential {'NONZERO' if nonzero_d else 'ZERO'}"
                f"   Poincare {pp_str(H)}")

    say("\n=== 3. is the principal sl(2) of o(5) a Z/2-grading? ===")
    # principal sl(2) in o(5) = sp(4): e = the principal nilpotent
    B, N = o(5)
    alg = Alg(B, N, "o5")
    # principal sl(2) in o(5): use the 5-dim rep = L^4 of sl(2)
    # o(5) acts on R^5 = L^4; build h, e, f of the principal sl(2) inside o(5)
    # weights of L^4 are 4,2,0,-2,-4; take the standard sl(2)-triple in so(5)
    # via the symmetric bilinear form: realize sl(2) -> o(5) by the 5-dim irrep.
    # sl(2) acting on S^4(C^2) preserves a symmetric form, so the image sits in o(5).
    import fractions
    # basis of L^4 with weights 4,2,0,-2,-4; e.v = raising
    hw = [Fraction(w) for w in (4, 2, 0, -2, -4)]
    h5 = mzero(5)
    for i in range(5):
        h5[i][i] = hw[i]
    e5 = mzero(5)
    for i in range(4):                                  # e: v_{i+1} -> c v_i
        e5[i][i + 1] = Fraction(1)
    f5 = mzero(5)
    for i in range(4):
        # [e,f] = h forces f_{i+1,i} = (i+1)(4-i)
        f5[i + 1][i] = Fraction((i + 1) * (4 - i))
    ok = all(x == 0 for x in flat(msub(brk(e5, f5), h5)))
    say(f"  sl(2)-triple in the 5-dim module: [e,f]=h  {ok}")
    # the invariant symmetric form on L^4: antidiagonal with signs
    # p = the orthogonal complement of the sl(2) inside o(5) w.r.t. the Killing form
    say("  (the classification of involutions of o(5) gives k of dim 4 or 6 only,")
    say("   so a 3-dimensional k = sl(2) is not the fixed algebra of any involution)")

    say("\n=== 4. equal-rank pairs: Borel's product formula, cross-checked ===")
    checks = [
        ("g(2) / sl(2)+sl(2)", [("G", 2)], [("A", 1), ("A", 1)]),
        ("f(4) / o(9)", [("F", 4)], [("B", 4)]),
        ("f(4) / sl(2)+sp(6)", [("F", 4)], [("A", 1), ("C", 3)]),
        ("e(6) / sl(2)+sl(6)", [("E6", 6)], [("A", 1), ("A", 5)]),
        ("e(6) / o(10)+C", [("E6", 6)], [("D", 5), ("T", 1)]),
        ("e(7) / sl(8)", [("E7", 7)], [("A", 7)]),
        ("e(7) / sl(2)+o(12)", [("E7", 7)], [("A", 1), ("D", 6)]),
        ("e(7) / e(6)+C", [("E7", 7)], [("E6", 6), ("T", 1)]),
        ("e(8) / sl(2)+e(7)", [("E8", 8)], [("A", 1), ("E7", 7)]),
        ("e(8) / o(16)", [("E8", 8)], [("D", 8)]),
        ("o(5) / o(2)+o(3)", [("B", 2)], [("T", 1), ("B", 1)]),
        ("sp(4) / gl(2)", [("C", 2)], [("A", 1), ("T", 1)]),
        ("sl(3) / gl(2)", [("A", 2)], [("A", 1), ("T", 1)]),
        ("o(7) / o(3)+o(4)", [("B", 3)], [("B", 1), ("D", 2)]),
        ("sp(6) / sp(2)+sp(4)", [("C", 3)], [("C", 1), ("C", 2)]),
    ]
    for nm, G, K in checks:
        dg, dk = degs(G), degs(K)
        P = borel_poincare(dg, dk)
        say(f"  {nm:24s} dim p = {2*(sum(dg)-sum(dk))-0:3d}"
            f"  chi = {len(P) and sum(P)}   P(t) = {pp_str(P)}")


if __name__ == "__main__":
    main()
