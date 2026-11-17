#!/usr/bin/env python3
"""ham.py -- Sect. 8 of the text (Problem 2) of Sect. 10): the Hamiltonian
algebras with a (2,4)-structure; Lemmas 8.1.1, 8.2.1, Theorem 8.3.1,
Statement 8.3.3.

The algebra.  p = 2, M = 2n+1, O = O(M; 1) (all heights 1), indeterminates
x, u_1..u_n, v_1..v_n, and the standard non-alternating shape
    omega = (dx)^(2) + sum du_i dv_i,
whose Hamiltonian fields are
    H_f = (d_x f) d_x + sum_i ((d_{u_i} f) d_{v_i} + (d_{v_i} f) d_{u_i}).
g := h_I(M; 1)' = Span{H_f | f non-constant, f != the top monomial}, of
dimension 2^M - 2 (the derived algebra; simple for M >= 4 by [KoKCh]).  Its
(2,4)-structure is that of the parity of the degree in x ([LLo2]).  Everything is realized by honest matrices: the fields act on
O, a 2^M-dimensional space, so squares are squares of operators.

The grading.  deg x = t, deg u_i = t + d_i, deg v_i = t - d_i, the field H_f
having degree deg f - 2t.  With t = 100 and d = (1, 3, ...) the degree-0 part
is exactly t = Span{H_{u_i v_i}}, and n_- = g_{-1} + n_-(g_0), the standard
grading refined by the torus, of dimension (n+1)^2.

What is computed:
  1. closure, dimensions, the x-parity grading, which basis vectors square
     outside g;
  2. that t is a maximal torus: its centralizer is t + N with ad_N nilpotent;
  3. the triangular decomposition and the count of n_-;
  4. the restricted closure L_f = g + {y^2 | y in g_-} as OPERATORS, which is
     restricted, and its induced modules M_L(lambda), of dimension
     2^{dim n_-(L_f)}; irreducibility by the n_+-singular vectors;
  5. for n = 1: the induced module over u_{2,4}(g) itself, of dimension
     2^a 4^b with b counted by the x-parity, and its reducibility.

Output: ham.log
"""
import itertools, random, sys, time
from collections import defaultdict
from wk import (GF, MatAlg, Induced, Space, nullspace, mat_mul, flat,
                triangular)

LOG = open("ham.log", "a")


def out(s=""):
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


# --------------------------------------------------------------- the algebra

def setup_ham(F, n, t=100, d=None):
    M = 2 * n + 1
    N = 1 << M
    if d is None:
        d = [2 * i + 1 for i in range(n)]
    pair = {0: 0}
    for i in range(n):
        pair[1 + i] = 1 + n + i
        pair[1 + n + i] = 1 + i
    vdeg = [t] + [t + d[i] for i in range(n)] + [t - d[i] for i in range(n)]

    def field(f):
        """the matrix of H_f, f a monomial bitmask"""
        A = [[0] * N for _ in range(N)]
        for a in range(M):
            if not (f >> a) & 1:
                continue
            c = f & ~(1 << a)          # d_a f
            b = pair[a]                # the field is c * d_b
            for g in range(N):
                if (g >> b) & 1:
                    h = g & ~(1 << b)  # d_b g
                    if c & h == 0:
                        A[c | h][g] ^= 1
        return A

    top = N - 1
    mons = [f for f in range(1, N) if f != top]
    basis = [field(f) for f in mons]
    names = []
    for f in mons:
        s = ""
        for a in range(M):
            if (f >> a) & 1:
                s += "x" if a == 0 else ("u%d" % a if a <= n else
                                         "v%d" % (a - n))
        names.append(s)
    deg = [sum(vdeg[a] for a in range(M) if (f >> a) & 1) - 2 * t
           for f in mons]
    xodd = [f & 1 for f in mons]
    alg = MatAlg(F, basis, names)
    return alg, deg, xodd, mons


def sq_in(alg, A):
    return alg.in_alg(mat_mul(alg.F, A, A))


def is_zero(A):
    return not any(any(r) for r in A)


# --------------------------------------------------------------- module

class InducedSq(Induced):
    """M(lambda) = u_2(L) (x)_{u(t + n_+)} K_lambda for a restricted L whose
    n_- is closed under the 2-map: a letter y_r with y_r^2 = y_r^{[2]} != 0
    is used at most once, and y_r y_r = y_r^{[2]}, expressed in n_-.  The
    letters must be ordered by degree, so that y_r^{[2]} involves only
    letters of lower index (its degree is twice that of y_r)."""

    def __init__(self, alg, neg, tor, pos, lam):
        F = alg.F
        self.g, self.F = alg, F
        self.neg, self.tor, self.pos, self.lam = neg, tor, pos, lam
        self.orders = [2] * len(neg)
        self.k = len(neg)
        self.mons = [tuple(a) for a in itertools.product(*[range(2)] *
                                                         self.k)]
        self.index = {a: i for i, a in enumerate(self.mons)}
        self.dim = len(self.mons)
        self._lmul, self._act = {}, {}
        self.sq = {}
        for r, i in enumerate(neg):
            Q = mat_mul(F, alg.basis[i], alg.basis[i])
            if not is_zero(Q):
                v = alg.coords(Q)
                for j in range(alg.n):
                    if v[j]:
                        assert j in neg, "a square left n_-"
                        assert neg.index(j) < r, "letter order"
                self.sq[r] = v

    def lmul(self, r, mon):
        key = (r, mon)
        if key in self._lmul:
            return self._lmul[key]
        first = next((s for s in range(self.k) if mon[s] > 0), None)
        if first is None or first > r:
            a = list(mon); a[r] = 1
            res = {self.index[tuple(a)]: 1}
        elif first == r:
            rest = list(mon); rest[r] = 0
            rest = tuple(rest)
            res = (self._lmul_vec(self.sq[r], {self.index[rest]: 1})
                   if r in self.sq else {})
        else:
            s = first
            rest = list(mon); rest[s] -= 1
            rest = tuple(rest)
            part = self.lmul(r, rest)
            res = {}
            for i, c in part.items():
                for j, c2 in self.lmul(s, self.mons[i]).items():
                    self._addto(res, j, self.F.mul(c, c2))
            br = self.g.bracket(self._e(self.neg[r]), self._e(self.neg[s]))
            for j, c in self._neg_coords(br):
                for i, c2 in self.lmul(j, rest).items():
                    self._addto(res, i, self.F.mul(c, c2))
        self._lmul[key] = res
        return res


# --------------------------------------------------------------- checks

def torus_maximal(alg, tor):
    """The centralizer C of t, and a search over its F_2-points for an
    element z outside t with ad_z semisimple.  Over a perfect field, A is
    semisimple if and only if A lies in K[A^2] (Frobenius is bijective on a
    reduced algebra and on nothing else)."""
    F = alg.F
    C = [i for i in range(alg.n)
         if all(alg.bracket(alg_e(alg, h), alg_e(alg, i)) == [0] * alg.n
                for h in tor)]
    found = []
    for coef in itertools.product([0, 1], repeat=len(C)):
        if not any(c for c, i in zip(coef, C) if i not in tor):
            continue
        v = [0] * alg.n
        for c, i in zip(coef, C):
            v[i] = c
        A = alg.ad(v)
        A2 = mat_mul(F, A, A)
        S = Space(F, alg.n * alg.n)
        P = [[1 if r == c else 0 for c in range(alg.n)] for r in range(alg.n)]
        for _ in range(alg.n + 1):
            S.add(flat(P))
            P = mat_mul(F, P, A2)
        if flat(A) in S:
            found.append([alg.names[i] for c, i in zip(coef, C) if c])
    return [alg.names[i] for i in C], found


def alg_e(alg, i):
    return [1 if j == i else 0 for j in range(alg.n)]


def closure(alg, deg, xodd):
    """L_f = g + span{y^2 | y an x-odd basis vector}, as matrices."""
    F = alg.F
    basis = list(alg.basis)
    names = list(alg.names)
    ldeg = list(deg)
    S = Space(F, alg.m * alg.m)
    for b in basis:
        S.add(flat(b))
    new = []
    for i in range(alg.n):
        if xodd[i]:
            Q = mat_mul(F, alg.basis[i], alg.basis[i])
            if not is_zero(Q) and S.add(flat(Q)):
                basis.append(Q)
                names.append("(" + alg.names[i] + ")^2")
                ldeg.append(2 * deg[i])
                new.append(i)
    L = MatAlg(F, basis, names)
    return L, ldeg, new


def nsing_blocks(F, M_, deg_of_mon, pos, stop=None):
    blocks = defaultdict(list)
    for i, a in enumerate(M_.mons):
        blocks[deg_of_mon(a)].append(i)
    total = 0
    for dd, idx in blocks.items():
        rows = defaultdict(dict)
        for c, i in enumerate(idx):
            for p in pos:
                for tt, xx in M_.act(M_._e(p), M_.mons[i]).items():
                    rows[(p, tt)][c] = xx
        R = []
        for key, dct in rows.items():
            r = [0] * len(idx)
            for c, xx in dct.items():
                r[c] = xx
            R.append(r)
        total += len(nullspace(F, R, len(idx))) if R else len(idx)
        if stop and total > stop:
            return total
    return total


def pred(F, lam):
    """the criterion of o^{(1)}_I(2n+1): lambda_i, lambda_i + lambda_j not
    in F_2"""
    vals = list(lam) + [F.add(lam[i], lam[j]) for i in range(len(lam))
                        for j in range(i + 1, len(lam))]
    return all(v not in (0, 1) for v in vals)


# --------------------------------------------------------------- main

def run(n, k, nlam, full=False):
    F = GF(k)
    t0 = time.time()
    alg, deg, xodd, mons = setup_ham(F, n)
    alg.build_struct()
    out("== n = %d (M = %d) over %s: dim g = %d" % (n, 2 * n + 1, F.name,
                                                   alg.n))
    # x-parity grading is a grading
    bad = 0
    for i in range(alg.n):
        for j in range(alg.n):
            v = alg.struct[i][j]
            for kk in range(alg.n):
                if v[kk] and xodd[kk] != (xodd[i] ^ xodd[j]):
                    bad += 1
    out("  x-parity is a Z/2-grading: %s;  dim g_+ = %d, dim g_- = %d"
        % (bad == 0, xodd.count(0), xodd.count(1)))
    sq = [sq_in(alg, alg.basis[i]) for i in range(alg.n)]
    out("  g_+ basis squares inside g: %s" %
        all(sq[i] for i in range(alg.n) if not xodd[i]))
    outside = [alg.names[i] for i in range(alg.n) if not sq[i]]
    out("  basis vectors squaring OUTSIDE g: %s" % outside)
    four = all(is_zero(mat_mul(F, mat_mul(F, alg.basis[i], alg.basis[i]),
                               mat_mul(F, alg.basis[i], alg.basis[i])))
               for i in range(alg.n) if xodd[i])
    out("  y^4 = 0 for every x-odd basis vector: %s" % four)
    neg, tor, pos = triangular(alg, deg)
    out("  degree 0 = %s" % [alg.names[i] for i in tor])
    out("  dim n_- = %d, dim n_+ = %d  ((n+1)^2 = %d)" %
        (len(neg), len(pos), (n + 1) ** 2))
    xo = [i for i in neg if xodd[i]]
    out("  x-odd in n_-: %s;  of them squaring outside g: %s" %
        ([alg.names[i] for i in xo],
         [alg.names[i] for i in xo if not sq[i]]))
    Cn, found = torus_maximal(alg, tor)
    out("  centralizer of t: %s; F_2-points outside t with ad semisimple: "
        "%s" % (Cn, found or "none"))
    a = sum(1 for i in neg if sq[i])
    b = sum(1 for i in neg if not sq[i])
    a24 = sum(1 for i in neg if not xodd[i])
    b24 = sum(1 for i in neg if xodd[i])
    out("  by the x-parity: a = %d, b = %d, 2^a 4^b = 2^%d" %
        (a24, b24, a24 + 2 * b24))
    out("  by the squares:  a = %d, b = %d, bound 2^%d  "
        "(n^2+3n+1 = %d)" % (a, b, a + 2 * b, n * n + 3 * n + 1))

    # the restricted closure as operators
    L, ldeg, new = closure(alg, deg, xodd)
    L.build_struct()
    Lneg, Ltor, Lpos = triangular(L, ldeg)
    out("  L_f: dim %d (= dim g + %d), degree 0 = %d elements, "
        "dim n_-(L_f) = %d" % (L.n, len(new), len(Ltor), len(Lneg)))
    closed = all(L.in_alg(mat_mul(F, B, B)) for B in L.basis)
    out("  L_f closed under the square of operators: %s" % closed)
    # torus degrees for monomials
    random.seed(11)
    lams = list(itertools.product(F.els, repeat=n))
    if nlam and len(lams) > nlam:
        lams = random.sample(lams, nlam)
    res = []
    for lam in lams:
        MM = InducedSq(L, Lneg, Ltor, Lpos, list(lam))
        if not res and MM.dim <= 64:
            ok, badp = MM.check_representation()
            out("  M_L(lambda) is a representation of L_f: %s" % ok)
        if not res:
            out("  dim M_L(lambda) = %d" % MM.dim)
        s = nsing_blocks(F, MM, lambda m: sum(m[r] * ldeg[Lneg[r]]
                                               for r in range(len(Lneg))),
                         Lpos)
        res.append((lam, s))
        if nlam:
            out("    lambda = %s: dim of n_+-singular vectors %d%s"
                % (lam, s, "  IRREDUCIBLE" if s == 1 else ""))
    nirr = sum(1 for lam, s in res if s == 1)
    npr = sum(1 for lam, s in res if pred(F, lam))
    mism = [lam for lam, s in res if (s == 1) != pred(F, lam)]
    out("  %d weights: irreducible %d, predicted by the o_I criterion %d, "
        "mismatches %s" % (len(res), nirr, npr, mism or "none"))
    if full:
        # the u_{2,4} induced module, x-parity orders
        orders = [4 if xodd[i] else 2 for i in neg]
        lam = lams[0]
        MM = Induced(alg, neg, orders, tor, pos, list(lam))
        ok, bad = MM.check_representation()
        s = nsing_blocks(F, MM, lambda m: sum(m[r] * deg[neg[r]]
                                               for r in range(len(neg))), pos)
        out("  u_{2,4} module over g, lambda = %s: dim %d, representation "
            "%s, singular vectors %d" % (lam, MM.dim, ok, s))
    out("  (%.0f s)" % (time.time() - t0))
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]); k = int(sys.argv[2])
    nlam = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(n, k, nlam, full=(n == 1))


def naive(n, k, lam):
    """the u_{2,4}(g)-module with b counted by the x-parity, at one weight"""
    F = GF(k)
    alg, deg, xodd, mons = setup_ham(F, n)
    alg.build_struct()
    neg, tor, pos = triangular(alg, deg)
    orders = [4 if xodd[i] else 2 for i in neg]
    MM = Induced(alg, neg, orders, tor, pos, list(lam))
    s = nsing_blocks(F, MM, lambda m: sum(m[r] * deg[neg[r]]
                                           for r in range(len(neg))), pos)
    out("== n = %d over %s: u_{2,4} module over g (x-parity orders), "
        "lambda = %s: dim %d, singular vectors %d" % (n, F.name, lam,
                                                      MM.dim, s))
