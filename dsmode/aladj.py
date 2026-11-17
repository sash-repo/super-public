#!/usr/bin/env python3
"""Which homological elements x of gl(m|n) suffice: Subsec. 1.7.7 of the text.

The four elements (22) of the text live in gl(1|1|1|1) = gl(2|2) in the
ALTERNATING format, the format in which every simple root is grey (isotropic):

    x0 = E12 + E34
    x1 = E12 + E14 + E34
    x2 = E12 + E43
    x3 = E12 + E23 + E14 - E43

What this script checks, over Q and over F_p for p = 2, 3, 5, 7:

  A. the four elements: parity, x^2 = 0, the pair (a,b) = (rank A, rank B) where
     A = x|_{V_ev} : V_ev --> V_od and B = x|_{V_od} : V_od --> V_ev, and the
     superdimension of g_x for g = gl(2|2), sl(2|2), psl(2|2); and the same
     four elements inside gl(1|1|1|1|1|1), where g_x = gl(1|1) (Example 1.7.9);
  B. the conjugating automorphisms x1 = Q x0 Q^{-1}, x3 = Q' x2 Q'^{-1}, found by
     the normal-form algorithm and verified as matrix identities (Example 1.7.9);
  C. the pair (a,b) is a G_ev-invariant; the supertranspose interchanges a and b;
  D. Proposition 1.7.8, verified constructively on EVERY homological x of
     gl(m|n) over F_q: x^2 = 0 iff AB = BA = 0; the G_ev-orbits are exactly the
     pairs (a,b) with a + b <= min(m,n); DS_x(V) = Ker x/Im x has sdim
     (m-a-b|n-a-b) and gl(V)_x = gl(DS_x V), so g_x depends on a+b ALONE;
  E. the elements x_r of Hypothesis 1.7.1 realize all k+1 classes of g_x
     (Proposition 1.7.8, item 4));
  F. iso-sets: supp(x0), supp(x2) are iso-sets for the diagonal torus,
     supp(x1), supp(x3) are not (Remark 1.7.10).

Usage:  python3 aladj.py
"""
import sys
import os
from fractions import Fraction
from itertools import product, combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank as ex_rank, nullspace as ex_null, solve as ex_solve, backend



def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)


CHECKS = {"pass": 0, "fail": 0}


def check(name, ok, detail=""):
    CHECKS["pass" if ok else "fail"] += 1
    say("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                         ("   " + detail) if detail else ""))
    return ok


# ------------------------------------------------------------------ field --
def red(x, p):
    return x % p if p else Fraction(x)


def mred(M, p):
    return [[red(c, p) for c in row] for row in M]


def zeros(n, m=None):
    m = n if m is None else m
    return [[0] * m for _ in range(n)]


def mmul(A, B, p):
    n, k, m = len(A), len(B), len(B[0])
    C = zeros(n, m)
    for i in range(n):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            if not a:
                continue
            Bt = B[t]
            for j in range(m):
                if Bt[j]:
                    C[i][j] += a * Bt[j]
    return mred(C, p)


def madd(A, B, p, s=1):
    return mred([[A[i][j] + s * B[i][j] for j in range(len(A[0]))]
                 for i in range(len(A))], p)


def mzero(A):
    return all(not c for row in A for c in row)


def meq(A, B):
    return all(A[i][j] == B[i][j] for i in range(len(A)) for j in range(len(A[0])))


def unit(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def Eij(N, i, j):
    """the matrix unit, 0-based indices"""
    M = zeros(N)
    M[i][j] = 1
    return M


def minv(A, p):
    """inverse of a square matrix over Q or F_p, by solving A X = I column-wise"""
    n = len(A)
    cols = []
    for j in range(n):
        b = [1 if i == j else 0 for i in range(n)]
        x = ex_solve(A, b, p=p)
        assert x is not None, "singular matrix"
        cols.append([red(v, p) for v in x])
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ------------------------------------------------------- superspace / gl ---
class GL:
    """gl(V) for V with the given parity string, over Q (p=None) or F_p."""

    def __init__(self, par, p=None):
        self.par = list(par)
        self.N = len(par)
        self.p = p
        self.EV = [i for i in range(self.N) if par[i] == 0]
        self.OD = [i for i in range(self.N) if par[i] == 1]
        self.m, self.n = len(self.EV), len(self.OD)
        # basis of gl(V), sorted: even matrix units first
        self.basis = ([(i, j) for i in range(self.N) for j in range(self.N)
                       if (par[i] + par[j]) % 2 == 0]
                      + [(i, j) for i in range(self.N) for j in range(self.N)
                         if (par[i] + par[j]) % 2 == 1])
        self.dim_ev = self.m ** 2 + self.n ** 2
        self.dim_od = 2 * self.m * self.n
        self.index = {ij: k for k, ij in enumerate(self.basis)}

    def parity(self, X):
        """0, 1 or None (inhomogeneous / zero -> 0)"""
        ev = any(X[i][j] for (i, j) in self.basis[:self.dim_ev])
        od = any(X[i][j] for (i, j) in self.basis[self.dim_ev:])
        if ev and od:
            return None
        return 1 if od else 0

    def sq(self, X):
        return mmul(X, X, self.p)

    def br(self, X, Y, pX, pY):
        s = -1 if (pX and pY) else 1
        return madd(mmul(X, Y, self.p), mmul(Y, X, self.p), self.p, -s)

    def coords(self, X):
        return [X[i][j] for (i, j) in self.basis]

    def blocks(self, X):
        """A : V_ev --> V_od  and  B : V_od --> V_ev, as matrices"""
        A = [[X[i][j] for j in self.EV] for i in self.OD]
        B = [[X[i][j] for j in self.OD] for i in self.EV]
        return A, B

    def ab(self, X):
        A, B = self.blocks(X)
        return (ex_rank(A, p=self.p) if A and A[0] else 0,
                ex_rank(B, p=self.p) if B and B[0] else 0)

    # ---------------------------------------------------------- ad and DS --
    def ad_ranks(self, X, sub=None):
        """ranks of ad_X : g_ev --> g_od and g_od --> g_ev, for g = gl or a
        subalgebra given by a basis `sub` = (list of even, list of odd matrices)"""
        if sub is None:
            ev = [Eij(self.N, i, j) for (i, j) in self.basis[:self.dim_ev]]
            od = [Eij(self.N, i, j) for (i, j) in self.basis[self.dim_ev:]]
        else:
            ev, od = sub
        rows_ev = [self.coords(self.br(X, Y, 1, 0)) for Y in ev]
        rows_od = [self.coords(self.br(X, Y, 1, 1)) for Y in od]
        r_ev = ex_rank(rows_ev, p=self.p) if rows_ev else 0
        r_od = ex_rank(rows_od, p=self.p) if rows_od else 0
        return r_ev, r_od

    def sdim_gx(self, X, sub=None):
        """sdim of g_x = Ker ad_x / Im ad_x"""
        if sub is None:
            d_ev, d_od = self.dim_ev, self.dim_od
        else:
            d_ev, d_od = len(sub[0]), len(sub[1])
        r_ev, r_od = self.ad_ranks(X, sub)
        return (d_ev - r_ev - r_od, d_od - r_od - r_ev)

    def sdim_DS_V(self, X):
        """sdim of Ker x / Im x inside the tautological module"""
        a, b = self.ab(X)
        return (self.m - a - b, self.n - a - b)


def sl_basis(g):
    """a basis of sl(V) = Ker str, split into even and odd parts"""
    ev, od = [], []
    seen_diag = None
    for (i, j) in g.basis:
        if i != j:
            (ev if (g.par[i] + g.par[j]) % 2 == 0 else od).append(Eij(g.N, i, j))
    # diagonal part of sl: str = sum (-1)^{p(i)} E_ii = 0
    diag = []
    for k in range(1, g.N):
        # E_00 * s_k - E_kk * s_0 with s_i = (-1)^{p(i)} has zero supertrace
        s0 = 1 if g.par[0] == 0 else -1
        sk = 1 if g.par[k] == 0 else -1
        M = zeros(g.N)
        M[0][0] = sk
        M[k][k] = -s0
        diag.append(mred(M, g.p))
    ev = diag + ev
    return ev, od


def psl_quotient(g, ev, od):
    """for sl(n|n): drop the identity from the even part, giving a basis of a
    complement; returns (ev', od) plus the projection of coordinates"""
    one = unit(g.N)
    rows = [g.coords(M) for M in ev]
    # keep a maximal independent subset of ev that stays independent modulo K*1
    keep, cur = [], [g.coords(one)]
    for k, r in enumerate(rows):
        if ex_rank(cur + [r], p=g.p) > ex_rank(cur, p=g.p):
            cur.append(r)
            keep.append(k)
    return [ev[k] for k in keep], od


def sdim_quotient_gx(g, X, ev, od, mod_one=False):
    """sdim g_x for the subalgebra with basis (ev, od); if mod_one, work in the
    quotient by the centre K*1 (which is even)"""
    if not mod_one:
        return g.sdim_gx(X, (ev, od))
    one = [g.coords(unit(g.N))]
    rows_ev = [g.coords(g.br(X, Y, 1, 0)) for Y in ev]     # land in the odd part
    rows_od = [g.coords(g.br(X, Y, 1, 1)) for Y in od]     # land in the even part
    r_ev = ex_rank(rows_ev, p=g.p) if rows_ev else 0
    # modulo the centre, the even image is Im + K*1 minus the line K*1
    r_od = (ex_rank(one + rows_od, p=g.p) - 1) if rows_od else 0
    # kernels: ad_x acts on the quotient, so a class is killed iff [x,Y] in K*1;
    # for the odd part K*1 does not meet it, for the even part the source is a
    # quotient of dimension len(ev) - 1... but ev already excludes 1 by
    # construction, so len(ev) counts the quotient
    return (len(ev) - r_ev - r_od, len(od) - r_od - r_ev)


# --------------------------------------------------- the normal-form proof --
def normal_form(g, X):
    """Constructive proof of the classification.

    Input: X odd with X^2 = 0 in gl(V).  Output: (P, canon, (a,b)) with P even
    invertible and  P X P^{-1} = canon,  where canon is the normal shape of
    the orbit (a,b): b matrix units V_od --> V_ev on the first b even/odd index
    pairs, then a matrix units V_ev --> V_od on the next a pairs.
    """
    p = g.p
    A, B = g.blocks(X)                    # A: V_ev -> V_od,  B: V_od -> V_ev
    m, n = g.m, g.n
    a = ex_rank(A, p=p) if A and A[0] else 0
    b = ex_rank(B, p=p) if B and B[0] else 0

    def kernel(M, dim_src):
        """basis of Ker M, M acting on column vectors of length dim_src"""
        if not M or not M[0] or mzero(M):
            return [[1 if i == j else 0 for i in range(dim_src)]
                    for j in range(dim_src)]
        return [[red(v, p) for v in w] for w in ex_null(M, p=p)]

    def grow(inside, pool):
        """greedily extend the list `inside` by vectors of `pool` that raise the
        rank; returns only the added vectors"""
        cur = [list(v) for v in inside]
        r = ex_rank(cur, p=p) if cur else 0
        added = []
        for v in pool:
            if ex_rank(cur + [list(v)], p=p) > r:
                cur.append(list(v))
                added.append(list(v))
                r += 1
        return added

    def std(d):
        return [[1 if i == k else 0 for i in range(d)] for k in range(d)]

    def apply(M, v):
        if not M or not M[0]:
            return []
        return [red(sum(M[i][j] * v[j] for j in range(len(v))), p)
                for i in range(len(M))]

    kerA = kernel(A, m)                     # dim m - a, contains Im B
    kerB = kernel(B, n)                     # dim n - b, contains Im A

    # U1 = a complement of Ker B in V_od (dim b);  U0 = one of Ker A in V_ev
    U1 = grow(kerB, std(n))
    U0 = grow(kerA, std(m))
    assert len(U1) == b and len(U0) == a, (len(U1), b, len(U0), a)

    imB_vecs = [apply(B, w) for w in U1]    # b independent vectors inside Ker A
    imA_vecs = [apply(A, u) for u in U0]    # a independent vectors inside Ker B

    rest_ev = grow(imB_vecs, kerA)          # completes Im B to a basis of Ker A
    rest_od = grow(imA_vecs, kerB)          # completes Im A to a basis of Ker B
    assert len(rest_ev) == m - a - b and len(rest_od) == n - a - b

    #   even:  B(w_1..w_b) | U0 (dim a) | rest of Ker A
    #   odd :  w_1..w_b    | A(U0)      | rest of Ker B
    ev_new = imB_vecs + U0 + rest_ev
    od_new = U1 + imA_vecs + rest_od
    assert len(ev_new) == m and len(od_new) == n

    # P^{-1} takes the standard basis to the new one, column-wise
    Pinv = zeros(g.N)
    for k, v in enumerate(ev_new):
        for i in range(m):
            Pinv[g.EV[i]][g.EV[k]] = v[i]
    for k, v in enumerate(od_new):
        for i in range(n):
            Pinv[g.OD[i]][g.OD[k]] = v[i]
    Pinv = mred(Pinv, p)
    P = minv(Pinv, p)

    canon = zeros(g.N)
    for k in range(b):                     # B-type: od -> ev
        canon[g.EV[k]][g.OD[k]] = 1
    for t in range(a):                     # A-type: ev -> od
        canon[g.OD[b + t]][g.EV[b + t]] = 1
    canon = mred(canon, p)
    return P, canon, (a, b)


# ------------------------------------------------------------------ A ------
def alt_par(m, n):
    """the alternating format of the paper: ev, od, ev, od, ..."""
    par, i, j = [], m, n
    while i or j:
        if i:
            par.append(0)
            i -= 1
        if j:
            par.append(1)
            j -= 1
    return par


def four_elements(g):
    """the four elements (22) of gl(2|2) in the alternating format (1-based names)"""
    E = lambda i, j: Eij(g.N, i - 1, j - 1)
    x0 = madd(E(1, 2), E(3, 4), g.p)
    x1 = madd(madd(E(1, 2), E(1, 4), g.p), E(3, 4), g.p)
    x2 = madd(E(1, 2), E(4, 3), g.p)
    x3 = madd(madd(madd(E(1, 2), E(2, 3), g.p), E(1, 4), g.p), E(4, 3), g.p, -1)
    return [("x0 = E12+E34", x0), ("x1 = E12+E14+E34", x1),
            ("x2 = E12+E43", x2), ("x3 = E12+E23+E14-E43", x3)]


def part_A(p):
    g = GL(alt_par(2, 2), p)
    field = "Q" if p is None else "F_%d" % p
    say("\n--- A. the four elements (22) in gl(1|1|1|1) over %s ---" % field)
    ev, od = sl_basis(g)
    pev, pod = psl_quotient(g, ev, od)
    say("    dim gl = %d (%d|%d),  dim sl = %d,  dim psl = %d"
        % (g.dim_ev + g.dim_od, g.dim_ev, g.dim_od, len(ev) + len(od),
           len(pev) + len(pod)))
    out = {}
    for name, x in four_elements(g):
        check("%-22s is odd" % name, g.parity(x) == 1)
        check("%-22s x^2 = 0" % name, mzero(g.sq(x)))
        a, b = g.ab(x)
        sd_gl = g.sdim_gx(x)
        sd_sl = g.sdim_gx(x, (ev, od))
        sd_psl = sdim_quotient_gx(g, x, pev, pod, mod_one=True)
        say("      (a,b) = (%d,%d)   rank x = %d   sdim DS_x(V) = %s"
            % (a, b, a + b, str(g.sdim_DS_V(x)).replace(", ", "|")))
        say("      sdim gl(2|2)_x = %s   sdim sl(2|2)_x = %s   sdim psl(2|2)_x = %s"
            % (str(sd_gl).replace(", ", "|"), str(sd_sl).replace(", ", "|"),
               str(sd_psl).replace(", ", "|")))
        out[name.split()[0]] = (a, b, sd_gl, sd_sl, sd_psl)
    check("g_x(x0) = g_x(x2) although (a,b) differ",
          out["x0"][2] == out["x2"][2] and out["x0"][:2] != out["x2"][:2],
          "(0,2) vs (1,1), both give %s" % str(out["x0"][2]).replace(", ", "|"))
    check("g_x(x1) = g_x(x0)", out["x1"][2] == out["x0"][2])
    check("g_x(x3) = g_x(x2)", out["x3"][2] == out["x2"][2])
    check("all four give the same sl(2|2)_x",
          len({out[k][3] for k in out}) == 1,
          str(out["x0"][3]).replace(", ", "|"))
    check("all four give the same psl(2|2)_x",
          len({out[k][4] for k in out}) == 1,
          str(out["x0"][4]).replace(", ", "|"))
    return out


def part_A3(p):
    """the same four elements inside gl(3|3): there g_x is NOT zero, so the
    coincidence g_x(x0) = g_x(x2) says something"""
    g = GL(alt_par(3, 3), p)
    field = "Q" if p is None else "F_%d" % p
    say("\n--- A'. the same four elements inside gl(1|1|1|1|1|1) over %s ---" % field)
    E = lambda i, j: Eij(g.N, i - 1, j - 1)
    els = [("x0 = E12+E34", madd(E(1, 2), E(3, 4), g.p)),
           ("x1 = E12+E14+E34", madd(madd(E(1, 2), E(1, 4), g.p), E(3, 4), g.p)),
           ("x2 = E12+E43", madd(E(1, 2), E(4, 3), g.p)),
           ("x3 = E12+E23+E14-E43",
            madd(madd(madd(E(1, 2), E(2, 3), g.p), E(1, 4), g.p), E(4, 3), g.p, -1))]
    ev, od = sl_basis(g)
    seen = {}
    for nm, x in els:
        check("%-22s odd, x^2 = 0" % nm, g.parity(x) == 1 and mzero(g.sq(x)))
        a, b = g.ab(x)
        sd = g.sdim_gx(x)
        say("      (a,b) = (%d,%d)   sdim DS_x(V) = %s   sdim gl(3|3)_x = %s   "
            "sdim sl(3|3)_x = %s"
            % (a, b, str(g.sdim_DS_V(x)).replace(", ", "|"),
               str(sd).replace(", ", "|"),
               str(g.sdim_gx(x, (ev, od))).replace(", ", "|")))
        seen[nm.split()[0]] = ((a, b), sd)
    want = GL(alt_par(1, 1), p)
    check("all four give gl(3|3)_x of the superdimension of gl(1|1) = (2|2), "
          "which is NOT zero",
          all(v[1] == (want.dim_ev, want.dim_od) for v in seen.values()),
          "the two orbits (0,2) and (1,1) give one and the same g_x")


def part_E(m, n, p):
    """the elements of Hypothesis 1gen) -- sums of the vectors of pairwise
    non-adjacent POSITIVE grey simple roots -- realize every isomorphism class"""
    g = GL(alt_par(m, n), p)
    field = "Q" if p is None else "F_%d" % p
    k = min(m, n)
    say("\n--- E. gl(%d|%d) over %s: the elements of the Hypothesis ---"
        % (m, n, field))
    classes = set()
    for r in range(k + 1):
        x = zeros(g.N)
        for j in range(1, r + 1):
            x[2 * j - 2][2 * j - 1] = 1          # E_{2j-1, 2j}, 1-based
        x = mred(x, p)
        ab = g.ab(x)
        sd = g.sdim_gx(x)
        want = GL(alt_par(m - r, n - r), p)
        say("      x_%d = sum_{1<=j<=%d} E_{2j-1,2j}:  (a,b) = %s,  "
            "sdim gl_x = %s = sdim gl(%d|%d)"
            % (r, r, str(ab), str(sd).replace(", ", "|"), m - r, n - r))
        check("  x_%d is homological, has (a,b) = (0,%d) and gives gl(%d|%d)"
              % (r, r, m - r, n - r),
              mzero(g.sq(x)) and ab == (0, r)
              and sd == (want.dim_ev, want.dim_od))
        classes.add(sd)
    check("the %d elements of the Hypothesis realize all %d classes of g_x"
          % (k + 1, k + 1), len(classes) == k + 1)


# ------------------------------------------------------------------ B ------
def part_B(p):
    g = GL(alt_par(2, 2), p)
    field = "Q" if p is None else "F_%d" % p
    say("\n--- B. the conjugating automorphisms over %s ---" % field)
    els = dict(four_elements(g))
    E = lambda i, j: Eij(g.N, i - 1, j - 1)
    # exp(-ad E24) takes x0 to x1
    G1 = madd(unit(g.N), E(2, 4), g.p, -1)            # 1 - E24, even, unipotent
    G1i = madd(unit(g.N), E(2, 4), g.p)
    got = mmul(mmul(G1, els["x0 = E12+E34"], g.p), G1i, g.p)
    check("g (E12+E34) g^{-1} = E12+E14+E34 for g = 1 - E24",
          meq(got, els["x1 = E12+E14+E34"]))
    check("1 - E24 is even", g.parity(madd(G1, unit(g.N), g.p, -1)) == 0)
    # the second one, produced by the normal-form algorithm and verified here
    for src, dst in [("x2 = E12+E43", "x3 = E12+E23+E14-E43"),
                     ("x0 = E12+E34", "x1 = E12+E14+E34")]:
        Ps, cs, abs_ = normal_form(g, els[src])
        Pd, cd, abd = normal_form(g, els[dst])
        check("normal form of %-22s is the normal shape (a,b)=%s" % (src, str(abs_)),
              meq(mmul(mmul(Ps, els[src], g.p), minv(Ps, g.p), g.p), cs))
        check("normal form of %-22s is the normal shape (a,b)=%s" % (dst, str(abd)),
              meq(mmul(mmul(Pd, els[dst], g.p), minv(Pd, g.p), g.p), cd))
        if abs_ == abd:
            Q = mmul(minv(Pd, g.p), Ps, g.p)
            check("an explicit even Q with Q %s Q^{-1} = %s"
                  % (src.split()[0], dst.split()[0]),
                  meq(mmul(mmul(Q, els[src], g.p), minv(Q, g.p), g.p), els[dst])
                  and g.parity(madd(Q, unit(g.N), g.p, -1)) in (0, None))


# ------------------------------------------------------------------ C ------
def part_C(p):
    g = GL(alt_par(2, 2), p)
    field = "Q" if p is None else "F_%d" % p
    say("\n--- C. (a,b) is a G_ev-invariant; the supertranspose swaps it (%s) ---"
        % field)
    els = dict(four_elements(g))
    x0, x2 = els["x0 = E12+E34"], els["x2 = E12+E43"]
    check("(a,b) differ for x0 and x2, so they are not G_ev-conjugate",
          g.ab(x0) != g.ab(x2), "%s vs %s" % (g.ab(x0), g.ab(x2)))
    # random even conjugations do not move (a,b)
    tested = 0
    vals = set()
    rng = range(p) if p else range(-2, 3)
    for t in rng:
        for (i, j) in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            if g.par[i] != g.par[j]:
                continue
            Q = madd(unit(g.N), Eij(g.N, i, j), g.p, t)
            if ex_rank(Q, p=p) < g.N:
                continue
            for nm, x in els.items():
                y = mmul(mmul(Q, x, g.p), minv(Q, g.p), g.p)
                vals.add((nm, g.ab(y) == g.ab(x)))
                tested += 1
    check("(a,b) unchanged by %d even conjugations" % tested,
          all(v for _, v in vals) and tested > 0)
    # supertranspose  X --> -X^{st}
    def st(X):
        Y = zeros(g.N)
        for i in range(g.N):
            for j in range(g.N):
                s = -1 if (g.par[i] + g.par[j]) % 2 == 0 else \
                    (1 if g.par[j] == 0 else -1)
                Y[i][j] = s * X[j][i]
        return mred(Y, g.p)
    for nm, x in els.items():
        y = st(x)
        check("supertranspose swaps (a,b) of %-22s" % nm,
              g.ab(y) == (g.ab(x)[1], g.ab(x)[0]) and mzero(g.sq(y)),
              "%s --> %s" % (g.ab(x), g.ab(y)))


# ------------------------------------------------------------------ D ------
def part_D(m, n, q):
    """every homological x of gl(m|n) over F_q, brought to normal form"""
    g = GL(alt_par(m, n), q)
    say("\n--- D. gl(%d|%d) over F_%d: every homological x, exhaustively ---"
        % (m, n, q))
    odd_slots = [(i, j) for i in range(g.N) for j in range(g.N)
                 if (g.par[i] + g.par[j]) % 2 == 1]
    total, hom = 0, 0
    byab = {}
    bad = []
    for vals in product(range(q), repeat=len(odd_slots)):
        total += 1
        X = zeros(g.N)
        for (i, j), v in zip(odd_slots, vals):
            X[i][j] = v
        if not mzero(g.sq(X)):
            continue
        hom += 1
        A, B = g.blocks(X)
        # x^2 = 0 <=> AB = 0 and BA = 0
        AB = mmul(A, B, q)
        BA = mmul(B, A, q)
        if not (mzero(AB) and mzero(BA)):
            bad.append(("AB/BA", X))
        ab = g.ab(X)
        sd = g.sdim_gx(X)
        sdV = g.sdim_DS_V(X)
        byab.setdefault(ab, {"count": 0, "gx": set(), "dsv": set()})
        byab[ab]["count"] += 1
        byab[ab]["gx"].add(sd)
        byab[ab]["dsv"].add(sdV)
        # the constructive part: conjugate to the normal shape
        P, canon, ab2 = normal_form(g, X)
        if ab2 != ab or not meq(mmul(mmul(P, X, q), minv(P, q), q), canon):
            bad.append(("normal form", X))
    say("    %d odd matrices, %d of them homological" % (total, hom))
    check("x^2 = 0 <=> AB = 0 and BA = 0 (%d elements)" % hom,
          not any(t == "AB/BA" for t, _ in bad))
    check("each of the %d homological x conjugated to its normal shape" % hom,
          not any(t == "normal form" for t, _ in bad))
    ks = set()
    for ab in sorted(byab):
        d = byab[ab]
        gx = list(d["gx"])[0]
        say("      (a,b) = %-6s  %6d elements   sdim DS_x(V) = %-8s sdim gl_x = %s"
            % (str(ab), d["count"], str(list(d["dsv"])[0]).replace(", ", "|"),
               str(gx).replace(", ", "|")))
        check("  (a,b)=%s: one value of g_x and of DS_x(V)" % str(ab),
              len(d["gx"]) == 1 and len(d["dsv"]) == 1)
        ks.add((ab[0] + ab[1], gx))
    check("g_x depends on a+b alone (%d pairs (a,b), %d values of a+b)"
          % (len(byab), len({k for k, _ in ks})),
          len(ks) == len({k for k, _ in ks}))
    exp_orbits = sum(1 for a in range(min(m, n) + 1)
                     for b in range(min(m, n) + 1 - a))
    check("the number of orbits is (k+1)(k+2)/2 = %d for k = min(m,n) = %d"
          % (exp_orbits, min(m, n)), len(byab) == exp_orbits)
    check("the number of isomorphism classes of g_x is k+1 = %d" % (min(m, n) + 1),
          len({k for k, _ in ks}) == min(m, n) + 1)
    for (a, b) in byab:
        gx = list(byab[(a, b)]["gx"])[0]
        want = GL(alt_par(m - a - b, n - a - b), q)
        check("  (a,b)=(%d,%d): gl_x has sdim of gl(%d|%d)" % (a, b, m - a - b, n - a - b),
              gx == (want.dim_ev, want.dim_od),
              "%s = %s" % (str(gx).replace(", ", "|"),
                           str((want.dim_ev, want.dim_od)).replace(", ", "|")))
    return byab


# ------------------------------------------------------------------ F ------
def part_F():
    """Gorelik's iso-sets for the supports of the four elements (22)"""
    say("\n--- F. Gorelik's iso-sets, in gl(2|2), alternating format ---")
    par = alt_par(2, 2)
    # roots e_i - e_j, i != j; parity p(i)+p(j)
    def rt(i, j):
        return (i, j)

    def is_odd(r):
        return (par[r[0]] + par[r[1]]) % 2 == 1

    def add(r, s):
        # e_i - e_j + e_k - e_l is a root iff j == k or l == i (and the result != 0)
        i, j = r
        k, l = s
        if j == k and i != l:
            return (i, l)
        if l == i and k != j:
            return (k, j)
        return None

    supports = {
        "x0 = E12+E34": [rt(0, 1), rt(2, 3)],
        "x1 = E12+E14+E34": [rt(0, 1), rt(0, 3), rt(2, 3)],
        "x2 = E12+E43": [rt(0, 1), rt(3, 2)],
        "x3 = E12+E23+E14-E43": [rt(0, 1), rt(1, 2), rt(0, 3), rt(3, 2)],
    }
    for nm, S in supports.items():
        pool = [r for r in S if is_odd(r)] + [(j, i) for (i, j) in S if is_odd((i, j))]
        witness = None
        for r in pool:
            for s in pool:
                t = add(r, s)
                if t is not None and not is_odd(t):
                    witness = (r, s, t)
        say("      %-22s iso-set: %s%s"
            % (nm, "YES" if witness is None else "NO",
               "" if witness is None else
               "   (e%d-e%d) + (e%d-e%d) = e%d-e%d is even"
               % (witness[0][0] + 1, witness[0][1] + 1, witness[1][0] + 1,
                  witness[1][1] + 1, witness[2][0] + 1, witness[2][1] + 1)))
        supports[nm] = witness
    check("supp(x0), supp(x2) are iso-sets",
          supports["x0 = E12+E34"] is None and supports["x2 = E12+E43"] is None)
    check("supp(x1), supp(x3) are NOT iso-sets for the diagonal Cartan subalgebra",
          supports["x1 = E12+E14+E34"] is not None
          and supports["x3 = E12+E23+E14-E43"] is not None)


# ----------------------------------------------------------------- main ----
def main():
    say("exactlin backend: %s" % backend())
    for p in (None, 2, 3, 5, 7):
        part_A(p)
    for p in (None, 2, 3, 5, 7):
        part_A3(p)
    for p in (None, 2, 3, 5):
        part_B(p)
        part_C(p)
    for (m, n, q) in [(1, 1, 2), (1, 1, 3), (2, 2, 2), (2, 2, 3), (1, 2, 3),
                      (2, 3, 2), (1, 3, 3), (2, 2, 5), (3, 3, 2)]:
        part_D(m, n, q)
    for p in (None, 2, 3, 5):
        for (m, n) in [(2, 2), (2, 3), (3, 3), (3, 5)]:
            part_E(m, n, p)
    part_F()
    say("\n=== %d checks, %d passed, %d failed ==="
        % (CHECKS["pass"] + CHECKS["fail"], CHECKS["pass"], CHECKS["fail"]))
    return 1 if CHECKS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
