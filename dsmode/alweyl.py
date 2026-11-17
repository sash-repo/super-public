#!/usr/bin/env python3
"""The Weyl-group step: Subsec. 1.7.7 of the text, Corollary 1.7.11,
Proposition 1.7.12 and Statement 1.7.13.

What this script checks, over Q and over F_p:

  A. the normal shape of Proposition 1.7.8 IS a sum of root vectors of an
     iso-set in the sense of [Go, 2.1.1]: a + b pairwise orthogonal, linearly
     independent isotropic roots, of which b are positive and a NEGATIVE
     (Corollary 1.7.11);

  B. the Weyl group W = S_m x S_n of gl(m|n): it preserves the type of an
     isotropic root, so it can never carry the (a,b)-shape to the (a',b')-one;
     what it does give is [Go, Cor. 4.7.2(ii)] -- two iso-sets of the same
     cardinality are W-conjugate AFTER the signs are thrown in, w(S' u -S') =
     S'' u -S''.  Exhaustive over all iso-sets of gl(m|n) for m, n <= 3
     (Proposition 1.7.12);

  C. the diagrammatic form of the recipe: orthogonality of two grey simple roots
     is non-adjacency in the Dynkin diagram, and the maximal number of grey
     nodes is attained in the alternating format (all parity formats of gl(m|n),
     m, n <= 3);

  D. psl(n|n): the cone of the homological elements is strictly larger than in
     gl(n|n), by the family X' = {x = (A,B) | AB = BA = c*Id, c =/= 0} of
     [Go, 3.3.4], whose elements are homological in psl and NOT in gl, whose
     support is not an iso-set for the diagonal torus, and whose (a,b) = (n,n)
     is outside the range of Proposition 1.7.8.  Exhaustive over F_3 and F_5 for
     psl(2|2) (Statement 1.7.13);

  E. the outer automorphisms of psl(2|2): the even derivations vanishing on the
     even part, their exponentials, and what the group they generate does to
     the orbits (a,b).

Usage:  python3 alweyl.py
"""
import sys
import os
from itertools import product, combinations, permutations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)

from exactlin import rank as ex_rank, backend
from aladj import (GL, Eij, zeros, unit, mmul, madd, mred, meq, mzero,
                   sl_basis, alt_par)

CHECKS = {"pass": 0, "fail": 0}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)


def check(name, ok, detail=""):
    CHECKS["pass" if ok else "fail"] += 1
    say("  [%s] %s%s" % ("ok" if ok else "FAIL", name,
                         ("   " + detail) if detail else ""))
    return ok


# ---------------------------------------------------------------- roots ----
class Roots:
    """the root system of gl(V) with respect to the diagonal torus, for the
    given parity string of the basis of V"""

    def __init__(self, par):
        self.par = list(par)
        self.N = len(par)
        self.s = [1 if q == 0 else -1 for q in par]     # (eps_i, eps_i)
        self.EV = [i for i in range(self.N) if par[i] == 0]
        self.OD = [i for i in range(self.N) if par[i] == 1]
        self.m, self.n = len(self.EV), len(self.OD)
        self.odd = [(i, j) for i in range(self.N) for j in range(self.N)
                    if i != j and par[i] != par[j]]
        self.even = [(i, j) for i in range(self.N) for j in range(self.N)
                     if i != j and par[i] == par[j]]

    def vec(self, r):
        """eps_i - eps_j as a vector in Z^N"""
        i, j = r
        v = [0] * self.N
        v[i] += 1
        v[j] -= 1
        return tuple(v)

    def form(self, u, v):
        return sum(self.s[i] * u[i] * v[i] for i in range(self.N))

    def is_root(self, v):
        """is the vector v a root, and of which parity"""
        nz = [i for i in range(self.N) if v[i]]
        if len(nz) != 2:
            return None
        i, j = nz
        if {v[i], v[j]} != {1, -1}:
            return None
        return "od" if self.par[i] != self.par[j] else "ev"

    def indep(self, S):
        rows = [list(self.vec(r)) for r in S]
        return ex_rank(rows) == len(S)

    def is_isoset(self, S):
        """Gorelik 2.1.1: linearly independent, and alpha + beta is not an even
        root for any two alpha, beta in Delta_od cap (S u -S)"""
        if not S or not self.indep(S):
            return False
        big = [self.vec(r) for r in S] + [tuple(-c for c in self.vec(r)) for r in S]
        for u in big:
            for v in big:
                w = tuple(u[i] + v[i] for i in range(self.N))
                if self.is_root(w) == "ev":
                    return False
        return True

    def typ(self, r):
        """'B' if the root vector maps V_od --> V_ev (block B), 'A' otherwise"""
        i, j = r
        return "B" if self.par[i] == 0 else "A"

    def ab_of_set(self, S):
        a = sum(1 for r in S if self.typ(r) == "A")
        return (a, len(S) - a)

    def simple(self):
        return [(i, i + 1) for i in range(self.N - 1)]

    def grey(self):
        return [k for k in range(self.N - 1) if self.par[k] != self.par[k + 1]]


def perms_W(rt):
    """the Weyl group S_m x S_n of gl(m|n): permutations of the basis of V
    preserving the parity, as index maps"""
    out = []
    for pe in permutations(rt.EV):
        for po in permutations(rt.OD):
            f = {}
            for k, i in enumerate(rt.EV):
                f[i] = pe[k]
            for k, i in enumerate(rt.OD):
                f[i] = po[k]
            out.append(f)
    return out


def act(f, S):
    return frozenset((f[i], f[j]) for (i, j) in S)


# ------------------------------------------------------------------- A -----
def part_A(m, n, p=None):
    """the normal shape of Proposition 1.7.8 is a sum of root vectors of an
    iso-set, b of the roots positive and a negative"""
    say("\n=== A. the normal shape is an iso-set: gl(%d|%d), p=%s ===" % (m, n, p))
    par = alt_par(m, n)
    rt = Roots(par)
    g = GL(par, p)
    k = min(m, n)
    smax = [(2 * j - 2, 2 * j - 1) for j in range(1, k + 1)]
    smaxpm = set(smax) | {(j, i) for (i, j) in smax}
    for b in range(k + 1):
        for a in range(k + 1 - b):
            if a + b == 0:
                continue
            # canon = sum_{1<=j<=b} E_{2j-1,2j} + sum_{b<j<=a+b} E_{2j,2j-1}
            supp = [(2 * j - 2, 2 * j - 1) for j in range(1, b + 1)]
            supp += [(2 * j - 1, 2 * j - 2) for j in range(b + 1, a + b + 1)]
            X = zeros(rt.N)
            for (i, j) in supp:
                X[i][j] = 1
            X = mred(X, p)
            tag = "(a,b)=(%d,%d)" % (a, b)
            check("%s  x is odd and x^2 = 0" % tag,
                  g.parity(X) == 1 and mzero(g.sq(X)))
            check("%s  ranks (a,b) as claimed" % tag, g.ab(X) == (a, b),
                  str(g.ab(X)))
            check("%s  supp(x) is an iso-set" % tag, rt.is_isoset(supp))
            vecs = [rt.vec(r) for r in supp]
            check("%s  the roots are isotropic" % tag,
                  all(rt.form(v, v) == 0 for v in vecs))
            check("%s  pairwise orthogonal" % tag,
                  all(rt.form(vecs[u], vecs[v]) == 0
                      for u in range(len(vecs)) for v in range(u + 1, len(vecs))))
            check("%s  linearly independent" % tag, rt.indep(supp) if supp else True)
            pos = sum(1 for (i, j) in supp if i < j)
            neg = len(supp) - pos
            check("%s  %d positive and %d negative roots" % (tag, b, a),
                  (pos, neg) == (b, a), "got (%d,%d)" % (pos, neg))
            allsimple = all(abs(i - j) == 1 for (i, j) in supp)
            check("%s  all the roots are simple (up to sign)" % tag, allsimple)
            check("%s  supp(x) lies in S_max u (-S_max) for the maximal iso-set "
                  "S_max of non-adjacent grey simple roots" % tag,
                  set(supp) <= smaxpm)


# ------------------------------------------------------------------- B -----
def all_isosets(rt, maxr=None):
    out = []
    top = maxr if maxr is not None else min(rt.m, rt.n)
    for r in range(1, top + 1):
        for S in combinations(rt.odd, r):
            if rt.is_isoset(S):
                out.append(frozenset(S))
    return out


def part_B(m, n):
    """what the Weyl group can and cannot do"""
    say("\n=== B. the Weyl group W = S_%d x S_%d of gl(%d|%d) ===" % (m, n, m, n))
    par = alt_par(m, n)
    rt = Roots(par)
    W = perms_W(rt)
    iso = all_isosets(rt)
    k = min(m, n)
    say("  %d iso-sets, |W| = %d" % (len(iso), len(W)))

    check("every iso-set has pairwise distinct indices",
          all(len({i for (i, j) in S} | {j for (i, j) in S}) == 2 * len(S)
              for S in iso))
    check("every iso-set has cardinality <= min(m,n) = %d" % k,
          all(len(S) <= k for S in iso))
    check("a set of pairwise orthogonal independent isotropic roots is an iso-set",
          all(rt.is_isoset(S) ==
              (rt.indep(S) and
               all(rt.form(rt.vec(u), rt.vec(v)) == 0
                   for u in S for v in S if u != v))
              for r in range(1, k + 1) for S in combinations(rt.odd, r)))

    # (i) W preserves the type, hence the pair (a,b)
    ok = all(rt.ab_of_set(act(f, S)) == rt.ab_of_set(S) for S in iso for f in W)
    check("W preserves the pair (a,b) of an iso-set", ok)

    # (ii) the W-orbits on iso-sets are exactly the pairs (a,b)
    orbits = []
    seen = set()
    for S in iso:
        if S in seen:
            continue
        orb = {act(f, S) for f in W}
        seen |= orb
        orbits.append((rt.ab_of_set(S), orb))
    labels = sorted(o[0] for o in orbits)
    want = sorted((a, b) for b in range(k + 1) for a in range(k + 1 - b)
                  if a + b >= 1)
    check("the W-orbits on iso-sets are exactly the pairs (a,b) with 1<=a+b<=%d" % k,
          labels == want, "%s vs %s" % (labels, want))

    # (iii) Gorelik 4.7.2(ii): the orbits merge once the signs are thrown in
    def closure(S):
        return frozenset(list(S) + [(j, i) for (i, j) in S])
    byr = {}
    for S in iso:
        byr.setdefault(len(S), []).append(S)
    ok = True
    for r, lst in sorted(byr.items()):
        cls = {closure(S) for S in lst}
        reps = []
        seen2 = set()
        for C in cls:
            if C in seen2:
                continue
            orb = {frozenset(act(f, C)) for f in W}
            seen2 |= orb
            reps.append(C)
        ok = ok and len(reps) == 1
        say("    cardinality %d: %d iso-sets, %d classes of S u (-S) modulo W"
            % (r, len(lst), len(reps)))
    check("two iso-sets of the same cardinality are W-conjugate modulo the signs "
          "[Go, Cor. 4.7.2(ii)]", ok)

    # (iv) what the POSITIVE iso-sets, and the SIMPLE ones, realize
    pos_iso = [S for S in iso if all(i < j for (i, j) in S)]
    simple = set(rt.simple())
    simp_iso = [S for S in iso if set(S) <= simple]
    ab_all = {rt.ab_of_set(S) for S in iso}
    ab_pos = {rt.ab_of_set(S) for S in pos_iso}
    ab_simp = {rt.ab_of_set(S) for S in simp_iso}
    say("    (a,b) realized: by all iso-sets %s" % sorted(ab_all))
    say("                    by the positive ones %s" % sorted(ab_pos))
    say("                    by the simple ones   %s" % sorted(ab_simp))
    check("the iso-sets of POSITIVE roots realize every pair (a,b) except "
          "(%d,0)" % k,
          ab_pos == ab_all - {(k, 0)},
          "%s vs %s" % (sorted(ab_pos), sorted(ab_all)))
    check("the iso-sets of SIMPLE roots realize fewer pairs (a,b) when k > 1",
          (ab_simp <= ab_all) and (ab_simp != ab_all if k > 1 else True),
          "%s vs %s" % (sorted(ab_simp), sorted(ab_all)))
    check("of the pairs with a+b = k = %d, the simple roots realize only (0,%d)"
          % (k, k),
          {ab for ab in ab_simp if sum(ab) == k} == {(0, k)},
          str(sorted(ab for ab in ab_simp if sum(ab) == k)))
    # the maximal iso-set of the recipe, and the signs
    smax = frozenset((2 * j - 2, 2 * j - 1) for j in range(1, k + 1))
    check("S_max = {alpha_1, alpha_3, ...} is an iso-set of %d pairwise "
          "non-adjacent grey simple roots" % k,
          rt.is_isoset(smax) and len(smax) == k and
          all(abs(i - u) >= 2 for (i, j) in smax for (u, v) in smax if i != u))
    smaxpm = set(smax) | {(j, i) for (i, j) in smax}
    reach = set()
    for S in iso:
        if set(S) <= smaxpm:
            reach.add(rt.ab_of_set(S))
    check("every pair (a,b) is realized inside S_max u (-S_max) "
          "[Go, Cor. 4.7.2(iii)]", reach == ab_all,
          "%s vs %s" % (sorted(reach), sorted(ab_all)))


# ------------------------------------------------------------------- C -----
def part_C(m, n):
    """the diagrammatic recipe: non-adjacency and the maximal number of
    grey nodes"""
    say("\n=== C. the Dynkin diagram of gl(%d|%d): non-adjacency, grey nodes ==="
        % (m, n))
    k = min(m, n)
    best = []
    for par in product([0, 1], repeat=m + n):
        if sum(1 for q in par if q == 0) != m:
            continue
        rt = Roots(par)
        simple = rt.simple()
        grey = rt.grey()
        # orthogonality of two grey simple roots  <==>  non-adjacency
        for u in grey:
            for v in grey:
                if u == v:
                    continue
                orth = rt.form(rt.vec(simple[u]), rt.vec(simple[v])) == 0
                if orth != (abs(u - v) >= 2):
                    check("orthogonal <==> non-adjacent in %s" % (par,), False,
                          "nodes %d, %d" % (u, v))
                    return
        # the largest set of pairwise non-adjacent grey simple roots
        top = 0
        for r in range(len(grey), 0, -1):
            for S in combinations(grey, r):
                if all(abs(u - v) >= 2 for u in S for v in S if u != v):
                    top = max(top, r)
            if top:
                break
        best.append((len(grey), top, tuple(par)))
    check("for grey simple roots, orthogonality is non-adjacency in the diagram",
          True)
    maxgrey = max(b[0] for b in best)
    alt = [b for b in best if b[0] == maxgrey]
    check("the maximal number of grey nodes is %d = %s"
          % (maxgrey, "2min(m,n)-1" if m == n else "2min(m,n)"),
          maxgrey == (2 * k - 1 if m == n else 2 * k))
    check("with the maximal number of grey nodes the recipe gives min(m,n) = %d" % k,
          all(b[1] == k for b in alt), str(alt))
    ourfmt = [b for b in best if b[2] == tuple(alt_par(m, n))][0]
    check("the alternating format Par_alt of the paper gives min(m,n) = %d, "
          "with %d grey nodes" % (k, ourfmt[0]), ourfmt[1] == k)
    others = [b for b in best if b[0] < maxgrey]
    attain = [b for b in best if b[1] == k]
    say("    %d formats in all, %d of them reach min(m,n) = %d; the maximal "
        "number of grey nodes is %d" % (len(best), len(attain), k, maxgrey))
    check("the clause cannot be DROPPED: the distinguished format gives 1 < %d"
          % k,
          [b for b in best if b[2] == tuple([0] * m + [1] * n)][0][1] == 1
          if k > 1 else True)
    check("but it can be relaxed: a format with fewer grey nodes may reach "
          "min(m,n) too",
          any(b[0] < maxgrey and b[1] == k for b in best) if m + n > 4 else True,
          "%d such formats" % len([b for b in best if b[0] < maxgrey and b[1] == k]))
    check("no format gives more than min(m,n) = %d" % k,
          all(b[1] <= k for b in best))


# ------------------------------------------------------------------- D -----
class PSL:
    """psl(n|n) = sl(n|n)/K.1 in the alternating format"""

    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.par = alt_par(n, n)
        self.g = GL(self.par, p)
        self.N = 2 * n
        ev, od = sl_basis(self.g)
        # a basis of sl_ev modulo K.1
        one = self.g.coords(unit(self.N))
        keep, cur = [], [one]
        for kk, M in enumerate(ev):
            r = self.g.coords(M)
            if ex_rank(cur + [r], p=p) > ex_rank(cur, p=p):
                cur.append(r)
                keep.append(kk)
        self.ev = [ev[kk] for kk in keep]
        self.od = od
        self.one = one

    def homological(self, X):
        """x^2 in K.1"""
        S = self.g.sq(X)
        rows = [self.one, self.g.coords(S)]
        return ex_rank(rows, p=self.p) <= 1

    def in_gl_cone(self, X):
        return mzero(self.g.sq(X))

    def sdim_gx(self, X):
        g = self.g
        rows_ev = [g.coords(g.br(X, Y, 1, 0)) for Y in self.ev]   # --> odd
        rows_od = [g.coords(g.br(X, Y, 1, 1)) for Y in self.od]   # --> even/K.1
        r_ev = ex_rank(rows_ev, p=self.p) if rows_ev else 0
        r_od = (ex_rank([self.one] + rows_od, p=self.p) - 1) if rows_od else 0
        return (len(self.ev) - r_ev - r_od, len(self.od) - r_od - r_ev)

    def gx_is_commutative(self, X):
        """is the odd part of g_x = Ker/Im commutative (i.e. [u,v] in Im + K.1)?"""
        g = self.g
        # a basis of Ker(ad_x on the odd part) modulo Im(ad_x from the even part)
        ker = []
        rows = []
        for Y in self.od:
            r = g.coords(g.br(X, Y, 1, 1))
            if ex_rank([self.one] + rows + [r], p=self.p) == \
               ex_rank([self.one] + rows, p=self.p):
                ker.append(Y)
            rows.append(r)
        # this is a crude spanning set; refine by solving properly
        kerb = []
        base = [self.one]
        # kernel: solve for the combinations Y with [x,Y] in K.1
        import itertools
        cols = [g.coords(g.br(X, Y, 1, 1)) for Y in self.od]
        from exactlin import nullspace as ex_null
        # rows of the matrix: the images; we need combinations c with sum c_i img_i
        # in K.1, i.e. rank condition -- solve [imgs ; one]^T
        M = [[cols[i][t] for i in range(len(cols))] + [self.one[t]]
             for t in range(len(cols[0]))]
        ns = ex_null(M, p=self.p)
        for v in ns:
            Y = zeros(self.N)
            for i in range(len(cols)):
                if v[i]:
                    Y = madd(Y, [[v[i] * c for c in row] for row in self.od[i]],
                             self.p)
            if not mzero(Y):
                kerb.append(mred(Y, self.p))
        im = [g.coords(g.br(X, Y, 1, 0)) for Y in self.ev]
        imr = ex_rank(im, p=self.p)
        ok = True
        for u in range(len(kerb)):
            for v in range(u, len(kerb)):
                Z = g.br(kerb[u], kerb[v], 1, 1)      # lands in the even part
                rows2 = [self.one] + [g.coords(g.br(X, Y, 1, 1)) for Y in self.od]
                # [u,v] must be in Im(ad_x from the odd part) + K.1
                if ex_rank(rows2 + [g.coords(Z)], p=self.p) > \
                   ex_rank(rows2, p=self.p):
                    ok = False
        return ok, len(kerb), imr


def part_D(n, p, exhaustive=False):
    """psl(n|n): the cone is strictly bigger"""
    say("\n=== D. psl(%d|%d) over F_%s: the extra homological elements ==="
        % (n, n, p))
    P = PSL(n, p)
    g = P.g
    rt = Roots(P.par)
    say("  sdim psl(%d|%d) = (%d|%d)" % (n, n, len(P.ev), len(P.od)))

    # the family X' : A = B = Id in the alternating format
    X = zeros(P.N)
    for j in range(n):
        X[2 * j][2 * j + 1] = 1
        X[2 * j + 1][2 * j] = 1
    X = mred(X, p)
    check("X' element: odd", g.parity(X) == 1)
    check("X' element: x^2 = 1, so NOT homological in gl(%d|%d)" % (n, n),
          not P.in_gl_cone(X) and meq(g.sq(X), unit(P.N)))
    check("X' element: homological in psl(%d|%d)" % (n, n), P.homological(X))
    check("X' element: (a,b) = (%d,%d), outside a+b <= %d" % (n, n, n),
          g.ab(X) == (n, n))
    supp = [(i, j) for i in range(P.N) for j in range(P.N) if X[i][j]]
    check("X' element: supp(x) is NOT an iso-set for the diagonal torus",
          not rt.is_isoset(supp))
    sd = P.sdim_gx(X)
    say("    sdim psl(%d|%d)_x = %s for x in X'" % (n, n, sd))
    check("X' element: sdim g_x = (0|2) = sdim psl(1|1) [Go, 3.3.4]", sd == (0, 2))
    ok, dk, di = P.gx_is_commutative(X)
    check("X' element: g_x is commutative, i.e. psl(1|1)", ok)

    # the rank-n element of the gl cone, for comparison
    Y = zeros(P.N)
    for j in range(n):
        Y[2 * j][2 * j + 1] = 1
    Y = mred(Y, p)
    sdY = P.sdim_gx(Y)
    check("x of rank %d in the gl cone: homological in psl too" % n,
          P.homological(Y) and P.in_gl_cone(Y))
    check("its sdim g_x is the same, (0|2)", sdY == (0, 2), str(sdY))

    if exhaustive and n == 2:
        say("  exhaustive sweep of psl(2|2)_od over F_%d" % p)
        odb = P.od
        cone_gl, cone_extra = 0, 0
        seen = {}
        for co in product(range(p), repeat=len(odb)):
            if not any(co):
                continue
            Z = zeros(P.N)
            for i, c in enumerate(co):
                if c:
                    Z = madd(Z, [[c * v for v in row] for row in odb[i]], p)
            Z = mred(Z, p)
            if not P.homological(Z):
                continue
            if P.in_gl_cone(Z):
                cone_gl += 1
                key = ("gl", g.ab(Z), P.sdim_gx(Z))
            else:
                cone_extra += 1
                key = ("X'", g.ab(Z), P.sdim_gx(Z))
            seen[key] = seen.get(key, 0) + 1
        say("    %d homological x with x^2 = 0 in gl, %d with x^2 in K*.1"
            % (cone_gl, cone_extra))
        for key in sorted(seen):
            say("      %-4s (a,b)=%s  sdim g_x=%s : %d elements"
                % (key[0], key[1], key[2], seen[key]))
        check("the extra family is non-empty over F_%d" % p, cone_extra > 0)
        check("every extra element has (a,b) = (2,2)",
              all(k[1] == (2, 2) for k in seen if k[0] == "X'"))
        check("every extra element gives sdim g_x = (0|2)",
              all(k[2] == (0, 2) for k in seen if k[0] == "X'"))
        check("in the gl cone, sdim g_x depends on a+b alone",
              len({(sum(k[1]), k[2]) for k in seen if k[0] == "gl"}) ==
              len({sum(k[1]) for k in seen if k[0] == "gl"}))
        return seen
    return None


# ------------------------------------------------------------------- E -----
def part_E_quiet(p):
    """the even derivations of psl(2|2) vanishing on the even part: the unknown
    is delta on the odd part, the conditions are

        [delta x, y] + [x, delta y] = 0  in  g_ev/K.1   (delta is a derivation),
        delta([h,x]) = [h, delta x]      for h even     (delta is g_ev-linear).

    The central term of the first condition is a free unknown, one per pair."""
    P = PSL(2, p)
    g = P.g
    od, ev = P.od, P.ev
    d = len(od)
    de = g.dim_ev

    def odco(X):
        return g.coords(X)[de:]

    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    ncen = len(pairs)
    rows = []
    for pi, (i, j) in enumerate(pairs):
        contrib = []
        for k in range(d):
            contrib.append((k, i, g.coords(g.br(od[k], od[j], 1, 1))))
            contrib.append((k, j, g.coords(g.br(od[i], od[k], 1, 1))))
        for t in range(len(P.one)):
            row = [0] * (d * d + ncen)
            for (k, l, vec) in contrib:
                row[k * d + l] += vec[t]
            row[d * d + pi] = -P.one[t]
            rows.append(row)
    # g_ev-linearity
    for h in ev:
        M = [odco(g.br(h, Y, 0, 1)) for Y in od]      # M[i][l] = coeff of od_l
        for i in range(d):
            for t in range(d):
                row = [0] * (d * d + ncen)
                # delta([h,od_i]) = sum_l M[i][l] * sum_k D[k][l] od_k
                for l in range(d):
                    if M[i][l]:
                        row[t * d + l] += M[i][l]
                # [h, delta od_i] = sum_k D[k][i] * sum_l M[k][l] od_l
                for k in range(d):
                    if M[k][t]:
                        row[k * d + i] -= M[k][t]
                rows.append(row)
    from exactlin import nullspace as ex_null
    ns = ex_null(rows, p=p)
    # keep only the D-part
    return [v[:d * d] for v in ns], P


def part_E(p):
    """the outer automorphisms of psl(2|2): do they merge the orbits (0,2)
    and (1,1)?"""
    say("\n=== E. the outer automorphisms of psl(2|2) over F_%s ===" % p)
    ns, P = part_E_quiet(p)
    g = P.g
    od = P.od
    d = len(od)
    de = g.dim_ev
    say("  the space of the even derivations of psl(2|2) vanishing on the even "
        "part has dimension %d" % len(ns))
    check("these derivations form an sl(2), i.e. a 3-dimensional space",
          len(ns) == 3, str(len(ns)))
    if len(ns) != 3:
        return ns, P

    def mat(v):
        return [[v[k * d + i] % p for i in range(d)] for k in range(d)]

    def odco(X):
        return list(g.coords(X)[de:])

    def from_co(c):
        Y = zeros(P.N)
        for i in range(d):
            if c[i] % p:
                Y = madd(Y, [[c[i] * u for u in row] for row in od[i]], p)
        return mred(Y, p)

    def apply(E, X):
        c = odco(X)
        return from_co([sum(E[k][i] * c[i] for i in range(d)) % p
                        for k in range(d)])

    # the group generated by the exponentials of the nilpotent derivations
    gens = []
    for v in ns:
        D = mat(v)
        for t in range(1, p):
            Dt = [[(t * u) % p for u in row] for row in D]
            E = [[1 if i == j else 0 for j in range(d)] for i in range(d)]
            T = [row[:] for row in E]
            fact, ok = 1, True
            for s in range(1, d + 2):
                T = mmul(T, Dt, p)
                if mzero(T):
                    break
                fact *= s
                if fact % p == 0:
                    ok = False
                    break
                E = madd(E, [[(pow(fact % p, p - 2, p) * u) % p for u in row]
                             for row in T], p)
            else:
                ok = False
            if ok:
                gens.append(E)
    grp = {tuple(tuple(r) for r in E) for E in gens}
    grp.add(tuple(tuple(1 if i == j else 0 for j in range(d))
                  for i in range(d)))
    while True:
        new = set()
        for A in grp:
            for B in gens:
                C = mmul([list(r) for r in A], B, p)
                new.add(tuple(tuple(r) for r in C))
        if new <= grp:
            break
        grp |= new
    say("  the group generated by the exponentials has %d elements" % len(grp))
    check("the group generated is non-trivial", len(grp) > 1)

    ok = True
    for E in grp:
        Em = [list(r) for r in E]
        for u in range(d):
            for v in range(u, d):
                D1 = madd(g.br(apply(Em, od[u]), apply(Em, od[v]), 1, 1),
                          g.br(od[u], od[v], 1, 1), p, -1)
                if ex_rank([P.one, g.coords(D1)], p=p) > 1:
                    ok = False
    check("every element of the group is an automorphism of psl(2|2) trivial "
          "on the even part", ok)

    x1 = mred(Eij(4, 0, 1), p)                              # (a,b) = (0,1)
    x0 = mred(madd(Eij(4, 0, 1), Eij(4, 2, 3), p), p)       # (a,b) = (0,2)
    x2 = mred(madd(Eij(4, 0, 1), Eij(4, 3, 2), p), p)       # (a,b) = (1,1)
    check("x_0, x_1 and x_2 are homological in psl(2|2)",
          all(P.homological(z) for z in (x0, x1, x2)))
    reach = {}
    for name, X in (("x_1 (0,1)", x1), ("x_0 (0,2)", x0), ("x_2 (1,1)", x2)):
        got = set()
        for E in grp:
            Y = apply([list(r) for r in E], X)
            if mzero(Y):
                continue
            if not P.homological(Y):
                got.add(("NOT homological", g.ab(Y)))
            else:
                got.add(("gl cone" if P.in_gl_cone(Y) else "X'", g.ab(Y)))
        reach[name] = got
        say("  the orbit of %s: %s" % (name, sorted(got, key=str)))
    check("the outer automorphisms preserve the homological cone of psl(2|2)",
          all("NOT homological" not in {t for (t, _) in s}
              for s in reach.values()))
    check("they do NOT merge the orbits (0,2) and (1,1)",
          (1, 1) not in {ab for (_, ab) in reach["x_0 (0,2)"]} and
          (0, 2) not in {ab for (_, ab) in reach["x_2 (1,1)"]})
    check("they DO merge the rank-2 elements of the gl cone with the extra "
          "family X' of [Go, 3.3.4]",
          any(t == "X'" for (t, _) in reach["x_0 (0,2)"]))
    check("they do not do so for the elements of rank 1",
          all(t != "X'" for (t, _) in reach["x_1 (0,1)"]))
    return ns, P


# ----------------------------------------------------------------- main ----
def main():
    say("exactlin backend: %s" % backend())
    for (m, n) in [(1, 1), (2, 2), (2, 3), (3, 3)]:
        for p in (None, 2, 3, 5):
            part_A(m, n, p)
    for (m, n) in [(1, 1), (2, 2), (1, 2), (2, 3), (3, 3)]:
        part_B(m, n)
    for (m, n) in [(2, 2), (2, 3), (3, 3)]:
        part_C(m, n)
    for p in (3, 5):
        part_D(2, p, exhaustive=True)
    for p in (3, 5):
        part_D(3, p)
    for p in (3, 5):
        part_E(p)
    say("\n=== %d checks, %d passed, %d failed ==="
        % (CHECKS["pass"] + CHECKS["fail"], CHECKS["pass"], CHECKS["fail"]))
    return 1 if CHECKS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
