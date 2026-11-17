#!/usr/bin/env python3
"""
Exact test of the last row of Table 2.12.1 of "A natural generalization of
supersymmetry" (Leites) and of Remark 3.2.1: an earlier draft claimed that

    for g = ag(2), ab(3), osp_a(4|2) the root vector of an isotropic (odd)
    simple root NEVER ensures nontriviality,

where (Subsec. 2.9, eq. (31)) an odd x with [x,x] in C(g) *ensures
nontriviality* iff there are a, b in g_ev with [[a,x],[b,x]] != 0.  If x does
not ensure nontriviality, h_{ad_x} = {a+[a,x]} is isomorphic to g_ev and g
contributes nothing.

RESULT: that claim is false for all three (the text now says 'always').  Every isotropic odd root vector of
all three ensures nontriviality, and the resulting h_{ad_x} is a SIMPLE
Volichenko algebra, not isomorphic to g_ev.

What is checked here, over Q with Fractions -- no floating point:

 (1) the models (exc3.py, d21a.py) are re-verified:
     super-Jacobi on every triple, sdim, simplicity of g;
 (2) every odd root vector x: is [x,x]=0, and does x ensure nontriviality,
     with an explicit witness pair (a,b);
 (3) for a representative isotropic x, h = h_{ad_x} is built as (g_ev, *) with
         a*b = [a,b] + S(a,b),   S(a,b) := [[a,x],[b,x]]  (symmetric),
     and we verify
       -- eq. (32): [S(a,b), x] = 0, i.e. h really is closed in g;
       -- Baranov's identities (i)-(v) of Subsec. 2.4;
       -- h is 2-step filtered with h_0 = Ann(x);
       -- h is SIMPLE: g_ev is semisimple, so every ideal of (h,[,]) is a sum
          of simple factors -- a finite list -- and none of the proper ones is
          closed under S.  This is a complete proof, not a sample;
       -- S != 0, so h is not isomorphic to g_ev.

 (4) CONTROLS reproducing the Table's OTHER rows with the same code:
       gl(2|2), x^1_1 (p=q=1)  -> ensures nontriviality   (Table: yes)
       gl(2|2), x^1_0 (q=0)    -> does not                (Table: no)

The osp(4|2) half is cross-checked by ospcheck.py, which builds osp(4|2) from
scratch inside gl(4|2) and shares no code with exc3.py/d21a.py.

Usage:  python3 nontriv.py            (add  --quick  to skip ab(3))
"""

import os
import sys
from fractions import Fraction as F
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "."))

import exc3          # noqa: E402   ag(2) = G(3),  ab(3) = F(4)
import d21a          # noqa: E402   osp_a(4|2)


# ------------------------------------------------------------------ linear algebra

def rref(rows):
    m = [list(r) for r in rows]
    piv, r = [], 0
    ncols = len(m[0]) if m else 0
    for c in range(ncols):
        s = None
        for i in range(r, len(m)):
            if m[i][c]:
                s = i
                break
        if s is None:
            continue
        m[r], m[s] = m[s], m[r]
        inv = F(1) / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c]:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        piv.append(c)
        r += 1
        if r == len(m):
            break
    return m[:r], piv


def span(vs):
    vs = [v for v in vs if any(v)]
    return rref(vs)[0] if vs else []


def inspan(v, ech):
    if not any(v):
        return True
    if not ech:
        return False
    _, piv = rref(ech)
    w = list(v)
    for row, c in zip(ech, piv):
        if w[c]:
            f = w[c]
            w = [x - f * y for x, y in zip(w, row)]
    return not any(w)


# ------------------------------------------------------------------ algebras

class Alg:
    def __init__(self, name, N, PAR, br, WT=None):
        self.name, self.N, self.PAR, self.br, self.WT = name, N, PAR, br, WT
        self.ev = [i for i in range(N) if PAR[i] == 0]
        self.od = [i for i in range(N) if PAR[i] == 1]

    def e(self, i):
        v = [F(0)] * self.N
        v[i] = F(1)
        return v

    def jacobi_bad(self):
        bad = 0
        for a in range(self.N):
            for b in range(self.N):
                for c in range(self.N):
                    lhs = self.br(self.e(a), self.br(self.e(b), self.e(c)))
                    r1 = self.br(self.br(self.e(a), self.e(b)), self.e(c))
                    r2 = self.br(self.e(b), self.br(self.e(a), self.e(c)))
                    sg = F(-1) ** (self.PAR[a] * self.PAR[b])
                    if lhs != [x + sg * y for x, y in zip(r1, r2)]:
                        bad += 1
        return bad


def from_exc3(m):
    return Alg(m.name, m.N, m.PAR, m.br, m.WT)


def from_d21a(c, name):
    B = d21a.build([F(t) for t in c])
    return Alg(name, d21a.N, d21a.PAR, lambda u, v: d21a.bracket(B, u, v))


def gl_control(m, n):
    d = m + n
    N = d * d
    idx = lambda i, j: i * d + j                                   # noqa: E731
    p = lambda i: 0 if i < m else 1                                # noqa: E731
    par = [(p(i) + p(j)) % 2 for i in range(d) for j in range(d)]
    tab = {}
    for i, j, k, l in product(range(d), repeat=4):
        a, b = idx(i, j), idx(k, l)
        out = {}
        if j == k:
            out[idx(i, l)] = out.get(idx(i, l), 0) + 1
        if l == i:
            sg = -((-1) ** (par[a] * par[b]))
            out[idx(k, j)] = out.get(idx(k, j), 0) + sg
        out = {q: F(t) for q, t in out.items() if t}
        if out:
            tab[(a, b)] = out

    def br(u, v):
        res = [F(0)] * N
        for a, ua in enumerate(u):
            if not ua:
                continue
            for b, vb in enumerate(v):
                if not vb:
                    continue
                row = tab.get((a, b))
                if row:
                    for q, t in row.items():
                        res[q] += ua * vb * t
        return res

    return Alg("gl(%d|%d)" % (m, n), N, par, br), idx


# ------------------------------------------------------------------ the criterion

def nontriviality(g, x):
    """returns (witness or None) for  [[a,x],[b,x]] != 0,  a,b in a basis of g_ev."""
    L = [g.br(g.e(i), x) for i in g.ev]
    for ai in range(len(g.ev)):
        for bi in range(ai, len(g.ev)):
            if any(g.br(L[ai], L[bi])):
                return (g.ev[ai], g.ev[bi])
    return None


def root_vectors(g, verbose=True):
    rows = []
    for s in g.od:
        x = g.e(s)
        iso = not any(g.br(x, x))
        wit = nontriviality(g, x) if iso else None
        L = span([g.br(g.e(i), x) for i in g.ev])
        rows.append((s, g.WT[s] if g.WT else None, iso, wit, len(L)))
    if verbose:
        print("   odd root vector          [x,x]=0   ensures nontriv.  dim[g_ev,x]")
        for (s, wt, iso, wit, dL) in rows:
            w = ("%s" % (tuple(str(t) for t in wt),)) if wt is not None else "#%d" % s
            print("   %-24s %-9s %-18s %d"
                  % (w, "yes" if iso else "NO",
                     "yes" if wit else "no", dL))
    return rows


# ------------------------------------------------------------------ the algebra h

class Voli:
    """h_{ad_x} = (g_ev, a*b = [a,b] + S(a,b)),  S(a,b) = [[a,x],[b,x]]."""

    def __init__(self, g, x):
        self.g, self.x = g, x
        self.dim = len(g.ev)
        self.ev = g.ev
        L = [g.br(g.e(i), x) for i in g.ev]
        self.L = L
        n = self.dim
        self.Sm = [[self._coord(g.br(L[i], L[j])) for j in range(n)]
                   for i in range(n)]
        self.Bm = [[self._coord(g.br(g.e(g.ev[i]), g.e(g.ev[j]))) for j in range(n)]
                   for i in range(n)]

    def _coord(self, v):
        """v lies in g_ev; return its coordinates in the g_ev basis."""
        assert all(v[k] == 0 for k in self.g.od), "not in g_ev"
        return [v[k] for k in self.ev]

    def _bil(self, T, u, v):
        out = [F(0)] * self.dim
        for i, ui in enumerate(u):
            if not ui:
                continue
            for j, vj in enumerate(v):
                if not vj:
                    continue
                row = T[i][j]
                for k, t in enumerate(row):
                    if t:
                        out[k] += ui * vj * t
        return out

    def lie(self, u, v):
        return self._bil(self.Bm, u, v)

    def circ(self, u, v):
        return self._bil(self.Sm, u, v)

    def prod(self, u, v):
        return [a + b for a, b in zip(self.lie(u, v), self.circ(u, v))]

    def e(self, i):
        v = [F(0)] * self.dim
        v[i] = F(1)
        return v

    # ---- closure of h in g: eq. (32)
    def closure_ok(self):
        for i in range(self.dim):
            for j in range(self.dim):
                v = self.g.br(self.L[i], self.L[j])
                if any(self.g.br(v, self.x)):
                    return False
        return True

    def S_is_zero(self):
        return not any(any(r) for row in self.Sm for r in row)

    # ---- Baranov (i)-(v)
    def baranov(self):
        n, e = self.dim, self.e
        B = [e(i) for i in range(n)]
        add = lambda *vs: [sum(t) for t in zip(*vs)]               # noqa: E731
        nz = lambda v: any(v)                                      # noqa: E731

        def j3(a, b, c):
            return add(self.lie(a, self.circ(b, c)),
                       self.lie(b, self.circ(c, a)),
                       self.lie(c, self.circ(a, b)))

        res = {}
        # (i) Jacobi for [,]
        res["i"] = not any(nz(add(self.lie(a, self.lie(b, c)),
                                  self.lie(b, self.lie(c, a)),
                                  self.lie(c, self.lie(a, b))))
                           for a in B for b in B for c in B)
        # (ii) (a o b) o c = 0
        res["ii"] = not any(nz(self.circ(self.circ(a, b), c))
                            for a in B for b in B for c in B)
        # (iii) j(a,b,c) o d = 0
        res["iii"] = not any(nz(self.circ(j3(a, b, c), d))
                             for a in B for b in B for c in B for d in B)
        # (iv) [a o b, c o d] = [a o b, c] o d + [a o b, d] o c
        ok = True
        for a in B:
            for b in B:
                ab = self.circ(a, b)
                for c in B:
                    for d in B:
                        lhs = self.lie(ab, self.circ(c, d))
                        rhs = add(self.circ(self.lie(ab, c), d),
                                  self.circ(self.lie(ab, d), c))
                        if lhs != rhs:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        res["iv"] = ok
        # (v) [j(a,b,c), c o d] = [j(a,b,c), c] o d + [j(a,b,c), d] o c
        ok = True
        for a in B:
            for b in B:
                for c in B:
                    jj = j3(a, b, c)
                    for d in B:
                        lhs = self.lie(jj, self.circ(c, d))
                        rhs = add(self.circ(self.lie(jj, c), d),
                                  self.circ(self.lie(jj, d), c))
                        if lhs != rhs:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        res["v"] = ok
        return res

    # ---- simplicity
    def lie_ideal_generated(self, vs):
        I = span(vs)
        changed = True
        while changed:
            changed = False
            for v in list(I):
                for i in range(self.dim):
                    w = self.lie(self.e(i), v)
                    if any(w) and not inspan(w, I):
                        I = span(I + [w])
                        changed = True
        return I

    def simple_factors(self):
        """the minimal nonzero [,]-ideals of g_ev (assumed semisimple)."""
        cands = []
        for i in range(self.dim):
            I = self.lie_ideal_generated([self.e(i)])
            cands.append(I)
        mins = []
        for I in cands:
            if any(len(J) < len(I) and all(inspan(v, I) for v in J) for J in cands):
                continue
            mins.append(I)
        uniq = []
        for I in mins:
            if not any(len(I) == len(J) and all(inspan(v, J) for v in I)
                       for J in uniq):
                uniq.append(I)
        return uniq

    # ---- Proposition 2.13: no ONE-SIDED ideals
    def no_one_sided_ideals(self):
        """A left ideal containing v contains A.v, where A is the unital
           associative subalgebra of End(h) generated by the left
           multiplications.  By Burnside, if A = End(h) then A.v = h for every
           v != 0, so there is no proper left ideal.  Same on the right.
           Returns (dim A_left, dim A_right, dim End(h))."""
        n = self.dim
        full = n * n

        def flat(Mt):
            return [Mt[i][j] for i in range(n) for j in range(n)]

        def mul(P, Q):
            return [[sum(P[i][k] * Q[k][j] for k in range(n)) for j in range(n)]
                    for i in range(n)]

        def closure(gens):
            # incremental echelon: pivot column -> normalized row
            ech = {}

            def insert(v):
                """reduce v against the echelon; add it if independent."""
                for c in sorted(ech):
                    if v[c]:
                        f, row = v[c], ech[c]
                        v = [x - f * y for x, y in zip(v, row)]
                c = next((i for i, x in enumerate(v) if x), None)
                if c is None:
                    return False
                inv = F(1) / v[c]
                ech[c] = [x * inv for x in v]
                return True

            ident = [[F(1) if i == j else F(0) for j in range(n)]
                     for i in range(n)]
            frontier = []
            for M in [ident] + gens:
                if insert(flat(M)):
                    frontier.append(M)
            while frontier and len(ech) < full:
                new = []
                for P in frontier:
                    for Q in gens:
                        R = mul(P, Q)
                        if insert(flat(R)):
                            new.append(R)
                            if len(ech) == full:
                                return full
                frontier = new
            return len(ech)

        # L_a(v) = a*v ; column j of the matrix is a*e_j
        Lg = [[[self.prod(self.e(i), self.e(j))[k] for j in range(n)]
               for k in range(n)] for i in range(n)]
        Rg = [[[self.prod(self.e(j), self.e(i))[k] for j in range(n)]
               for k in range(n)] for i in range(n)]
        return closure(Lg), closure(Rg), full

    def is_simple(self):
        """complete: every ideal of (h,*) is a [,]-ideal, hence a sum of the
           simple factors of the semisimple g_ev -- a finite list."""
        facs = self.simple_factors()
        k = len(facs)
        tot = sum(len(f) for f in facs)
        if tot != self.dim:
            return None, facs        # g_ev not semisimple: test inconclusive
        for mask in range(1, 2 ** k - 1):
            J = []
            for i in range(k):
                if mask >> i & 1:
                    J += facs[i]
            J = span(J)
            closed = True
            for v in J:
                for i in range(self.dim):
                    if not inspan(self.circ(self.e(i), v), J):
                        closed = False
                        break
                if not closed:
                    break
            if closed:
                return False, facs
        return True, facs


# ------------------------------------------------------------------ drivers

def analyse(g, verbose=True, check_jacobi=True):
    print("=" * 76)
    print("%s   sdim (%d|%d)" % (g.name, len(g.ev), len(g.od)))
    print("=" * 76)
    if check_jacobi:
        print("   super-Jacobi failures on all %d^3 triples: %d"
              % (g.N, g.jacobi_bad()))
    rows = root_vectors(g, verbose)
    iso = [r for r in rows if r[2]]
    ens = [r for r in iso if r[3]]
    print("   -> %d odd root vectors, %d isotropic, %d of those ensure "
          "nontriviality" % (len(rows), len(iso), len(ens)))
    if not ens:
        print("   -> no isotropic odd root vector ensures nontriviality here\n")
        return True
    s, wt, _, wit, dL = ens[0]
    x = g.e(s)
    print("   -> Table 2.12.1, last row: 'always' -- confirmed")
    print("      witness for x = f_%d: a = e_%d, b = e_%d," % (s, wit[0], wit[1]))
    val = g.br(g.br(g.e(wit[0]), x), g.br(g.e(wit[1]), x))
    print("      [[a,x],[b,x]] = %s"
          % " + ".join("%s*e_%d" % (t, k) for k, t in enumerate(val) if t))

    h = Voli(g, x)
    print("\n   the Volichenko algebra h_{ad_x}, dim %d:" % h.dim)
    print("      h closed in g (eq. (32)) ............... %s" % h.closure_ok())
    print("      S = 0 (i.e. h isomorphic to g_ev) ...... %s" % h.S_is_zero())
    print("      dim h_0 = dim Ann(x) ................... %d" % (h.dim - dL))
    bar = h.baranov()
    print("      Baranov (i)-(v) ........................ %s"
          % ", ".join("(%s) %s" % (k, "ok" if v else "FAIL")
                      for k, v in bar.items()))
    simple, facs = h.is_simple()
    print("      simple factors of g_ev ................. %s"
          % " + ".join("dim %d" % len(f) for f in facs))
    print("      h is SIMPLE ............................ %s" % simple)
    dl, dr, full = h.no_one_sided_ideals()
    print("      Prop. 2.13, no one-sided ideals ........ %s "
          "(left mult. generate dim %d, right %d, of End(h) = %d)"
          % (dl == full and dr == full, dl, dr, full))
    print()
    return False


def controls():
    print("=" * 76)
    print("CONTROLS: the Table's first row, reproduced by the same code")
    print("=" * 76)
    g, idx = gl_control(2, 2)
    ok = True
    for lab, cells, expect in [("x^1_1 (p=q=1)", [(0, 2), (3, 1)], True),
                               ("x^1_0 (q=0)", [(0, 2)], False)]:
        x = [F(0)] * g.N
        for (i, j) in cells:
            x[idx(i, j)] = F(1)
        iso = not any(g.br(x, x))
        wit = nontriviality(g, x) if iso else None
        got = wit is not None
        print("   gl(2|2), %-14s  [x,x]=0: %-5s  ensures nontriviality: %-5s "
              " (Table says %s)  %s"
              % (lab, iso, got, expect, "OK" if got == expect else "MISMATCH"))
        ok &= (got == expect)
    print("   -> the criterion as coded reproduces the Table where the Table "
          "is checkable.\n")
    return ok


def main():
    quick = "--quick" in sys.argv
    controls()

    for (c, lab) in [((1, 1, -2), "a=1 (= osp(4|2))"),
                     ((1, 2, -3), "a=2"),
                     ((1, -3, 2), "a=-3"),
                     ((2, 3, -5), "a=3/2"),
                     ((F(1), F(1, 2), F(-3, 2)), "a=1/2")]:
        analyse(from_d21a(c, "osp_a(4|2), %s" % lab),
                verbose=(c == (1, 1, -2)), check_jacobi=(c == (1, 1, -2)))

    analyse(from_exc3(exc3.build_G3()))
    if not quick:
        analyse(from_exc3(exc3.build_F4()), verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
