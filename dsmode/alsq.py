#!/usr/bin/env python3
"""The squaring on g_x = Ker ad_x / Im ad_x for p = 2: Subsec. 1.4 of the text.

For p = 2 the subspace Im ad_x need not be closed under the squaring, so the
quotient g_x need not inherit it.  This script computes, for a Lie
superalgebra g over K = GF(2^k) given by its structure constants AND its
squaring, and for a homological x (odd, x^2 = 0):

  * Ker ad_x, Im ad_x, and the check Im subset Ker;
  * whether Im ad_x is an ideal of Ker ad_x with respect to the BRACKET;
  * the obstruction (eq. (5) of the text)

        theta_x : (Im ad_x)_od --> (g_x)_ev,   y |--> y^2 mod Im ad_x,

    which is additive and Frobenius-semilinear, hence vanishes iff it vanishes
    on a basis; g_x is a Lie superalgebra with squaring iff theta_x = 0.

Models:
  1) the 4-dimensional Lie superalgebra of Example 1.4.3     -- theta_x != 0;
  2) its T*-extension of Example 1.4.4, WITH an even NIS     -- theta_x != 0;
  3) gl(m|n) and sl(m|n) for p = 2, matrix model             -- theta_x = 0
     (Proposition 1.4.8), together with the identity [x,a]^2 = [x, a[x,a]]
     on which its proof rests.

Usage:  python3 alsq.py [k]      (k = 2 by default, K = GF(2^k))
"""
import sys
from itertools import combinations

from gf2k import GF, IRRED, rank, kernel

def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)


# --------------------------------------------------------------- algebra ----
class SLA:
    """Lie superalgebra over GF(2^k) by structure constants + squaring."""

    def __init__(self, name, parity, act, sqr, names=None):
        self.name = name
        self.parity = parity
        self.n = len(parity)
        self.act = act          # act[i][j] = coordinates of [e_i, e_j]
        self.sqr = sqr          # sqr[i] = coordinates of e_i^2 (odd i)
        self.names = names or ["e%d" % i for i in range(self.n)]
        self.EV = [i for i in range(self.n) if parity[i] == 0]
        self.OD = [i for i in range(self.n) if parity[i] == 1]

    def zero(self):
        return [GF(0)] * self.n

    def e(self, i):
        v = self.zero()
        v[i] = GF(1)
        return v

    def bracket(self, u, v):
        r = self.zero()
        for i in range(self.n):
            if not u[i]:
                continue
            for j in range(self.n):
                if not v[j]:
                    continue
                c = u[i] * v[j]
                row = self.act[i][j]
                for kk in range(self.n):
                    if row[kk]:
                        r[kk] = r[kk] + c * row[kk]
        return r

    def squaring(self, v):
        """v odd: v^2 = sum c_i^2 e_i^2 + sum_{i<j} c_i c_j [e_i, e_j]."""
        r = self.zero()
        idx = [i for i in range(self.n) if v[i]]
        for i in idx:
            c2 = v[i] * v[i]
            for kk in range(self.n):
                if self.sqr[i][kk]:
                    r[kk] = r[kk] + c2 * self.sqr[i][kk]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                c = v[i] * v[j]
                row = self.act[i][j]
                for kk in range(self.n):
                    if row[kk]:
                        r[kk] = r[kk] + c * row[kk]
        return r

    def is_odd(self, v):
        return all(not v[i] for i in self.EV)

    # ------------------------------------------------------------ axioms ---
    def check_axioms(self, sample=None):
        """Jacobi, [x^2,y] = [x,[x,y]], [x^2,x] = 0, and [z,z] = 0 (z even)."""
        n, bad = self.n, {"jacobi": 0, "sq_der": 0, "sq_x": 0, "alt": 0,
                          "parity": 0}
        rng = range(n) if sample is None else sample
        for i in rng:
            for j in range(n):
                for kk in range(n):
                    lhs = self.bracket(self.e(i), self.bracket(self.e(j), self.e(kk)))
                    rhs = [p + q for p, q in zip(
                        self.bracket(self.bracket(self.e(i), self.e(j)), self.e(kk)),
                        self.bracket(self.e(j), self.bracket(self.e(i), self.e(kk))))]
                    if any(p + q for p, q in zip(lhs, rhs)):
                        bad["jacobi"] += 1
        for i in self.OD:
            x = self.e(i)
            x2 = self.squaring(x)
            for j in range(n):
                lhs = self.bracket(x2, self.e(j))
                rhs = self.bracket(x, self.bracket(x, self.e(j)))
                if any(p + q for p, q in zip(lhs, rhs)):
                    bad["sq_der"] += 1
            if any(self.bracket(x2, x)):
                bad["sq_x"] += 1
            if any(self.sqr[i][kk] for kk in self.OD):
                bad["parity"] += 1          # x^2 must be even
        for i in self.EV:
            if any(self.bracket(self.e(i), self.e(i))):
                bad["alt"] += 1
        return bad

    # ------------------------------------------------------------- DS ------
    def ad_rows(self, x):
        return [self.bracket(x, self.e(i)) for i in range(self.n)]

    def im_basis(self, x, rows=None):
        rows = rows if rows is not None else self.ad_rows(x)
        rk, red = rank([r for r in rows if any(r)] or [self.zero()], self.n)
        return [r for r in red if any(r)]

    def ker_basis(self, x, rows=None):
        rows = rows if rows is not None else self.ad_rows(x)
        out = []
        for c in kernel(rows, self.n):
            v = self.zero()
            for i in range(self.n):
                if c[i]:
                    for kk in range(self.n):
                        v[kk] = v[kk] + c[i] * self.e(i)[kk]
            out.append(v)
        return out

    def in_span(self, v, red):
        """is v in the row space of the reduced list red?"""
        base = len(red)
        r, _ = rank([row[:] for row in red] + [v[:]], self.n)
        return r == base

    def theta(self, x, verbose=False):
        """The obstruction. Returns (dim of the image of theta_x, witnesses)."""
        rows = self.ad_rows(x)
        im = self.im_basis(x, rows)
        ker = self.ker_basis(x, rows)
        # Im subset Ker ?
        im_in_ker = all(not any(self.bracket(x, v)) for v in im)
        # Im an ideal of Ker w.r.t. the bracket ?
        ideal = True
        for z in ker:
            for v in im:
                if not self.in_span(self.bracket(z, v), im):
                    ideal = False
        # theta on the odd part of Im: split the reduced basis by parity.
        odd_im = []
        for v in im:
            ev = [v[i] if self.parity[i] == 0 else GF(0) for i in range(self.n)]
            od = [v[i] if self.parity[i] == 1 else GF(0) for i in range(self.n)]
            for part in (ev, od):
                pass
            if any(od):
                odd_im.append(od)
        # a basis of (Im)_od : Im is parity-graded (ad_x is odd), reduce again
        rk, odd_im = rank(odd_im, self.n) if odd_im else (0, [])
        witnesses = []
        cur = [r[:] for r in im]
        base = len(im)
        for y in odd_im:
            y2 = self.squaring(y)
            if not self.in_span(y2, im):
                witnesses.append((y, y2))
                r, red = rank(cur + [y2], self.n)
                if r > base:
                    cur, base = red, r
        return len(witnesses), witnesses, im, ker, im_in_ker, ideal

    def report(self, x, label):
        cnt, wit, im, ker, im_in_ker, ideal = self.theta(x)
        say("  %-22s rank ad_x=%-3d dim Ker=%-3d dim Im=%-3d "
            "Im<=Ker:%s ideal:%s theta_x=%s"
            % (label, len(im), len(ker), len(im),
               "yes" if im_in_ker else "NO", "yes" if ideal else "NO",
               "0" if cnt == 0 else "NON-ZERO (%d witnesses)" % cnt))
        for y, y2 in wit[:3]:
            say("      witness y = %s,  y^2 = %s  not in Im ad_x"
                % (self.show(y), self.show(y2)))
        return cnt


def show_vec(alg, v):
    out = []
    for i in range(alg.n):
        if v[i]:
            c = "" if v[i] == GF(1) else "%s*" % v[i]
            out.append(c + alg.names[i])
    return " + ".join(out) if out else "0"


SLA.show = lambda self, v: show_vec(self, v)


# ------------------------------------------ 1) the example of Example 1.4.3 ----
def example_143():
    """a, b even; x, y odd; [a,x] = y, x^2 = 0, y^2 = b, y and b central."""
    names = ["a", "b", "x", "y"]
    parity = [0, 0, 1, 1]
    n = 4
    z = [GF(0)] * n
    act = [[z[:] for _ in range(n)] for _ in range(n)]
    ey = [GF(0), GF(0), GF(0), GF(1)]
    act[0][2] = ey[:]     # [a,x] = y
    act[2][0] = ey[:]     # p = 2: no signs
    sqr = [z[:] for _ in range(n)]
    sqr[3] = [GF(0), GF(1), GF(0), GF(0)]   # y^2 = b
    return SLA("Example 1.4.3", parity, act, sqr, names)


# --------------------------------- 2) its T*-extension: NIS, theta_x != 0 ---
def tstar(h):
    """g = h (+) h^*, h^* an abelian ideal, coadjoint action, f^2 = 0.

    B((u,f),(v,g)) = f(v) + g(u) is an even NIS on g."""
    n = h.n
    N = 2 * n
    parity = h.parity + h.parity          # dual of a p-graded space, same p
    names = h.names + [s + "*" for s in h.names]
    z = [GF(0)] * N
    act = [[z[:] for _ in range(N)] for _ in range(N)]
    for i in range(n):
        for j in range(n):
            # [e_i, e_j] inside h
            act[i][j] = h.act[i][j] + [GF(0)] * n
            # [e_i, f_j](e_l) = f_j([e_i, e_l])  ==>  coefficient of f_l
            co = [GF(0)] * N
            for l in range(n):
                co[n + l] = h.act[i][l][j]
            act[i][n + j] = co
            act[n + j][i] = co[:]
    sqr = [z[:] for _ in range(N)]
    for i in range(n):
        if h.parity[i] == 1:
            sqr[i] = h.sqr[i] + [GF(0)] * n
    g = SLA("T*-extension of " + h.name, parity, act, sqr, names)

    def B(u, v):
        r = GF(0)
        for i in range(n):
            r = r + u[n + i] * v[i] + v[n + i] * u[i]
        return r
    g.B = B
    return g


def check_nis(g):
    """symmetry, evenness, nondegeneracy, invariance of g.B."""
    n = g.n
    bad = {"sym": 0, "even": 0, "inv": 0}
    for i in range(n):
        for j in range(n):
            if g.B(g.e(i), g.e(j)) != g.B(g.e(j), g.e(i)):
                bad["sym"] += 1
            if g.parity[i] != g.parity[j] and g.B(g.e(i), g.e(j)):
                bad["even"] += 1
            for kk in range(n):
                lhs = g.B(g.bracket(g.e(i), g.e(j)), g.e(kk))
                rhs = g.B(g.e(i), g.bracket(g.e(j), g.e(kk)))
                if lhs != rhs:
                    bad["inv"] += 1
    gram = [[g.B(g.e(i), g.e(j)) for j in range(n)] for i in range(n)]
    rk, _ = rank(gram, n)
    return bad, rk


# ------------------------------------------- 3) gl(m|n), sl(m|n) at p = 2 ----
def mat_mul(A, B, N):
    return [[sum((A[i][k] * B[k][j] for k in range(N)), GF(0))
             for j in range(N)] for i in range(N)]


def mat_add(A, B, N):
    return [[A[i][j] + B[i][j] for j in range(N)] for i in range(N)]


def gl_model(m, nn, special=False):
    """Basis E_ij of gl(m|nn) (or of sl = ker str) at p = 2, matrix squaring."""
    N = m + nn

    def par(i, j):
        return (0 if i < m else 1) ^ (0 if j < m else 1)

    mats, names, parity = [], [], []
    if not special:
        for i in range(N):
            for j in range(N):
                A = [[GF(0)] * N for _ in range(N)]
                A[i][j] = GF(1)
                mats.append(A)
                names.append("E%d%d" % (i + 1, j + 1))
                parity.append(par(i, j))
    else:
        # sl = ker str; at p = 2, str = tr.  Basis: all E_ij with i != j,
        # and the diagonal matrices of zero trace.
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                A = [[GF(0)] * N for _ in range(N)]
                A[i][j] = GF(1)
                mats.append(A)
                names.append("E%d%d" % (i + 1, j + 1))
                parity.append(par(i, j))
        for i in range(N - 1):
            A = [[GF(0)] * N for _ in range(N)]
            A[i][i] = GF(1)
            A[i + 1][i + 1] = GF(1)
            mats.append(A)
            names.append("H%d" % (i + 1))
            parity.append(0)
    n = len(mats)

    def coords(A):
        v = [GF(0)] * n
        rest = [row[:] for row in A]
        for idx, M in enumerate(mats):
            # find the leading position of M
            pos = None
            for i in range(N):
                for j in range(N):
                    if M[i][j]:
                        pos = (i, j)
                        break
                if pos:
                    break
            i, j = pos
            c = rest[i][j]
            if c:
                v[idx] = c
                for a in range(N):
                    for b in range(N):
                        if M[a][b]:
                            rest[a][b] = rest[a][b] + c * M[a][b]
        assert not any(any(r) for r in rest), "not in the algebra"
        return v

    z = [GF(0)] * n
    act = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            P = mat_add(mat_mul(mats[i], mats[j], N),
                        mat_mul(mats[j], mats[i], N), N)
            act[i][j] = coords(P)
    sqr = [z[:] for _ in range(n)]
    for i in range(n):
        if parity[i] == 1:
            sqr[i] = coords(mat_mul(mats[i], mats[i], N))
    g = SLA(("sl" if special else "gl") + "(%d|%d)" % (m, nn),
            parity, act, sqr, names)
    g.mats, g.N, g.coords = mats, N, coords
    return g


def identity_check(g, trials):
    """[x,a]^2 = [x, a[x,a]] for x odd with x^2 = 0 and a even (matrices)."""
    N = g.N
    bad = 0
    for x, a in trials:
        X = mat_of(g, x)
        A = mat_of(g, a)
        assert not any(any(r) for r in mat_mul(X, X, N)), "x^2 != 0"
        Y = mat_add(mat_mul(X, A, N), mat_mul(A, X, N), N)      # [x,a]
        lhs = mat_mul(Y, Y, N)
        AY = mat_mul(A, Y, N)
        rhs = mat_add(mat_mul(X, AY, N), mat_mul(AY, X, N), N)  # [x, aY]
        if any(any(c for c in row) for row in mat_add(lhs, rhs, N)):
            bad += 1
    return bad


def mat_of(g, v):
    N = g.N
    M = [[GF(0)] * N for _ in range(N)]
    for idx in range(g.n):
        if v[idx]:
            for i in range(N):
                for j in range(N):
                    if g.mats[idx][i][j]:
                        M[i][j] = M[i][j] + v[idx] * g.mats[idx][i][j]
    return M


def sweep(g, maxsum=3, cap=None, gray=False):
    """All F_2-combinations of at most `maxsum` odd basis vectors that are
    homological; report how many have theta_x != 0."""
    tot = homs = bad = 0
    worst = []
    for r in range(1, maxsum + 1):
        for combo in combinations(g.OD, r):
            tot += 1
            x = g.zero()
            for i in combo:
                x[i] = GF(1)
            if any(g.squaring(x)):
                continue
            homs += 1
            cnt, wit, im, ker, ik, ideal = g.theta(x)
            if cnt or not ik or not ideal:
                bad += 1
                worst.append(("+".join(g.names[i] for i in combo), cnt, ik, ideal))
            if cap and homs >= cap:
                return tot, homs, bad, worst
    return tot, homs, bad, worst


# ------------------------------------------------------------------ main ----
def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    GF.k, GF.mod = k, IRRED[k]
    A = GF(0b10)                      # a = t, a generator of GF(2^k)
    say("K = GF(2^%d), a = t;  all arithmetic exact." % k)

    say("\n=== 1. Example 1.4.3: the squaring can be lost ===")
    h = example_143()
    say("  axioms:", h.check_axioms())
    say("  basis: %s, parity %s, [a,x]=y, x^2=0, y^2=b" % (h.names, h.parity))
    h.report(h.e(2), "x")
    say("  ==> Im ad_x = <y> is an ideal of Ker ad_x for the BRACKET, but")
    say("      y^2 = b is not in it: g_x inherits no squaring.")

    say("\n=== 2. Example 1.4.4: the same, inside an algebra WITH an even NIS ===")
    g = tstar(h)
    say("  axioms:", g.check_axioms())
    bad, rk = check_nis(g)
    say("  NIS: violations %s, rank of the Gram matrix %d of %d" % (bad, rk, g.n))
    g.report(g.e(2), "x")
    say("  ==> an even NIS does NOT make theta_x vanish.")

    say("\n=== 3. Proposition 1.4.8: gl(m|n) and sl(m|n) at p = 2 (matrix model) ===")
    for (m, nn) in [(1, 1), (2, 1), (2, 2), (1, 2), (3, 1), (2, 3)]:
        for special in (False, True):
            g = gl_model(m, nn, special)
            ax = g.check_axioms(sample=range(min(g.n, 6)))
            # homological elements: the x_k = sum E_{2j-1,2j}, and all
            # single odd basis vectors, and all their sums of size <= 3
            tot, homs, badcnt, worst = sweep(g, maxsum=3)
            say("  %-9s dim=%-3d axioms(sample)=%s  homological x tried: %d, "
                "theta_x != 0 for %d" % (g.name, g.n, ax, homs, badcnt))
            if worst:
                say("     ", worst[:5])
    say("  identity [x,a]^2 = [x, a[x,a]] (associative model):")
    for (m, nn) in [(2, 2), (3, 2), (2, 3)]:
        g = gl_model(m, nn)
        trials = []
        for i in g.OD:
            x = g.e(i)
            if any(g.squaring(x)):
                continue
            for j in g.EV:
                trials.append((x, g.e(j)))
        for i, j in combinations(g.OD, 2):
            x = g.zero()
            x[i] = GF(1)
            x[j] = GF(1)
            if any(g.squaring(x)):
                continue
            for l in g.EV:
                trials.append((x, g.e(l)))
        say("    gl(%d|%d): %d pairs (x,a), failures: %d"
            % (m, nn, len(trials), identity_check(g, trials)))

    say("\ndone.")


if __name__ == "__main__":
    main()
