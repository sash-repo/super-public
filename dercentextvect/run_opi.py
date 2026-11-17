"""Run the checks of opi.py; log to run_opi.log."""

import os
import sys
from itertools import combinations

from opi import (E, bracket, coords, echelon, rank, reduce_by, nullspace,
                 same_span,
                 LieAlg, der_dim, der_brute, o_Pi, chevalley_span, psl4)

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


CHECKS = []


def check(name, got, want):
    ok = (got == want)
    CHECKS.append(ok)
    say("  [%s] %-58s got %s, want %s" % ("ok" if ok else "FAIL", name, got, want))
    return ok


def note(name, got):
    say("  [--] %-58s %s" % (name, got))


# --------------------------------------------------------------- self-tests

def selftests():
    say("\n=== self-tests ===")
    # abelian of dim 3: Der = gl(3)
    sc = [[0] * 3 for _ in range(3)]
    check("Der(abelian_3) = gl(3)", der_dim(sc, 3), 9)
    check("Der(abelian_3), brute force", der_brute(sc, 3), 9)

    # sl(2) over F_2: basis e,h,f with [e,f]=h, [h,e]=[h,f]=0 (p=2)
    N = 2
    b = [E(N, 1, 2), E(N, 1, 1) ^ E(N, 2, 2), E(N, 2, 1)]
    g = LieAlg(N, b, "sl(2)")
    check("dim sl(2)/F_2", g.dim, 3)
    check("dim center sl(2)/F_2", len(g.center()), 1)
    check("Der sl(2)/F_2: solver = brute force",
          der_dim(g.struct(), g.dim), der_brute(g.struct(), g.dim))

    # gl(2) over F_2, dim 4, another solver-vs-brute-force comparison
    b = [E(2, i, j) for i in (1, 2) for j in (1, 2)]
    g = LieAlg(2, b, "gl(2)")
    check("Der gl(2)/F_2: solver = brute force",
          der_dim(g.struct(), g.dim), der_brute(g.struct(), g.dim))

    # a derivation really is a derivation: spot-check the solver's output on
    # o^{(2)}_Pi(6) by testing ad_x for x in the algebra
    g = chevalley_span(3)
    sc = g.struct()
    n = g.dim
    inner = []
    for k in range(n):
        D = 0
        for j in range(n):
            c = sc[k][j]
            for a in range(n):
                if (c >> a) & 1:
                    D |= 1 << (a * n + j)
        inner.append(D)
    ok = True
    for D in inner:
        for b_ in range(n):
            for c_ in range(b_ + 1, n):
                lhs = 0
                for k in range(n):
                    if (sc[b_][c_] >> k) & 1:
                        for a in range(n):
                            if (D >> (a * n + k)) & 1:
                                lhs ^= 1 << a
                for m in range(n):
                    if (D >> (m * n + b_)) & 1:
                        lhs ^= sc[m][c_]
                    if (D >> (m * n + c_)) & 1:
                        lhs ^= sc[b_][m]
                if lhs:
                    ok = False
    check("every ad_x satisfies the Leibniz system", ok, True)


# ------------------------------------------------------- the algebras o_Pi

def algebras(n):
    say("\n=== o_Pi(%d), p = 2 ===" % (2 * n))
    g = o_Pi(n)
    note("dim o_Pi(%d)" % (2 * n), g.dim)
    g1 = g.derived()
    note("dim o_Pi(%d)^{(1)}" % (2 * n), g1.dim)
    g2 = g1.derived()
    note("dim o_Pi(%d)^{(2)}" % (2 * n), g2.dim)

    ch = chevalley_span(n)
    note("dim Span(Chevalley basis mod 2)", ch.dim)
    same = same_span(ch.basis, g2.basis)
    check("Span(Chevalley) = o_Pi(%d)^{(2)}" % (2 * n), same, True)

    # 2n(n-1) roots + (n-1) independent h's
    check("dim = 2n(n-1) + (n-1) = %d" % (2 * n * (n - 1) + n - 1),
          g2.dim, 2 * n * (n - 1) + n - 1)

    # h_{n-1} = h_n mod 2
    N = 2 * n
    h_nm1 = (E(N, n - 1, n - 1) ^ E(N, n, n)
             ^ E(N, N - 1, N - 1) ^ E(N, N, N))
    h_last = (E(N, n - 1, n - 1) ^ E(N, n, n)
              ^ E(N, N - 1, N - 1) ^ E(N, N, N))
    check("h_{n-1} = h_n mod 2", h_nm1 == h_last, True)

    ctr = g2.center()
    note("dim center o_Pi(%d)^{(2)}" % (2 * n), len(ctr))
    if ctr:
        I = 0
        for i in range(1, N + 1):
            I |= E(N, i, i)
        cvecs = []
        for x in ctr:
            M = 0
            for k in range(g2.dim):
                if (x >> k) & 1:
                    M ^= g2.basis[k]
            cvecs.append(M)
        check("the center is spanned by the unit matrix",
              same_span(cvecs, [I]), True)
    return g2


def derivations(g2, label):
    sc = g2.struct()
    n = g2.dim
    dder = der_dim(sc, n)
    ctr = len(g2.center())
    inner = n - ctr
    note("dim %s" % label, n)
    note("dim center", ctr)
    note("dim Der %s" % label, dder)
    note("dim ad(%s) = dim - dim center" % label, inner)
    note("dim out der %s = dim H^1(%s;%s)" % (label, label, label), dder - inner)
    return dder - inner


def quotient_by_centre(g2):
    """Structure constants of g2/c, c the center."""
    sc = g2.struct()
    n = g2.dim
    c = echelon(g2.center())
    keep = []
    for k in range(n):
        v = 1 << k
        if reduce_by(echelon(c + keep), v):
            keep.append(v)
    # basis of the quotient: images of the chosen e_k
    idx = [v.bit_length() - 1 for v in keep]
    m = len(idx)

    def project(mask):
        """Reduce a coordinate vector modulo c, express in the kept indices."""
        v = mask
        for b in c:
            p = b.bit_length() - 1
            if (v >> p) & 1:
                v ^= b
        out = 0
        for a, k in enumerate(idx):
            if (v >> k) & 1:
                out |= 1 << a
        assert v == sum(1 << k for a, k in enumerate(idx) if (out >> a) & 1)
        return out

    q = [[0] * m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            q[a][b] = project(sc[idx[a]][idx[b]])
    return q, m


def q1():
    say("\n=== Q1: dim out der (o^{(2)}_Pi(8)/c) = 26 ? ===")
    g2 = algebras(4)
    sc, m = quotient_by_centre(g2)
    note("dim o^{(2)}_Pi(8)/c", m)
    # the quotient is centerless?
    rows = []
    for j in range(m):
        for k in range(m):
            r = 0
            for i in range(m):
                if (sc[i][j] >> k) & 1:
                    r |= 1 << i
            if r:
                rows.append(r)
    ctr = len(nullspace(rows, m))
    note("dim center of the quotient", ctr)
    d = der_dim(sc, m)
    note("dim Der (o^{(2)}_Pi(8)/c)", d)
    out = d - (m - ctr)
    check("dim out der (o^{(2)}_Pi(8)/c)", out, 26)
    return out


def q3():
    say("\n=== Q3: dim out der o^{(2)}_Pi(6) = 7 ? ===")
    g2 = algebras(3)
    out = derivations(g2, "o^{(2)}_Pi(6)")
    check("dim out der o^{(2)}_Pi(6)", out, 7)
    return out


def q2():
    say("\n=== Q2: psl(4) ~ o^{(2)}_Pi(6) via the correspondence (44) ? ===")
    N4, sl, names, I = psl4()
    N6 = 6

    # the map (44), on the six upper E^{i,j} and the two Cartan elements
    upper = {
        (2, 3): [(1, 2), (5, 4)],
        (1, 2): [(2, 3), (6, 5)],
        (3, 4): [(2, 6), (3, 5)],
        (1, 3): [(1, 3), (6, 4)],
        (2, 4): [(1, 6), (3, 4)],
        (1, 4): [(1, 5), (2, 4)],
    }
    img = {}
    for (i, j), cells in upper.items():
        img[(i, j)] = 0
        for (a, b) in cells:
            img[(i, j)] ^= E(N6, a, b)
        # "the transposed basis elements go to the transposed images"
        img[(j, i)] = 0
        for (a, b) in cells:
            img[(j, i)] ^= E(N6, b, a)
    h1 = E(N6, 2, 2) ^ E(N6, 3, 3) ^ E(N6, 5, 5) ^ E(N6, 6, 6)
    h2 = E(N6, 1, 1) ^ E(N6, 2, 2) ^ E(N6, 4, 4) ^ E(N6, 5, 5)

    src, tgt = [], []
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                src.append(E(N4, i, j))
                tgt.append(img[(i, j)])
    src.append(E(N4, 1, 1) ^ E(N4, 2, 2))
    tgt.append(h1)
    src.append(E(N4, 2, 2) ^ E(N4, 3, 3))
    tgt.append(h2)
    src.append(I)
    tgt.append(0)                       # phi(I) = 0: the map factors through psl

    check("the 15 source vectors span sl(4)", rank(src), 15)
    check("dim image = 14 = dim psl(4)", rank(tgt), 14)

    o2 = chevalley_span(3)
    inside = all(reduce_by(o2.basis, t) == 0 for t in tgt)
    check("the image lies in o^{(2)}_Pi(6)", inside, True)
    check("the image is all of o^{(2)}_Pi(6)", rank(tgt) == o2.dim, True)

    # homomorphism test on every pair of the 15 source basis vectors
    sbasis = echelon(src)
    # express phi on the echelon basis: build the linear map by coordinates
    def phi(x):
        c = coords(src, x)
        assert c is not None
        out = 0
        for k in range(len(src)):
            if (c >> k) & 1:
                out ^= tgt[k]
        return out

    # phi must be well defined: src is not independent (I = sum of the others?)
    check("the 16 source vectors are dependent (rank 15)", rank(src), 15)

    bad = []
    for a in range(len(src)):
        for b in range(a + 1, len(src)):
            lhs = phi(bracket(N4, src[a], src[b]))
            rhs = bracket(N6, tgt[a], tgt[b])
            if lhs != rhs:
                bad.append((a, b))
    check("phi([x,y]) = [phi x, phi y] on all %d pairs"
          % (len(src) * (len(src) - 1) // 2), len(bad), 0)
    if bad:
        say("      failing pairs (source indices):", bad[:20],
            "..." if len(bad) > 20 else "")
        nm = names + ["I"]
        for (a, b) in bad[:10]:
            say("      [%s , %s]" % (nm[a], nm[b]))
    return len(bad) == 0


def simple_relative(n):
    """The simple relative of o_Pi(2n): o^{(2)}_Pi(2n), modulo its center."""
    g2 = chevalley_span(n)
    sc, m = quotient_by_centre(g2)
    return g2, sc, m


def sweep(ns):
    say("\n=== dim out der of the simple relative of o_Pi(2n) ===")
    for n in ns:
        g2, sc, m = simple_relative(n)
        g3 = g2.derived()
        perfect = same_span(g3.basis, g2.basis)
        rows = []
        for j in range(m):
            for k in range(m):
                r = 0
                for i in range(m):
                    if (sc[i][j] >> k) & 1:
                        r |= 1 << i
                if r:
                    rows.append(r)
        ctr = len(nullspace(rows, m))
        d = der_dim(sc, m)
        out = d - (m - ctr)
        say("  n=%d: dim o^{(2)}_Pi(%d) = %2d, center %d, simple relative %2d, "
            "perfect %s, dim Der %3d, dim out %2d"
            % (n, 2 * n, g2.dim, len(g2.center()), m, perfect, d, out))


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_opi.log"), "w")
    say("o_Pi(2n) at p = 2 --- Subsecs. 3.3, 3.4 of the text")
    selftests()
    q1()
    q2()
    q3()
    sweep([3, 4, 5])
    say("\n=== summary: %d/%d checks passed ==="
        % (sum(CHECKS), len(CHECKS)))
    LOG.close()
    return 0 if all(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
