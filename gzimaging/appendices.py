"""Independent check of the quantitative claims of the appendices of the imaging text
(numbers first computed by H. Andersen).

Nothing here re-runs her code: each claim is re-derived from scratch, symbolically
where a symbolic derivation exists, and the outcome is compared against the number
she quotes.  Run:  python3 appendices.py   (writes appendices.log)

CONVENTION.  omega^{ij} is the Poisson bivector, fixed by requiring that
{f,h} = omega^{ij} d_i f d_j h be the standard Poisson bracket: for omega = W eps
with eps = [[0,1],[-1,0]] that is omega^{ij} = eps/W, which is MINUS the matrix
inverse of omega_{ij} in dimension 2.  The pairing <A,B> is quadratic in
omega^{ij}, so that sign does not reach it, and the index expression of eq. (2)
of the draft comes out as PLUS Grozman's own g = ac' - 2bb' + ca' (p. 34 of [G3]),
whence <II,II> = +2K.  H. Andersen's code gave -2K because its omega_pair() returns
tr(W A W B), and tr(W A W B) = -omega^{ka} omega^{lb} A_kl B_ab: the index
expression contracts the FIRST index of omega against the FIRST index of A,
which costs one transposition of the antisymmetric omega.  The mathematics is
the same either way; the sign printed in the paper must be the one eq. (2) gives.

Sections
  A   the omega-pairing on a surface:  <II,II> = -2K, tr(omega^{-1}II) = 0,
      Cayley-Hamilton, and the identification of the pairing with Grozman's
      own leading block g
  B   naturality under a NON-area-preserving reparametrization
  C   the pointwise invariants: SL(2,R) = SO(2,1)^0 and the first fundamental
      theorem; orbit dimensions; the spherical cap / ridge pair
  D   homogeneity under Sigma -> lambda Sigma
  E   Subsection 2.3: nondegeneracy is b.e != 0; the rotation stabilizer
  F   Section 5: the exact finite-s SNR law, the table, the baselines, the
      smoothing scale, the optimal scale s* = L sqrt(2k+1), the discrete sum,
      and a ramp-filtered CT noise spectrum
"""

import sys

import numpy as np
import sympy as sp

LOG = None
PASS = []


def say(s=""):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def head(s):
    say()
    say("== " + s + " ==")


def check(name, ok, detail=""):
    PASS.append((name, bool(ok)))
    say("  [%s] %s%s" % ("ok" if ok else "FAIL", name, ("   " + detail) if detail else ""))


EPS = sp.Matrix([[0, 1], [-1, 0]])


def raise_idx(om):
    """the Poisson bivector omega^{ij} of a 2-form omega_{ij} on a surface."""
    return -om.inv()


def pairing(A, B, om=EPS):
    """<A,B> = omega^{ka} omega^{lb} A_kl B_ab."""
    W = raise_idx(om)
    return sp.simplify(sum(W[k, a] * W[l, b] * A[k, l] * B[a, b]
                           for k in range(2) for a in range(2)
                           for l in range(2) for b in range(2)))


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------

def surf_data(r, x, y):
    """g, II, K, H, omega of a parametrized surface r(x,y) in R^3."""
    rx, ry = r.diff(x), r.diff(y)
    g = sp.Matrix([[rx.dot(rx), rx.dot(ry)], [rx.dot(ry), ry.dot(ry)]])
    g = sp.simplify(g)
    nn = rx.cross(ry)
    nrm = sp.sqrt(sp.simplify(nn.dot(nn)))
    n = nn / nrm
    II = sp.Matrix([[r.diff(x, 2).dot(n), r.diff(x, y).dot(n)],
                    [r.diff(x, y).dot(n), r.diff(y, 2).dot(n)]])
    II = sp.simplify(II)
    K = sp.simplify(II.det() / g.det())
    H = sp.simplify((g.inv() * II).trace() / 2)
    om = sp.sqrt(sp.simplify(g.det())) * EPS
    return g, II, K, H, om


def graph(f, u, v):
    return sp.Matrix([u, v, f])


# ---------------------------------------------------------------------------
# A.  the omega-pairing on a surface
# ---------------------------------------------------------------------------

def section_A():
    head("A. the omega-pairing on a surface")

    a, b, c, a1, b1, c1 = sp.symbols("a b c a1 b1 c1")
    A = sp.Matrix([[a, b], [b, c]])
    B = sp.Matrix([[a1, b1], [b1, c1]])

    p = sp.expand(pairing(A, B))
    grozman_g = a * c1 - 2 * b * b1 + c * a1          # Grozman's g, p. 34 of [G3]
    check("<A,B> = ac' - 2bb' + ca' = Grozman's own leading block g",
          sp.simplify(p - grozman_g) == 0, "got %s" % p)
    polar = sp.expand(A.trace() * B.trace() - (A * B).trace())
    check("<A,B> = tr A tr B - tr AB", sp.simplify(p - polar) == 0)
    check("<A,A> = 2 det A", sp.simplify(pairing(A, A) - 2 * A.det()) == 0)

    S = raise_idx(EPS) * A
    check("tr(omega^{-1}A) = 0 for symmetric A: no linear contraction exists",
          sp.simplify(S.trace()) == 0)
    check("Cayley-Hamilton for the traceless S: S^2 + det(S) 1 = 0",
          sp.simplify(S * S + S.det() * sp.eye(2)) == sp.zeros(2, 2))
    check("<A,B> = -tr(S_A S_B): the index expression of (2) is MINUS the trace form",
          sp.simplify(pairing(A, B) + (raise_idx(EPS) * A * raise_idx(EPS) * B).trace()) == 0)
    say("  (whence the value -2K: its omega_pair() returns the trace form.  The")
    say("   content is the same, the sign printed in the paper must follow eq. (2).)")

    # on a genuine surface, in an arbitrary (non-area-preserving) chart
    u, v = sp.symbols("u v", real=True)
    f = u**2 / 2 + sp.Rational(3, 2) * v**2 + u * v / 3 + u**3 / 5
    g, II, K, H, om = surf_data(graph(f, u, v), u, v)
    check("<II,II>_omega = 2K  (arbitrary chart, symbolic)",
          sp.simplify(pairing(II, II, om) - 2 * K) == 0)
    check("det(omega^{-1}II) = K", sp.simplify((raise_idx(om) * II).det() - K) == 0)

    t = sp.Symbol("t")
    mixed = sp.Rational(1, 2) * sp.diff(sp.det(A + t * B), t).subs(t, 0)
    check("<A,B> = 2 x (the mixed determinant), the polarization of det",
          sp.simplify(pairing(A, B) - 2 * mixed) == 0)
    check("<A,g>_eps = 2H in a g-orthonormal frame (H needs g as an argument)",
          sp.simplify(pairing(A, sp.eye(2)) - (a + c)) == 0)
    return f, u, v


# ---------------------------------------------------------------------------
# B.  naturality under a non-area-preserving reparametrization
# ---------------------------------------------------------------------------

def section_B(f, u, v):
    head("B. naturality under a NON-area-preserving change of chart")

    a, bb = sp.symbols("a b", real=True)
    U, V = 2 * a + a**3, bb / 3          # phi: (a,b) -> (u,v)
    J = sp.Matrix([[sp.diff(U, a), sp.diff(U, bb)], [sp.diff(V, a), sp.diff(V, bb)]])
    say("  phi: u = 2a + a^3, v = b/3;  det Dphi = %s  (not 1, not even constant)"
        % sp.simplify(J.det()))

    r = graph(f, u, v)
    g1, II1, K1, H1, om1 = surf_data(r, u, v)
    old = pairing(II1, II1, om1)

    # THE SAME surface, parametrized through phi
    r2 = r.subs({u: U, v: V})
    g2, II2, K2, H2, om2 = surf_data(r2, a, bb)
    new = pairing(II2, II2, om2)

    sub = {u: U, v: V}
    check("K in the new chart = K composed with phi", sp.simplify(K2 - K1.subs(sub)) == 0)
    check("H likewise", sp.simplify(H2 - H1.subs(sub)) == 0)
    d = sp.simplify(new - old.subs(sub))
    check("<II,II> likewise, although phi does not preserve the area",
          d == 0, "difference = %s" % d)
    say("  => the omega-contraction is natural under ALL reparametrizations, so the")
    say("     'proper subgroup' of Proposition 3.3 fails for the INDUCED area form.")

    def poisson(F, G, om, x, y):
        W = raise_idx(om)
        dF, dG = [sp.diff(F, x), sp.diff(F, y)], [sp.diff(G, x), sp.diff(G, y)]
        return sum(W[i, j] * dF[i] * dG[j] for i in range(2) for j in range(2))

    pb_old = poisson(K1, H1, om1, u, v)
    pb_new = poisson(K2, H2, om2, a, bb)
    # compared at sample points: the symbolic difference is a rational function of
    # degree ~16 whose simplification costs more than it proves
    pts = [(sp.Rational(1, 3), sp.Rational(2, 5)), (sp.Rational(-2, 7), sp.Rational(1, 2)),
           (sp.Rational(3, 4), sp.Rational(-1, 3))]
    ok = True
    for A0, B0 in pts:
        lhs = pb_new.subs({a: A0, bb: B0})
        rhs = pb_old.subs({u: U.subs({a: A0}), v: V.subs({bb: B0})})
        ok = ok and sp.simplify(sp.nsimplify(lhs - rhs)) == 0
    check("{K,H}_omega is natural too (the third ingredient of (2)), at %d points"
          % len(pts), ok)


# ---------------------------------------------------------------------------
# C.  the pointwise invariants
# ---------------------------------------------------------------------------

def section_C():
    head("C. the pointwise invariants of (omega, II) and of a pair")

    a, b, c, a1, b1, c1 = sp.symbols("a b c a1 b1 c1")
    p, q, r, s = sp.symbols("p q r s")
    A = sp.Matrix([[a, b], [b, c]])
    B = sp.Matrix([[a1, b1], [b1, c1]])
    M = sp.Matrix([[p, q], [r, s]])

    check("SL(2) preserves eps",
          sp.simplify(M.T * EPS * M - (p * s - q * r) * EPS) == sp.zeros(2, 2))
    check("det is SL(2)-invariant on Sym^2, so SL(2,R) maps onto SO(2,1)^0",
          sp.simplify(sp.expand((M.T * A * M).det() - (p * s - q * r)**2 * A.det())) == 0)
    check("<.,.> is the associated bilinear form: <MA,MB> = det(M)^2 <A,B>",
          sp.simplify(pairing(M.T * A * M, M.T * B * M)
                      - (p * s - q * r)**2 * pairing(A, B)) == 0)
    say("  The quadratic form det = ac - b^2 on the 3-space Sym^2 is nondegenerate of")
    say("  signature (1,2), so the image of SL(2,R) in GL(Sym^2) is SO(2,1)^0.  By the")
    say("  first fundamental theorem for the orthogonal group, the polynomial invariants")
    say("  of ONE vector are generated by its square and of TWO by the three pairwise")
    say("  products; two vectors in a 3-space admit no determinant, so SO-invariants =")
    say("  O-invariants.  Hence the ring is generated by <A,A>, <A,B>, <B,B> -- integral")
    say("  generation, not merely functional independence.")

    def orbit_dim(vs):
        gens = [np.array([[1., 0.], [0., -1.]]), np.array([[0., 1.], [0., 0.]]),
                np.array([[0., 0.], [1., 0.]])]
        cols = []
        for X in gens:
            col = []
            for Vm in vs:
                D = X.T @ Vm + Vm @ X
                col += [D[0, 0], D[0, 1], D[1, 1]]
            cols.append(col)
        return np.linalg.matrix_rank(np.array(cols).T, tol=1e-9)

    rng = np.random.default_rng(823)
    d1, d2 = set(), set()
    for _ in range(400):
        A1 = rng.normal(size=(2, 2)); A1 = A1 + A1.T
        B1 = rng.normal(size=(2, 2)); B1 = B1 + B1.T
        d1.add(orbit_dim([A1]))
        d2.add(orbit_dim([A1, B1]))
    check("one form: ambient 3, generic orbit 2 -> exactly 1 invariant (K)",
          max(d1) == 2, "orbit dims seen %s" % sorted(d1))
    check("a pair: ambient 6, generic orbit 3 -> exactly 3 invariants",
          max(d2) == 3, "orbit dims seen %s" % sorted(d2))

    inv = [pairing(A, A), pairing(A, B), pairing(B, B)]
    Jac = sp.Matrix([[sp.diff(i, x) for x in (a, b, c, a1, b1, c1)] for i in inv])
    check("their differentials have rank 3 at a generic point",
          Jac.subs({a: 1, b: 2, c: 5, a1: 3, b1: -1, c1: 2}).rank() == 3)

    # the spherical cap and the ridge
    M0 = sp.Matrix([[sp.sqrt(2), 0], [0, 1 / sp.sqrt(2)]])
    cap = sp.eye(2)
    ridge = sp.simplify(M0.T * cap * M0)
    check("M = diag(sqrt2, 1/sqrt2) is symplectic (det = 1)", sp.simplify(M0.det() - 1) == 0)
    check("it carries II = 1 to diag(2, 1/2)",
          ridge == sp.Matrix([[2, 0], [0, sp.Rational(1, 2)]]))
    check("K agrees (1 = 1) while H does not (1 against 5/4)",
          cap.det() == ridge.det() == 1 and cap.trace() / 2 == 1
          and ridge.trace() / 2 == sp.Rational(5, 4))
    check("<II,II> agrees as it must: %s = %s"
          % (pairing(cap, cap), pairing(ridge, ridge)),
          sp.simplify(pairing(cap, cap) - pairing(ridge, ridge)) == 0)

    # the two quadrics really are surfaces with these data at the origin
    x, y = sp.symbols("x y", real=True)
    for name, f, kap in (("cap", (x**2 + y**2) / 2, (1, 1)),
                         ("ridge", x**2 + y**2 / 4, (2, sp.Rational(1, 2)))):
        g, II, K, H, om = surf_data(graph(f, x, y), x, y)
        g0 = g.subs({x: 0, y: 0})
        II0 = II.subs({x: 0, y: 0})
        check("  the quadric %s has g = 1 and II = diag%s at the origin"
              % (name, kap), g0 == sp.eye(2) and II0 == sp.diag(*kap))


# ---------------------------------------------------------------------------
# D.  homogeneity
# ---------------------------------------------------------------------------

def section_D():
    head("D. homogeneity under Sigma -> lambda Sigma")

    lam, u, v = sp.symbols("lambda u v", positive=True)
    f = u**2 / 2 + sp.Rational(3, 2) * v**2 + u * v / 3
    g, II, K, H, om = surf_data(graph(f, u, v), u, v)
    gl, IIl, oml = lam**2 * g, lam * II, lam**2 * om
    check("under a rescaling g, omega -> lambda^2, II -> lambda; then K has weight -2",
          sp.simplify(IIl.det() / gl.det() - K / lam**2) == 0)
    check("H has weight -1", sp.simplify((gl.inv() * IIl).trace() / 2 - H / lam) == 0)
    check("<II,II> has weight -2",
          sp.simplify(pairing(IIl, IIl, oml) - pairing(II, II, om) / lam**2) == 0)

    say("  Index count for a term of hGz(II,II') carrying p factors omega^{ij} and q")
    say("  factors omega_{ij}: the two arguments and the two derivatives supply 6 lower")
    say("  indices, the target keeps 2, so 2p - 2q = 4, and the weight is")
    say("  lambda^{2-2p+2q} = lambda^{2-2(p-q)} = lambda^{-2}, whatever q may be.")
    for q in range(4):
        p = q + 2
        check("  p = %d, q = %d gives weight lambda^{%d}" % (p, q, 2 - 2 * p + 2 * q),
              2 - 2 * p + 2 * q == -2)


# ---------------------------------------------------------------------------
# E.  Subsection 2.3
# ---------------------------------------------------------------------------

def section_E():
    head("E. Subsection 2.3: a nondegenerate 2-form on (x,y,z,t)")

    b1, b2, b3, e1, e2, e3 = sp.symbols("b1 b2 b3 e1 e2 e3", real=True)
    bvec, evec = [b1, b2, b3], [e1, e2, e3]
    Om = sp.zeros(4, 4)
    for (i, j), (k, sg) in {(0, 1): (2, 1), (1, 2): (0, 1), (0, 2): (1, -1)}.items():
        Om[i, j] = sg * bvec[k]
        Om[j, i] = -Om[i, j]
    for i in range(3):
        Om[3, i], Om[i, 3] = evec[i], -evec[i]
    det = sp.factor(sp.expand(Om.det()))
    bdote = b1 * e1 + b2 * e2 + b3 * e3
    check("det(omega) = (b.e)^2, so nondegeneracy is exactly b.e != 0",
          sp.simplify(det - bdote**2) == 0, "det = %s" % det)

    def stab_dim(bv, ev):
        gens = [np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float),
                np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float),
                np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float)]
        rows = [list(X @ bv) + list(X @ ev) for X in gens]
        return 3 - np.linalg.matrix_rank(np.array(rows), tol=1e-9)

    rng = np.random.default_rng(1)
    dims = set()
    for _ in range(500):
        bv, ev = rng.normal(size=3), rng.normal(size=3)
        if abs(bv @ ev) > 1e-6:
            dims.add(stab_dim(bv, ev))
    check("generic (b,e): the stabilizer in SO(3) has dimension 0",
          dims == {0}, "dims seen %s over 500 draws" % sorted(dims))
    bv = np.array([1.0, 2.0, -0.5])
    check("b parallel to e, still nondegenerate: dimension 1, a single axis",
          stab_dim(bv, 3.0 * bv) == 1)


# ---------------------------------------------------------------------------
# F.  Section 5
# ---------------------------------------------------------------------------

def dfact(n):
    r = 1
    while n > 1:
        r, n = r * n, n - 2
    return r


def He(k, x):
    """the probabilists' Hermite polynomial He_k, evaluated on an array."""
    h0, h1 = np.ones_like(x), x
    if k == 0:
        return h0
    for n in range(1, k):
        h0, h1 = h1, x * h1 - n * h0
    return h1


def Mk(k):
    """max |He_k(u)| e^{-u^2/2}."""
    u = np.linspace(-15, 15, 3000001)
    return float(np.max(np.abs(He(k, u)) * np.exp(-u**2 / 2)))


def section_F():
    head("F. Section 5: the signal-to-noise law")

    # (i) I_k against the closed form, by quadrature with the exact k-th derivative
    from scipy.integrate import quad
    s = 1.3
    for k in range(5):
        gk = lambda x: (He(k, np.array(x / s)) * np.exp(-x**2 / (2 * s**2))
                        / (s**(k + 1) * np.sqrt(2 * np.pi)))**2
        num = quad(gk, -40 * s, 40 * s, limit=400)[0]
        closed = dfact(2 * k - 1) / (2**(k + 1) * np.sqrt(np.pi) * s**(2 * k + 1))
        check("I_%d: quadrature / closed form = 1" % k, abs(num / closed - 1) < 1e-10,
              "ratio %.12f" % (num / closed))

    M = {k: Mk(k) for k in range(5)}
    C = {k: M[k] * np.sqrt(2.0**k / dfact(2 * k - 1)) for k in range(5)}
    say("  M_k = " + ", ".join("%d:%.5f" % (k, M[k]) for k in range(5)))
    say("  C_k = " + ", ".join("%d:%.5f" % (k, C[k]) for k in range(5)))
    check("C_k are all of order 1", all(0.5 < C[k] < 2 for k in range(5)))
    check("C_k reproduce the values 1.000, 0.858, 1.155, 1.008, 1.171",
          all(abs(C[k] - v) < 2e-3
              for k, v in zip(range(5), [1.000, 0.858, 1.155, 1.008, 1.171])))

    # (ii) the exact finite-s law.  A Gaussian feature of scale L smoothed at scale s
    # has effective scale sqrt(L^2+s^2) and its peak attenuated by L/sqrt(L^2+s^2);
    # the noise of the k-th derivative estimate is sigma sqrt(h I_k(s)).  Whence
    #        SNR(k)/SNR(0) = C_k (s/sqrt(L^2+s^2))^k .
    def ratio(k, s_, L, exact=True):
        r = s_ / np.sqrt(L**2 + s_**2) if exact else s_ / L
        return C[k] * r**k

    say()
    say("  table (6), recomputed:")
    say("    s      L    k    s/L    draft    asympt     exact    exact vs order 2")
    rows = [(1.0, 2.0, 3, "1/8"), (1.0, 2.0, 4, "1/14"),
            (1.0, 5.0, 3, "1/124"), (1.0, 5.0, 4, "1/534")]
    got = []
    for s_, L, k, drafted in rows:
        asy, exa, b2 = (ratio(k, s_, L, False), ratio(k, s_, L), ratio(2, s_, L))
        say("  %4.1f %6.1f %4d   %4.2f   %6s   1/%-8.1f 1/%-8.1f 1/%.1f"
            % (s_, L, k, s_ / L, drafted, 1 / asy, 1 / exa, b2 / exa))
        got.append((1 / exa, b2 / exa))
    want = [(11.1, 2.6), (21.3, 4.9), (131.5, 5.8), (577.2, 25.6)]
    check("the exact and the order-2 columns reproduce the table",
          all(abs(g[0] - w[0]) < 0.15 and abs(g[1] - w[1]) < 0.1
              for g, w in zip(got, want)))
    check("the draft's table is the asymptotic law, correctly evaluated",
          all(abs(1 / ratio(k, s_, L, False) - float(d.split("/")[1])) < 0.6
              for s_, L, k, d in rows))

    # (iii) the smoothing scale bringing order 4 within a factor 2 of order 0, L = 5
    from scipy.optimize import brentq
    L = 5.0
    s_asy = L * (0.5 / C[4])**0.25
    s_exa = brentq(lambda t: ratio(4, t, L) - 0.5, 0.1, 500.0)
    say()
    say("  the '4 mm' claim of 5.3: asymptotic %.2f mm, exact %.2f mm = %.2f L"
        % (s_asy, s_exa, s_exa / L))
    check("the asymptotic inversion gives 4.0 mm", abs(s_asy - 4.04) < 0.02)
    check("the exact one gives 6.9 mm, i.e. LARGER than the feature",
          abs(s_exa - 6.87) < 0.05 and s_exa > L)

    # (iv) the optimal smoothing scale of an order-k descriptor
    say()
    def snr(k, t, L=5.0):
        return C[k] * t**(k + 0.5) / (L**2 + t**2)**((k + 1) / 2.0)
    for k in range(5):
        # d/dt log SNR(k) = (k+1/2)/t - (k+1) t/(L^2+t^2)
        dlog = lambda t: (k + 0.5) / t - (k + 1) * t / (L**2 + t**2)
        st = brentq(dlog, 1e-6, 1000.0)
        check("the optimal scale at order %d is L sqrt(2k+1) = %.4f L"
              % (k, np.sqrt(2 * k + 1)), abs(st / L - np.sqrt(2 * k + 1)) < 1e-8,
              "found %.6f L" % (st / L))
    say("  at each order's OWN optimal scale the penalty nearly disappears:")
    for k in (2, 3, 4):
        rk = snr(k, L * np.sqrt(2 * k + 1)) / snr(0, L)
        say("    k = %d: SNR(k)/SNR(0) = 1/%.2f, but at s* = %.2f L"
            % (k, 1 / rk, np.sqrt(2 * k + 1)))

    # (v) the discrete sum against the continuum integral
    say()
    say("  continuum I_k against the discrete sum, ratio discrete/continuum:")
    say("   s/h       k=0     k=1     k=2     k=3     k=4")
    for soh in (1.00, 1.25, 1.50, 1.67, 2.00):
        n = np.arange(-400, 401)
        x = n / soh                       # samples in units of s
        line = "  %4.2f  " % soh
        row = []
        for k in range(5):
            d = He(k, x) * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
            disc = float(np.sum(d**2)) / soh
            cont = dfact(2 * k - 1) / (2**(k + 1) * np.sqrt(np.pi))
            row.append(disc / cont)
            line += "  %6.4f" % (disc / cont)
        say(line)
        if abs(soh - 1.00) < 1e-9:
            check("at s/h = 1 the k=3 ratio is 0.954 and the k=4 ratio 1.125",
                  abs(row[3] - 0.954) < 0.01 and abs(row[4] - 1.125) < 0.01)
        if abs(soh - 1.25) < 1e-9:
            check("by s/h = 1.25 every order is within 0.5 %",
                  all(abs(r - 1) < 0.005 for r in row))

    # (vi) white noise against a ramp-filtered, apodized CT noise power spectrum
    say()
    say("  white noise against P(f) ~ f^2 exp(-(f/f0)^2), f0 = 0.35/mm, s = 1 mm, L = 5 mm:")
    f0, s_, Lc = 0.35, 1.0, 5.0

    def nz(k, white):
        P = (lambda f: np.ones_like(f)) if white else (lambda f: f**2 * np.exp(-(f / f0)**2))
        g = lambda f: P(f) * (2 * np.pi * f)**(2 * k) * np.exp(-(2 * np.pi * f * s_)**2)
        return quad(g, 0, 20, limit=400)[0]

    facs = []
    for k in (1, 2, 3, 4):
        rw = ratio(k, s_, Lc)
        fac = np.sqrt((nz(k, False) / nz(0, False)) / (nz(k, True) / nz(0, True)))
        facs.append(fac)
        say("    k = %d   white 1/%-8.1f CT 1/%-8.1f CT worse by %.2fx"
            % (k, 1 / rw, fac / rw, fac))
    check("the CT spectrum costs about a factor 2 at k = 3 and 4, the exponent untouched",
          abs(facs[2] - 2.0) < 0.15 and abs(facs[3] - 2.06) < 0.15)


def main():
    global LOG
    LOG = open("appendices.log", "w")
    say("Independent check of the appendices of the imaging text")
    say("=" * 74)
    f, u, v = section_A()
    section_B(f, u, v)
    section_C()
    section_D()
    section_E()
    section_F()
    say()
    say("=" * 74)
    bad = [n for n, ok in PASS if not ok]
    say("%d checks, %d passed, %d failed" % (len(PASS), len(PASS) - len(bad), len(bad)))
    for n in bad:
        say("  FAILED: " + n)
    LOG.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
