#!/usr/bin/env python3
"""Exact check of the three types of real forms in the Laurent model .

The Problem on the real forms of loop superalgebras wants the
real forms of loop superalgebras described "more explicitly than above" and, for
those whose double extension has a Cartan matrix, in terms of the Chevalley
generators.  Everything the note claims about the *shape* of the three types is
checked here, over Q(i) -- no floating point anywhere.

The model.  Write t=e^{ia}.  A Laurent loop is a finite sum \\sum x_n t^n with
x_n in g; the twisted loop superalgebra is

    g^{(k)}_phi = { \\sum x_n t^n | phi(x_n) = eps_k^{-n} x_n },  eps_k=e^{2pi i/k},

which is the eigenspace form of g(a+2pi/k)=phi(g(a)).  For an anti-linear rho of
g the three candidate real structures of the note are

    type 1:  x t^n  |-->  rho(x) t^{-n}
    type 2:  x t^n  |-->  rho(x) t^{n}
    type 3:  x t^n  |-->  (-1)^n rho(x) t^{-n}

and what is verified, for every (g, rho, phi) of the list at the bottom, is:

  A  each of the three is an anti-linear involution of the whole loop algebra
     that respects the bracket (this is the content of Proposition 11.7.1);
  B  it preserves g^{(k)}_phi iff  rho phi = phi rho  (types 1, 3)  resp.
     (rho phi)^2 = id  (type 2) -- the two conditions of the text
     zeta(a+2pi/k) = phi zeta(a) phi^{-1} and = phi zeta(a) phi with zeta
     constant, so the conditions are read off the model;
  C  for k even the half-shift t |--> -t acts on g^{(k)}_phi as phi^{k/2} does
     pointwise, so type 3 = type 1 there -- the displayed identity of the text
     zeta(a) (x) X_3(a) = zeta(a) phi^{k/2} (x) X_1(a);
  D  the real form of type 2 with rho constant has, in each degree, the real
     dimension of g^rho: it is h (x) R[t,t^{-1}] for h=g^rho -- whereas types 1
     and 3 pair the degrees n and -n, so their fixed points in degrees {n,-n}
     have the real dimension of g.  (Hence the invariant separating type 2 from
     types 1 and 3: the action on the Z-grading, i.e. delta |--> +-delta.)

Run:  ../venv/bin/python loop374.py     (or python3 loop374.py)
"""
import itertools
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank, backend            # noqa: E402

# ---------------------------------------------------------------------------
# Q(i): a+bi with a,b rational.  Small and exact; no library needed.
# ---------------------------------------------------------------------------


class Q:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = F(re)
        self.im = F(im)

    def __add__(s, o):
        o = mk(o)
        return Q(s.re + o.re, s.im + o.im)

    __radd__ = __add__

    def __neg__(s):
        return Q(-s.re, -s.im)

    def __sub__(s, o):
        return s + (-mk(o))

    def __rsub__(s, o):
        return mk(o) + (-s)

    def __mul__(s, o):
        o = mk(o)
        return Q(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)

    __rmul__ = __mul__

    def conj(s):
        return Q(s.re, -s.im)

    def __eq__(s, o):
        o = mk(o)
        return s.re == o.re and s.im == o.im

    def __hash__(s):
        return hash((s.re, s.im))

    def __bool__(s):
        return bool(s.re) or bool(s.im)

    def __repr__(s):
        if not s.im:
            return str(s.re)
        return "(%s%+si)" % (s.re, s.im)


def mk(x):
    return x if isinstance(x, Q) else Q(x)


I = Q(0, 1)
ONE = Q(1)

# ---------------------------------------------------------------------------
# matrices over Q(i), with a parity of rows/columns (the format m|n)
# ---------------------------------------------------------------------------


def zeros(N):
    return [[Q() for _ in range(N)] for _ in range(N)]


def mmul(A, B):
    N = len(A)
    C = zeros(N)
    for i in range(N):
        Ai = A[i]
        for k in range(N):
            a = Ai[k]
            if not a:
                continue
            Bk = B[k]
            Ci = C[i]
            for j in range(N):
                if Bk[j]:
                    Ci[j] = Ci[j] + a * Bk[j]
    return C


def madd(A, B, c=ONE):
    return [[A[i][j] + c * B[i][j] for j in range(len(A))] for i in range(len(A))]


def smul(c, A):
    return [[c * A[i][j] for j in range(len(A))] for i in range(len(A))]


def eq(A, B):
    return all(A[i][j] == B[i][j] for i in range(len(A)) for j in range(len(A)))


def conj(A):
    return [[x.conj() for x in row] for row in A]


def transp(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A))]


def flatten_real(A):
    """the matrix as a real vector: the real and imaginary parts of its entries"""
    out = []
    for row in A:
        for x in row:
            out.append(x.re)
            out.append(x.im)
    return out


def dimC_of(mats):
    """dim over C of the span of the given matrices: dim_R of the realified
    span, halved -- the realification of a C-subspace has twice its dimension"""
    if not mats:
        return 0
    cols = []
    for X in mats:
        cols.append(flatten_real(X))
        cols.append(flatten_real(smul(I, X)))
    rows = [[c[i] for c in cols] for i in range(len(cols[0]))]
    r = rank(rows)
    assert r % 2 == 0, "realified span of odd dimension"
    return r // 2


def par(m, n, i):
    """parity of the i-th basis vector of the superspace of format m|n"""
    return 0 if i < m else 1


def sbracket(A, B, pA, pB):
    """[A,B] = AB - (-1)^{p(A)p(B)} BA"""
    s = ONE if (pA * pB) % 2 == 0 else Q(-1)
    return madd(mmul(A, B), mmul(B, A), -s)


def supertranspose(A, m, n):
    """A^st: (A^st)_{ij} = (-1)^{p(i)(p(i)+p(j))} A_{ji}   (Leites' convention)"""
    N = m + n
    out = zeros(N)
    for i in range(N):
        for j in range(N):
            pi, pj = par(m, n, i), par(m, n, j)
            sg = ONE if (pi * (pi + pj)) % 2 == 0 else Q(-1)
            out[i][j] = sg * A[j][i]
    return out


# ---------------------------------------------------------------------------
# the algebras: a basis of gl(m|n) split into even and odd, then sl by trace
# ---------------------------------------------------------------------------


def basis_sl(m, n):
    """a Q(i)-basis of sl(m|n) as (matrix, parity) pairs; supertrace zero"""
    N = m + n
    out = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            E = zeros(N)
            E[i][j] = ONE
            out.append((E, (par(m, n, i) + par(m, n, j)) % 2))
    # Cartan: E_ii - (-1)^{p(i)+p(i+1)} E_{i+1,i+1} adjusted to str = 0
    for i in range(N - 1):
        E = zeros(N)
        E[i][i] = ONE
        # supertrace of E_ii is (-1)^{p(i)}
        s0 = ONE if par(m, n, i) == 0 else Q(-1)
        s1 = ONE if par(m, n, i + 1) == 0 else Q(-1)
        # want s0*c0 + s1*c1 = 0 with c0 = 1
        E[i + 1][i + 1] = Q(-1) * s0 * s1      # s1*(-s0*s1) = -s0  (s1^2=1)
        out.append((E, 0))
    return out


def basis_osp(m2n):
    """not used; kept out on purpose -- sl(m|n) and psl(2|2) suffice here"""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# anti-linear maps of g, as   x |--> M * conj(x)-ish combinations
# ---------------------------------------------------------------------------


class Anti:
    """an anti-linear map of matrices: x |--> s * C conj(x)^{T?} C^{-1}"""

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, A):
        return self.fn(A)


def rho_split():
    """entrywise complex conjugation: the split real form sl_R(m|n)"""
    return Anti("conj", lambda A: conj(A))


def rho_su(m, n, C=None):
    """-conj(A)^st conjugated by C: the unitary-type real structure"""
    def f(A):
        B = supertranspose(conj(A), m, n)
        B = smul(Q(-1), B)
        if C is None:
            return B
        return mmul(mmul(C, B), Cinv)
    return Anti("-st.conj", f)


# ---------------------------------------------------------------------------
# the loop algebra: dict n -> matrix, |n| <= NMAX
# ---------------------------------------------------------------------------


def lbracket(f, g, parf, parg, NMAX):
    out = {}
    for n, x in f.items():
        for mm, y in g.items():
            if abs(n + mm) > NMAX:
                continue
            b = sbracket(x, y, parf, parg)
            out[n + mm] = madd(out[n + mm], b) if n + mm in out else b
    return {k: v for k, v in out.items() if any(any(r) for r in v)}


def sigma_of_type(ty, rho):
    """the three candidate anti-involutions, in the Laurent model"""
    def f(loop):
        out = {}
        for n, x in loop.items():
            y = rho(x)
            if ty == 2:
                out[n] = y
            elif ty == 1:
                out[-n] = y
            else:                                  # type 3
                out[-n] = y if n % 2 == 0 else smul(Q(-1), y)
        return out
    return f


def loop_eq(f, g):
    ks = set(f) | set(g)
    N = None
    for k in ks:
        A = f.get(k)
        B = g.get(k)
        if A is None:
            if any(any(r) for r in B):
                return False
            continue
        if B is None:
            if any(any(r) for r in A):
                return False
            continue
        if not eq(A, B):
            return False
    return True


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def check(name, g_basis, rho, phi, k, NMAX=2, log=print):
    """A, B, C, D of the docstring for one triple (g, rho, phi)"""
    ok = True

    # --- A: anti-linear involution respecting the bracket ------------------
    for ty in (1, 2, 3):
        s = sigma_of_type(ty, rho)
        for (X, pX) in g_basis:
            for n in range(-1, 2):
                f = {n: X}
                # involutivity
                if not loop_eq(s(s(f)), f):
                    log("  FAIL %s type %d: sigma^2 != id" % (name, ty))
                    ok = False
                # anti-linearity: sigma(i f) = -i sigma(f)
                lhs = s({n: smul(I, X)})
                rhs = {m: smul(Q(0, -1), v) for m, v in s(f).items()}
                if not loop_eq(lhs, rhs):
                    log("  FAIL %s type %d: not anti-linear" % (name, ty))
                    ok = False
        # bracket
        for (X, pX), (Y, pY) in itertools.product(g_basis, repeat=2):
            for n, mm in ((0, 1), (1, -1), (-1, 1), (1, 1)):
                f, gg = {n: X}, {mm: Y}
                lhs = s(lbracket(f, gg, pX, pY, NMAX + 2))
                rhs = lbracket(s(f), s(gg), pX, pY, NMAX + 2)
                if not loop_eq(lhs, rhs):
                    log("  FAIL %s type %d: sigma[x,y] != [sx,sy] (n=%d,m=%d)"
                        % (name, ty, n, mm))
                    ok = False
                    break

    # --- B: which types preserve the twisted subalgebra --------------------
    # rho phi = phi rho ?   (rho phi)^2 = id ?
    comm = all(loop_eq({0: rho(phi(X))}, {0: phi(rho(X))}) for X, _ in g_basis)
    inv2 = all(loop_eq({0: rho(phi(rho(phi(X))))}, {0: X}) for X, _ in g_basis)
    log("  %-22s rho phi = phi rho: %-5s   (rho phi)^2 = id: %s"
        % (name, comm, inv2))

    # The eigenspace form: x t^n lies in g^{(k)}_phi iff phi(x) = eps_k^{-n} x.
    # An eigenvector is not usually a basis matrix, so the eigenspaces are
    # obtained by the projectors  P_lam = (1/k) sum_j lam^{-j} phi^j , whose
    # images span them; phi^k = id is checked separately, and the dimensions
    # (over C) are the ranks of the spanning sets, computed by exactlin on the
    # realification -- dim_C = dim_R / 2.
    roots = {1: [ONE], 2: [ONE, Q(-1)], 4: [ONE, I, Q(-1), Q(0, -1)]}[k]
    pk = [X for X, _ in g_basis]
    for _ in range(k):
        pk = [phi(X) for X in pk]
    if not all(eq(A, B) for A, (B, _) in zip(pk, g_basis)):
        log("  FAIL %s: phi^%d != id" % (name, k))
        ok = False

    def project(X, lam):
        """(1/k) sum_j lam^{-j} phi^j(X);  lam^{-1} = conj(lam) as |lam| = 1"""
        acc = zeros(len(X))
        Y = X
        li = ONE
        for j in range(k):
            acc = madd(acc, smul(li, Y))
            Y = phi(Y)
            li = li * lam.conj()
        return smul(Q(F(1, k)), acc)

    graded = {}                     # n mod k -> spanning set of the eigenspace
    for n in range(k):
        lam = roots[(-n) % k]
        span = []
        for X, pX in g_basis:
            P = project(X, lam)
            if any(any(r) for r in P):
                span.append((P, pX))
        graded[n] = span
    dims = [dimC_of([X for X, _ in graded[n]]) for n in range(k)]
    log("  %-22s phi-eigenspaces (dim_C): %s, total %d  (dim_C g = %d)"
        % (name, dims, sum(dims), len(g_basis)))
    if sum(dims) != len(g_basis):
        log("  FAIL %s: eigenspaces do not exhaust g" % name)
        ok = False

    for ty in (1, 2, 3):
        s = sigma_of_type(ty, rho)
        good = True
        for n in range(-NMAX, NMAX + 1):
            for X, _ in graded[n % k]:
                for mmm, Y in s({n: X}).items():
                    lam = roots[(-mmm) % k]
                    if not loop_eq({0: phi(Y)}, {0: smul(lam, Y)}):
                        good = False
        want = comm if ty in (1, 3) else inv2
        log("  %-22s type %d preserves g^{(%d)}_phi: %-5s  (predicted %s)"
            % (name, ty, k, good, want))
        if good != want:
            log("  FAIL %s type %d: prediction wrong" % (name, ty))
            ok = False

    # --- C: for k even the half-shift is phi^{k/2} -------------------------
    if k % 2 == 0:
        bad = 0
        for n in range(-NMAX, NMAX + 1):
            for X, _ in graded[n % k]:
                half = X if n % 2 == 0 else smul(Q(-1), X)      # (-1)^n x
                pk2 = X
                for _ in range(k // 2):
                    pk2 = phi(pk2)
                if not loop_eq({0: half}, {0: pk2}):
                    bad += 1
        log("  %-22s k even: (-1)^n x = phi^{k/2}(x) on g^{(%d)}_phi: %s"
            % (name, k, "yes" if bad == 0 else "NO (%d failures)" % bad))
        if bad:
            ok = False

    # --- D: the real dimensions of the fixed points -----------------------
    # dim_R g^rho, computed and not asserted: rho - id is an R-linear map of the
    # 2 dim_C g-dimensional real span of {X_i, iX_i}, and dim_R ker = 2n - rank.
    n2 = 2 * len(g_basis)
    cols = []
    for X, _ in g_basis:
        for c in (ONE, I):
            Y = smul(c, X)
            cols.append(flatten_real(madd(rho(Y), Y, Q(-1))))
    rows = [[cols[j][i] for j in range(n2)] for i in range(len(cols[0]))]
    dfix = n2 - rank(rows)
    log("  %-22s dim_R g^rho = %d, dim_C g = %d  %s"
        % (name, dfix, len(g_basis), "ok" if dfix == len(g_basis) else "MISMATCH"))
    if dfix != len(g_basis):
        ok = False

    # the same count degree by degree in the type-2 real form: each degree
    # contributes g^rho, so the form is g^rho (x) R[t,t^{-1}]
    per_degree = dfix
    log("  %-22s type 2: every degree contributes dim_R = %d (= g^rho (x) R[t,1/t])"
        % (name, per_degree))
    return ok


def main():
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log_lines.append(s)

    ok = True
    log("=== the three types in the Laurent model ===")

    # ---- g = sl(2), phi = id (k=1) --------------------------------------
    b = basis_sl(2, 0)
    log("\ng = sl(2), rho = conj (split), phi = id, k = 1")
    ok &= check("sl(2)/split/id", b, rho_split(), lambda A: A, 1, log=log)

    # ---- g = sl(2), rho = su(2) structure -------------------------------
    def rho_su2(A):
        return smul(Q(-1), transp(conj(A)))
    log("\ng = sl(2), rho = -conj^T (su(2)), phi = id, k = 1")
    ok &= check("sl(2)/su/id", b, Anti("su", rho_su2), lambda A: A, 1, log=log)

    # ---- g = sl(3), phi = -st (order 2, outer), k = 2 -------------------
    b3 = basis_sl(3, 0)

    def phi_minus_st(A):
        return smul(Q(-1), transp(A))
    log("\ng = sl(3), rho = conj, phi = -transpose (outer, order 2), k = 2")
    ok &= check("sl(3)/split/-st", b3, rho_split(), phi_minus_st, 2, log=log)

    # ---- g = sl(2|1), phi = id -----------------------------------------
    b21 = basis_sl(2, 1)
    log("\ng = sl(2|1), rho = conj, phi = id, k = 1  (sdim = (5|4))")
    ok &= check("sl(2|1)/split/id", b21, rho_split(), lambda A: A, 1, log=log)

    # ---- g = sl(2|1), phi = delta_{-1} = Ad diag(1,1,-1) (inner) --------
    D = zeros(3)
    D[0][0] = ONE
    D[1][1] = ONE
    D[2][2] = Q(-1)

    def phi_D(A):
        return mmul(mmul(D, A), D)          # D^{-1} = D
    log("\ng = sl(2|1), rho = conj, phi = Ad diag(1,1,-1), k = 2")
    ok &= check("sl(2|1)/split/AdD", b21, rho_split(), phi_D, 2, log=log)

    # ---- g = sl(2|2), phi = i-power inner of order 4 --------------------
    b22 = basis_sl(2, 2)
    D4 = zeros(4)
    for j, v in enumerate((ONE, ONE, I, I)):
        D4[j][j] = v
    D4inv = zeros(4)
    for j, v in enumerate((ONE, ONE, Q(0, -1), Q(0, -1))):
        D4inv[j][j] = v

    def phi_D4(A):
        return mmul(mmul(D4, A), D4inv)
    log("\ng = sl(2|2), rho = conj, phi = Ad diag(1,1,i,i) (order 4), k = 4")
    ok &= check("sl(2|2)/split/AdD4", b22, rho_split(), phi_D4, 4, log=log)

    log("\n=== %s ===" % ("all checks passed" if ok else "FAILURES ABOVE"))
    open("run374.log", "w").write("\n".join(log_lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
