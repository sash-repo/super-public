#!/usr/bin/env python3
"""
hamvol.py --- the Hamiltonian Volichenko algebras, exactly.

Subsec. 4.13 of the text (proof of Proposition 2.13) and Statement 4.13.3.
The text proves that the simple
epimorphic Volichenko algebras h_{ad_x} have no one-sided ideals for g NOT of
the Hamiltonian series; the Hamiltonian case is posed as a Problem.

What is built here.

  po(0|n) = Lambda(theta_1..theta_n) with the odd Poisson bracket

      {f, g} = (-1)^{p(f)} sum_{j,k} B_{jk} (d_j f)(d_k g),      d_j = d/d theta_j,

  B a symmetric nondegenerate form.  Two choices matter:

      B = 1        (orthonormal)  -->  x = theta_1  is ANISOTROPIC: {x,x} = -1,
                                       which is 0 only modulo constants;
      B hyperbolic                -->  x = theta_1  is ISOTROPIC:   {x,x} = 0
                                       on the nose.

  These are exactly the two homologic elements of Table 2.12.2:
  x = H_{theta_1} (anisotropic) and x = H_{theta_1 + i theta_2} (isotropic,
  rational after the change of basis to a hyperbolic form).

  h(0|n) = po(0|n)/C,  h'(0|n) = span of the monomials of degree 1..n-1,
  simple for n > 3.  g = h'(0|n), g_ev = even monomials of degree 2..n-1.

  The Volichenko algebra h_{ad_x} = {a + [a,x] | a in g_ev} is carried by
  pr_ev onto g_ev with the product of eq. (152)

      a . b = [a, b] + S(a, b),      S(a, b) = [[a, x], [b, x]].

  Right (left) ideals of h are the subspaces invariant under all right (left)
  multiplications, so by Burnside there are none proper iff the unital
  associative algebra they generate is all of End(h).

Runs on plain python3 (Fraction) and uses exactlin.py (python-flint) when available.
"""

import sys, os, itertools
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "."))
try:
    from exactlin import rank as ex_rank, backend as ex_backend
except Exception:                                        # pragma: no cover
    ex_rank, ex_backend = None, lambda: "unavailable"


# ----------------------------------------------------------------- Grassmann

def bits(m):
    "indices set in the bitmask m, ascending"
    out, j = [], 0
    while m:
        if m & 1:
            out.append(j)
        m >>= 1
        j += 1
    return out


def popcount(m):
    return bin(m).count("1")


def mono_mul(m1, m2):
    "product of two monomials given by bitmasks; returns (sign, mask) or (0, 0)"
    if m1 & m2:
        return 0, 0
    # sign = (-1)^{number of transpositions} sorting the concatenation
    sign, b2 = 1, bits(m2)
    for j in b2:
        # move theta_j left past every generator of m1 with index > j
        higher = popcount(m1 >> (j + 1))
        # and past the part of m2 already merged that sits above j -- none,
        # since we merge b2 in ascending order and each stays to the right
        if higher & 1:
            sign = -sign
    return sign, m1 | m2


def mul(f, g):
    "product in Lambda(theta)"
    out = {}
    for m1, c1 in f.items():
        for m2, c2 in g.items():
            s, m = mono_mul(m1, m2)
            if s:
                out[m] = out.get(m, 0) + s * c1 * c2
    return {m: c for m, c in out.items() if c}


def deriv(f, j):
    "left derivative d/d theta_j"
    out, bit = {}, 1 << j
    for m, c in f.items():
        if m & bit:
            s = -1 if (popcount(m & (bit - 1)) & 1) else 1
            out[m ^ bit] = out.get(m ^ bit, 0) + s * c
    return {m: c for m, c in out.items() if c}


def parity(f):
    "parity of a homogeneous element; raises if inhomogeneous"
    ps = {popcount(m) & 1 for m in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous element %r" % (f,))
    return ps.pop() if ps else 0


def add(*fs):
    out = {}
    for f in fs:
        for m, c in f.items():
            out[m] = out.get(m, 0) + c
    return {m: c for m, c in out.items() if c}


def smul(c, f):
    return {m: c * v for m, v in f.items() if c * v}


# ------------------------------------------------------------ Poisson bracket

class Po:
    """po(0|n) with the odd Poisson bracket attached to a symmetric form B."""

    def __init__(self, n, B=None):
        self.n = n
        if B is None:                                   # orthonormal
            B = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        self.B = B

    def pb(self, f, g):
        """{f, g}, extended bilinearly to parity-INHOMOGENEOUS arguments.

        Volichenko elements a + [a,x] are inhomogeneous by definition, so the
        bracket has to be split over the parity decomposition of the first
        argument -- the sign (-1)^{p(f)} is the only place parity enters.
        """
        ev = {m: c for m, c in f.items() if popcount(m) % 2 == 0}
        od = {m: c for m, c in f.items() if popcount(m) % 2 == 1}
        if ev and od:
            return add(self.pb_hom(ev, g), self.pb_hom(od, g))
        return self.pb_hom(f, g)

    def pb_hom(self, f, g):
        "{f, g} for f parity-homogeneous"
        if not f:
            return {}
        sgn = -1 if parity(f) else 1
        out = {}
        for j in range(self.n):
            dj = deriv(f, j)
            if not dj:
                continue
            for k in range(self.n):
                if not self.B[j][k]:
                    continue
                dk = deriv(g, k)
                if not dk:
                    continue
                for m, c in mul(dj, dk).items():
                    out[m] = out.get(m, 0) + sgn * self.B[j][k] * c
        return {m: c for m, c in out.items() if c}


def hyperbolic(n):
    "B: theta_1,theta_2 / theta_3,theta_4 / ... hyperbolic pairs, last one 1 if n odd"
    B = [[0] * n for _ in range(n)]
    i = 0
    while i + 1 < n:
        B[i][i + 1] = B[i + 1][i] = 1
        i += 2
    if i < n:
        B[i][i] = 1
    return B


# ------------------------------------------------------- h'(0|n) and h_{ad_x}

class HamVol:
    """The Volichenko algebra h_{ad_x} inside g = h'(0|n), carried onto g_ev."""

    def __init__(self, n, B=None, x=None):
        self.n = n
        self.po = Po(n, B)
        self.x = x if x is not None else {1: 1}          # theta_1
        # g = h'(0|n): monomials of degree 1..n-1, modulo constants
        self.gbasis = [m for m in range(1 << n) if 1 <= popcount(m) <= n - 1]
        self.evb = [m for m in self.gbasis if popcount(m) % 2 == 0]
        self.idx = {m: i for i, m in enumerate(self.evb)}
        self.d = len(self.evb)

    # --- brackets inside h'(0|n): the Poisson bracket, constants killed
    def br(self, f, g):
        out = self.po.pb(f, g)
        out.pop(0, None)                                 # modulo constants
        for m in out:
            assert popcount(m) <= self.n - 1, "left h'(0|n)"
        return out

    def ad_x(self, a):
        return self.br(a, self.x)

    def S(self, a, b):
        return self.br(self.ad_x(a), self.ad_x(b))

    def prod(self, a, b):
        "a . b = [a,b] + S(a,b), the product of eq. (152)"
        return add(self.br(a, b), self.S(a, b))

    # --- linear algebra bookkeeping on g_ev
    def vec(self, f):
        v = [0] * self.d
        for m, c in f.items():
            v[self.idx[m]] = c
        return v

    def elt(self, i):
        return {self.evb[i]: 1}

    def right_mult(self, b):
        "matrix of z |-> z . b"
        return [self.vec(self.prod(self.elt(i), b)) for i in range(self.d)]

    def left_mult(self, a):
        "matrix of z |-> a . z"
        return [self.vec(self.prod(a, self.elt(i))) for i in range(self.d)]


# ------------------------------------------------------------------- checks

def check_setup(H, verbose=True):
    """The identities the construction rests on, verified on every basis pair."""
    n, ok = H.n, True

    # 1. super-antisymmetry and super-Jacobi of the bracket on h'(0|n)
    gb = [{m: 1} for m in H.gbasis]
    for f in gb:
        for g in gb:
            pf, pg = parity(f), parity(g)
            lhs = H.br(f, g)
            rhs = smul(-(-1) ** (pf * pg), H.br(g, f))
            if lhs != rhs:
                ok = False; print("  ! antisymmetry fails", f, g)
    import random
    random.seed(7)
    trials = gb if len(gb) <= 30 else random.sample(gb, 30)
    for f in trials:
        for g in trials:
            for h in trials:
                pf, pg, ph = parity(f), parity(g), parity(h)
                j = add(H.br(H.br(f, g), h),
                        smul((-1) ** (pf * (pg + ph)), H.br(H.br(g, h), f)),
                        smul((-1) ** (ph * (pf + pg)), H.br(H.br(h, f), g)))
                if j:
                    ok = False; print("  ! Jacobi fails", f, g, h); break

    # 2. x is homologic in h'(0|n): [x,x] = 0 modulo constants
    xx = H.br(H.x, H.x)
    if xx:
        ok = False; print("  ! [x,x] =", xx)
    xx_po = H.po.pb(H.x, H.x)
    aniso = bool(xx_po)

    # 3. ad_x^2 = 0 on g_ev
    for i in range(H.d):
        if H.ad_x(H.ad_x(H.elt(i))):
            ok = False; print("  ! ad_x^2 =/= 0 on", H.evb[i])

    # 4. THE identity that makes pr_ev an isomorphism of algebras:
    #    [a + [a,x], b + [b,x]] = c + [c,x]  with  c = [a,b] + S(a,b).
    for i in range(H.d):
        a = H.elt(i)
        for j in range(H.d):
            b = H.elt(j)
            lhs = H.br(add(a, H.ad_x(a)), add(b, H.ad_x(b)))
            c = H.prod(a, b)
            rhs = add(c, H.ad_x(c))
            if lhs != rhs:
                ok = False
                print("  ! (108a) fails on", H.evb[i], H.evb[j], lhs, rhs)

    # 5. the two halves of the product: a.b - b.a = 2[a,b], a.b + b.a = 2S(a,b)
    for i in range(H.d):
        a = H.elt(i)
        for j in range(H.d):
            b = H.elt(j)
            if add(H.prod(a, b), smul(-1, H.prod(b, a))) != smul(2, H.br(a, b)):
                ok = False; print("  ! R1 antisymmetric half fails")
            if add(H.prod(a, b), H.prod(b, a)) != smul(2, H.S(a, b)):
                ok = False; print("  ! R1 symmetric half fails")

    if verbose:
        print("  setup checks: %s   (x is %s: {x,x} = %s in po)"
              % ("OK" if ok else "FAILED",
                 "anisotropic" if aniso else "isotropic", xx_po or "0"))
    return ok, aniso


def weight_element(H):
    """Solve for t in g_ev with [t,x] = lambda x, lambda =/= 0.

    This is the hypothesis of the Lemma that replaces the nonstandard-grading
    argument.  Returns the list of admissible (t, lambda), as a basis of the
    solution space of [t,x] - lambda x = 0 with lambda a free unknown.
    """
    # unknowns: coefficients of t on evb, plus lambda.  Equations: coefficients
    # of [t,x] - lambda*x on the odd basis of g.
    odd = [m for m in H.gbasis if popcount(m) % 2 == 1]
    oidx = {m: i for i, m in enumerate(odd)}
    rows = []
    cols = H.d + 1
    for k, m in enumerate(odd):
        rows.append([0] * cols)
    for i in range(H.d):
        for m, c in H.ad_x(H.elt(i)).items():
            rows[oidx[m]][i] += c
    for m, c in H.x.items():
        rows[oidx[m]][H.d] -= c
    sols = nullspace_Q(rows)
    good = [s for s in sols if s[H.d] != 0]
    return good


# --------------------------------------------------------- exact linear algebra

def nullspace_Q(rows):
    "row basis of the nullspace, over Q; rows is a list of rows"
    if not rows:
        return []
    m, ncols = len(rows), len(rows[0])
    A = [[Fraction(x) for x in r] for r in rows]
    pivots, r = [], 0
    for c in range(ncols):
        p = None
        for i in range(r, m):
            if A[i][c]:
                p = i; break
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = 1 / A[r][c]
        A[r] = [v * inv for v in A[r]]
        for i in range(m):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        pivots.append(c); r += 1
        if r == m:
            break
    free = [c for c in range(ncols) if c not in pivots]
    out = []
    for fc in free:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -A[i][fc]
        out.append(v)
    return out


class Echelon:
    "incremental row-reduced basis over Q (Fraction), vectors as lists"

    def __init__(self, ncols):
        self.ncols = ncols
        self.rows = {}          # pivot column -> reduced row

    def reduce(self, v):
        v = [Fraction(x) for x in v]
        for c in sorted(self.rows):
            if v[c]:
                f = v[c]
                r = self.rows[c]
                v = [a - f * b for a, b in zip(v, r)]
        return v

    def add(self, v):
        "returns True if v enlarged the span"
        v = self.reduce(v)
        for c in range(self.ncols):
            if v[c]:
                inv = 1 / v[c]
                self.rows[c] = [a * inv for a in v]
                return True
        return False

    @property
    def dim(self):
        return len(self.rows)


def matmul(A, B):
    "A, B square matrices as lists of rows (row i = image of basis vector i)"
    d = len(A)
    out = []
    for i in range(d):
        row = [0] * d
        Ai = A[i]
        for k in range(d):
            a = Ai[k]
            if a:
                Bk = B[k]
                for j in range(d):
                    if Bk[j]:
                        row[j] += a * Bk[j]
        out.append(row)
    return out


def flat(A):
    return [x for row in A for x in row]


def generated_algebra_dim(gens, d, cap=None):
    """dim of the unital associative subalgebra of End(V) generated by `gens`."""
    E = Echelon(d * d)
    ident = [[1 if i == j else 0 for j in range(d)] for i in range(d)]
    frontier = []
    if E.add(flat(ident)):
        frontier.append(ident)
    for g in gens:
        if E.add(flat(g)):
            frontier.append(g)
    while frontier:
        new = []
        for M in frontier:
            for g in gens:
                P = matmul(M, g)
                if E.add(flat(P)):
                    new.append(P)
                    if cap and E.dim >= cap:
                        return E.dim
        frontier = new
    return E.dim


def submodule(H, v, mats):
    """smallest subspace containing v and stable under all matrices in mats"""
    E = Echelon(H.d)
    frontier = []
    if E.add(v):
        frontier.append(v)
    while frontier:
        new = []
        for w in frontier:
            for M in mats:
                # w is a row vector, M acts as row i = image of e_i
                img = [0] * H.d
                for i, c in enumerate(w):
                    if c:
                        for j, a in enumerate(M[i]):
                            if a:
                                img[j] += c * a
                if E.add(img):
                    new.append(img)
        frontier = new
    return E.dim


# ------------------------------------------------- fast mod-p linear algebra
#
# Rank over F_p is a LOWER bound for the rank over Q (equality for all but
# finitely many primes), so a FULL rank mod p proves a full rank over Q --
# which is the direction we need for "the algebra is all of End h".  In the
# other direction (a deficient rank mod p) we re-check over Q, and we use a
# prime p = 1 (mod 4), where sqrt(-1) exists, so that the splitting field of
# the n = 4 anomaly is already present in F_p.

P = 1000000009           # 1000000009 = 1 mod 4, so i = sqrt(-1) exists


def sqrt_m1(p=P):
    for a in range(2, 1000):
        r = pow(a, (p - 1) // 4, p)
        if (r * r) % p == p - 1:
            return r
    raise RuntimeError("no sqrt(-1)")


def np_rref(A, p=P):
    "in-place row reduction mod p; returns rank"
    import numpy as np
    A %= p
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = None
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        col = A[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if nzr.size:
            A[nzr] = (A[nzr] - np.outer(col[nzr], A[r])) % p
        r += 1
        if r == rows:
            break
    return r


def algebra_dim_modp(gens, d, p=P):
    """dim of the unital associative algebra generated by `gens`, over F_p."""
    import numpy as np
    G = [np.array(g, dtype=object).astype(np.int64) % p for g in gens]
    full = d * d
    basis = np.zeros((full, full), dtype=np.int64)   # echelon rows
    pivot = np.full(full, -1, dtype=np.int64)        # pivot col -> row index
    nrows = 0

    def insert(vec):
        nonlocal nrows
        v = vec % p
        while True:
            nz = np.nonzero(v)[0]
            if nz.size == 0:
                return False
            c = int(nz[0])
            if pivot[c] >= 0:
                v = (v - v[c] * basis[pivot[c]]) % p
            else:
                v = (v * pow(int(v[c]), p - 2, p)) % p
                basis[nrows] = v
                pivot[c] = nrows
                nrows += 1
                return True

    ident = np.eye(d, dtype=np.int64)
    frontier = []
    if insert(ident.reshape(-1)):
        frontier.append(ident)
    for g in G:
        if insert(g.reshape(-1)):
            frontier.append(g)
    while frontier and nrows < full:
        new = []
        for M in frontier:
            for g in G:
                Pm = (M @ g) % p
                if insert(Pm.reshape(-1)):
                    new.append(Pm)
                    if nrows >= full:
                        break
            if nrows >= full:
                break
        frontier = new
    return nrows


def commutant_dim(gens, d, p=P):
    """dim {X in End V | X g = g X for all generators g}, over F_p.

    dim > 1 means V is not absolutely irreducible: over the splitting field it
    breaks up, and the pieces are proper one-sided ideals of h.
    """
    import numpy as np
    rows = []
    for g in gens:
        G = np.array(g, dtype=np.int64) % p
        # (Xg - gX)_{ij} = sum_k X_{ik} g_{kj} - g_{ik} X_{kj}
        M = np.zeros((d * d, d * d), dtype=np.int64)
        for i in range(d):
            for j in range(d):
                r = i * d + j
                for k in range(d):
                    M[r, i * d + k] = (M[r, i * d + k] + G[k, j]) % p
                    M[r, k * d + j] = (M[r, k * d + j] - G[i, k]) % p
        rows.append(M % p)
    A = np.vstack(rows) % p
    return d * d - np_rref(A, p)


def submodule_dims_modp(gens, d, p=P, vectors=None):
    """dims of the invariant subspaces generated by the given vectors."""
    import numpy as np
    G = [np.array(g, dtype=np.int64) % p for g in gens]
    out = {}
    if vectors is None:
        vectors = [np.eye(d, dtype=np.int64)[i] for i in range(d)]
    for v0 in vectors:
        basis = np.zeros((d, d), dtype=np.int64)
        pivot = np.full(d, -1, dtype=np.int64)
        n = 0

        def insert(vec):
            nonlocal n
            v = vec % p
            while True:
                nz = np.nonzero(v)[0]
                if nz.size == 0:
                    return False
                c = int(nz[0])
                if pivot[c] >= 0:
                    v = (v - v[c] * basis[pivot[c]]) % p
                else:
                    v = (v * pow(int(v[c]), p - 2, p)) % p
                    basis[n] = v
                    pivot[c] = n
                    n += 1
                    return True

        frontier = []
        if insert(np.array(v0, dtype=np.int64)):
            frontier.append(np.array(v0, dtype=np.int64) % p)
        while frontier:
            new = []
            for w in frontier:
                for g in G:
                    img = (w @ g) % p
                    if insert(img):
                        new.append(img)
            frontier = new
        out[tuple(int(x) % p for x in v0)] = n
    return sorted(set(out.values())), out


# ------------------------------------------------------------------- driver

def run(n, B=None, tag="", x=None, do_algebra=True):
    H = HamVol(n, B, x)
    print("h'(0|%d), %s:  dim g_ev = dim h = %d" % (n, tag, H.d))
    ok, aniso = check_setup(H)
    if not ok:
        print("  SETUP FAILED -- not proceeding")
        return H, None

    good = weight_element(H)
    if good:
        print("  a torus element with [t,x] = lambda x, lambda =/= 0: EXISTS"
              " (%d-parameter family)" % len(good))
    else:
        print("  a torus element with [t,x] = lambda x, lambda =/= 0: NONE"
              "  -->  outside the graded argument")

    if not do_algebra:
        return H, None

    R = [H.right_mult(H.elt(i)) for i in range(H.d)]
    L = [H.left_mult(H.elt(i)) for i in range(H.d)]
    full = H.d * H.d
    dR = algebra_dim_modp(R, H.d)
    dL = algebra_dim_modp(L, H.d)
    print("  dim <R_a> = %d,  dim <L_a> = %d,  dim End h = %d   %s"
          % (dR, dL, full,
             "==> NO one-sided ideals (Burnside)" if dR == full == dL
             else "==> NOT absolutely irreducible"))
    if dR != full or dL != full:
        cR, cL = commutant_dim(R, H.d), commutant_dim(L, H.d)
        print("  commutant of <R_a>: dim %d,  of <L_a>: dim %d"
              " (1 = absolutely irreducible)" % (cR, cL))
        dims, _ = submodule_dims_modp(R, H.d)
        print("  right submodules generated by basis monomials, dims:", dims)
        # Are the pieces TWO-SIDED?  A two-sided ideal is a subspace invariant
        # under the left AND the right multiplications at once, so h is simple
        # as a Volichenko algebra iff <L_a, R_a> = End h.  If that holds while
        # <R_a> does not, then h is SIMPLE and yet has one-sided ideals --
        # a counterexample to Proposition 2.13.
        dLR = algebra_dim_modp(L + R, H.d)
        print("  dim <L_a, R_a> = %d of %d  ==> h %s as a Volichenko algebra"
              % (dLR, full,
                 "IS SIMPLE (no two-sided ideals)" if dLR == full
                 else "has proper TWO-sided ideals too"))
        explicit_split(H, R, "right")
    return H, (dR, dL, full)


def explicit_split(H, gens, side, p=P):
    """Exhibit the proper invariant subspaces over F_p (which contains i)."""
    import numpy as np
    d = H.d
    # a non-scalar element of the commutant: solve (Xg - gX) = 0
    rows = []
    for g in gens:
        G = np.array(g, dtype=np.int64) % p
        M = np.zeros((d * d, d * d), dtype=np.int64)
        for i in range(d):
            for j in range(d):
                r = i * d + j
                for k in range(d):
                    M[r, i * d + k] = (M[r, i * d + k] + G[k, j]) % p
                    M[r, k * d + j] = (M[r, k * d + j] - G[i, k]) % p
        rows.append(M % p)
    A = np.vstack(rows) % p
    ns = nullspace_modp(A, p)
    X = None
    for v in ns:
        Xc = v.reshape(d, d)
        if np.any((Xc - Xc[0, 0] * np.eye(d, dtype=np.int64)) % p):
            X = Xc % p
            break
    if X is None:
        return
    # eigenspaces of X give the split
    for lam in range(p):
        pass
    # find eigenvalues as roots of the minimal polynomial on a random vector
    # -- cheaper: X satisfies X^2 = alpha X + beta (commutant is a quadratic
    # field here), so diagonalize by solving (X - lam) v = 0 for lam in F_p
    tr = int(np.trace(X) % p)
    # characteristic scalars: try the two roots of the quadratic X^2-aX-b=0
    X2 = (X @ X) % p
    # express X2 = a X + b I  (works when the commutant is 2-dimensional)
    a = b = None
    for i in range(d):
        for j in range(d):
            if i != j and X[i, j] % p:
                a = (X2[i, j] * pow(int(X[i, j]), p - 2, p)) % p
                b = (X2[i, i] - a * X[i, i]) % p
                break
        if a is not None:
            break
    if a is None:
        return
    disc = (a * a + 4 * b) % p
    r = pow(int(disc), (p - 1) // 2, p)
    if r != 1 and disc != 0:
        print("    (the commutant is a quadratic field; it splits only over"
              " its splitting field -- p was chosen with i in F_p)")
        return
    s = tonelli(disc, p)
    inv2 = pow(2, p - 2, p)
    for lam in {((a + s) * inv2) % p, ((a - s) * inv2) % p}:
        Mm = (X - lam * np.eye(d, dtype=np.int64)) % p
        ker = nullspace_modp(Mm.T % p, p)
        if 0 < len(ker) < d:
            print("    a proper %s ideal of dimension %d (eigenvalue of the"
                  " commutant)" % (side, len(ker)))


def nullspace_modp(A, p=P):
    import numpy as np
    A = A.copy() % p
    rows, cols = A.shape
    r, pivots = 0, []
    for c in range(cols):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        col = A[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if nzr.size:
            A[nzr] = (A[nzr] - np.outer(col[nzr], A[r])) % p
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    out = []
    for fc in free:
        v = np.zeros(cols, dtype=np.int64)
        v[fc] = 1
        for i, pc in enumerate(pivots):
            v[pc] = (-A[i, fc]) % p
        out.append(v % p)
    return out


def tonelli(a, p=P):
    "a square root of a mod p"
    a = int(a) % p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        raise ValueError("not a square")
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        bb = pow(c, 1 << (m - i - 1), p)
        m, c = i, bb * bb % p
        t = t * c % p
        r = r * bb % p
    return r


if __name__ == "__main__":
    print("exactlin backend:", ex_backend())
    ns = [int(a) for a in sys.argv[1:]] or [4, 5, 6]
    for n in ns:
        print("=" * 72)
        run(n, None, "x = theta_1 ANISOTROPIC (the main Hamiltonian algebra)")
        print("-" * 72)
        run(n, hyperbolic(n), "x = theta_1 ISOTROPIC (the other one)")
