"""Non-split supermanifolds (CP^1, O) of dimension 1|m: exact engine.

Everything is done with honest transition functions on the standard cover of CP^1
by U (coordinate x) and V (coordinate y = 1/x), and exact rational arithmetic.

Conventions (those of Leites--Tikhomirov, arXiv:2503.02416, and of
Bunegina--Onishchik, J. Math. Sci. 82 (1996) 3503--3527):

  the line bundle L_n has transition  eta = x^n xi,  so O(n) := sheaf of sections
  of L_n; the retract of our supermanifolds corresponds to E = + L_{-k_j}, i.e.,
  the SPLIT transition functions are

      y = x^{-1},   eta_j = x^{-k_j} xi_j.

A non-split structure is produced from a cocycle z (a degree-2 derivation given
on U cap V) by composing the split transition with exp(z), which is what
Bunegina--Onishchik's formula (47) does.

A vector field is stored by its action on the coordinates of U:
    D = A d/dx + sum_j B_j d/dxi_j          (coefficients on the LEFT,
                                             derivatives are the LEFT ones),
and then D = D(y) d/dy + sum_j D(eta_j) d/deta_j in the chart V, so that D is
global iff A, B_j are regular on U and D(y), D(eta_j) are regular on V.

"Regular on V" is tested WITHOUT inverting the coordinate change: O(V) is the
span of the products Y^p H_S with p >= 0, and membership in that span is one
exact linear-algebra problem.
"""

from fractions import Fraction
from itertools import combinations
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import nullspace, rank            # noqa: E402


# ---------------------------------------------------------------- Laurent part

def lp(*pairs):
    """Laurent polynomial from (exponent, coefficient) pairs."""
    out = {}
    for e, c in pairs:
        c = Fraction(c)
        if c:
            out[e] = out.get(e, Fraction(0)) + c
    return {e: c for e, c in out.items() if c}


def lp_mul(f, g):
    out = {}
    for e1, c1 in f.items():
        for e2, c2 in g.items():
            e = e1 + e2
            out[e] = out.get(e, Fraction(0)) + c1 * c2
    return {e: c for e, c in out.items() if c}


def lp_add(f, g):
    out = dict(f)
    for e, c in g.items():
        out[e] = out.get(e, Fraction(0)) + c
    return {e: c for e, c in out.items() if c}


def lp_scal(f, c):
    c = Fraction(c)
    return {} if not c else {e: v * c for e, v in f.items()}


def lp_dx(f):
    return {e - 1: c * e for e, c in f.items() if e != 0}


def lp_pow(f, n):
    """f^n for a MONOMIAL f = c x^e and any integer n."""
    assert len(f) == 1, "lp_pow is only used on monomials"
    (e, c), = f.items()
    return {e * n: c ** n if n >= 0 else Fraction(1, 1) / (c ** (-n))}


# ------------------------------------------------------- Grassmann coefficients

# A GElem is a dict  S -> Laurent polynomial,  where S is a sorted tuple of odd
# indices (1-based); it represents  sum_S f_S(x) xi_S  with xi_S = xi_{s1}...

def g_zero():
    return {}


def g_from(S, f):
    return {tuple(S): dict(f)} if f else {}


def g_add(a, b):
    out = {S: dict(f) for S, f in a.items()}
    for S, f in b.items():
        out[S] = lp_add(out.get(S, {}), f)
    return {S: f for S, f in out.items() if f}


def g_scal(a, c):
    return {S: lp_scal(f, c) for S, f in a.items() if lp_scal(f, c)}


def _merge_sign(S, T):
    """sign and union of xi_S * xi_T, or None if they share an index."""
    if set(S) & set(T):
        return None, None
    merged = list(S) + list(T)
    sign = 1
    # bubble sort, counting transpositions
    arr = merged[:]
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sign = -sign
    return sign, tuple(arr)


def g_mul(a, b):
    out = {}
    for S, f in a.items():
        for T, g in b.items():
            sign, U = _merge_sign(S, T)
            if sign is None:
                continue
            prod = lp_scal(lp_mul(f, g), sign)
            if prod:
                out[U] = lp_add(out.get(U, {}), prod)
    return {S: f for S, f in out.items() if f}


def g_dx(a):
    return {S: lp_dx(f) for S, f in a.items() if lp_dx(f)}


def g_dxi(a, j):
    """LEFT derivative with respect to xi_j."""
    out = {}
    for S, f in a.items():
        if j not in S:
            continue
        pos = S.index(j)
        sign = (-1) ** pos
        T = tuple(u for u in S if u != j)
        out[T] = lp_add(out.get(T, {}), lp_scal(f, sign))
    return {S: f for S, f in out.items() if f}


def g_parity(a):
    """0/1 if homogeneous, None if not."""
    par = {len(S) % 2 for S in a}
    return par.pop() if len(par) == 1 else (0 if not par else None)


# ------------------------------------------------------------ the supermanifold

class SM:
    """(CP^1, O) of dimension 1|m given by transition functions on U cap V."""

    def __init__(self, ks, Y, H, name="", npar=0):
        self.m = len(ks)
        self.ks = list(ks)
        self.Y = Y                      # y as a function of (x, xi)
        self.H = list(H)                # eta_j as functions of (x, xi)
        self.name = name
        self.npar = npar                # odd parameters of the base, indexed
        #                                 m+1, ...: they are NOT coordinates

    # --- constructors -------------------------------------------------------

    @staticmethod
    def split(ks, name="", npar=0):
        m = len(ks)
        Y = g_from((), lp((-1, 1)))
        H = [g_from((j + 1,), lp((-ks[j], 1))) for j in range(m)]
        return SM(ks, Y, H, name or f"split{tuple(ks)}", npar)

    @staticmethod
    def from_cocycle(ks, zx=None, zxi=None, name="", npar=0):
        """Compose the split transition with exp(z).

        zx   : GElem, the value z(x)      (must square to zero here)
        zxi  : dict j -> GElem, the values z(xi_j)
        """
        m = len(ks)
        zx = zx or g_zero()
        zxi = zxi or {}
        # x' = x + z(x)   (exp(z) = id + z on the coordinates for our cocycles)
        nil = zx
        assert not g_mul(nil, nil), "z(x)^2 must vanish"
        xp = g_add(g_from((), lp((1, 1))), nil)
        xip = [g_add(g_from((j + 1,), lp((0, 1))), zxi.get(j + 1, g_zero()))
               for j in range(m)]
        Y = _pow_nilpotent(xp, -1)
        H = [g_mul(_pow_nilpotent(xp, -ks[j]), xip[j]) for j in range(m)]
        return SM(ks, Y, H, name or f"ns{tuple(ks)}", npar)

    # --- the algebra of global vector fields --------------------------------

    def regular_V_basis(self, deg):
        """Spanning set of O(V) truncated: Y^p H_S for 0 <= p <= deg."""
        out = []
        pars = _subsets_of(range(self.m + 1, self.m + self.npar + 1))
        for S in _subsets(self.m):
            base = g_from((), lp((0, 1)))
            for j in S:
                base = g_mul(base, self.H[j - 1])
            for P in pars:                    # the base is Lambda(tau_1, ...)
                b = g_mul(g_from(P, lp((0, 1))), base) if P else base
                cur = b
                for p in range(deg + 1):
                    if cur:
                        out.append(cur)
                    cur = g_mul(cur, self.Y)
        return out

    def vector_fields(self, parity, degU=12, degV=None):
        """Basis of the global vector fields of the given parity (0 or 1).

        Returns a list of (A, [B_1..B_m]) with exact rational coefficients.
        degU bounds the x-degree of the coefficients on U; the answer is
        checked for stability against degU by the caller.
        """
        degV = degV or degU + 4 + max(self.ks + [0])
        m = self.m
        # ---- unknowns: the coefficients of A and of the B_j on U
        slots = []                       # (which, S, p)
        allS = _subsets(m + self.npar)
        for S in allS:
            if len(S) % 2 == parity:     # A has the parity of D
                for p in range(degU + 1):
                    slots.append(("A", S, p))
        for j in range(1, m + 1):
            for S in allS:
                if len(S) % 2 == (parity + 1) % 2:   # B_j has the other parity
                    for p in range(degU + 1):
                        slots.append((("B", j), S, p))
        # ---- the images D(Y), D(eta_j) as linear functions of the unknowns
        targets = [self.Y] + self.H
        cols = []
        for which, S, p in slots:
            A = g_zero()
            B = [g_zero()] * m
            mono = g_from(S, lp((p, 1)))
            if which == "A":
                A = mono
            else:
                B = list(B)
                B[which[1] - 1] = mono
            cols.append([_apply(A, B, T) for T in targets])
        # ---- the subspace of things regular on V
        regV = self.regular_V_basis(degV)
        # ---- assemble one big linear system: sum_u c_u D_u(T) - sum_w d_w w = 0
        nslot, nreg = len(slots), len(regV)
        rows = {}                        # monomial key -> row index
        entries = []                     # (row, col, value)

        def put(elem, col, sign=1):
            for S, f in elem.items():
                for e, c in f.items():
                    key = (S, e)
                    r = rows.setdefault(key, len(rows))
                    entries.append((r, col, sign * c))

        for u in range(nslot):
            for t in range(len(targets)):
                for S, f in cols[u][t].items():
                    for e, c in f.items():
                        key = (t, S, e)
                        r = rows.setdefault(key, len(rows))
                        entries.append((r, u, c))
        for t in range(len(targets)):
            for w, elem in enumerate(regV):
                for S, f in elem.items():
                    for e, c in f.items():
                        key = (t, S, e)
                        r = rows.setdefault(key, len(rows))
                        entries.append((r, nslot + t * nreg + w, -c))
        ncol = nslot + len(targets) * nreg
        mat = [[Fraction(0)] * ncol for _ in range(len(rows))]
        for r, c, v in entries:
            mat[r][c] += v
        ns = nullspace(mat) if mat else []
        # ---- project onto the slot coordinates and re-extract a basis
        proj = [list(v[:nslot]) for v in ns]
        proj = [v for v in proj if any(v)]
        if not proj:
            return []
        basis = _row_basis(proj)
        fields = []
        for v in basis:
            A = g_zero()
            B = [g_zero() for _ in range(m)]
            for c, (which, S, p) in zip(v, slots):
                if not c:
                    continue
                mono = g_from(S, lp((p, c)))
                if which == "A":
                    A = g_add(A, mono)
                else:
                    B[which[1] - 1] = g_add(B[which[1] - 1], mono)
            fields.append((A, B))
        return fields

    # --- transitivity -------------------------------------------------------

    def transitive(self, degU=12, verbose=False):
        """Is  v(M,O) -> T_p(M,O)  onto at EVERY point of CP^1?

        Point sampling cannot prove this: the locus where the evaluation drops
        rank is the common zero set of polynomials, so one tests it by a gcd.
        On the chart U the even part is onto off the zeros of the gcd of the
        A's, and the odd part is onto off the zeros of the gcd of the maximal
        minors of the matrix of the B's.  The point y = 0 is then read off in
        the chart V.
        """
        ev = self.evaluations(degU)
        bad = []
        for chart in ("U", "V"):
            e = _drop_locus(ev[chart], self.m)
            if e["even"] is not None:
                bad.append((chart, "even", e["even"]))
            if e["odd"] is not None:
                bad.append((chart, "odd", e["odd"]))
            if verbose:
                print(f"    chart {chart}: even {'onto' if e['even'] is None else 'drops on ' + _pstr(e['even'])}"
                      f", odd {'onto' if e['odd'] is None else 'drops on ' + _pstr(e['odd'])}")
        # a drop on the chart U at a point that lies in U is fatal; the gcd may
        # also vanish only "at infinity", which the chart V settles.
        for chart, which, g in bad:
            if _has_affine_root(g):
                return False
        return True

    def evaluations(self, degU=12):
        """Values of the fields on the reduction, in both charts."""
        evU = {"even": [], "odd": []}
        evV = {"even": [], "odd": []}
        for par, tag in ((0, "even"), (1, "odd")):
            for A, B in self.vector_fields(par, degU):
                evU[tag].append((A.get((), {}), [b.get((), {}) for b in B]))
                # the same field in the chart V
                Ay = _apply(A, B, self.Y)
                Be = [_apply(A, B, h) for h in self.H]
                # restrict to the reduction and rewrite x = 1/y
                evV[tag].append((_flip(Ay.get((), {})),
                                 [_flip(b.get((), {})) for b in Be]))
        return {"U": evU, "V": evV}

    def sdim_vect(self, degU=12):
        return (len(self.vector_fields(0, degU)),
                len(self.vector_fields(1, degU)))


# ------------------------------------------------------------------- utilities

def _subsets(m):
    return _subsets_of(range(1, m + 1))


def _subsets_of(idx):
    idx = list(idx)
    out = []
    for r in range(len(idx) + 1):
        out.extend(combinations(idx, r))
    return out


def _apply(A, B, F):
    """D(F) for D = A d/dx + sum B_j d/dxi_j."""
    out = g_mul(A, g_dx(F))
    for j, b in enumerate(B, start=1):
        out = g_add(out, g_mul(b, g_dxi(F, j)))
    return out


def _pow_nilpotent(elem, n):
    """(x + nu)^n for nu nilpotent, exact, via the binomial series."""
    body = g_from((), elem.get((), {}))
    nu = {S: f for S, f in elem.items() if S != ()}
    assert len(body.get((), {})) == 1, "the body must be a monomial"
    res = g_from((), lp_pow(body[()], n))
    term = g_from((), lp_pow(body[()], n))
    binom = Fraction(1)
    cur = nu
    i = 1
    while cur:
        binom = binom * Fraction(n - i + 1, i)
        # (x+nu)^n = x^n sum_i C(n,i) (nu/x)^i
        piece = g_scal(g_mul(g_from((), lp_pow(body[()], n - i)), cur), binom)
        res = g_add(res, piece)
        cur = g_mul(cur, nu)
        i += 1
        if i > 10:
            break
    del term
    return res


def _flip(f):
    """Rewrite a Laurent polynomial in x as one in y = 1/x."""
    return {-e: c for e, c in f.items()}


def _row_basis(rows):
    """A basis of the row space, as a subset-free echelon basis."""
    mat = [list(r) for r in rows]
    piv, basis = [], []
    for r in mat:
        cur = list(r)
        for p, b in zip(piv, basis):
            if cur[p]:
                f = cur[p] / b[p]
                cur = [a - f * c for a, c in zip(cur, b)]
        nz = next((i for i, c in enumerate(cur) if c), None)
        if nz is not None:
            piv.append(nz)
            basis.append(cur)
    return basis


# ------------------------------------------------- polynomials for the gcd test

def _poly(f):
    """Laurent dict -> coefficient list (must be a genuine polynomial)."""
    if not f:
        return []
    assert min(f) >= 0, f"not regular on this chart: {f}"
    return [f.get(i, Fraction(0)) for i in range(max(f) + 1)]


def _pnorm(p):
    while p and not p[-1]:
        p.pop()
    return p


def _pmul(a, b):
    if not a or not b:
        return []
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        if u:
            for j, v in enumerate(b):
                out[i + j] += u * v
    return _pnorm(out)


def _psub(a, b):
    n = max(len(a), len(b))
    out = [(a[i] if i < len(a) else Fraction(0))
           - (b[i] if i < len(b) else Fraction(0)) for i in range(n)]
    return _pnorm(out)


def _pmod(a, b):
    a, b = _pnorm(list(a)), _pnorm(list(b))
    while a and len(a) >= len(b):
        f = a[-1] / b[-1]
        sh = len(a) - len(b)
        a = _psub(a, [Fraction(0)] * sh + [f * c for c in b])
    return a


def _pgcd(a, b):
    a, b = _pnorm(list(a)), _pnorm(list(b))
    while b:
        a, b = b, _pmod(a, b)
    return [c / a[-1] for c in a] if a else []


def _pgcd_all(ps):
    g = []
    for p in ps:
        g = _pgcd(g, p) if g else _pnorm(list(p))
        if len(g) == 1:
            return g
    return g


def _pdet(rows):
    """Determinant of a square matrix of polynomials, Laplace along row 0."""
    n = len(rows)
    if n == 1:
        return _pnorm(list(rows[0][0]))
    out = []
    for j in range(n):
        minor = [[rows[i][c] for c in range(n) if c != j] for i in range(1, n)]
        term = _pmul(rows[0][j], _pdet(minor))
        if j % 2:
            term = _psub([], term)
        out = _psub([], _psub([], _padd(out, term)))
    return _pnorm(out)


def _padd(a, b):
    return _psub(a, _psub([], b))


def _drop_locus(ev, m):
    """gcd of the even evaluations and of the maximal minors of the odd ones.

    Returns {"even": g or None, "odd": g or None}; None means "onto at every
    point of this chart", a polynomial means "drops exactly at its roots".
    """
    out = {}
    evens = [_poly(A) for A, _ in ev["even"]]
    g = _pgcd_all([p for p in evens if p])
    out["even"] = None if len(g) == 1 else (g or [Fraction(0)])
    rows = [[_poly(b) for b in B] for _, B in ev["odd"]]
    if m == 0:
        out["odd"] = None
        return out
    minors = []
    for sel in combinations(range(len(rows)), m):
        minors.append(_pdet([rows[i] for i in sel]))
    minors = [p for p in minors if p]
    g = _pgcd_all(minors) if minors else []
    out["odd"] = None if len(g) == 1 else (g or [Fraction(0)])
    return out


def _pstr(g):
    if g == [Fraction(0)] or not g:
        return "all of it"
    return " + ".join(f"{c}x^{i}" for i, c in enumerate(g) if c)


def _has_affine_root(g):
    """A non-constant polynomial has a root in the algebraically closed field;
    the zero polynomial (stored as [0]) means the rank drops identically."""
    return not (len(g) == 1 and g[0] != 0)


# ------------------------------------------------------- cohomological formulas

def h0(n):
    return n + 1 if n >= 0 else 0


def h1(n):
    return -n - 1 if n <= -2 else 0


def summands_naive(ks, q):
    """The WRONG prediction: T_q read as the direct sum
    Lambda^q E (x) Theta  +  Lambda^{q+1} E (x) E^*.  Kept because it is the
    trap: those two are the quotient and the sub of a NON-SPLIT extension."""
    m = len(ks)
    out = []
    if q >= 0:
        for I in combinations(range(m), q):
            out.append(2 - sum(ks[i] for i in I))
    for J in combinations(range(m), q + 1):
        for r in range(m):
            out.append(ks[r] - sum(ks[i] for i in J))
    return out


def summands_BO(ks, q):
    """The degrees of the line-bundle summands of T_q, after B--O's Prop. 18.

    T_q = (+_{|I|=q} ST^{(I)}_q) + (+_{|J|=q+1, r not in J} L_{k_r - k_J}),
    where ST^{(I)}_q is spanned by xi_I d/dx and xi_I xi_j d/dxi_j (j not in I)
    and is isomorphic to
        (m-q) L_{-k_I} + L_{2-k_I}      if k_i = 0 for all i not in I,
        (m-q-1) L_{-k_I} + 2 L_{1-k_I}  otherwise.
    """
    m = len(ks)
    out = []
    if q >= 0:
        for I in combinations(range(m), q):
            kI = sum(ks[i] for i in I)
            rest = [i for i in range(m) if i not in I]
            if all(ks[i] == 0 for i in rest):
                out += [-kI] * (m - q) + [2 - kI]
            else:
                out += [-kI] * (m - q - 1) + [1 - kI] * 2
    for J in combinations(range(m), q + 1):
        kJ = sum(ks[i] for i in J)
        for r in range(m):
            if r not in J:
                out.append(ks[r] - kJ)
    return out


def sdim_split_predicted(ks, naive=False):
    """sdim v(M, Lambda E) as  + H^0(T_q),  E = + O(-k_j)."""
    m = len(ks)
    ev = od = 0
    f = summands_naive if naive else summands_BO
    for q in range(-1, m + 1):
        d = sum(h0(n) for n in f(ks, q))
        if q % 2 == 0:
            ev += d
        else:
            od += d
    return ev, od
