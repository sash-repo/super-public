#!/usr/bin/env python3
"""
Sherman's symmetries of the Duflo--Serganova functor,
A. Sherman, "On symmetries of the Duflo-Serganova functor",
Israel J. Math. 269:1 (2025) 31--66 = arXiv:2201.12808, in characteristic p.

Over C, Sherman's Theorem 3.12 says: if
    x = x_{b_1} + ... + x_{b_r},   b_i mutually orthogonal isotropic roots,
    t' = sum_i [g_{b_i}, g_{-b_i}] = <h_1, ..., h_r>   (a maximal toral
                                                        subalgebra of [x,g]),
then the Lie superalgebra

    g(x, t') := (c(t')/t')_x        (c(t') = centralizer of t' in g)

acts on the functor DS_x, contains g_x (Lemma 3.6), and equals

    g(x, t') = g_x  x  K^{0|r},

the extra odd abelian factor being spanned by the classes of the NEGATIVE root
vectors y_{b_1}, ..., y_{b_r}.

This script recomputes all of that from scratch, by exact linear algebra, for
the matrix families of Sections 3 and 4 of the text --- gl, sl, psl(n|n) and
gl, psl(n|n+pk) in the alternating format --- over Q and over F_p, p = 2,3,5,
and reports for every x_r:

  * sdim g_x                             (reproduces the tables of the text),
  * dim t' and whether t' is toral and t' <= [x,g]   (Sherman's hypotheses),
  * sdim g(x,t') and whether it equals sdim g_x + (0|r),
  * whether the classes of y_1..y_r are non-zero, odd, and central in g(x,t'),
  * whether there is h in g_ev with [h,x] = x        (the G_m of GHSS Rem 5.8).

This is Statement 1.7.4 of the text.  In tables (54)-(56) the DS-homology g_x does NOT
separate the ranks r for psl(n|n) --- that is exactly why nds(psl(n|n)) = n-1
is one less than df = n --- whereas g_x x K^{0|r} does.

Usage:  python3 sherman.py          (gl, sl, psl(n|n), n = 2, 3, 4)
        python3 sherman.py npk      (gl, psl(n|n+pk))
"""

import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank, rref, nullspace, backend          # noqa: E402


# --------------------------------------------------------------- the format

def parities(a, b):
    """Alternating format (ev,od,ev,od,...), surplus at the end -- as in the
    text and in dscheck.py."""
    par, ev, od, i = [None] * (a + b), 0, 0, 0
    while ev < a and od < b:
        par[i] = 0; ev += 1; i += 1
        par[i] = 1; od += 1; i += 1
    while ev < a:
        par[i] = 0; ev += 1; i += 1
    while od < b:
        par[i] = 1; od += 1; i += 1
    return par


class Amb:
    """The ambient associative superalgebra Mat(a|b) = gl(a|b) as K^{n*n}."""

    def __init__(self, a, b, p):
        self.a, self.b, self.p = a, b, (p or None)   # None = char 0 (over Q)
        self.n = a + b
        self.par = parities(a, b)
        self.N = self.n * self.n

    def idx(self, i, j):
        return i * self.n + j

    def zero(self):
        return [0] * self.N

    def E(self, i, j):
        v = self.zero()
        v[self.idx(i, j)] = 1
        return v

    def epar(self, i, j):
        return (self.par[i] + self.par[j]) % 2

    def norm(self, v):
        if self.p:
            return [c % self.p for c in v]
        return list(v)

    def mul(self, u, v):
        """Matrix product of two dense vectors."""
        n, w = self.n, self.zero()
        for i in range(n):
            row = u[i * n:(i + 1) * n]
            if not any(row):
                continue
            for k in range(n):
                c = row[k]
                if not c:
                    continue
                for j in range(n):
                    d = v[k * n + j]
                    if d:
                        w[i * n + j] += c * d
        return self.norm(w)

    def bracket(self, u, pu, v, pv):
        """Supercommutator [u,v] = uv - (-1)^{pu pv} vu of homogeneous u, v."""
        s = -1 if (pu and pv) else 1
        uv, vu = self.mul(u, v), self.mul(v, u)
        return self.norm([x - s * y for x, y in zip(uv, vu)])

    def homparts(self, v):
        """Split a dense vector into its even and odd parts."""
        ev, od = self.zero(), self.zero()
        n = self.n
        for i in range(n):
            for j in range(n):
                c = v[i * n + j]
                if c:
                    (od if self.epar(i, j) else ev)[i * n + j] = c
        return ev, od

    def show(self, v):
        n, out = self.n, []
        for i in range(n):
            for j in range(n):
                c = v[i * n + j]
                if c:
                    out.append(f"{c}*E{i+1},{j+1}" if c != 1 else f"E{i+1},{j+1}")
        return " + ".join(out) if out else "0"


# ------------------------------------------------------------- linear algebra

def span_rank(rows, p):
    return rank(rows, p) if rows else 0


def proj_factory(C, N, p):
    """Projection K^N ->> K^N / span(C), as a map onto the non-pivot columns."""
    if not C:
        return (lambda v: list(v)), N
    R, piv = rref(C, p)
    nonpiv = [j for j in range(N) if j not in piv]

    def f(v):
        w = list(v)
        for row, j in zip(R, piv):
            c = w[j]
            if c:
                w = [x - c * y for x, y in zip(w, row)]
                if p:
                    w = [x % p for x in w]
        return [w[j] for j in nonpiv]

    return f, len(nonpiv)


def in_span(v, rows, p):
    """Is v in the span of rows?"""
    if not rows:
        return not any(c % p if p else c for c in v)
    return rank(rows, p) == rank(rows + [list(v)], p)


# ------------------------------------------------------ the graded subquotient

class SubQuot:
    """A graded subalgebra V of the ambient, modulo a central ideal C.

    V is given by bases of its even and odd parts (lists of dense vectors);
    C likewise.  All of C must be central in V (true in every use below: C is
    either the scalars, or t', which is central in c(t')).
    """

    def __init__(self, amb, Vev, Vod, Cev=(), Cod=()):
        self.amb = amb
        self.Vev, self.Vod = list(Vev), list(Vod)
        self.Cev, self.Cod = list(Cev), list(Cod)
        p = amb.p
        self.pi, _ = proj_factory(self.Cev + self.Cod, amb.N, p)
        self.dim = (span_rank(self.Vev, p) - span_rank(self.Cev, p),
                    span_rank(self.Vod, p) - span_rank(self.Cod, p))

    def ds(self, x):
        """sdim of (V/C)_x = Ker ad_x / Im ad_x, x odd with x^2 = 0.

        The bases self.V* are linearly independent, so the dimension of
        {v in span(B) : [x,v] in C} is len(B) - rank([x,b] mod C : b in B).
        """
        amb, p = self.amb, self.amb.p
        out = []
        for eps, (B, Bother, C) in enumerate(((self.Vev, self.Vod, self.Cev),
                                              (self.Vod, self.Vev, self.Cod))):
            M = [self.pi(amb.bracket(x, 1, b, eps)) for b in B]
            ker = len(B) - (span_rank(M, p) if M else 0) - span_rank(C, p)
            IM = [self.pi(amb.bracket(x, 1, b, 1 - eps)) for b in Bother]
            im = span_rank(IM, p) if IM else 0
            out.append(ker - im)
        return tuple(out)

    def im_of_parity(self, x, eps):
        """A spanning set of the parity-eps part of Im ad_x inside V/C."""
        amb = self.amb
        src = self.Vod if eps == 0 else self.Vev
        return [self.pi(amb.bracket(x, 1, b, 1 - eps)) for b in src]

    def classes(self, vecs, x, eps=1):
        """For homogeneous vectors of parity eps: which survive in (V/C)_x
        (in Ker ad_x, not in Im ad_x + C), and how many are independent there."""
        amb, p = self.amb, self.amb.p
        IM = self.im_of_parity(x, eps)
        base = span_rank(IM, p) if IM else 0
        alive, cls = [], []
        for v in vecs:
            zero = not any(c % p if p else c
                           for c in self.pi(amb.bracket(x, 1, v, eps)))
            c = self.pi(v)
            new = not in_span(c, IM, p)
            alive.append(zero and new)
            if zero and new:
                cls.append(c)
        indep = (span_rank(IM + cls, p) - base) if cls else 0
        return alive, indep

    def ker_basis(self, x, eps):
        """Ambient basis of {v of parity eps in V : [x,v] in C}."""
        amb, p = self.amb, self.amb.p
        B = self.Vev if eps == 0 else self.Vod
        rows = [self.pi(amb.bracket(x, 1, b, eps)) for b in B]
        if not rows or not rows[0]:
            return list(B)
        MT = [[rows[i][j] for i in range(len(rows))] for j in range(len(rows[0]))]
        out = []
        for c in nullspace(MT, p):
            v = amb.zero()
            for coef, b in zip(c, B):
                if coef:
                    v = [xx + coef * yy for xx, yy in zip(v, b)]
            out.append(amb.norm(v))
        return out

    def is_central_in_ds(self, v, pv, x):
        """Is the class of v central in the homology (V/C)_x?  I.e. does
        [v, Ker ad_x] sit inside Im ad_x + C?"""
        amb, p = self.amb, self.amb.p
        for eps in (0, 1):
            IM = self.im_of_parity(x, (eps + pv) % 2)
            for b in self.ker_basis(x, eps):
                w = self.pi(amb.bracket(v, pv, b, eps))
                if not in_span(w, IM, p):
                    return False
        return True


# --------------------------------------------------------------- the algebras

def build(amb, kind):
    """Bases of the even and odd parts of gl / sl / psl, and the central ideal."""
    n = amb.n
    Vev, Vod = [], []
    if kind == "gl":
        for i in range(n):
            for j in range(n):
                (Vod if amb.epar(i, j) else Vev).append(amb.E(i, j))
        return Vev, Vod, [], []
    # sl: supertrace zero.  str(E_ii) = (-1)^{par_i}
    for i in range(n):
        for j in range(n):
            if i != j:
                (Vod if amb.epar(i, j) else Vev).append(amb.E(i, j))
    sgn = [(-1) ** amb.par[i] for i in range(n)]
    for i in range(1, n):
        # sgn[0] E_00 * ? : take v = sgn[i] E_00 - sgn[0] E_ii  (str = 0)
        v = amb.zero()
        v[amb.idx(0, 0)] = sgn[i]
        v[amb.idx(i, i)] = -sgn[0]
        Vev.append(amb.norm(v))
    one = amb.norm([1 if i == j else 0 for i in range(n) for j in range(n)])
    if kind == "sl":
        return Vev, Vod, [], []
    if kind == "psl":
        assert in_span(one, Vev, amb.p), "1 is not in sl -- psl undefined"
        return Vev, Vod, [one], []
    raise ValueError(kind)


def x_of_rank(amb, r):
    """x_r = sum_{1<=j<=r} E_{2j-1,2j} in the alternating format."""
    v = amb.zero()
    for j in range(r):
        v[amb.idx(2 * j, 2 * j + 1)] = 1
    return v


def y_of_rank(amb, r):
    return [amb.E(2 * j + 1, 2 * j) for j in range(r)]


def h_of_rank(amb, r):
    """h_j = [x_j, y_j] = E_{2j-1,2j-1} + E_{2j,2j}."""
    hs = []
    for j in range(r):
        v = amb.zero()
        v[amb.idx(2 * j, 2 * j)] = 1
        v[amb.idx(2 * j + 1, 2 * j + 1)] = 1
        hs.append(v)
    return hs


# ------------------------------------------------------------------ the checks

def ad_matrix(amb, h, B, pb):
    """ad_h on span(B), all of parity pb, in the rref basis of span(B)."""
    p = amb.p
    R, piv = rref(B, p)
    out = []
    for b in R:
        cur = amb.bracket(h, 0, b, pb)
        row = [0] * len(R)
        for k, (rr, j) in enumerate(zip(R, piv)):
            c = cur[j]
            if c:
                cur = [xx - c * yy for xx, yy in zip(cur, rr)]
                if p:
                    cur = [xx % p for xx in cur]
                row[k] = c % p if p else c
        assert not any(c % p if p else c for c in cur), "span not ad_h-stable"
        out.append(row)
    return out


def is_toral(amb, h, Vev, Vod):
    """Is ad_h semisimple with eigenvalues in the prime field?
    In char p that is ad_h^p = ad_h; in char 0 we test that the minimal
    polynomial divides X(X-1)(X+1)(X-2)(X+2) -- the only eigenvalues that
    occur for the h_j at hand."""
    return all(_toral_block(amb, ad_matrix(amb, h, B, pb))
               for B, pb in ((Vev, 0), (Vod, 1)) if B)


def _toral_block(amb, A):
    m = len(A)
    p = amb.p

    def mm(X, Y):
        Z = [[sum(X[i][k] * Y[k][j] for k in range(m)) for j in range(m)]
             for i in range(m)]
        return [[c % p for c in row] for row in Z] if p else Z

    def sub(X, lam):
        return [[X[i][j] - (lam if i == j else 0) for j in range(m)]
                for i in range(m)]

    if p:
        P = A
        for _ in range(p - 1):
            P = mm(P, A)
        return all(not ((P[i][j] - A[i][j]) % p) for i in range(m)
                   for j in range(m))
    P = mm(mm(A, sub(A, 1)), sub(A, -1))
    P = mm(P, mm(sub(A, 2), sub(A, -2)))
    return all(c == 0 for row in P for c in row)


def centralizer(amb, h_list, Vev, Vod):
    """Basis of the centralizer of <h_list> inside span(Vev)+span(Vod)."""
    p = amb.p
    out = []
    for B, pb in ((Vev, 0), (Vod, 1)):
        rows = []
        for b in B:
            img = []
            for h in h_list:
                img += amb.bracket(h, 0, b, pb)
            rows.append(img)
        # coefficient vectors c with sum_i c_i [h, b_i] = 0: right kernel of
        # the transpose of the above.
        MT = [[rows[i][j] for i in range(len(rows))]
              for j in range(len(rows[0]))] if rows else []
        ker = nullspace(MT, p) if MT else []
        basis = []
        for c in ker:
            v = amb.zero()
            for coef, b in zip(c, B):
                if coef:
                    v = [xx + coef * yy for xx, yy in zip(v, b)]
            basis.append(amb.norm(v))
        out.append(basis)
    return out


def scaling_element(amb, x, Vev, p):
    """Is there h in g_ev with [h,x] = x?  (the G_m of GHSS, Remark 5.8)"""
    rows = [amb.bracket(b, 0, x, 1) for b in Vev]
    # solve sum c_i [b_i, x] = x
    M = [[rows[i][j] for i in range(len(rows))] for j in range(amb.N)]
    from exactlin import solve
    sol = solve(M, list(x), p)
    return sol is not None


def report(a, b, kind, p, ranks=None):
    amb = Amb(a, b, p)
    p = amb.p                       # None in characteristic 0
    Vev, Vod, Cev, Cod = build(amb, kind)
    g = SubQuot(amb, Vev, Vod, Cev, Cod)
    name = f"{kind}({a}|{b})"
    pp = f"p={p if p else 0}"
    print(f"\n=== {name}, {pp} : sdim g = {g.dim[0]}|{g.dim[1]} ===")
    rmax = min(a, b)
    for r in (ranks or range(1, rmax + 1)):
        x = x_of_rank(amb, r)
        assert not any(amb.mul(x, x)), "x^2 != 0"
        gx = g.ds(x)
        hs = h_of_rank(amb, r)
        ys = y_of_rank(amb, r)
        # t' inside g (mod the centre C)
        piC, _ = proj_factory(Cev + Cod, amb.N, p)
        dim_t = span_rank([piC(h) for h in hs], p)
        # t' subset of [x,g] ?
        IM = [piC(amb.bracket(x, 1, bb, 1)) for bb in Vod]
        t_in_im = all(in_span(piC(h), IM, p) for h in hs)
        toral = all(is_toral(amb, h, Vev, Vod) for h in hs)
        # c(t') and g(x,t') = (c(t')/(t'+C))_x
        Cev2, Cod2 = centralizer(amb, hs, Vev, Vod)
        cq = SubQuot(amb, Cev2, Cod2, list(Cev) + hs, list(Cod))
        gxt = cq.ds(x)
        alive, indep = cq.classes(ys, x, 1)
        central = [cq.is_central_in_ds(y, 1, x) for y in ys]
        gm = scaling_element(amb, x, Vev, p)
        expect = (gx[0], gx[1] + r)
        flag = "= g_x x K^{0|%d}" % r if gxt == expect else \
               "DIFFERS from g_x x K^{0|%d}" % r
        print(f"  r={r}: sdim g_x = ({gx[0]}|{gx[1]}), "
              f"rank ad_x = {(g.dim[0]+g.dim[1]-gx[0]-gx[1])//2}")
        print(f"        dim t' = {dim_t} (toral {toral}, t'<=[x,g] {t_in_im}), "
              f"sdim c(t') = ({span_rank(Cev2,p)}|{span_rank(Cod2,p)})")
        print(f"        sdim g(x,t') = ({gxt[0]}|{gxt[1]})  {flag}")
        print(f"        y_j alive {sum(alive)}/{r}, independent {indep}, "
              f"central {sum(central)}/{r}; exists h with [h,x]=x: {gm}")


if __name__ == "__main__":
    print(f"exactlin backend: {backend()}")
    which = sys.argv[1] if len(sys.argv) > 1 else "nn"
    if which == "nn":
        for p in (0, 2, 3, 5):
            for n in (2, 3, 4):
                for kind in ("gl", "sl", "psl"):
                    report(n, n, kind, p)
    elif which == "npk":
        # gl, psl(n|n+pk) of Section 4
        for p, cases in ((2, [(1, 3), (2, 4), (3, 5), (2, 6)]),
                         (3, [(1, 4), (2, 5), (3, 6)]),
                         (5, [(1, 6), (2, 7)])):
            for (n, m) in cases:
                for kind in ("gl", "psl"):
                    report(n, m, kind, p)
