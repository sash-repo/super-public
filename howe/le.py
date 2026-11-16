"""
le.py -- exact engine for the Lie superalgebra le(n) = Le(n) (Leites/Buttin algebra)
and for its induced modules I(V) = U(le(n)) (x)_{U(le(n)_{>=0})} V.

Conventions follow Kochetkov, "Discrete modules for the Lie superalgebras of the
series Le" (Yaroslavl 1988), pp. 142-147:

    W(n,n)  = vector fields in x_1..x_n (even), xi_1..xi_n (odd),
    omega   = sum_i dx_i dxi_i,
    le(n)   = { D in W(n,n) : L_D(omega) = 0 } = Pi(C[x,xi]) / C.1  with the antibracket.

A generating function f (a polynomial in x, xi) gives the vector field

    D_f = sum_i ( d f/d x_i * d/d xi_i  +  (-1)^{p(f)} d f/d xi_i * d/d x_i ),
    p(D_f) = p(f) + 1,      deg D_f = deg f - 2.

This normalisation is fixed by Kochetkov's Cartan subalgebra:
    e_i = D_{x_i xi_i} = xi_i delta_i - x_i partial_i,          delta_i = d/d xi_i.

Weights are taken in the basis (e_1,...,e_n); the e_i-weight of D_g for a monomial
g = x^a xi^S is  (#S cap {i}) - a_i.  So

    partial_i := -D_{xi_i}  has weight  eps_i,      delta_i := D_{x_i}  has weight -eps_i.

Positive roots: ht(nu) = sum_i (n+1-i) nu_i > 0.  Then
    n_+  = < D_{x_i xi_j} : j < i >  +  < D_{xi_i xi_j} : i < j >     (pe(n)_{+1} = Lambda^2)
    n_-  = < D_{x_i xi_j} : j > i >  +  < D_{x_i x_j}   : i <= j >    (pe(n)_{-1} = S^2)
    h    = < D_{x_i xi_i} >
and L_0 = pe(n), L_{-1} = <partial_i, delta_i> (commutative, sdim n|n).

Everything is exact (Fraction).  No dependencies.
"""

from fractions import Fraction as F
from itertools import combinations

# ---------------------------------------------------------------- polynomials
# A monomial is (a, S): a = tuple of x-exponents (length n), S = sorted tuple of
# xi-indices.  A polynomial is a dict {monomial: Fraction}.


def mono_par(m):
    return len(m[1]) % 2


def merge_sign(S1, S2):
    """sign of xi^S1 * xi^S2 when sorted; None if the product vanishes."""
    if set(S1) & set(S2):
        return None, None
    sign = 1
    for a in reversed(S1):
        # move a leftwards past the elements of S2 smaller than a
        sign *= (-1) ** sum(1 for b in S2 if b < a)
    return sign, tuple(sorted(S1 + S2))


def pmul(p, q):
    out = {}
    for (a1, S1), c1 in p.items():
        for (a2, S2), c2 in q.items():
            sg, S = merge_sign(S1, S2)
            if sg is None:
                continue
            a = tuple(u + v for u, v in zip(a1, a2))
            k = (a, S)
            out[k] = out.get(k, 0) + sg * c1 * c2
    return {k: v for k, v in out.items() if v}


def padd(p, q, c=1):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, 0) + c * v
        if out[k] == 0:
            del out[k]
    return out


def pscal(p, c):
    if c == 0:
        return {}
    return {k: c * v for k, v in p.items()}


def dx(p, i):
    out = {}
    for (a, S), c in p.items():
        if a[i] == 0:
            continue
        b = list(a)
        b[i] -= 1
        k = (tuple(b), S)
        out[k] = out.get(k, 0) + c * a[i]
    return {k: v for k, v in out.items() if v}


def dxi(p, i):
    """left derivative d/d xi_i"""
    out = {}
    for (a, S), c in p.items():
        if i not in S:
            continue
        pos = S.index(i)
        k = (a, S[:pos] + S[pos + 1:])
        out[k] = out.get(k, 0) + c * (-1) ** pos
    return {k: v for k, v in out.items() if v}


def is_hom_par(p):
    ps = {mono_par(m) for m in p}
    assert len(ps) <= 1, "inhomogeneous parity"
    return ps.pop() if ps else 0


def pdeg(p):
    ds = {sum(a) + len(S) for (a, S) in p}
    assert len(ds) <= 1, "inhomogeneous degree"
    return ds.pop() if ds else None


def wt(m, n):
    a, S = m
    return tuple((1 if i in S else 0) - a[i] for i in range(n))


def pwt(p, n):
    ws = {wt(m, n) for m in p}
    assert len(ws) == 1, "not a weight vector"
    return ws.pop()


# ------------------------------------------------------- vector fields (check)
class VF:
    """vector field: comp[g] = D(g) for g = 0..n-1 (x_g), n..2n-1 (xi_{g-n})"""

    def __init__(self, n, comp, par):
        self.n = n
        self.comp = {g: v for g, v in comp.items() if v}
        self.par = par

    def apply(self, p):
        n = self.n
        out = {}
        for g, coef in self.comp.items():
            d = dx(p, g) if g < n else dxi(p, g - n)
            out = padd(out, pmul(coef, d))
        return out

    def bracket(self, other):
        n = self.n
        sg = (-1) ** (self.par * other.par)
        comp = {}
        for g in range(2 * n):
            gp = {((0,) * n, (g - n,)): F(1)} if g >= n else {(tuple(1 if k == g else 0 for k in range(n)), ()): F(1)}
            comp[g] = padd(self.apply(other.comp.get(g, {})),
                           other.apply(self.comp.get(g, {})), -sg)
        return VF(n, comp, (self.par + other.par) % 2)


def D(f, n):
    """the vector field D_f of the generating function f"""
    q = is_hom_par(f)
    comp = {}
    for i in range(n):
        comp[n + i] = dx(f, i)                    # D_f(xi_i)
        comp[i] = pscal(dxi(f, i), (-1) ** q)     # D_f(x_i)
    return VF(n, comp, (q + 1) % 2)


# ------------------------------------------------------------- the antibracket
def abr(f, g, n):
    """{f,g} with D_{{f,g}} = [D_f, D_g]"""
    q = is_hom_par(f)
    out = {}
    for i in range(n):
        out = padd(out, pmul(dx(f, i), dxi(g, i)))
        out = padd(out, pmul(dxi(f, i), dx(g, i)), (-1) ** q)
    return out


def par_le(f):
    """parity of D_f"""
    return (is_hom_par(f) + 1) % 2


# ------------------------------------------------------------- linear algebra
def rref(rows, ncols):
    """rows: list of dict col->Fraction.  Returns (pivots, reduced rows)."""
    rows = [dict(r) for r in rows]
    piv = []
    out = []
    for r in rows:
        for p, q in zip(piv, out):
            if p in r:
                c = r[p]
                for k, v in q.items():
                    r[k] = r.get(k, 0) - c * v
                    if r[k] == 0:
                        del r[k]
        if not r:
            continue
        p = min(r)
        c = r[p]
        r = {k: F(v, 1) / c for k, v in r.items()}
        for i, q in enumerate(out):
            if p in q:
                c2 = q[p]
                for k, v in r.items():
                    q[k] = q.get(k, 0) - c2 * v
                    if q[k] == 0:
                        del q[k]
        piv.append(p)
        out.append(r)
    order = sorted(range(len(piv)), key=lambda i: piv[i])
    return [piv[i] for i in order], [out[i] for i in order]


def nullspace(rows, cols):
    """rows = list of dict col->val (equations).  cols = ordered list of unknowns.
    Returns a basis of the kernel as list of dict col->val."""
    idx = {c: i for i, c in enumerate(cols)}
    eqs = [{idx[k]: v for k, v in r.items() if k in idx} for r in rows]
    eqs = [e for e in eqs if e]
    piv, red = rref(eqs, len(cols))
    free = [i for i in range(len(cols)) if i not in set(piv)]
    basis = []
    for fcol in free:
        sol = {fcol: F(1)}
        for p, r in zip(piv, red):
            if fcol in r:
                sol[p] = -r[fcol]
        basis.append({cols[k]: v for k, v in sol.items() if v})
    return basis
