"""
Exact engine: deformations of the distinguished stringy Lie
superalgebras, i.e. the second cohomology H^2(g; g) of a Z-graded Lie
superalgebra with LAURENT coefficients, and its comparison with the
POLYNOMIAL case.

Everything is exact rational arithmetic.  The brackets are not re-derived:
they are imported from `yb.py`, where each of the twelve
distinguished stringy Lie superalgebras was built and Jacobi-checked.

What is computed
----------------
A deformation of the bracket is an even 2-cochain c with

    [x,y]_eps := [x,y] + eps c(x,y),      eps^2 = 0,

and the Jacobi identity modulo eps^2 is the cocycle condition

    J_c(x,y,z) := (-1)^{p_x p_z} ( [c(x,y),z] + c([x,y],z) )
                + (-1)^{p_y p_x} ( [c(y,z),x] + c([y,z],x) )
                + (-1)^{p_z p_y} ( [c(z,x),y] + c([z,x],y) )  =  0.

We never write down a Chevalley-Eilenberg differential with its signs: the
identity above IS the definition used, and it is calibrated in `test_defco.py`
against the undeformed bracket (c = the bracket itself is a cocycle, being the
derivative of the rescaling family) and against sl(2), where H^2 = 0.

A coboundary is  (delta phi)(x,y) = [phi x, y] + [x, phi y] - phi([x,y])
for an even degree-preserving phi (the infinitesimal of conjugation by 1+eps phi).

The window method
-----------------
g = sum_d g_d has finite-dimensional homogeneous components but is infinite.
We work in a degree window W = [-N, N] (or [lo, N] for polynomial
coefficients):

  * unknowns   c(e_i, e_j) for deg e_i, deg e_j, deg e_i + deg e_j in W;
  * equations  J_c(e_i,e_j,e_k) = 0 for the triples all of whose pairwise
               degree sums lie in W (so that every c-value the identity asks
               for is one of the unknowns);
  * Z = the solution space, B = the image of delta on the phi supported in W.

Since the equations are a SUBSET of the full cocycle identities, Z contains
the restriction of every true cocycle, and B is contained in Z.  Restricting
both to a smaller CORE window [-M, M] gives

        h(N, M) := dim Z|_core - dim B|_core,

a decreasing-in-N upper bound for the number of classes of H^2 visible on the
core.  h(N,M) = 0 for growing N is the computational form of "every cocycle is
a coboundary on the core".

Weights
-------
If g carries an INNER grading element (an E in g_0 with [E,x] = (deg x) x),
then H^*(g;g) is concentrated in weight 0 by Cartan's formula, and the
degree-preserving cochains above are the whole story.  `grading_element`
decides this for each algebra; where E is only an outer derivation
(svect^L_lambda(1|2) for generic lambda) the other weights are computed too,
by `weight` != 0.
"""

from fractions import Fraction as Q
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from exactlin import rank, nullspace                      # FLINT if present
import yb
from stringy import add, smul


# =========================================================== graded algebra

class Graded:
    """A yb.py algebra presented by its homogeneous components.

    A basis vector is an ELEMENT (a dict key -> Fraction), so that the
    divergence-free algebras, whose homogeneous components are kernels, are
    handled by the same code as the monomial ones.
    """

    def __init__(self, alg, name=None, polynomial=False):
        self.alg = alg
        self.name = name or getattr(alg, "name", alg.__class__.__name__)
        self.polynomial = polynomial          # a >= 0 instead of a in Z
        self._bas = {}
        self._coord = {}
        self._step = None

    # ----------------------------------------------------------- the basis

    def basis(self, d):
        """List of (element, parity) spanning g_d; cached."""
        if d in self._bas:
            return self._bas[d]
        out = []
        for p in (0, 1):
            if hasattr(self.alg, "homog_basis"):            # svect: a kernel
                vecs = self.alg.homog_basis(d, (p,))
            elif hasattr(self.alg, "homog_keys"):
                vecs = [{k: Q(1)} for k in self.alg.homog_keys(d, (p,))]
            else:                                           # witt, m^L(1)
                vecs = [{k: Q(1)} for k in self._keys_of_degree(d)
                        if self.alg.par(k) == p]
            for v in vecs:
                if self.polynomial and any(self._expo(k) < 0 for k in v):
                    continue
                out.append((v, p))
        self._bas[d] = out
        return out

    def _expo(self, key):
        return self.alg.expo(key) if hasattr(self.alg, "expo") else key[0]

    def _keys_of_degree(self, d, pad=4):
        """The keys of degree d.  The span of the Laurent exponent must follow
        d: deg is affine in it, so |a| <= |d| + pad always suffices here."""
        span = abs(d) + pad
        return [k for k in self.alg.keys(range(-span, span + 1))
                if self.alg.deg(k) == d]

    def dim(self, d):
        return len(self.basis(d))

    def step(self, span=12):
        """The gcd of the occupied degrees.

        yb.py grades the contact algebras by the CONTACT degree (deg t = 2),
        so witt and k^M(1|1) occupy the even degrees only; a window must be
        counted in steps of that gcd or half of it is empty.
        """
        if self._step is None:
            from math import gcd
            s = 0
            for d in range(-span, span + 1):
                if self.dim(d):
                    s = gcd(s, abs(d))
            self._step = s or 1
        return self._step

    def degrees(self, N):
        """The window of 2N+1 (or, for polynomial coefficients, N+1) steps."""
        st = self.step()
        if self.polynomial:
            lo = min(d for d in range(-4 * st, 1) if self.dim(d))
            return [d for d in range(lo, N * st + 1) if (d - lo) % st == 0]
        return [st * m for m in range(-N, N + 1)]

    # ------------------------------------------------- coordinates in g_d

    def coords(self, d, elt):
        """Express an element of degree d in the basis of g_d."""
        if not elt:
            return [Q(0)] * self.dim(d)
        bas = self.basis(d)
        if not bas:
            raise ValueError("nonzero element in an empty component %d" % d)
        keys, piv = self._coord.get(d, (None, None))
        if keys is None:
            keys = sorted({k for v, _ in bas for k in v}, key=repr)
            cols = [[v.get(k, Q(0)) for k in keys] for v, _ in bas]
            piv = _rref_solver(cols, len(keys))
            self._coord[d] = (keys, piv)
        rhs = [elt.get(k, Q(0)) for k in keys]
        sol = piv(rhs)
        if sol is None:
            raise ValueError("element of degree %d not in the span" % d)
        return sol

    # ------------------------------------------------------- the bracket

    def brk(self, x, y):
        return self.alg.brk(x, y)

    def homog_degree(self, elt):
        ds = {self.alg.deg(k) for k in elt}
        if len(ds) != 1:
            raise ValueError("not homogeneous: %r" % (ds,))
        return ds.pop()

    # -------------------------------------------------- the grading element

    def grading_element(self, probe=3):
        """Return E in g_0 with [E,x] = (deg x) x, or None if there is none.

        The test is a linear one: solve for the coefficients of E in the basis
        of g_0 against the requirement on every basis vector of degree in
        [-probe, probe].
        """
        b0 = self.basis(0)
        if not b0:
            return None
        cols, rhs = [], []
        targets = []
        for d in range(-probe, probe + 1):
            for v, _ in self.basis(d):
                targets.append((d, v))
        rows = {}
        for j, (E, _) in enumerate(b0):
            for i, (d, v) in enumerate(targets):
                w = self.brk(E, v)
                for k, c in w.items():
                    rows.setdefault((i, k), {})[j] = Q(c)
        for i, (d, v) in enumerate(targets):
            for k, c in v.items():
                rows.setdefault((i, k), {})
        keys = sorted(rows, key=repr)
        mat = [[rows[k].get(j, Q(0)) for j in range(len(b0))] for k in keys]
        tgt = []
        for (i, k) in keys:
            d, v = targets[i]
            tgt.append(Q(d) * v.get(k, Q(0)))
        sol = _solve(mat, tgt)
        if sol is None:
            return None
        E = {}
        for j, (v, _) in enumerate(b0):
            if sol[j]:
                E = add(E, smul(sol[j], v))
        return E


# ====================================================== cochains in a window

class Window:
    """The weight-w even (or odd) 2-cochains supported on a degree window."""

    def __init__(self, g, degs, weight=0, cochain_parity=0):
        self.g = g
        self.degs = list(degs)
        self.win = set(self.degs)
        self.w = weight
        self.cp = cochain_parity
        # -------- the basis vectors of the window, globally indexed
        self.vec = []                       # (elt, parity, degree)
        self.at = {}                        # degree -> list of global indices
        for d in self.degs:
            self.at[d] = []
            for v, p in g.basis(d):
                self.at[d].append(len(self.vec))
                self.vec.append((v, p, d))
        # -------- the pairs and the unknowns
        self.pairs = {}                     # (i,j) i<=j -> first unknown index
        self.ptarget = {}                   # (i,j) -> list of target indices
        n = 0
        for i in range(len(self.vec)):
            vi, pi, di = self.vec[i]
            for j in range(i, len(self.vec)):
                vj, pj, dj = self.vec[j]
                if i == j and (pi + pj) % 2 == 0 and pi == 0:
                    continue                # c(x,x) = 0 for x even (skew)
                if i == j and pi == 0:
                    continue
                if di + dj not in self.win:
                    continue
                tgt_d = di + dj + self.w
                tgt_p = (pi + pj + self.cp) % 2
                tl = [t for t, (_, p, _) in enumerate(self._bas(tgt_d))
                      if p == tgt_p]
                if not tl:
                    continue
                self.pairs[(i, j)] = n
                self.ptarget[(i, j)] = (tgt_d, tl)
                n += len(tl)
        self.nunk = n

    # the basis of an arbitrary degree (targets may leave the window)
    def _bas(self, d):
        return [(v, p, d) for v, p in self.g.basis(d)]

    # ------------------------------------------------------------- c(i,j)

    def cval(self, i, j):
        """c(e_i, e_j) as {(target degree, target index): linear form}.

        A linear form is a dict {unknown -> Fraction}.  Skew-supersymmetry
        c(y,x) = -(-1)^{p_x p_y} c(x,y) is applied when j < i.
        """
        sgn = Q(1)
        if j < i:
            pi, pj = self.vec[i][1], self.vec[j][1]
            sgn = Q(-1) * Q((-1) ** (pi * pj))
            i, j = j, i
        if (i, j) not in self.pairs:
            return None                       # outside the window: undefined
        base = self.pairs[(i, j)]
        d, tl = self.ptarget[(i, j)]
        return d, {t: {base + a: sgn} for a, t in enumerate(tl)}

    def cval_elt(self, elt, d, j):
        """c(v, e_j) for an arbitrary homogeneous v of degree d (elt in g_d)."""
        if d not in self.win:
            return None
        out = {}
        co = self.g.coords(d, elt)
        for a, i in enumerate(self.at[d]):
            if not co[a]:
                continue
            r = self.cval(i, j)
            if r is None:
                return None
            td, lf = r
            for t, form in lf.items():
                acc = out.setdefault((td, t), {})
                for u, c in form.items():
                    acc[u] = acc.get(u, Q(0)) + co[a] * c
        return out

    # ------------------------------------------------------ the equations

    def cocycle_rows(self):
        """Rows of the linear system J_c = 0 (sparse: list of dicts)."""
        rows = []
        V = self.vec
        nv = len(V)
        for i in range(nv):
            for j in range(i, nv):
                if V[i][2] + V[j][2] not in self.win:
                    continue
                for k in range(j, nv):
                    if V[j][2] + V[k][2] not in self.win:
                        continue
                    if V[i][2] + V[k][2] not in self.win:
                        continue
                    r = self._triple(i, j, k)
                    if r is not None:
                        rows.extend(r)
        return rows

    def _triple(self, i, j, k):
        """The rows of J_c(e_i,e_j,e_k) = 0, or None if a c-value is missing."""
        V = self.vec
        sig = V[i][2] + V[j][2] + V[k][2] + self.w
        acc = {}                                   # (target index) -> form
        for (x, y, z) in ((i, j, k), (j, k, i), (k, i, j)):
            vx, px, dx = V[x]
            vy, py, dy = V[y]
            vz, pz, dz = V[z]
            # the three terms of the super Jacobi identity carry the sign
            # (-1)^{p(first) p(third)} -- the same shape in each cyclic slot.
            s = Q((-1) ** (px * pz))
            # --- [c(x,y), z]
            r = self.cval(x, y)
            if r is None:
                return None
            td, lf = r
            for t, form in lf.items():
                ft = self.g.basis(td)[t][0]
                br = self.g.brk(ft, vz)
                if not br:
                    continue
                co = self.g.coords(sig, br)
                for a, c in enumerate(co):
                    if not c:
                        continue
                    cell = acc.setdefault(a, {})
                    for u, cc in form.items():
                        cell[u] = cell.get(u, Q(0)) + s * c * cc
            # --- c([x,y], z)
            bxy = self.g.brk(vx, vy)
            if bxy:
                got = self.cval_elt(bxy, dx + dy, z)
                if got is None:
                    return None
                for (td2, t), form in got.items():
                    ft = self.g.basis(td2)[t][0]
                    # c([x,y],z) already IS an element of degree sigma
                    co = self.g.coords(sig, ft)
                    for a, c in enumerate(co):
                        if not c:
                            continue
                        cell = acc.setdefault(a, {})
                        for u, cc in form.items():
                            cell[u] = cell.get(u, Q(0)) + s * c * cc
        rows = []
        for a, cell in acc.items():
            cell = {u: c for u, c in cell.items() if c}
            if cell:
                rows.append(cell)
        return rows

    # ---------------------------------------------------- the coboundaries

    def coboundaries(self):
        """A spanning set of B, as dense vectors of length self.nunk."""
        V = self.vec
        out = []
        for d in self.degs:
            src = self.at[d]
            tdeg = d + self.w
            tb = self.g.basis(tdeg)
            for a, i in enumerate(src):
                pi = V[i][1]
                for t, (ft, pt) in enumerate(tb):
                    if (pt + pi + self.cp) % 2:
                        continue
                    out.append(self._delta(i, ft, tdeg))
        return [v for v in out if any(v)]

    def _delta(self, i0, f0, tdeg):
        """delta(phi) where phi(e_{i0}) = f0 and phi = 0 on the other basis."""
        V = self.vec
        vec = [Q(0)] * self.nunk
        for (i, j), base in self.pairs.items():
            vi, pi, di = V[i]
            vj, pj, dj = V[j]
            td, tl = self.ptarget[(i, j)]
            val = {}
            # delta(phi)(x,y) = [phi x, y] + (-1)^{p(phi)p(x)}[x, phi y]
            #                   - phi([x,y]).
            # For an ODD parameter eps (eps^2 = 0) the trivializing map is
            # Id + eps phi with phi ODD, and [x, eps phi y] = (-1)^{p(x)} eps
            # [x, phi y]: the middle term carries the sign.  Without it the
            # coboundaries of an odd cochain are not cocycles, and B|core comes
            # out LARGER than Z|core -- the negative h that gave this away.
            if i == i0:
                val = add(val, self.g.brk(f0, vj))
            if j == i0:
                sgn = Q((-1) ** (self.cp * pi))
                val = add(val, smul(sgn, self.g.brk(vi, f0)))
            # - phi([x,y])
            b = self.g.brk(vi, vj)
            if b and (di + dj) in self.win:
                co = self.g.coords(di + dj, b)
                for a, s in enumerate(co):
                    if s and self.at[di + dj][a] == i0:
                        val = add(val, smul(-s, f0))
            if not val:
                continue
            co = self.g.coords(td, val)
            for a, t in enumerate(tl):
                vec[base + a] = co[t]
        return vec

    # ------------------------------------------------------- the core mask

    def core_columns(self, M):
        """Indices of the unknowns whose pair lives in the core [-M, M]."""
        V = self.vec
        cols = []
        for (i, j), base in self.pairs.items():
            di, dj = V[i][2], V[j][2]
            if abs(di) <= M and abs(dj) <= M and abs(di + dj) <= M:
                _, tl = self.ptarget[(i, j)]
                cols.extend(range(base, base + len(tl)))
        return sorted(cols)


# ============================================================ the computation

def h2_core(g, N, M, weight=0, cochain_parity=0, degs=None, verbose=False):
    """dim Z|core - dim B|core for a window of N steps and a core of M steps."""
    if degs is None:
        degs = g.degrees(N)
    W = Window(g, degs, weight * g.step(), cochain_parity)
    if W.nunk == 0:
        return {"unknowns": 0, "Z": 0, "B": 0, "h": 0, "vectors": len(W.vec)}
    rows = W.cocycle_rows()
    mat = [[r.get(c, Q(0)) for c in range(W.nunk)] for r in rows]
    # NB: exactlin.nullspace(M, p) takes a PRIME MODULUS as its second
    # argument, not the column count -- passing the latter computes modulo a
    # composite and silently returns nonsense.  Over Q it takes one argument.
    Z = nullspace(mat) if mat else _identity_rows(W.nunk)
    B = W.coboundaries()
    cols = W.core_columns(M * g.step())
    Zc = [[z[c] for c in cols] for z in Z]
    Bc = [[b[c] for c in cols] for b in B]
    rZ = rank(Zc) if Zc else 0
    rB = rank(Bc) if Bc else 0
    res = {"unknowns": W.nunk, "rows": len(rows), "vectors": len(W.vec),
           "Zdim": len(Z), "Z": rZ, "B": rB, "h": rZ - rB, "core_cols": len(cols)}
    if verbose:
        print("   %-22s N=%-3s M=%-3s unk=%-6d rows=%-6d  Z|c=%-4d B|c=%-4d  h=%d"
              % (g.name, N, M, W.nunk, len(rows), rZ, rB, rZ - rB))
    return res


def _identity_rows(n):
    return [[Q(1) if i == j else Q(0) for j in range(n)] for i in range(n)]


# ------------------------------------------------------------------ solvers

def _rref_solver(cols, nrows):
    """Return a function rhs -> solution of (cols) x = rhs, or None."""
    ncols = len(cols)
    aug = [[cols[j][i] for j in range(ncols)] for i in range(nrows)]

    def solve(rhs):
        return _solve(aug, rhs)
    return solve


def _solve(mat, rhs):
    """Exact solution of mat x = rhs (any solution), or None."""
    if not mat:
        return [] if not any(rhs) else None
    m = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    ncols = len(mat[0])
    piv, r = {}, 0
    for c in range(ncols):
        p = None
        for i in range(r, len(m)):
            if m[i][c]:
                p = i
                break
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        inv = Q(1) / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c]:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        piv[c] = r
        r += 1
    for i in range(r, len(m)):
        if m[i][ncols]:
            return None
    x = [Q(0)] * ncols
    for c, i in piv.items():
        x[c] = m[i][ncols]
    return x


# ------------------------------------------------------------- the algebras

def algebras(laurent=True):
    """The models used in the paper, as Graded objects."""
    poly = not laurent
    out = []
    out.append(Graded(yb.Witt(), "witt" if laurent else "vect(1)", poly))
    for n in (1, 2, 3):
        out.append(Graded(yb.Contact(n),
                          ("k^L(1|%d)" if laurent else "k(1|%d)") % n, poly))
    if laurent:
        for n in (1, 2):
            out.append(Graded(yb.Ramond(n), "k^M(1|%d)" % n, poly))
    for m in (1,):
        out.append(Graded(yb.Vect(m),
                          ("vect^L(1|%d)" if laurent else "vect(1|%d)") % m, poly))
    out.append(Graded(yb.MPer(), "m^L(1)" if laurent else "m(1)", poly))
    return out
