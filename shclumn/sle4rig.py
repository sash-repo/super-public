#!/usr/bin/env python3
r"""Is sle'(4) rigid?  The degree -2 part of H^2(sle'(4); sle'(4)), exactly.

This is the computation that decides the dichotomy for n = 4: either
Delta_* kappa = 0 and sle'(4) is rigid on the same footing as
sle'(n) for n > 4, or H^2(sle'(4); sle'(4)) != 0 with a cocycle of degree -2.

    g = sle'(n) = Im Delta / C.1,   Delta = sum_i d/dq_i d/dxi_i,
    g_k = (polynomials of degree k+2 in Im Delta),  g_{-1} = <q_i, xi_i>,
    g_0 = spe(n),  parity of f = (Grassmann degree of f) + 1.

WHY A FINITE COMPUTATION SUFFICES.  Let c be a 2-cocycle of degree -2, i.e.
c(g_i, g_j) subset g_{i+j-2}.  Then

  Lemma (propagation).  If c vanishes on every pair g_i ^ g_j with i+j <= 1,
  then c = 0.

  Proof.  Induction on T = i+j >= 2.  In the cocycle identity for (x,y,z) with
  x in g_{-1}, y in g_i, z in g_j, i+j = T, every term except [x, c(y,z)] is a
  value of c on a pair of total degree T-1, hence 0 by induction; so
  [g_{-1}, c(y,z)] = 0 with c(y,z) in g_{T-2}, T-2 >= 0, and c(y,z) = 0 by
  transitivity.  For T <= 0 the target g_{T-2} is 0 anyway.               QED

So a degree -2 cocycle is determined by its two components of total degree 1,

    c^(-1,2): g_{-1} (x) g_2 --> g_{-1},    c^(0,1): g_0 (x) g_1 --> g_{-1},

and it suffices to bound the space of those.  We impose the cocycle equations
of triple-degree 1 (which involve, besides these, the three components of total
degree 2) and compare the result with the coboundaries.  If the two coincide,
H^2(g;g)_{-2} = 0 -- rigorously, by the Lemma.

WHY WE MAY RESTRICT TO sl(4)-INVARIANT COCHAINS.  sl(4) subset g_0 acts
semisimply on every (finite-dimensional) weight component of C^.(g;g), and it
acts trivially on cohomology; hence H(C^inv) = H(C)^{sl(4)} = H(C).

Everything is over Q, in exact arithmetic; no SuperLie.

Usage:  python3 sle4rig.py [n]        (n = 4 by default; n = 3, 5 for control)
"""
import sys
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, '.')
from sle import monomials, delta, bracket, vec, rref            # noqa: E402


# --------------------------------------------------------------------- basics
def wt(mon):
    """gl(n)-weight of a monomial q^a xi^S: a - indicator(S)."""
    a, S = mon
    w = list(a)
    for i in S:
        w[i] -= 1
    return tuple(w)


def par(mon):
    """Parity in b(n) = Pi(C[q,xi]): Grassmann degree + 1."""
    return (len(mon[1]) + 1) % 2


class Algebra:
    """The Z-graded Lie superalgebra g = sle'(n) up to degree kmax."""

    def __init__(self, n, kmax):
        self.n, self.kmax = n, kmax
        self.mon = {}          # polynomial degree -> list of monomials
        self.idx = {}          # polynomial degree -> {monomial: position}
        self.basis = {}        # k -> list of elements (dicts monomial->coeff)
        self.piv = {}          # k -> list of pivot monomials (echelon form)
        self.wt = {}           # k -> list of weights
        self.par = {}          # k -> list of parities
        self._brc = {}         # bracket cache (per instance!)
        self._adT = {}         # transposed ad(E) tables
        for k in range(-1, kmax + 1):
            self._build(k)

    def _P(self, m):
        if m not in self.mon:
            self.mon[m] = monomials(self.n, m)
            self.idx[m] = {mo: i for i, mo in enumerate(self.mon[m])}
        return self.mon[m]

    def _build(self, k):
        """g_k = Delta(P_{k+4}) inside P_{k+2}, in echelon form."""
        m = k + 2
        src, tgt = self._P(m + 2), self._P(m)
        rows = [vec(delta({mo: 1}, self.n), self.idx[m]) for mo in src]
        R = [r for r in rref(rows, len(tgt)) if r]
        R.sort(key=lambda r: min(r))
        bas, piv = [], []
        for r in R:
            p = min(r)
            piv.append(tgt[p])
            bas.append({tgt[c]: Fraction(v) for c, v in r.items()})
        self.basis[k], self.piv[k] = bas, piv
        self.wt[k] = [wt(p) for p in piv]
        self.par[k] = [par(p) for p in piv]
        # weight- and parity-homogeneity of the echelon basis
        for b, w, p in zip(bas, self.wt[k], self.par[k]):
            assert all(wt(mo) == w for mo in b), "basis vector not weight-homogeneous"
            assert all(par(mo) == p for mo in b), "basis vector not parity-homogeneous"

    def dim(self, k):
        return len(self.basis[k]) if -1 <= k <= self.kmax else 0

    def coord(self, k, f):
        """Coordinates of a polynomial f in the echelon basis of g_k."""
        if k < -1 or k > self.kmax:
            assert not f, "element outside the truncation is non-zero"
            return {}
        f = dict(f)
        out = {}
        for i, (b, p) in enumerate(zip(self.basis[k], self.piv[k])):
            if p in f:
                lam = f[p] / b[p]
                if lam:
                    out[i] = lam
                for mo, c in b.items():
                    v = f.get(mo, 0) - lam * c
                    if v:
                        f[mo] = v
                    else:
                        f.pop(mo, None)
        assert not f, "element not in g_k (leftover: %r)" % (list(f)[:3],)
        return out

    # ---- brackets, as sparse structure constants [g_i basis a, g_j basis b]
    def br(self, i, j, a, b):
        if i + j < -1:
            return {}          # constants: sle'(n) = sb'(n)/C.1
        key = (i, j, a, b)
        c = self._brc.get(key)
        if c is None:
            f = bracket(self.basis[i][a], self.basis[j][b], self.n)
            f = {mo: v for mo, v in f.items() if v}
            c = self.coord(i + j, f) if f else {}
            self._brc[key] = c
        return c

    def adT(self, k, E, tag):
        """Transpose of ad(E)|g_k: a0 -> [(a, coefficient of e_{a0} in [E,e_a])]."""
        key = (k, tag)
        t = self._adT.get(key)
        if t is None:
            t = {}
            for a in range(self.dim(k)):
                for x, lam in self.bracket_vec(0, E, k, {a: Fraction(1)}).items():
                    t.setdefault(x, []).append((a, lam))
            self._adT[key] = t
        return t

    def bracket_vec(self, i, u, j, v):
        """Bracket of two coordinate vectors u in g_i, v in g_j."""
        out = {}
        for a, ca in u.items():
            for b, cb in v.items():
                for c, cc in self.br(i, j, a, b).items():
                    val = out.get(c, 0) + ca * cb * cc
                    if val:
                        out[c] = val
                    else:
                        out.pop(c, None)
        return out


# ------------------------------------------------------------------ cochains
# A 2-cochain component c^(i,j) is a dict {(a, b): {c: coeff}} giving the value
# of c on (basis_a of g_i, basis_b of g_j).  For i = j only the pairs a <= b are
# stored, the rest being fixed by c(y,x) = -(-1)^{p(x)p(y)} c(x,y).

def sl_zero(w1, w2):
    """True if w1 - w2 is a multiple of (1,...,1), i.e. sl(n)-weight 0."""
    d = [x - y for x, y in zip(w1, w2)]
    return all(x == d[0] for x in d)


def sym_sign(pa, pb):
    return -1 if (pa * pb) % 2 == 0 else 1     # c(y,x) = sign * c(x,y)


class Cochains:
    """Weight-0 and sl(n)-invariant bases of the degree -2 cochain spaces."""

    def __init__(self, g):
        self.g = g
        n = g.n
        # sl(n) raising operators e_{a,a+1} = q_a xi_{a+1} in g_0
        self.raise_ = []
        for a in range(n - 1):
            mo = ([0] * n, (a + 1,))
            mo[0][a] = 1
            f = {(tuple(mo[0]), mo[1]): Fraction(1)}
            self.raise_.append(g.coord(0, f))

    # ---- weight-0 basis of Hom(g_i (x) g_j, g_{i+j-2})
    def wt0_basis(self, i, j, shift=-2):
        g = self.g
        k = i + j + shift
        if g.dim(k) == 0 or g.dim(i) == 0 or g.dim(j) == 0:
            return []
        tgt = {}
        for t, w in enumerate(g.wt[k]):
            tgt.setdefault(w, []).append(t)
        # an sl(n)-weight is a gl(n)-weight modulo the centre: the cochain is
        # of sl(n)-weight 0 iff wt(t) - wt(a) - wt(b) = m (1,...,1), any m in Z.
        # (This is what makes the Pfaffian, of gl-weight -2(eps_1+...+eps_n),
        # an sl(V)-invariant cochain; see Remark "Why the classes exist".)
        n = g.n
        shifts = range(-(g.kmax + 4), g.kmax + 5)
        out = []
        for a, wa in enumerate(g.wt[i]):
            brange = range(g.dim(j)) if i != j else range(a, g.dim(j))
            for b in brange:
                if i == j and a == b and g.par[i][a] == 0:
                    continue        # c(x,x) = 0 for x even
                wb = g.wt[j][b]
                w = tuple(x + y for x, y in zip(wa, wb))
                for m in shifts:
                    for t in tgt.get(tuple(x + m for x in w), ()):
                        out.append((a, b, t))
        return out

    def act(self, i, j, E, tag, coch, shift=-2):
        """(E.c)(u,v) = [E, c(u,v)] - c([E,u],v) - c(u,[E,v]), E even in g_0."""
        g = self.g
        k = i + j + shift
        out = {}

        def add(a, b, valdict, scale):
            if i == j and a > b:
                a, b, scale = b, a, scale * sym_sign(g.par[i][b], g.par[j][a])
            if i == j and a == b and g.par[i][a] == 0:
                return
            d = out.setdefault((a, b), {})
            for c, v in valdict.items():
                nv = d.get(c, 0) + scale * v
                if nv:
                    d[c] = nv
                else:
                    d.pop(c, None)

        # expand to ordered pairs when the two slots are the same g_i
        items = []
        for (a, b), val in coch.items():
            items.append((a, b, val))
            if i == j and a != b:
                s = sym_sign(g.par[i][a], g.par[j][b])
                items.append((b, a, {t: s * x for t, x in val.items()}))
        Ti, Tj = g.adT(i, E, tag), g.adT(j, E, tag)
        for (a, b, val) in items:
            add(a, b, g.bracket_vec(0, E, k, val), 1)        # [E, c(a,b)]
            for x, lam in Ti.get(a, ()):                      # - c([E,x], b)
                add(x, b, val, -lam)
            for y, lam in Tj.get(b, ()):                      # - c(a, [E,y])
                add(a, y, val, -lam)
        return {p: v for p, v in out.items() if v}

    def full(self, i, j, coch):
        """Value of the cochain on an arbitrary ordered pair of basis vectors."""
        g = self.g
        def value(a, b):
            if i == j and a > b:
                v = coch.get((b, a), {})
                s = sym_sign(g.par[i][b], g.par[j][a])
                return {c: s * x for c, x in v.items()}
            return coch.get((a, b), {})
        return value

    def invariants(self, i, j, parity, shift=-2, verbose=True):
        """Basis of the sl(n)-invariant cochains of the given parity in
        Hom(g_i (x) g_j, g_{i+j-2}), as coefficient dicts over wt0_basis."""
        g = self.g
        k = i + j + shift
        allb = self.wt0_basis(i, j, shift)
        basis = [(a, b, t) for (a, b, t) in allb
                 if (g.par[i][a] + g.par[j][b] + g.par[k][t]) % 2 == parity]
        if not basis:
            return basis, []
        eqs = {}
        for e, E in enumerate(self.raise_):
            for p, (a, b, t) in enumerate(basis):
                img = self.act(i, j, E, e, {(a, b): {t: Fraction(1)}}, shift)
                for (x, y), val in img.items():
                    for c, v in val.items():
                        row = eqs.setdefault((e, x, y, c), {})
                        nv = row.get(p, 0) + v
                        if nv:
                            row[p] = nv
                        else:
                            row.pop(p, None)
        ker = kernel([r for r in eqs.values() if r], len(basis))
        if verbose:
            print("    Hom(g_%d (x) g_%d, g_%d)_%s : wt0 %d, sl(%d)-invariant %d"
                  % (i, j, k, "ev" if parity == 0 else "od",
                     len(basis), g.n, len(ker)))
        return basis, ker


# ------------------------------------------------------------- linear algebra
def kernel(rows, ncols):
    """Kernel of the matrix whose rows are dicts {col: value}."""
    rows = [dict(r) for r in rows if r]
    piv = {}
    for r in rows:
        r = dict(r)
        while r:
            c = min(r)
            if c in piv:
                f = r[c] / piv[c][c]
                for k, v in piv[c].items():
                    nv = r.get(k, 0) - f * v
                    if nv:
                        r[k] = nv
                    else:
                        r.pop(k, None)
            else:
                piv[c] = r
                break
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for fc in free:
        sol = {fc: Fraction(1)}
        for c in sorted(piv, reverse=True):
            r = piv[c]
            s = sum(r[k] * sol.get(k, 0) for k in r if k != c)
            if s:
                sol[c] = -s / r[c]
        out.append({k: v for k, v in sol.items() if v})
    return out


def kernel_of_columns(vectors, ncols=None):
    """Kernel of the map (coefficients) -> sum_i coeff_i * vectors[i]."""
    rows = {}
    for i, v in enumerate(vectors):
        for c, x in v.items():
            rows.setdefault(c, {})[i] = x
    return kernel(list(rows.values()), len(vectors))


def rank_of(vectors):
    return len(vectors) - len(kernel_of_columns(vectors))


# ------------------------------------------------ the Chevalley-Eilenberg d
# Convention (fixed empirically by d^2 = 0, see selftest):
#
#  (d phi)(x,y)   = (-1)^{px pc} [x, phi(y)] - (-1)^{py(pc+px)} [y, phi(x)]
#                   - phi([x,y])
#  (d c)(x,y,z)   = (-1)^{px pc} [x, c(y,z)] - (-1)^{py(pc+px)} [y, c(x,z)]
#                   + (-1)^{pz(pc+px+py)} [z, c(x,y)]
#                   - c([x,y],z) + (-1)^{pz py} c([x,z],y)
#                   - (-1)^{py px + pz px} c([y,z],x)
#
# where px = tilde x is the parity in b(n) and pc that of the cochain.

def sg(e):
    return -1 if e % 2 else 1


class C2:
    """A degree-(-2) 2-cochain: components (i,j) with i <= j, values in
    g_{i+j-2}.  comp[(i,j)][(a,b)] = {t: coeff}."""

    def __init__(self, g, parity, comp=None, shift=-2):
        self.g, self.parity, self.shift = g, parity, shift
        self.comp = comp if comp is not None else {}

    def val(self, i, a, j, b):
        g = self.g
        if i <= j:
            v = self.comp.get((i, j), {}).get((a, b) if i != j or a <= b else None)
            if i == j and a > b:
                v0 = self.comp.get((i, j), {}).get((b, a), {})
                s = sym_sign(g.par[i][b], g.par[j][a])
                return {t: s * x for t, x in v0.items()}
            return v or {}
        v0 = self.comp.get((j, i), {}).get((b, a), {})
        s = sym_sign(g.par[j][b], g.par[i][a])
        return {t: s * x for t, x in v0.items()}

    def d(self, di, a, dj, b, dk, c):
        """(dc)(x,y,z) for basis vectors x in g_di, y in g_dj, z in g_dk,
        returned as a coordinate dict in g_{di+dj+dk-2}."""
        g, pc, sh = self.g, self.parity, self.shift
        px, py, pz = g.par[di][a], g.par[dj][b], g.par[dk][c]
        out = {}

        def acc(v, s):
            for t, x in v.items():
                nv = out.get(t, 0) + s * x
                if nv:
                    out[t] = nv
                else:
                    out.pop(t, None)

        # module terms
        acc(g.bracket_vec(di, {a: Fraction(1)}, dj + dk + sh,
                          self.val(dj, b, dk, c)), sg(px * pc))
        acc(g.bracket_vec(dj, {b: Fraction(1)}, di + dk + sh,
                          self.val(di, a, dk, c)), -sg(py * (pc + px)))
        acc(g.bracket_vec(dk, {c: Fraction(1)}, di + dj + sh,
                          self.val(di, a, dj, b)), sg(pz * (pc + px + py)))
        # bracket terms
        for (u, du), (v, dv), (w, dw), s in (
                ((a, di), (b, dj), (c, dk), -1),
                ((a, di), (c, dk), (b, dj), sg(pz * py)),
                ((b, dj), (c, dk), (a, di), -sg(py * px + pz * px))):
            br = g.bracket_vec(du, {u: Fraction(1)}, dv, {v: Fraction(1)})
            for t, lam in br.items():
                acc(self.val(du + dv, t, dw, w), s * lam)
        return out


class C1:
    """A degree-(-2) 1-cochain: phi[i] = {a: {t: coeff}}, g_i -> g_{i-2}."""

    def __init__(self, g, parity, phi=None, shift=-2):
        self.g, self.parity, self.shift = g, parity, shift
        self.phi = phi if phi is not None else {}

    def val(self, i, a):
        return self.phi.get(i, {}).get(a, {})

    def d(self, di, a, dj, b):
        g, pc, sh = self.g, self.parity, self.shift
        px, py = g.par[di][a], g.par[dj][b]
        out = {}

        def acc(v, s):
            for t, x in v.items():
                nv = out.get(t, 0) + s * x
                if nv:
                    out[t] = nv
                else:
                    out.pop(t, None)

        acc(g.bracket_vec(di, {a: Fraction(1)}, dj + sh, self.val(dj, b)),
            sg(px * pc))
        acc(g.bracket_vec(dj, {b: Fraction(1)}, di + sh, self.val(di, a)),
            -sg(py * (pc + px)))
        for t, lam in g.bracket_vec(di, {a: Fraction(1)},
                                    dj, {b: Fraction(1)}).items():
            for u, x in self.val(di + dj, t).items():
                nv = out.get(u, 0) - lam * x
                if nv:
                    out[u] = nv
                else:
                    out.pop(u, None)
        return out


# ------------------------------------------------------------------- helpers
def coboundary(g, phi, pairs):
    """The 2-cochain d(phi) on the listed graded components (i <= j)."""
    c = C2(g, phi.parity, {}, shift=phi.shift)
    for (i, j) in pairs:
        comp = {}
        for a in range(g.dim(i)):
            brange = range(g.dim(j)) if i != j else range(a, g.dim(j))
            for b in brange:
                v = phi.d(i, a, j, b)
                if v:
                    comp[(a, b)] = v
        c.comp[(i, j)] = comp
    return c


def rand_phi(g, parity, shift, degs, seed=1, density=6):
    """A sparse random 1-cochain g_k -> g_{k+shift} of the given parity."""
    import random
    rnd = random.Random(seed)
    phi = {}
    for k in degs:
        kk = k + shift
        if g.dim(k) == 0 or g.dim(kk) == 0:
            continue
        d = {}
        for _ in range(density):
            a = rnd.randrange(g.dim(k))
            cands = [t for t in range(g.dim(kk))
                     if (g.par[kk][t] + g.par[k][a]) % 2 == parity]
            if not cands:
                continue
            t = rnd.choice(cands)
            d.setdefault(a, {})[t] = Fraction(rnd.randint(-3, 3) or 2)
        phi[k] = d
    return C1(g, parity, phi, shift=shift)


def selftest(g, verbose=True):
    n, K = g.n, g.kmax
    if verbose:
        print("  dims g_k, k = -1..%d: %s"
              % (K, [g.dim(k) for k in range(-1, K + 1)]))
        ev = sum(1 for p in g.par[0] if p == 0)
        print("  sdim g_0 = (%d|%d)   [spe(%d) = (%d|%d)]"
              % (ev, g.dim(0) - ev, n, n * n - 1, n * n))
    # Jacobi on all triples of low degree
    bad = 0
    for a in range(g.dim(-1)):
        for b in range(g.dim(0)):
            for c in range(g.dim(1)):
                x, y, z = {a: Fraction(1)}, {b: Fraction(1)}, {c: Fraction(1)}
                px, py = g.par[-1][a], g.par[0][b]
                t1 = g.bracket_vec(-1, x, 1, g.bracket_vec(0, y, 1, z))
                t2 = g.bracket_vec(0, y, 0, g.bracket_vec(-1, x, 1, z))
                t3 = g.bracket_vec(-1, g.bracket_vec(-1, x, 0, y), 1, z)
                for t, v in t2.items():
                    t1[t] = t1.get(t, 0) - sg(px * py) * v
                for t, v in t3.items():
                    t1[t] = t1.get(t, 0) - v
                if any(v for v in t1.values()):
                    bad += 1
    if verbose:
        print("  Jacobi on g_{-1} x g_0 x g_1: %s"
              % ("OK" if bad == 0 else "%d FAILURES" % bad))
    assert bad == 0
    # transitivity: g_m -> Hom(g_{-1}, g_{m-1}) injective for m >= 0
    for m in range(0, K + 1):
        rows = []
        for t in range(g.dim(m)):
            r = {}
            for a in range(g.dim(-1)):
                for u, v in g.br(-1, m, a, t).items():
                    r[(a, u)] = v
            rows.append(r)
        cols = {}
        for t, r in enumerate(rows):
            for c, v in r.items():
                cols.setdefault(c, {})[t] = v
        ker = kernel(list(cols.values()), g.dim(m))
        if verbose:
            print("  transitivity in degree %d: kernel of ad|g_{-1} is %d-dim"
                  % (m, len(ker)))
        assert not ker, "NOT transitive in degree %d" % m
    # d^2 = 0
    for parity in (0, 1):
        for shift in (0, -1):
            phi = rand_phi(g, parity, shift, range(-1, K + 1), seed=parity + 7 * shift)
            pairs = [(i, j) for i in range(-1, K + 1) for j in range(i, K + 1)
                     if -1 <= i + j + shift <= K and i + j <= K]
            c = coboundary(g, phi, pairs)
            bad = 0
            trials = 0
            for di in range(-1, 2):
                for dj in range(di, 2):
                    for dk in range(dj, 2):
                        if not (-1 <= di + dj + dk + shift <= K):
                            continue
                        for a in range(min(g.dim(di), 4)):
                            for b in range(min(g.dim(dj), 4)):
                                for cc in range(min(g.dim(dk), 4)):
                                    trials += 1
                                    v = c.d(di, a, dj, b, dk, cc)
                                    if any(x for x in v.values()):
                                        bad += 1
            if verbose:
                print("  d(d phi) = 0, parity %d, shift %2d: %d/%d triples "
                      "non-zero" % (parity, shift, bad, trials))
            assert bad == 0, "sign convention wrong (d^2 != 0)"
    return True


def levels(d, K):
    """Pairs/triples relevant for cochains of degree d (see the Lemma).

    Level 1 = pairs with i+j+d = -1 (values in g_{-1}); level 2 = pairs with
    i+j+d = 0; the equations are the cocycle identities on triples with
    i+j+k+d = -1, which involve exactly these two levels."""
    lev1 = [(i, j) for i in range(-1, K + 1) for j in range(i, K + 1)
            if i + j == -1 - d]
    lev2 = [(i, j) for i in range(-1, K + 1) for j in range(i, K + 1)
            if i + j == -d]
    tri = [(i, j, k) for i in range(-1, K + 1) for j in range(i, K + 1)
           for k in range(j, K + 1) if i + j + k == -1 - d]
    return lev1, lev2, tri


def run(n, d=-2, K=3, verbose=True):
    print("=" * 72)
    print("g = sle'(%d):  H^2(g;g) in degree %d" % (n, d))
    print("=" * 72)
    g = Algebra(n, K)
    if verbose:
        selftest(g, verbose)
    ch = Cochains(g)
    LEVEL1, LEVEL2, TRIPLES = levels(d, K)
    print("  level 1 %s   level 2 %s   triples %s" % (LEVEL1, LEVEL2, TRIPLES))
    out = {}
    for parity in (0, 1):
        print("\n  --- cochains of parity %s ---"
              % ("even" if parity == 0 else "odd"))
        cols, offs = [], {}
        for (i, j) in LEVEL1 + LEVEL2:
            basis, inv = ch.invariants(i, j, parity, shift=d)
            offs[(i, j)] = (len(cols), basis, inv)
            for k in inv:
                comp = {}
                for p, lam in k.items():
                    a, b, t = basis[p]
                    comp.setdefault((a, b), {})[t] = lam
                cols.append(C2(g, parity, {(i, j): comp}, shift=d))
        nun = len(cols)
        print("    unknowns (invariant, levels 1+2): %d" % nun)
        if nun == 0:
            out[parity] = (0, 0)
            continue
        eqs = {}
        for (di, dj, dk) in TRIPLES:
            for a in range(g.dim(di)):
                brange = range(a, g.dim(dj)) if di == dj else range(g.dim(dj))
                for b in brange:
                    crange = range(b, g.dim(dk)) if dj == dk else range(g.dim(dk))
                    for cc in crange:
                        for u, col in enumerate(cols):
                            for t, x in col.d(di, a, dj, b, dk, cc).items():
                                if x:
                                    eqs.setdefault((di, dj, dk, a, b, cc, t),
                                                   {})[u] = x
        print("    equations of triple-degree %d: %d" % (-1 - d, len(eqs)))
        ker = kernel(list(eqs.values()), nun)
        print("    solutions (levels 1+2, invariant): %d" % len(ker))
        # --- level-1 coordinates (positions in the wt0 bases)
        pos, off = {}, 0
        for (i, j) in LEVEL1:
            _, basis, _ = offs[(i, j)]
            for p, tr in enumerate(basis):
                pos[(i, j) + tr] = off + p
            off += len(basis)
        base1 = {}
        off = 0
        for (i, j) in LEVEL1:
            base1[(i, j)] = off
            off += len(offs[(i, j)][1])
        svecs = []
        for k in ker:
            v = {}
            for u, lam in k.items():
                for (i, j) in LEVEL1:
                    o, basis, inv = offs[(i, j)]
                    if o <= u < o + len(inv):
                        for p, x in inv[u - o].items():
                            v[base1[(i, j)] + p] = v.get(base1[(i, j)] + p, 0) + lam * x
            if v:
                svecs.append(v)
        ds = rank_of(svecs)
        print("    ... their level-1 parts span: %d" % ds)
        # --- coboundaries
        degs = sorted({x for (i, j) in LEVEL1 for x in (i, j, i + j)})
        cob = []
        for k in degs:
            kk = k + d
            hb = [(a, t) for a in range(g.dim(k)) for t in range(g.dim(kk))
                  if g.dim(kk) and sl_zero(g.wt[kk][t], g.wt[k][a])
                  and (g.par[k][a] + g.par[kk][t]) % 2 == parity]
            if not hb:
                continue
            eqs2 = {}
            for e, E in enumerate(ch.raise_):
                for p, (a, t) in enumerate(hb):
                    img = {}
                    for x, lam in g.bracket_vec(0, E, kk, {t: Fraction(1)}).items():
                        img[(a, x)] = img.get((a, x), 0) + lam
                    for x, lam in g.adT(k, E, e).get(a, ()):
                        img[(x, t)] = img.get((x, t), 0) - lam
                    for key, v in img.items():
                        if v:
                            row = eqs2.setdefault((e,) + key, {})
                            row[p] = row.get(p, 0) + v
            ker2 = kernel([r for r in eqs2.values() if r], len(hb))
            print("    invariant phi: g_%d -> g_%d : wt0 %d, invariant %d"
                  % (k, kk, len(hb), len(ker2)))
            for kv in ker2:
                phi = {}
                for p, lam in kv.items():
                    a, t = hb[p]
                    phi.setdefault(k, {}).setdefault(a, {})[t] = lam
                cob.append(C1(g, parity, phi, shift=d))
        bvecs = []
        for phi in cob:
            v = {}
            for (i, j) in LEVEL1:
                for a in range(g.dim(i)):
                    for b in range(g.dim(j)):
                        for t, x in phi.d(i, a, j, b).items():
                            key = (i, j, a, b, t)
                            if key in pos and x:
                                v[pos[key]] = x
            if v:
                bvecs.append(v)
        db = rank_of(bvecs)
        tog = rank_of(svecs + bvecs)
        print("    coboundaries at level 1: %d;  cocycles: %d;  together: %d"
              % (db, ds, tog))
        assert tog == ds, "coboundary outside the cocycle space -- BUG"
        print("    ==>  dim H^2(g;g)_{%d}^{%s} <= %d"
              % (d, "ev" if parity == 0 else "od", ds - db))
        out[parity] = (ds, db)
    tot = sum(a - b for a, b in out.values())
    print("\n  RESULT for sle'(%d), degree %d:  dim H^2(g;g) <= %d "
          "(even %d, odd %d)\n"
          % (n, d, tot, out[0][0] - out[0][1], out[1][0] - out[1][1]))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[0]) if args else 4
    d = int(args[1]) if len(args) > 1 else -2
    run(n, d)
