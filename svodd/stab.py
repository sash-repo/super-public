#!/usr/bin/env python3
"""
stab.py -- T_1: THE STABILISER ENGINE.

For a normal form f of the published tables it computes, in exact arithmetic,

  (1) the stabiliser Lie algebra
          s_f = { V in vect(p|q)_0 , V even : V(f) = 0 },
      graded by  deg(m d/dz) = deg(m) - 1 >= 0,  i.e. the Lie algebra of
      Stab_{Diff_0}(f).  Its degree-0 part is the linear part (torus +
      reductive), its positive part is the nilpotent part;

  (2) the representation  rho : s_f --> gl(Nt_f)  on the corrected first layer
          Nt_f = m_odd / (m I_f)_odd
      together with the invariant subspace
          N_f = (m^2)_odd / (m I_f)_odd  of germs critical at the marked point;

  (3) orbit dimensions of Stab on Nt_f, hence the generic orbit, hence the
      first-layer modality  mod(f) = dim Nt_f - dim(generic orbit);

  (4) group elements: exp(V) for V of positive degree is an honest element of
      Stab_{Diff_0}(f), and acts on Nt_f by the (nilpotent, hence finite)
      matrix exponential exp(rho(V)) -- so every claim "these two germs are
      equivalent" is shipped with a certificate.

CERTIFICATION.  Only the image rho(s_f) in gl(Nt_f) matters for the orbits, and
that image is computed twice, bracketing it:

  lower  L :  V polynomial of degree <= M with V(f) = 0 EXACTLY.  Every such V
              is an honest element of s_f, so rho(L) is contained in the truth.
  upper  U :  V of degree <= k-1 with V(f) in m^{k+2}, where k = top degree of
              Nt_f.  Every element of s_f, truncated, satisfies this, so the
              truth is contained in rho(U).

When dim rho(L) = dim rho(U) the action on the layer is known exactly, and the
engine says CERTIFIED.  (Vector fields of degree >= k act by 0 on Nt_f, since
they push m into m^{k+1}, whose odd part lies in (m I_f)_odd.)

Everything sits on sv.py.  Truncation in the x-degree is chosen so that no term
of V(f) is ever cut, and the layer is cross-checked against sv.stable().
"""

from fractions import Fraction
from itertools import combinations
import random
import sys

import sv


# --------------------------------------------------------------- monomials

def mdeg(m):
    """total degree of a monomial (x-exponents, xi-set)"""
    return sum(m[0]) + len(m[1])


def mkey(m):
    """a total order on monomials, refining the total degree"""
    return (mdeg(m), m[0], tuple(sorted(m[1])))


def build(terms):
    f = {}
    for c, a, S in terms:
        key = (tuple(a), frozenset(S))
        f[key] = f.get(key, Fraction(0)) + Fraction(c)
    return {k: v for k, v in f.items() if v}


def xdeg_max(f):
    return max((sum(m[0]) for m in f), default=0)


def deg_max(f):
    return max((mdeg(m) for m in f), default=0)


# ------------------------------------------------------- reduced row echelon

class LinSpace:
    """
    A subspace of the truncated ring, kept in reduced row echelon form and
    keyed by the leading (largest) monomial of each row.
    """

    def __init__(self):
        self.rows = {}                      # pivot monomial -> row (dict)

    def reduce(self, v):
        v = {k: Fraction(c) for k, c in v.items() if c}
        while True:
            hits = [m for m in v if m in self.rows]
            if not hits:
                return v
            m = max(hits, key=mkey)
            c = v[m] / self.rows[m][m]
            for k, val in self.rows[m].items():
                nv = v.get(k, Fraction(0)) - c * val
                if nv:
                    v[k] = nv
                elif k in v:
                    del v[k]

    def add(self, v):
        """returns True if v enlarged the space"""
        v = self.reduce(v)
        if not v:
            return False
        m = max(v, key=mkey)
        c = v[m]
        v = {k: val / c for k, val in v.items()}
        for pm, row in self.rows.items():             # back-substitute
            if m in row:
                cc = row[m]
                for k, val in v.items():
                    nv = row.get(k, Fraction(0)) - cc * val
                    if nv:
                        row[k] = nv
                    elif k in row:
                        del row[k]
        self.rows[m] = v
        return True

    def dim(self):
        return len(self.rows)

    def pivots(self):
        return set(self.rows)


# ------------------------------------------------------------ linear algebra

def nullspace(cols, nrows_keys):
    """
    cols: list of dicts (each a vector, keyed by anything hashable).
    Returns a basis of { c : sum_a c_a cols[a] = 0 } as lists of Fractions.
    """
    n = len(cols)

    def safe_key(k):
        try:
            return (0,) + mkey(k)[:1] + (str(k),)
        except (TypeError, IndexError):
            return (1, 0, str(k))
    keys = sorted(nrows_keys, key=safe_key)
    # matrix as list of rows, each row a list of n Fractions
    rows = []
    for k in keys:
        row = [cols[a].get(k, Fraction(0)) for a in range(n)]
        if any(row):
            rows.append(row)
    # Gaussian elimination
    pivcol = {}
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivcol[c] = r
        r += 1
        if r == len(rows):
            break
    import param
    if param.RECORD:
        param.record_rank(("null", r, n))
    free = [c for c in range(n) if c not in pivcol]
    basis = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for c, ri in pivcol.items():
            v[c] = -rows[ri][fc]
        basis.append(v)
    return basis


def rank_of(vectors):
    """rank of a list of dicts"""
    L = LinSpace()
    r = 0
    for v in vectors:
        if L.add(v):
            r += 1
    return r


def rank_rows(rows):
    """rank of a list of lists of Fractions"""
    rows = [list(r) for r in rows if any(r)]
    if not rows:
        return 0
    n = len(rows[0])
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                fct = rows[i][c]
                rows[i] = [a - fct * b for a, b in zip(rows[i], rows[r])]
        r += 1
        if r == len(rows):
            break
    return r


# ---------------------------------------------------------------- the layer

def layer(p, q, terms, D):
    """
    Returns (R, f, mI, basis, k, safe) where
        mI     = LinSpace spanned by (m I_f)_odd inside the truncated ring,
        basis  = monomial basis of Nt_f = m_odd/(m I_f)_odd, ordered by degree,
        k      = top degree occurring in that basis,
        safe   = the degree up to which the pivot information is trustworthy.
    """
    R = sv.Ring(p, q, D)
    f = build(terms)
    gens = []
    for i in range(p):
        g = sv.d_dx(R, f, i)
        if g:
            gens.append(g)
    for j in range(q):
        g = sv.d_dxi(R, f, j)
        if g:
            gens.append(g)

    allmon = R.basis()
    mmon = [m for m in allmon if mdeg(m) >= 1]

    mI = LinSpace()
    for g in gens:
        for m in sorted(mmon, key=mkey):
            v = sv.mul(R, {m: Fraction(1)}, g)
            odd = {mm: c for mm, c in v.items() if R.parity(mm) == 1}
            if odd:
                mI.add(odd)

    safe = D - xdeg_max(f)                       # below this nothing was cut
    piv = mI.pivots()
    free_odd = [m for m in allmon
                if R.parity(m) == 1 and mdeg(m) >= 1 and m not in piv]
    basis = sorted([m for m in free_odd if sum(m[0]) <= safe], key=mkey)

    # k = the top of the filtration of Nt_f by the degree, i.e. the largest j
    # with (m^j)_odd not inside (m I_f)_odd.  This is NOT the top degree of the
    # basis: when f is only quasi-homogeneous the ideal is graded for the
    # weighted degree, so a monomial of high degree can have a nonzero class of
    # low degree (VII_2: the class of y^3.xi_1).  Fields of degree >= k kill the
    # layer, so k is exactly how far the jet algebra has to be computed.
    k = 0
    for m in allmon:
        if R.parity(m) == 1 and 1 <= mdeg(m) and sum(m[0]) <= safe:
            if mI.reduce({m: Fraction(1)}):
                k = max(k, mdeg(m))
    return R, f, mI, basis, k, safe


def check_top_degree(R, mI, basis, k, safe, p, q):
    """
    The engine rests on  (m^{k+1})_odd  <  (m I_f)_odd,  which is what makes
    vector fields of degree >= k act by zero on the layer.  This is NOT
    automatic when f is only quasi-homogeneous, the ideal then being graded for
    the weighted degree and not for the standard one, so it is verified here:
    every odd monomial of degree > k inside the trustworthy range must reduce
    to zero modulo (m I_f)_odd.
    """
    top = 0
    for m in R.basis():
        if R.parity(m) != 1 or sum(m[0]) > safe:
            continue
        d = mdeg(m)
        if d < 1:
            continue
        r = mI.reduce({m: Fraction(1)})
        if r:
            top = max(top, d)
        if d > k and r:
            return False, d
    return top == k, top


# ----------------------------------------------------------- vector fields
# A vector field is a dict {('x',i): poly, ('xi',j): poly}; it is EVEN, i.e.
# the coefficients of d/dx_i are even and those of d/dxi_j are odd.

def vf_monomials(R, p, q, dlo, dhi):
    """basis of even vector fields vanishing at 0, of degree in [dlo,dhi]"""
    out = []
    for m in R.basis():
        d = mdeg(m) - 1
        if d < dlo or d > dhi:
            continue
        if R.parity(m) == 0:
            for i in range(p):
                out.append({('x', i): {m: Fraction(1)}})
        else:
            for j in range(q):
                out.append({('xi', j): {m: Fraction(1)}})
    return out


def vf_degree(V):
    return min((mdeg(m) - 1 for c in V.values() for m in c), default=0)


def apply_vf(R, V, f):
    tot = {}
    for (kind, idx), coef in V.items():
        df = sv.d_dx(R, f, idx) if kind == 'x' else sv.d_dxi(R, f, idx)
        if not df:
            continue
        for k, c in sv.mul(R, coef, df).items():
            tot[k] = tot.get(k, Fraction(0)) + c
    return {k: c for k, c in tot.items() if c}


def vf_bracket(R, V, W):
    """[V,W] for two EVEN vector fields: componentwise V(W_z) - W(V_z)"""
    out = {}
    for slot in set(V) | set(W):
        a = apply_vf(R, V, W.get(slot, {}))
        b = apply_vf(R, W, V.get(slot, {}))
        c = {}
        for m, v in a.items():
            c[m] = c.get(m, Fraction(0)) + v
        for m, v in b.items():
            c[m] = c.get(m, Fraction(0)) - v
        c = {m: v for m, v in c.items() if v}
        if c:
            out[slot] = c
    return out


def vf_combine(basisV, coeffs):
    out = {}
    for V, c in zip(basisV, coeffs):
        if not c:
            continue
        for slot, poly in V.items():
            tgt = out.setdefault(slot, {})
            for m, cc in poly.items():
                nv = tgt.get(m, Fraction(0)) + c * cc
                if nv:
                    tgt[m] = nv
                elif m in tgt:
                    del tgt[m]
    return {s: p for s, p in out.items() if p}


# ------------------------------------------------------- the stabiliser algebra

def stab_algebra(R, f, p, q, dlo, dhi, cut=None):
    """
    Basis of { V of degree in [dlo,dhi] : V(f) = 0 }, or, if cut is not None,
    of { V : V(f) has no term of degree < cut } -- the relaxed (upper) system.
    Returned both as vector fields and as coordinate vectors on the monomial
    basis of vect_[dlo,dhi], together with that basis.
    """
    basisV = vf_monomials(R, p, q, dlo, dhi)
    cols, keys = [], set()
    for V in basisV:
        w = apply_vf(R, V, f)
        if cut is not None:
            w = {m: c for m, c in w.items() if mdeg(m) < cut}
        cols.append(w)
        keys |= set(w)
    sols = nullspace(cols, keys)
    return [vf_combine(basisV, c) for c in sols], sols, basisV


def jet_algebra(R, f, p, q, k, M):
    """
    THE BRACKET.  Let S be the image of s_f in vect_[0,k-1] -- the only part
    of the stabiliser that acts on Nt_f, since vector fields of degree >= k
    kill the layer.  With  o = ord(f)  (lowest degree of f):

      lower   pi(L_M),  L_M = { deg V <= M : V(f) = 0 }               < S
      upper   pi(U_M),  U_M = { deg V <= M : V(f) in m^{o+M+1} }      > S

    because any V in s_f splits as V_{<=M} + V_{>M} with V_{>M}(f) in
    m^{o+M+1}.  Returns (lower basis, upper basis, dimL, dimU) with the bases
    given as vector fields of degree <= k-1.
    """
    o = min((mdeg(m) for m in f), default=0)
    _, solsL, basisV = stab_algebra(R, f, p, q, 0, M)
    _, solsU, _ = stab_algebra(R, f, p, q, 0, M, cut=o + M + 1)
    keep = [a for a, V in enumerate(basisV) if vf_degree(V) <= k - 1]

    def project(sols):
        rows = [[s[a] for a in keep] for s in sols]
        # a basis of the row space, echelonised
        out, cur = [], []
        for r in rows:
            if rank_rows(cur + [r]) > len(cur):
                cur.append(r)
                out.append(vf_combine([basisV[a] for a in keep], r))
        return out

    Lb, Ub = project(solsL), project(solsU)
    return Lb, Ub, len(Lb), len(Ub)


def filtration_dims(vfs, k):
    """dim of {V in span(vfs) : deg V >= d} for d = 0..k-1"""
    out = []
    for d in range(k):
        sel = []
        for V in vfs:
            Vd = {s: {m: c for m, c in poly.items() if mdeg(m) - 1 >= d}
                  for s, poly in V.items()}
            sel.append({s: pp for s, pp in Vd.items() if pp})
        # the subspace of the span whose elements have degree >= d is the
        # kernel of the truncation to degrees < d
        low = []
        for V in vfs:
            Vd = {s: {m: c for m, c in poly.items() if mdeg(m) - 1 < d}
                  for s, poly in V.items()}
            low.append({s: pp for s, pp in Vd.items() if pp})
        keys = sorted({(s, m) for V in low for s, poly in V.items()
                       for m in poly}, key=lambda t: (str(t[0]), mkey(t[1])))
        rows = [[V.get(s, {}).get(m, Fraction(0)) for (s, m) in keys]
                for V in low]
        out.append(len(vfs) - rank_rows(rows))
    return out


# --------------------------------------------------- the layer representation

def rho(R, V, basis, mI, index):
    """matrix of V on Nt_f, in the monomial basis; rows = images"""
    n = len(basis)
    M = [[Fraction(0)] * n for _ in range(n)]
    for jcol, g in enumerate(basis):
        h = apply_vf(R, V, {g: Fraction(1)})
        r = mI.reduce(h)
        for m, c in r.items():
            if m not in index:
                raise RuntimeError(f"image leaves the layer: {m}")
            M[index[m]][jcol] = c
    return M


def rho_image_dim(mats):
    """dimension of the span of a list of matrices"""
    rows = [sum(M, []) for M in mats]
    return rank_rows(rows)


def act(M, vec):
    n = len(vec)
    return [sum(M[i][j] * vec[j] for j in range(n)) for i in range(n)]


def orbit_dim(mats, vec):
    return rank_rows([act(M, vec) for M in mats])


def generic_orbit_dim(mats, n, trials=30, seed=1):
    rnd = random.Random(seed)
    best, witness = 0, None
    for _ in range(trials):
        v = [Fraction(rnd.randint(-9, 9)) for _ in range(n)]
        d = orbit_dim(mats, v)
        if d > best:
            best, witness = d, v
    return best, witness


def mat_exp(M):
    """exponential of a nilpotent matrix, exactly"""
    n = len(M)
    E = [[Fraction(1) if i == j else Fraction(0) for j in range(n)]
         for i in range(n)]
    T = [row[:] for row in E]
    for step in range(1, n + 2):
        T = [[sum(M[i][l] * T[l][j] for l in range(n)) / step for j in range(n)]
             for i in range(n)]
        if not any(any(x for x in row) for row in T):
            break
        E = [[E[i][j] + T[i][j] for j in range(n)] for i in range(n)]
    return E


def exp_vf(R, V, h, nmax=40):
    """exp(V) applied to a function h; finite because V raises the degree"""
    out = dict(h)
    term = dict(h)
    for n in range(1, nmax):
        term = apply_vf(R, V, term)
        term = {m: c / n for m, c in term.items()}
        if not term:
            break
        for m, c in term.items():
            nv = out.get(m, Fraction(0)) + c
            if nv:
                out[m] = nv
            elif m in out:
                del out[m]
    return out


# ------------------------------------------------------ invariants on the layer

def deg_monomials(n, d):
    """exponent tuples of degree d in n variables"""
    if n == 0:
        return [()] if d == 0 else []
    out = []

    def rec(i, left, acc):
        if i == n - 1:
            out.append(acc + (left,))
            return
        for e in range(left + 1):
            rec(i + 1, left - e, acc + (e,))
    rec(0, d, ())
    return out


def derivation_matrix(M, mons, pos):
    """
    matrix of the derivation  sum_i (M g)_i d/dg_i  on homogeneous polynomials
    of a fixed degree, in the monomial basis `mons` (pos = monomial -> column).
    """
    cols = []
    n = len(M)
    for a in mons:
        col = {}
        for i in range(n):
            if not a[i]:
                continue
            for j in range(n):
                if not M[i][j]:
                    continue
                b = list(a)
                b[i] -= 1
                b[j] += 1
                b = tuple(b)
                col[b] = col.get(b, Fraction(0)) + a[i] * M[i][j]
        cols.append({k: v for k, v in col.items() if v})
    return cols


def invariant_polys(mats, n, d, verbose=False):
    """
    Basis of the homogeneous degree-d polynomials on the layer annihilated by
    every generator.  A nonzero answer PROVES that the generic orbit is not
    open, i.e. that the first-layer modality is at least 1.
    """
    mons = deg_monomials(n, d)
    pos = {a: i for i, a in enumerate(mons)}
    # stack the derivations of a spanning subset first, then verify on all
    cand = None
    for M in mats:
        cols = derivation_matrix(M, mons, pos)
        keys = set()
        for c in cols:
            keys |= set(c)
        if cand is None:
            sols = nullspace(cols, keys)
            cand = sols
        else:
            if not cand:
                return []
            # restrict the derivation to the current candidate space
            newcols = []
            allk = set()
            for v in cand:
                w = {}
                for a, coef in zip(mons, v):
                    if not coef:
                        continue
                    for b, cc in cols[pos[a]].items():
                        w[b] = w.get(b, Fraction(0)) + coef * cc
                w = {k: c for k, c in w.items() if c}
                newcols.append(w)
                allk |= set(w)
            inner = nullspace(newcols, allk)
            cand = [[sum(co[t] * cand[t][i] for t in range(len(cand)))
                     for i in range(len(mons))] for co in inner]
        if not cand:
            return []
    return [{mons[i]: c for i, c in enumerate(v) if c} for v in (cand or [])]


def annihilator_forms(mats, n):
    """
    Bilinear forms A with  A^t M + M^t A = 0  for every M in mats, i.e. forms
    preserved by the whole algebra.  For such an A the covector u(g) = A g
    annihilates the tangent space of the orbit through g:

        u(g)^t (M g) = g^t A^t M g = 0,   A^t M being antisymmetric.

    So a nonzero A PROVES that no orbit through a g with A g /= 0 is open,
    hence that the first-layer modality is at least 1.  This is a proof, not a
    sample: it is one exact linear system in the n^2 entries of A.
    """
    cols = {(i, j): {} for i in range(n) for j in range(n)}
    keys = set()
    for t, M in enumerate(mats):
        for u in range(n):
            for v in range(u, n):
                key = (t, u, v)
                keys.add(key)
                for i in range(n):
                    if M[i][v]:
                        c = cols[(i, u)]
                        c[key] = c.get(key, Fraction(0)) + M[i][v]
                    if M[i][u]:
                        c = cols[(i, v)]
                        c[key] = c.get(key, Fraction(0)) + M[i][u]
    order = [(i, j) for i in range(n) for j in range(n)]
    sols = nullspace([cols[k] for k in order], keys)
    out = []
    for s in sols:
        A = [[Fraction(0)] * n for _ in range(n)]
        for (i, j), c in zip(order, s):
            A[i][j] = c
        out.append(A)
    return out


def form_kind(A, n):
    sym = all(A[i][j] == A[j][i] for i in range(n) for j in range(n))
    anti = all(A[i][j] == -A[j][i] for i in range(n) for j in range(n))
    return "symmetric" if sym else ("antisymmetric" if anti else "mixed")


def show_poly(P, names):
    out = []
    for a, c in sorted(P.items()):
        t = []
        for i, e in enumerate(a):
            if e == 1:
                t.append(names[i])
            elif e > 1:
                t.append(f"{names[i]}^{e}")
        out.append(("+ " if c > 0 else "- ")
                   + (f"{abs(c)}*" if abs(c) != 1 else "") + "".join(t))
    return " ".join(out).lstrip("+ ")


# ------------------------------------------------------------ quasi-homogeneity

def weight_space(p, q, f):
    """
    Solutions (w_1..w_p, v_1..v_q, delta) of  <exponent, w> = delta  for every
    monomial of f.  Returns (dim of the solution space, one solution or None,
    dim of the subspace with delta = 0) -- the last being the dimension of the
    diagonal torus in the stabiliser.
    """
    mons = sorted(f, key=mkey)
    n = p + q + 1
    cols = [dict() for _ in range(n)]
    for r, m in enumerate(mons):
        for i in range(p):
            if m[0][i]:
                cols[i][r] = Fraction(m[0][i])
        for j in sorted(m[1]):
            cols[p + j][r] = Fraction(1)
        cols[p + q][r] = Fraction(-1)
    sols = nullspace(cols, set(range(len(mons))))
    tor = [s for s in sols]
    # subspace with delta = 0
    zero = [s for s in sols if s[p + q] == 0]
    # dimension of {delta=0} inside the solution space
    if any(s[p + q] for s in sols):
        dim0 = len(sols) - 1
    else:
        dim0 = len(sols)
    pos = None
    for s in sols:
        if s[p + q] > 0 and all(x > 0 for x in s[:p + q]):
            pos = s
            break
    return len(sols), pos, dim0, mons


# ---------------------------------------------------------------- the driver

def analyse(name, p, q, terms, D, M=None, verbose=True, show_basis=False):
    """
    D is the truncation used for the layer; the stabiliser is solved in its own,
    larger, ring, big enough that no term of V(f) is ever cut.

    Both brackets are pushed all the way through to the orbits, and each yields
    a one-sided conclusion that holds whether or not they meet:

        S_L  <  S  <  S_U   ==>   mod_U  <=  mod(f)  <=  mod_L,

    so  mod_L = 0  PROVES finitely many classes, and  mod_U > 0  PROVES moduli.
    """
    R, f, mI, basis, k, safe = layer(p, q, terms, D)
    n = len(basis)
    index = {m: i for i, m in enumerate(basis)}

    chk = sv.stable(p, q, terms, D)
    ok_layer = (chk["layer"] == n)
    ok_rho_room = (2 * k - 1 <= safe) or p == 0

    if M is None:
        M = max(k - 1, 0) + deg_max(f)
    Ds = M + 1 + xdeg_max(f)
    Rs = sv.Ring(p, q, max(Ds, D))

    Lb, Ub, dimL, dimU = jet_algebra(Rs, f, p, q, k, M)
    certified = (dimL == dimU)

    def package(S):
        mats = [rho(R, V, basis, mI, index) for V in S]
        filt = filtration_dims(S, max(k, 1))
        lin = (filt[0] - filt[1]) if len(filt) > 1 else filt[0]
        g, _ = generic_orbit_dim(mats, n) if mats else (0, None)
        return mats, filt, lin, g, n - g

    matsL, filtL, linL, gL, modL = package(Lb)
    matsU, filtU, linU, gU, modU = package(Ub)

    if modL == 0:
        verdict = "finite"
    elif modU > 0:
        verdict = f"MODULI (>= {modU})"
    else:
        verdict = f"mod in [{modU},{modL}]"

    dimN = len([m for m in basis if mdeg(m) >= 2])
    if verbose:
        print(f"--- {name}   p|q = {p}|{q}   sdim O_f = "
              f"({chk['sdim'][0]}|{chk['sdim'][1]})")
        print(f"    Nt_f: dim {n}  (N_f: {dimN}, new stratum: {n-dimN}), "
              f"top degree k = {k}, layer cross-check "
              f"{'OK' if ok_layer else 'MISMATCH'}")
        if show_basis:
            print("    basis: " + ", ".join(sv.show_mon(m, "xy", "xi")
                                            for m in basis))
        print(f"    jet algebra S = s_f mod deg>={k}: dim {dimL} (lower) / "
              f"{dimU} (upper) at M={M}   "
              f"{'CERTIFIED' if certified else 'not certified'}")
        print(f"    filtration dims (deg>=0,1,...,{k-1}): {filtL}, "
              f"linear part {linL}")
        print(f"    generic orbit dim {gL} (lower) / {gU} (upper) of {n}")
        print(f"    first-layer modality in [{modU},{modL}]   ==>  {verdict}")
    return {
        "name": name, "p": p, "q": q, "sdim": chk["sdim"], "dimNt": n,
        "dimN": dimN, "k": k, "filt": filtL, "lin": linL,
        "dimL": dimL, "dimU": dimU, "M": M, "verdict": verdict,
        "certified": certified, "gdim": gL, "gdimU": gU,
        "modality": modL, "modL": modL, "modU": modU,
        "mats": matsL if certified else matsU, "matsL": matsL, "matsU": matsU,
        "basis": basis, "R": R, "f": f, "mI": mI,
        "index": index, "S": Lb if certified else Ub, "ok_layer": ok_layer,
        "ok_rho_room": ok_rho_room,
        "rhodim": rho_image_dim(matsL) if matsL else 0,
    }
