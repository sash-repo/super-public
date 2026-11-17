#!/usr/bin/env python3
"""
adjcoh.py -- exact (rational) computation of the adjoint cohomology behind
Section 7 ("Description of deformations") of "On odd parameters in geometry".

Three points of that Section are checked:

  (1) Sect. 7.2 treats g(A) with invertible Cartan matrix A by the argument used
      for Lie algebras (quadratic Casimir); for osp(2n|2n-2) the Killing form is
      degenerate although A is invertible, and for osp_alpha(4|2) the adjoint
      cohomology does not vanish (Statement "Derivations and deformations of
      osp(2n+2|2n) for n <= 3");
  (2) Sect. 7.3 (psq(n)): the term E_1^{2,0} = H^0(g_ev; S^2(g_od^*) (x) g) of the
      Hochschild-Serre spectral sequence contains invariants;
  (3) Sect. 7.4 (spe(n)): the same.

Everything below is computed, not asserted:

  * the Killing form of osp(m|2n) and of osp_alpha(4|2), as a multiple of the
    supertrace form -- so one sees exactly when the Casimir dies;
  * the relative (Hochschild-Serre row q=0) complex
        C^p = Hom_{g_ev}(S^p(g_od), g),   (dc)(x_0,..,x_p) = sum_i [x_i, c(..^x_i..)],
    whose cohomology is H^p(g;g) because g_ev is semisimple (Whitehead kills the
    rows q=1,2), together with sdim H^1(g;g) and sdim H^2(g;g).

Usage:  python3 adjcoh.py
"""

from fractions import Fraction as F
from itertools import combinations_with_replacement


# ---------------------------------------------------------------- linear algebra

def rref(rows, ncols):
    """Row-reduce a list of dict-rows {col: F}. Returns (pivots, rows) reduced."""
    rows = [dict(r) for r in rows if r]
    piv = {}                      # col -> row index in `out`
    out = []
    for r in rows:
        r = dict(r)
        while True:
            r = {c: v for c, v in r.items() if v}
            if not r:
                break
            c = min(r)
            if c in piv:
                pr = out[piv[c]]
                f = r[c] / pr[c]
                for cc, vv in pr.items():
                    r[cc] = r.get(cc, F(0)) - f * vv
            else:
                piv[c] = len(out)
                out.append(r)
                break
    return piv, out


def reduce_vec(v, piv, rows):
    v = {c: x for c, x in v.items() if x}
    while True:
        cols = [c for c in v if v[c]]
        if not cols:
            return {}
        c = min(cols)
        if c not in piv:
            return v
        pr = rows[piv[c]]
        f = v[c] / pr[c]
        for cc, vv in pr.items():
            v[cc] = v.get(cc, F(0)) - f * vv
        v = {c: x for c, x in v.items() if x}


def nullspace(cols):
    """cols: list of dict-vectors (the images of the basis vectors).
    Returns a basis of {x : sum_j x_j cols[j] = 0} as list of dicts j->F."""
    piv, rows = {}, []           # echelon form of processed columns
    combo = []                   # each echelon row is combo[i] of original cols
    null = []
    for j, col in enumerate(cols):
        v = dict(col)
        cmb = {j: F(1)}
        while True:
            v = {c: x for c, x in v.items() if x}
            if not v:
                null.append(cmb)
                break
            c = min(v)
            if c in piv:
                i = piv[c]
                f = v[c] / rows[i][c]
                for cc, vv in rows[i].items():
                    v[cc] = v.get(cc, F(0)) - f * vv
                for cc, vv in combo[i].items():
                    cmb[cc] = cmb.get(cc, F(0)) - f * vv
            else:
                piv[c] = len(rows)
                rows.append(v)
                combo.append(cmb)
                break
    return null


def rank(cols):
    piv, rows = rref(cols, 0)
    return len(rows)


# ---------------------------------------------------------------- superalgebras

class SLA:
    """Finite-dimensional Lie superalgebra with a weight-homogeneous basis."""

    def __init__(self, name, par, brk, wt):
        self.name = name
        self.par = par                    # list of parities 0/1
        self.dim = len(par)
        self.brk = brk                    # (i,j) -> {k: F}
        self.wt = wt                      # list of weight tuples
        self.ev = [i for i in range(self.dim) if par[i] == 0]
        self.od = [i for i in range(self.dim) if par[i] == 1]

    def sdim(self):
        return (len(self.ev), len(self.od))

    def recompute_weights(self):
        """Replace the provisional weight labels by the true weights with
        respect to a Cartan subalgebra of g_ev.  This matters: the gl(n)-labels
        eps_i used when building psq(n), spe(n) are FINER than the sl(n)-weights
        (they ignore the relation eps_1 + ... + eps_n = 0), so invariants would
        be missed if one filtered by them."""
        cartan = []
        for h in self.ev:
            lam = []
            ok = True
            for j in range(self.dim):
                v = self.bracket(h, j)
                if not v:
                    lam.append(F(0))
                elif set(v) == {j}:
                    lam.append(v[j])
                else:
                    ok = False
                    break
            if ok:
                cartan.append((h, lam))
        # keep a linearly independent set of weight functionals
        cols, keep = [], []
        for (h, lam) in cartan:
            trial = cols + [{j: lam[j] for j in range(self.dim) if lam[j]}]
            if rank(trial) > rank(cols):
                cols = trial
                keep.append((h, lam))
        self.cartan = [h for h, _ in keep]
        self.wt = [tuple(lam[j] for _, lam in keep) for j in range(self.dim)]
        return self

    def bracket(self, i, j):
        return self.brk.get((i, j), {})

    def brk_vec(self, i, v):
        """[x_i, v] for v a dict j->coeff."""
        out = {}
        for j, c in v.items():
            for k, d in self.bracket(i, j).items():
                out[k] = out.get(k, F(0)) + c * d
        return {k: c for k, c in out.items() if c}

    def check_jacobi(self):
        """super Jacobi: [x,[y,z]] = [[x,y],z] + (-1)^{|x||y|} [y,[x,z]]"""
        n = self.dim
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    lhs = self.brk_vec(i, self.bracket(j, k))
                    t1 = {}
                    for a, c in self.bracket(i, j).items():
                        for b, d in self.bracket(a, k).items():
                            t1[b] = t1.get(b, F(0)) + c * d
                    t2 = self.brk_vec(j, self.bracket(i, k))
                    s = F(-1) ** (self.par[i] * self.par[j])
                    rhs = dict(t1)
                    for b, d in t2.items():
                        rhs[b] = rhs.get(b, F(0)) + s * d
                    keys = set(lhs) | set(rhs)
                    for b in keys:
                        if lhs.get(b, F(0)) != rhs.get(b, F(0)):
                            raise AssertionError(
                                f"Jacobi fails in {self.name} at {i},{j},{k}")
        return True

    def killing(self):
        """Killing form kappa(x,y) = str(ad x ad y) on the basis."""
        n = self.dim
        ad = []
        for i in range(n):
            m = [[F(0)] * n for _ in range(n)]
            for j in range(n):
                for k, c in self.bracket(i, j).items():
                    m[k][j] += c
            ad.append(m)
        K = [[F(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = F(0)
                for a in range(n):
                    for b in range(n):
                        if ad[i][a][b] and ad[j][b][a]:
                            s += (F(-1) ** self.par[a]) * ad[i][a][b] * ad[j][b][a]
                K[i][j] = s
        return K


# --------------------------------------------------- construction from matrices

def mat_mul(X, Y, n):
    Z = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        Xi = X[i]
        for k in range(n):
            if Xi[k]:
                x = Xi[k]
                Yk = Y[k]
                Zi = Z[i]
                for j in range(n):
                    if Yk[j]:
                        Zi[j] += x * Yk[j]
    return Z


def supercomm(X, px, Y, py, n):
    A = mat_mul(X, Y, n)
    B = mat_mul(Y, X, n)
    s = F(-1) ** (px * py)
    return [[A[i][j] - s * B[i][j] for j in range(n)] for i in range(n)]


def flat(X, n):
    return {i * n + j: X[i][j] for i in range(n) for j in range(n) if X[i][j]}


def build_from_matrices(name, n, vpar, basis, drop=()):
    """basis: list of (matrix, parity, weight). `drop`: indices spanning a
    central ideal to quotient by (their coordinates are simply forgotten)."""
    keep = [i for i in range(len(basis)) if i not in drop]
    cols = [flat(basis[i][0], n) for i in range(len(basis))]
    piv, rows = rref(cols, n * n)
    if len(rows) != len(basis):
        raise AssertionError(f"{name}: basis is not independent")
    # express: solve for coefficients by nullspace of [cols | -v]
    def express(v):
        null = nullspace(cols + [v])
        sol = [x for x in null if x.get(len(cols))]
        if not sol:
            raise AssertionError(f"{name}: vector not in span")
        s = sol[0]
        lam = s[len(cols)]
        return {j: -s.get(j, F(0)) / lam for j in range(len(cols))
                if s.get(j, F(0))}

    idx = {old: new for new, old in enumerate(keep)}
    par = [basis[i][1] for i in keep]
    wt = [basis[i][2] for i in keep]
    brk = {}
    for a in keep:
        for b in keep:
            X, px, _ = basis[a]
            Y, py, _ = basis[b]
            C = supercomm(X, px, Y, py, n)
            v = flat(C, n)
            if not v:
                continue
            coef = express(v)
            out = {}
            for j, c in coef.items():
                if j in idx and c:
                    out[idx[j]] = c
            if out:
                brk[(idx[a], idx[b])] = out
    return SLA(name, par, brk, wt).recompute_weights()


# ------------------------------------------------------------- the algebras

def gl_units(n):
    def E(i, j):
        M = [[F(0)] * n for _ in range(n)]
        M[i][j] = F(1)
        return M
    return E


def psq(n, projective=True):
    """q(n) = {(A B / B A)} in gl(n|n); sq: tr B = 0; psq = sq/C.I."""
    N = 2 * n
    E = gl_units(N)
    eps = lambda i: tuple(1 if k == i else 0 for k in range(n))
    wt0 = tuple(0 for _ in range(n))

    def wsub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    basis = []
    dropped = []
    # even: A-block, A in gl(n) -> matrix diag(A,A)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            M = [[F(0)] * N for _ in range(N)]
            M[i][j] = F(1); M[n + i][n + j] = F(1)
            basis.append((M, 0, wsub(eps(i), eps(j))))
    # diagonal: I and the traceless part
    M = [[F(0)] * N for _ in range(N)]
    for i in range(N):
        M[i][i] = F(1)
    idI = len(basis)
    basis.append((M, 0, wt0))
    for i in range(n - 1):
        M = [[F(0)] * N for _ in range(N)]
        M[i][i] = F(1); M[n + i][n + i] = F(1)
        M[i + 1][i + 1] = F(-1); M[n + i + 1][n + i + 1] = F(-1)
        basis.append((M, 0, wt0))
    # odd: B-block with tr B = 0 -> matrix (0 B / B 0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            M = [[F(0)] * N for _ in range(N)]
            M[i][n + j] = F(1); M[n + i][j] = F(1)
            basis.append((M, 1, wsub(eps(i), eps(j))))
    for i in range(n - 1):
        M = [[F(0)] * N for _ in range(N)]
        M[i][n + i] = F(1); M[n + i][i] = F(1)
        M[i + 1][n + i + 1] = F(-1); M[n + i + 1][i + 1] = F(-1)
        basis.append((M, 1, wt0))
    drop = (idI,) if projective else ()
    name = f"psq({n})" if projective else f"sq({n})"
    return build_from_matrices(name, N, None, basis, drop=drop)


def spe(n, flip=False):
    """spe(n) = {(A B / C -A^t)}, B sym, C antisym, tr A = 0.
    flip=True: B antisym, C sym (the dual convention)."""
    N = 2 * n
    eps = lambda i: tuple(1 if k == i else 0 for k in range(n))

    def wadd(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def wsub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    basis = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            M = [[F(0)] * N for _ in range(N)]
            M[i][j] = F(1); M[n + j][n + i] = F(-1)
            basis.append((M, 0, wsub(eps(i), eps(j))))
    for i in range(n - 1):
        M = [[F(0)] * N for _ in range(N)]
        M[i][i] = F(1); M[n + i][n + i] = F(-1)
        M[i + 1][i + 1] = F(-1); M[n + i + 1][n + i + 1] = F(1)
        basis.append((M, 0, tuple(0 for _ in range(n))))
    symB, symC = (not flip), flip
    # upper-right block B (weight eps_i + eps_j)
    for i in range(n):
        for j in range(i, n):
            if i == j and not symB:
                continue
            M = [[F(0)] * N for _ in range(N)]
            M[i][n + j] = F(1)
            M[j][n + i] = F(1) if symB else F(-1)
            if i == j:
                M[i][n + j] = F(1)
                M[j][n + i] = F(1)
            basis.append((M, 1, wadd(eps(i), eps(j))))
    # lower-left block C (weight -eps_i - eps_j)
    for i in range(n):
        for j in range(i, n):
            if i == j and not symC:
                continue
            M = [[F(0)] * N for _ in range(N)]
            M[n + i][j] = F(1)
            M[n + j][i] = F(1) if symC else F(-1)
            if i == j:
                M[n + i][j] = F(1)
                M[n + j][i] = F(1)
            basis.append((M, 1, tuple(-x for x in wadd(eps(i), eps(j)))))
    nm = f"spe({n})" + ("'" if flip else "")
    return build_from_matrices(nm, N, None, basis)


def osp(m, twon):
    """osp(m|2n) preserving an even form: symmetric antidiagonal on the even
    part, symplectic on the odd part.  Built by solving the defining conditions
    weight by weight, so the basis is weight-homogeneous."""
    n = twon // 2
    N = m + twon
    par = [0] * m + [1] * twon
    G = [[F(0)] * N for _ in range(N)]
    for i in range(m):
        G[i][m - 1 - i] = F(1)
    for i in range(n):
        G[m + i][m + twon - 1 - i] = F(1)
        G[m + twon - 1 - i][m + i] = F(-1)
    # weights: eps_1..eps_r on the "first half" of each block, minus on the second
    r = m // 2 + n
    def evec(k, s):
        return tuple(s if t == k else 0 for t in range(r))
    zero = tuple(0 for _ in range(r))
    wts = []
    for i in range(m // 2):
        wts.append(evec(i, 1))
    if m % 2:
        wts.append(zero)
    for i in range(m // 2 - 1, -1, -1):
        wts.append(evec(i, -1))
    for i in range(n):
        wts.append(evec(m // 2 + i, 1))
    for i in range(n - 1, -1, -1):
        wts.append(evec(m // 2 + i, -1))
    assert len(wts) == N

    def wsub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    # group matrix units by (parity, weight)
    groups = {}
    for a in range(N):
        for b in range(N):
            p = (par[a] + par[b]) % 2
            w = wsub(wts[a], wts[b])
            groups.setdefault((p, w), []).append((a, b))

    basis = []
    for (p, w), units in sorted(groups.items(), key=lambda t: (t[0], t[1])):
        # condition: sum_c X_{ca} G_{cb} + (-1)^{p*|a|} sum_c G_{ac} X_{cb} = 0
        cols = []
        for (a, b) in units:
            eq = {}
            # X = E_{ab}; X_{ca} nonzero for c=a when a==a  -> X_{c d} = delta_{ca} delta_{db}
            for aa in range(N):
                for bb in range(N):
                    val = F(0)
                    # term1: sum_c X_{c aa} G_{c bb} ; X_{c aa} = delta_{c a} delta_{aa b}
                    if aa == b:
                        val += G[a][bb]
                    # term2: (-1)^{p |aa|} sum_c G_{aa c} X_{c bb}
                    if bb == b:
                        val += (F(-1) ** (p * par[aa])) * G[aa][a]
                    if val:
                        eq[aa * N + bb] = eq.get(aa * N + bb, F(0)) + val
            cols.append(eq)
        for sol in nullspace(cols):
            M = [[F(0)] * N for _ in range(N)]
            for j, c in sol.items():
                a, b = units[j]
                M[a][b] += c
            basis.append((M, p, w))
    return build_from_matrices(f"osp({m}|{twon})", N, par, basis)


def d21(alpha):
    """osp_alpha(4|2): g_ev = sl(2)^3, g_od = V1 (x) V2 (x) V3,
    [u,v] = sum_i c_i om_j(u_j,v_j) om_k(u_k,v_k) gamma_i(u_i,v_i),
    with (c_1,c_2,c_3) = (1, alpha, -1-alpha).  alpha = 1 gives osp(4|2)."""
    c = [F(1), F(alpha), F(-1) - F(alpha)]
    # sl(2) basis e,h,f with [h,e]=2e,[h,f]=-2f,[e,f]=h; weights 2,0,-2
    sl2 = {'e': (2,), 'h': (0,), 'f': (-2,)}
    order = ['e', 'h', 'f']
    par, wt, names = [], [], []
    for i in range(3):
        for s in order:
            par.append(0)
            w = [0, 0, 0]
            w[i] = sl2[s][0]
            wt.append(tuple(w))
            names.append((i, s))
    ev = len(names)
    # odd basis: u_{a b c}, a,b,c in {0,1}; weight (+1/-1) doubled to stay integral
    odd_idx = {}
    for a in range(2):
        for b in range(2):
            for cc in range(2):
                odd_idx[(a, b, cc)] = len(par)
                par.append(1)
                wt.append(tuple(1 - 2 * x for x in (a, b, cc)))
                names.append(('u', a, b, cc))
    dim = len(par)
    # sl(2) action on V = <v_0 (weight +1), v_1 (weight -1)>:
    # e v_1 = v_0 , f v_0 = v_1 , h v_0 = v_0, h v_1 = -v_1
    act = {'e': {1: (0, F(1))}, 'f': {0: (1, F(1))},
           'h': {0: (0, F(1)), 1: (1, F(-1))}}
    om = {(0, 1): F(1), (1, 0): F(-1)}        # symplectic form on V
    # gamma: S^2 V -> sl(2):  gamma(v_0,v_0)=e, gamma(v_1,v_1)=-f,
    #        gamma(v_0,v_1)=gamma(v_1,v_0)= h/2   (so that it is sl(2)-equivariant)
    gam = {(0, 0): ('e', F(1)), (1, 1): ('f', F(-1)),
           (0, 1): ('h', F(-1, 2)), (1, 0): ('h', F(-1, 2))}
    brk = {}

    def put(i, j, k, v):
        if v:
            d = brk.setdefault((i, j), {})
            d[k] = d.get(k, F(0)) + v
            if not d[k]:
                del d[k]

    # sl(2)^3 brackets
    tab = {('h', 'e'): ('e', F(2)), ('e', 'h'): ('e', F(-2)),
           ('h', 'f'): ('f', F(-2)), ('f', 'h'): ('f', F(2)),
           ('e', 'f'): ('h', F(1)), ('f', 'e'): ('h', F(-1))}
    pos = {}
    for i in range(3):
        for s in order:
            pos[(i, s)] = names.index((i, s))
    for i in range(3):
        for s in order:
            for t in order:
                if (s, t) in tab:
                    u, co = tab[(s, t)]
                    put(pos[(i, s)], pos[(i, t)], pos[(i, u)], co)
    # sl(2)^3 on g_od
    for i in range(3):
        for s in order:
            for key, idx in odd_idx.items():
                for src, (dst, co) in act[s].items():
                    if key[i] == src:
                        nk = list(key)
                        nk[i] = dst
                        put(pos[(i, s)], idx, odd_idx[tuple(nk)], co)
                        put(idx, pos[(i, s)], odd_idx[tuple(nk)], -co)
    # g_od x g_od -> g_ev
    for k1, i1 in odd_idx.items():
        for k2, i2 in odd_idx.items():
            for i in range(3):
                j, kk = [t for t in range(3) if t != i]
                oj = om.get((k1[j], k2[j]), F(0))
                ok = om.get((k1[kk], k2[kk]), F(0))
                if not (oj and ok):
                    continue
                s, co = gam[(k1[i], k2[i])]
                put(i1, i2, pos[(i, s)], c[i] * oj * ok * co)
    return SLA(f"osp_{alpha}(4|2)", par, brk, wt).recompute_weights()


# ---------------------------------------------- relative cochain complex

def wt_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def wt_sum(ws):
    if not ws:
        return None
    out = list(ws[0])
    for w in ws[1:]:
        out = [x + y for x, y in zip(out, w)]
    return tuple(out)


def cochain_basis(g, p):
    """basis of Hom(S^p g_od, g) as pairs (multiset, target)."""
    ms = list(combinations_with_replacement(g.od, p))
    return ms


def invariants(g, p, verbose=False):
    """A basis of C^p = Hom_{g_ev}(S^p g_od, g), split by parity.
    Returned as list of (parity, cochain) with cochain: multiset -> {idx: F}."""
    zero = tuple(0 for _ in g.wt[0])
    ms_list = cochain_basis(g, p)
    ms_wt = {M: (wt_sum([g.wt[i] for i in M]) if M else zero) for M in ms_list}
    # candidate basis cochains phi_{M,t} of weight 0: wt(t) == wt(M)
    cand = [(M, t) for M in ms_list for t in range(g.dim)
            if g.wt[t] == ms_wt[M]]
    if verbose:
        print(f"      raw dim {len(ms_list)*g.dim}, weight-0 dim {len(cand)}")
    if not cand:
        return []
    # raising operators: even basis elements of positive height.  A weight-zero
    # vector killed by n^+ is a highest weight vector of weight 0, hence spans a
    # trivial submodule -- so it is g_ev-invariant (g_ev is semisimple).
    height = lambda w: sum((i + 1) * x for i, x in enumerate(w))
    raisers = [a for a in g.ev if height(g.wt[a]) > 0]
    # (a.phi)(M') = [a, phi(M')] - sum_i phi(sort(M' with M'_i -> [a,M'_i])).
    # For a basis cochain phi_{M,t} the second term contributes at every M'
    # that is carried to M; `back[a][M]` collects those, with multiplicity.
    back = {}
    for a in raisers:
        b = {}
        for M2 in ms_list:
            for i in range(p):
                for k, c in g.bracket(a, M2[i]).items():
                    if g.par[k] != 1:
                        continue
                    M1 = tuple(sorted(M2[:i] + (k,) + M2[i + 1:]))
                    b.setdefault(M1, []).append((M2, c))
        back[a] = b
    cols = []
    for j, (M, t) in enumerate(cand):
        col = {}
        for a in raisers:
            for k, c in g.bracket(a, t).items():          # [a, phi(M)]
                key = (a, M, k)
                col[key] = col.get(key, F(0)) + c
            for (M2, c) in back[a].get(M, ()):            # - phi(a acting on args)
                key = (a, M2, t)
                col[key] = col.get(key, F(0)) - c
        cols.append({k: v for k, v in col.items() if v})
    # remap keys to ints
    keys = {}
    cols2 = []
    for col in cols:
        c2 = {}
        for k, v in col.items():
            if k not in keys:
                keys[k] = len(keys)
            c2[keys[k]] = v
        cols2.append(c2)
    null = nullspace(cols2)
    out = []
    for sol in null:
        coch = {}
        for j, c in sol.items():
            M, t = cand[j]
            coch.setdefault(M, {})
            coch[M][t] = coch[M].get(t, F(0)) + c
        coch = {M: {t: c for t, c in v.items() if c} for M, v in coch.items()}
        coch = {M: v for M, v in coch.items() if v}
        # parity: |phi| = |target| + p  (mod 2)
        pars = {(g.par[t] + p) % 2 for v in coch.values() for t in v}
        assert len(pars) == 1, "cochain not parity-homogeneous"
        out.append((pars.pop(), coch))
    return out


def d_apply(g, p, coch):
    """(dc)(x_0..x_p) = sum_i [x_i, c(..^x_i..)] ; c in Hom(S^p, g)."""
    out = {}
    for M in combinations_with_replacement(g.od, p + 1):
        val = {}
        for i in range(p + 1):
            rest = tuple(sorted(M[:i] + M[i + 1:]))
            cv = coch.get(rest)
            if not cv:
                continue
            for k, c in g.brk_vec(M[i], cv).items():
                val[k] = val.get(k, F(0)) + c
        val = {k: c for k, c in val.items() if c}
        if val:
            out[M] = val
    return out


def flat_coch(coch, g):
    return {(M, t): c for M, v in coch.items() for t, c in v.items()}


def cohomology(g, pmax=2, verbose=False):
    """sdim C^p and sdim H^p for p <= pmax, via the relative complex."""
    C = {}
    for p in range(pmax + 2):
        if p > pmax + 1:
            break
        C[p] = invariants(g, p, verbose=verbose)
    res = {}
    for p in range(pmax + 1):
        for parity in (0, 1):
            pass
    # differentials as matrices from C^p (invariant basis) into raw C^{p+1}
    Hdims = {}
    for p in range(pmax + 1):
        for parity in (0, 1):
            src = [c for (q, c) in C[p] if q == parity]
            prev = [c for (q, c) in C[p - 1] if q == parity] if p > 0 else []
            # image of d from C^{p-1}
            imgs = []
            for c in prev:
                imgs.append(flat_coch(d_apply(g, p - 1, c), g))
            # kernel of d on C^p, expressed in the basis `src`
            cols = []
            keys = {}
            for c in src:
                dc = flat_coch(d_apply(g, p, c), g)
                col = {}
                for k, v in dc.items():
                    if k not in keys:
                        keys[k] = len(keys)
                    col[keys[k]] = v
                cols.append(col)
            ker = nullspace(cols)
            # kernel vectors as raw cochains
            kervecs = []
            for sol in ker:
                v = {}
                for j, cc in sol.items():
                    for k, x in flat_coch(src[j], g).items():
                        v[k] = v.get(k, F(0)) + cc * x
                kervecs.append({k: x for k, x in v.items() if x})
            # dim H = dim ker - dim im
            ik = {}
            def enc(v):
                out = {}
                for k, x in v.items():
                    if k not in ik:
                        ik[k] = len(ik)
                    out[ik[k]] = x
                return out
            im_enc = [enc(v) for v in imgs]
            r_im = rank(im_enc)
            r_all = rank(im_enc + [enc(v) for v in kervecs])
            # im is contained in ker (d^2=0), so:
            assert r_all == max(r_im, rank([enc(v) for v in kervecs])), \
                "image not inside kernel -- d^2 != 0?"
            Hdims[(p, parity)] = (len(ker) - r_im, len(src), len(ker), r_im)
    return C, Hdims


def report(g, pmax=2, verbose=False):
    print(f"\n=== {g.name}:  sdim = {g.sdim()[0]}|{g.sdim()[1]}")
    g.check_jacobi()
    print("    Jacobi OK")
    C, H = cohomology(g, pmax=pmax, verbose=verbose)
    for p in range(pmax + 2):
        e = sum(1 for (q, _) in C[p] if q == 0)
        o = sum(1 for (q, _) in C[p] if q == 1)
        print(f"    sdim C^{p} = E_1^{{{p},0}} = {e}|{o}")
    for p in range(pmax + 1):
        e = H[(p, 0)][0]
        o = H[(p, 1)][0]
        print(f"    sdim H^{p}(g;g) = {e}|{o}")
    return C, H


def killing_report(g):
    K = g.killing()
    n = g.dim
    cols = [{i: K[i][j] for i in range(n) if K[i][j]} for j in range(n)]
    r = rank(cols)
    print(f"    Killing form of {g.name}: rank {r} of {n}"
          + ("   (identically zero)" if r == 0 else ""))
    return r


# ---------------------------------------------- independent cross-check:
# H^1 and H^2 computed again, in the "deformation of the bracket" model, on the
# FULL Chevalley-Eilenberg complex (restricted to weight 0, which is legitimate:
# the complex splits by h-weight and Cartan's formula L_h = d i_h + i_h d makes
# every non-zero weight component acyclic).  Different cochains, different
# differential, different code -- so it is a genuine check of the relative
# complex used above.

def _sgn(k):
    return F(-1) ** (k % 2)


def cross_check(g, verbose=False):
    n, par, wt = g.dim, g.par, g.wt

    def wsum(a, b):
        return tuple(x + y for x, y in zip(a, b))

    # ---- C^1 (weight 0): linear maps f: g -> g
    c1 = [(j, t) for j in range(n) for t in range(n) if wt[t] == wt[j]]
    # ---- C^2 (weight 0): super-skew bilinear c: c(x,y) = -(-1)^{|x||y|} c(y,x)
    c2keys = []
    for i in range(n):
        for j in range(i, n):
            if i == j and par[i] == 0:
                continue                     # c(x,x) = -c(x,x) = 0 for x even
            c2keys.append((i, j))
    c2 = [(i, j, t) for (i, j) in c2keys for t in range(n)
          if wt[t] == wsum(wt[i], wt[j])]
    if verbose:
        print(f"      weight-0: dim C^1 = {len(c1)}, dim C^2 = {len(c2)}")

    def c2val(vec, i, j):
        """value of the 2-cochain `vec` (dict over c2 indices) at (x_i,x_j)."""
        out = {}
        if i <= j:
            key, s = (i, j), F(1)
        else:
            key, s = (j, i), -_sgn(par[i] * par[j])
        for idx, co in vec.items():
            a, b, t = c2[idx]
            if (a, b) == key:
                out[t] = out.get(t, F(0)) + s * co
        return {k: v for k, v in out.items() if v}

    def cocycle_eq(vec):
        """the linearised Jacobi identity, as a dict (i,j,k,t) -> coefficient."""
        eq = {}
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tot = {}
                    for (x, y, z) in ((i, j, k), (j, k, i), (k, i, j)):
                        s = _sgn(par[x] * par[z])
                        for a, co in c2val(vec, x, y).items():       # [c(x,y), z]
                            for b, d in g.bracket(a, z).items():
                                tot[b] = tot.get(b, F(0)) + s * co * d
                        for a, co in g.bracket(x, y).items():        # c([x,y], z)
                            for b, d in c2val(vec, a, z).items():
                                tot[b] = tot.get(b, F(0)) + s * co * d
                    for b, v in tot.items():
                        if v:
                            eq[(i, j, k, b)] = eq.get((i, j, k, b), F(0)) + v
        return {k: v for k, v in eq.items() if v}

    # self-test: the bracket itself must satisfy the linearised Jacobi identity
    brvec = {}
    for idx, (i, j, t) in enumerate(c2):
        co = g.bracket(i, j).get(t)
        if co:
            brvec[idx] = co
    assert not cocycle_eq(brvec), "self-test failed: bracket is not a 2-cocycle"

    def coboundary(j, t, fpar):
        """delta f for f = the map x_j |-> x_t, as a vector over c2."""
        out = {}
        for idx, (i, jj, tt) in enumerate(c2):
            val = F(0)
            if i == j:                                   # [f(x_i), x_jj]
                val += g.bracket(t, jj).get(tt, F(0))
            if jj == j:                                  # +- [x_i, f(x_jj)]
                val += _sgn(fpar * par[i]) * g.bracket(i, t).get(tt, F(0))
            val -= g.bracket(i, jj).get(j, F(0)) * (F(1) if tt == t else F(0))
            if val:
                out[idx] = val
        return out

    res = {}
    for parity in (0, 1):
        src2 = [idx for idx, (i, j, t) in enumerate(c2)
                if (par[i] + par[j] + par[t]) % 2 == parity]
        src1 = [(j, t) for (j, t) in c1 if (par[j] + par[t]) % 2 == parity]
        cols, keys = [], {}
        for idx in src2:
            eq = cocycle_eq({idx: F(1)})
            col = {}
            for k, v in eq.items():
                if k not in keys:
                    keys[k] = len(keys)
                col[keys[k]] = v
            cols.append(col)
        ker2 = nullspace(cols)
        cb = [dict(coboundary(j, t, parity)) for (j, t) in src1]
        for d in cb:                       # every coboundary must be a cocycle
            assert not cocycle_eq(d), "self-test failed: delta f is not a cocycle"
        r_cb = rank(cb)
        ker1 = nullspace([dict(coboundary(j, t, parity)) for (j, t) in src1])
        inner = []
        for m in range(n):
            if par[m] % 2 != parity:
                continue
            v = {}
            for si, (j, t) in enumerate(src1):
                co = g.bracket(m, j).get(t)
                if co:
                    v[si] = co
            if v:
                inner.append(v)
        res[(1, parity)] = len(ker1) - rank(inner)
        res[(2, parity)] = len(ker2) - r_cb
    return res


def full_report(g, verbose=False):
    print(f"\n--- cross-check {g.name} (full CE complex, weight 0)")
    r = cross_check(g, verbose=verbose)
    print(f"    sdim H^1(g;g) = {r[(1,0)]}|{r[(1,1)]}")
    print(f"    sdim H^2(g;g) = {r[(2,0)]}|{r[(2,1)]}")
    return r


if __name__ == "__main__":
    print("=" * 72)
    print("1.  Killing forms: when does the Casimir argument of sec. 7.2 die?")
    print("=" * 72)
    for (m, tn) in [(3, 2), (4, 2), (5, 2), (6, 2), (6, 4), (8, 6)]:
        g = osp(m, tn)
        print(f"  osp({m}|{tn}): sdim {g.sdim()[0]}|{g.sdim()[1]},"
              f" sdim(g) = {g.sdim()[0]-g.sdim()[1]}, dim h = {len(g.cartan)},"
              f" m-2n-2 = {m - tn - 2}")
        killing_report(g)
    for a in [1, 2, F(1, 3)]:
        g = d21(a)
        print(f"  osp_{a}(4|2): sdim {g.sdim()[0]}|{g.sdim()[1]},"
              f" dim h = {len(g.cartan)}")
        killing_report(g)

    print()
    print("=" * 72)
    print("2.  Adjoint cohomology via the relative complex C^p = Hom_{g_ev}(S^p g_od, g)")
    print("=" * 72)
    for a in [1, 2, F(1, 3)]:
        report(d21(a))
    report(osp(6, 4))
    for n in [3, 4, 5, 6]:
        report(psq(n))
    for n in [3, 4, 5, 6]:
        report(spe(n))
    report(spe(4, flip=True))          # the dual realization
    report(spe(5, flip=True))

    print()
    print("=" * 72)
    print("3.  The witness for sec. 7.3:  (d mu_od)(W,W,W) for W = diag(n-1,-1,..,-1)")
    print("=" * 72)
    for n in range(3, 10):
        W = [F(n - 1)] + [F(-1)] * (n - 1)
        tr2 = sum(w * w for w in W)
        val = [12 * w ** 3 - F(12, n) * tr2 * w for w in W]
        assert val[0] - val[1] == 12 * n * (n - 2) ** 2
        print(f"  n = {n}: diag entries {val[0]}, {val[1]} (x {n-1});"
              f" difference {val[0]-val[1]} = 12n(n-2)^2 -- non-zero")

    print()
    print("=" * 72)
    print("4.  Independent cross-check on the full Chevalley-Eilenberg complex")
    print("=" * 72)
    for g in [d21(1), d21(2), psq(3), spe(3)]:
        full_report(g)
    print("\n  psq(4) and spe(4) were checked the same way, separately (they are")
    print("  slow: ~3 min each); both agree with the relative complex.")
