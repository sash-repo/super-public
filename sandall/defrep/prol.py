#!/usr/bin/env python3
"""
prol.py --- the Cartan prolong (M, g)_* of an irreducible module M = L^lambda
over g(A), degree by degree, and the highest root of g(A).

Everything is Z^n-graded by the Chevalley multidegree: a vector of M_d (d counts
the f's used) has degree -d, the image of e_i has degree +e_i, of f_i degree
-e_i.  The prolong of a graded pair is graded, so each component g_k is found
block by block, one Z^n-degree at a time, which keeps every linear system small.

    g_k = { phi : g_{-1} -> g_{k-1} :  [phi(v), w] = (-1)^{p(v)p(w)} [phi(w), v] }

(the Jacobi identity for phi, v, w with [v, w] = 0); the sign does not depend on
the parity of phi.  g_0 is the image of g(A) in gl(M) (the Lie closure of the
images of the Chevalley generators), optionally enlarged by the identity and by
the grading operators of the degenerate directions.

Arithmetic: over F_p (p > 0) through hwmod's Fp; for characteristic 0 use a
large prime and read a zero result as a zero over Q (a rank over F_p is a lower
bound for the rank over Q, so a kernel that is 0 mod p is 0 over Q).  Any other
field object with norm/is_zero/inv/zero/one (e.g. GF(2^m) below) also works.
"""
import sys
from hwmod import HWModule, Fp

# ----------------------------------------------------------------- fields

class GF2m:
    """GF(2^m) as ints (bit vectors) modulo an irreducible polynomial."""
    def __init__(self, m, poly):
        self.m, self.poly, self.p = m, poly, 2
    def norm(self, x):
        x = int(x) if not isinstance(x, int) else x
        # integers given in the text (+-1, 2, ...) are read mod 2 first when small
        if x < 0:
            x = (-x) & 1
        if x >= (1 << self.m):
            x = self._red(x)
        return x
    def _red(self, x):
        m, P = self.m, self.poly
        while x.bit_length() > m:
            x ^= P << (x.bit_length() - 1 - m)
        return x
    def mul(self, a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            a <<= 1
            b >>= 1
        return self._red(r)
    def is_zero(self, x):
        return x == 0
    def inv(self, x):
        # x^(2^m - 2)
        r, e, b = 1, (1 << self.m) - 2, x
        while e:
            if e & 1:
                r = self.mul(r, b)
            b = self.mul(b, b)
            e >>= 1
        return r
    zero = 0
    one = 1


def add(K, a, b):
    if isinstance(K, GF2m):
        return a ^ b
    return K.norm(a + b)

def mul(K, a, b):
    if isinstance(K, GF2m):
        return K.mul(a, b)
    return K.norm(a * b)

def neg(K, a):
    if isinstance(K, GF2m):
        return a
    return K.norm(-a)


# ---------------------------------------------------- linear algebra (dicts)

def nullspace_sparse(K, rows, ncols):
    """rows: list of dicts {col: val}.  Returns a basis of the kernel as dicts."""
    if type(K).__name__ == 'Fp' and K.p and ncols and rows:
        try:
            from flint import nmod_mat
            M = nmod_mat(len(rows), ncols, K.p)
            for i, r in enumerate(rows):
                for c, v in r.items():
                    M[i, c] = int(v) % K.p
            X, nul = M.nullspace()
            out = []
            for j in range(nul):
                out.append({i: int(X[i, j]) for i in range(ncols) if int(X[i, j])})
            return out
        except ImportError:
            pass
    # generic Gaussian elimination
    piv = {}          # pivot col -> reduced row (dict), row normalized to 1 at pivot
    order = []
    for r in rows:
        r = {c: v for c, v in r.items() if not K.is_zero(v)}
        for c in list(order):
            if c in r and not K.is_zero(r[c]):
                f = r[c]
                for cc, vv in piv[c].items():
                    r[cc] = add(K, r.get(cc, K.zero), neg(K, mul(K, f, vv)))
                    if K.is_zero(r[cc]):
                        del r[cc]
        if not r:
            continue
        c0 = min(r)
        iv = K.inv(r[c0])
        r = {c: mul(K, v, iv) for c, v in r.items()}
        # eliminate c0 from existing pivots
        for c in order:
            if c0 in piv[c]:
                f = piv[c][c0]
                pr = piv[c]
                for cc, vv in r.items():
                    pr[cc] = add(K, pr.get(cc, K.zero), neg(K, mul(K, f, vv)))
                    if K.is_zero(pr[cc]):
                        del pr[cc]
        piv[c0] = r
        order.append(c0)
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for fc in free:
        v = {fc: K.one}
        for c, r in piv.items():
            if fc in r:
                v[c] = neg(K, r[fc])
        out.append(v)
    return out


# ---------------------------------------------------------- module matrices

class Rep:
    """Matrices of e_i, f_i, h_i on L^lambda, with degrees and parities."""

    def __init__(self, A, par, p, lam, field=None, **kw):
        self.mod = HWModule(A, par, p, lam, field=field, **kw)
        m = self.mod
        self.K = m.K
        self.n = m.n
        self.A = A
        self.par = par
        self.index = {}
        self.deg = []
        self.parity = []
        for d in sorted(m.dim, key=lambda d: (sum(d), d)):
            for a in range(m.dim[d]):
                self.index[(d, a)] = len(self.deg)
                self.deg.append(tuple(-x for x in d))
                self.parity.append(sum(d[j] * par[j] for j in range(m.n)) % 2)
        self.N = len(self.deg)
        K = self.K
        # sparse matrices: dict col -> dict row -> val
        self.E, self.F, self.H = [], [], []
        for i in range(m.n):
            Ei, Fi, Hi = {}, {}, {}
            for d, md in m.dim.items():
                if not md:
                    continue
                mu = m.mu(d, i)
                for a in range(md):
                    col = self.index[(d, a)]
                    if not K.is_zero(mu):
                        Hi[col] = {col: mu}
                # f_i : M_d -> M_{d+e_i}
                d2 = list(d); d2[i] += 1; d2 = tuple(d2)
                Fm = m.F.get((d2, i))
                if Fm is not None:
                    for a in range(md):
                        col = self.index[(d, a)]
                        colv = {}
                        for r in range(len(Fm)):
                            v = Fm[r][a]
                            if not K.is_zero(v):
                                colv[self.index[(d2, r)]] = v
                        if colv:
                            Fi[col] = colv
                Em = m.E.get((d, i))
                if Em is not None:
                    d3 = list(d); d3[i] -= 1; d3 = tuple(d3)
                    for a in range(md):
                        col = self.index[(d, a)]
                        colv = {}
                        for r in range(len(Em)):
                            v = Em[r][a]
                            if not K.is_zero(v):
                                colv[self.index[(d3, r)]] = v
                        if colv:
                            Ei[col] = colv
            self.E.append(Ei); self.F.append(Fi); self.H.append(Hi)

    def sdim(self):
        return self.mod.sdim()


# ------------------------------------------------ operators as flat dicts
# an operator is a dict {(row, col): val}

def op_from_cols(cols):
    return {(r, c): v for c, cv in cols.items() for r, v in cv.items()}

def op_mul(K, X, Y):
    # (XY)[r,c] = sum_k X[r,k] Y[k,c]
    Xc = {}
    for (r, k), v in X.items():
        Xc.setdefault(k, []).append((r, v))
    out = {}
    for (k, c), w in Y.items():
        for r, v in Xc.get(k, ()):
            key = (r, c)
            out[key] = add(K, out.get(key, K.zero), mul(K, v, w))
    return {k: v for k, v in out.items() if not K.is_zero(v)}

def op_bracket(K, X, px, Y, py):
    XY = op_mul(K, X, Y)
    YX = op_mul(K, Y, X)
    s = -1 if (px and py) else 1          # [X,Y] = XY - (-1)^{pxpy} YX
    out = dict(XY)
    for k, v in YX.items():
        t = v if s == -1 else neg(K, v)
        out[k] = add(K, out.get(k, K.zero), t)
    return {k: v for k, v in out.items() if not K.is_zero(v)}


class Span:
    """Echelon basis of a space of operators of one Z^n-degree."""
    def __init__(self, K):
        self.K = K
        self.piv = {}     # key -> normalized reduced vector
        self.basis = []   # original (un-reduced) independent vectors
    def reduce(self, v):
        K = self.K
        v = dict(v)
        for key, r in self.piv.items():
            if key in v:
                f = v[key]
                for kk, vv in r.items():
                    v[kk] = add(K, v.get(kk, K.zero), neg(K, mul(K, f, vv)))
                    if K.is_zero(v[kk]):
                        del v[kk]
        return v
    def add(self, v):
        K = self.K
        r = self.reduce(v)
        if not r:
            return False
        k0 = min(r)
        iv = K.inv(r[k0])
        r = {k: mul(K, x, iv) for k, x in r.items()}
        for key in list(self.piv):
            pr = self.piv[key]
            if k0 in pr:
                f = pr[k0]
                for kk, vv in r.items():
                    pr[kk] = add(K, pr.get(kk, K.zero), neg(K, mul(K, f, vv)))
                    if K.is_zero(pr[kk]):
                        del pr[kk]
        self.piv[k0] = r
        self.basis.append(v)
        return True


def vadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


class Image:
    """rho(g(A)) in gl(M), graded by Z^n; basis per degree."""

    def __init__(self, rep, extra_identity=False, extra_grading=False, maxdeg=60):
        K = rep.K
        n = rep.n
        self.rep = rep
        self.K = K
        gens = []
        for i in range(n):
            e = tuple(1 if j == i else 0 for j in range(n))
            me = tuple(-x for x in e)
            gens.append((e, rep.par[i], op_from_cols(rep.E[i])))
            gens.append((me, rep.par[i], op_from_cols(rep.F[i])))
        self.spans = {}
        self.par = {}
        queue = []
        for dg, pg, X in gens:
            if self._add(dg, pg, X):
                queue.append((dg, pg, X))
        # closure under ad of the generators
        while queue:
            dg, pg, X = queue.pop()
            for dg2, pg2, Y in gens:
                d = vadd(dg, dg2)
                Z = op_bracket(K, Y, pg2, X, pg)
                if Z and self._add(d, (pg + pg2) % 2, Z):
                    queue.append((d, (pg + pg2) % 2, Z))
        zero = tuple([0] * n)
        if extra_grading:
            for j in range(n):
                D = {(a, a): K.norm(rep.deg[a][j]) for a in range(rep.N)
                     if not K.is_zero(K.norm(rep.deg[a][j]))}
                if D:
                    self._add(zero, 0, D)
        if extra_identity:
            self._add(zero, 0, {(a, a): K.one for a in range(rep.N)})
        self.basis = []   # (degree, parity, operator)
        for d, sp in sorted(self.spans.items()):
            for X in sp.basis:
                self.basis.append((d, self.par[d], X))

    def _add(self, d, par, X):
        sp = self.spans.setdefault(d, Span(self.K))
        self.par[d] = par
        return sp.add(X)

    def dims(self):
        ev = sum(1 for d, p, X in self.basis if p == 0)
        return ev, len(self.basis) - ev

    def highest_root(self):
        """Z^n-degree of the top positive root (largest height)."""
        pos = [d for d in self.spans if all(x >= 0 for x in d) and any(d)
               and self.spans[d].basis]
        top = max(pos, key=sum)
        return top


def labels_of(A, K, theta):
    """theta(h_i) = sum_j theta_j a_ij."""
    n = len(A)
    return [K.norm(sum(theta[j] * A[i][j] for j in range(n))) for i in range(n)]


# ------------------------------------------------------------- the prolong

def prolong(rep, img, kmax=6, verbose=True, out=None):
    """Dimensions (ev, od) of g_1, ..., g_kmax (stops at the first zero)."""
    K = rep.K
    N = rep.N
    sgn = lambda a, b: -1 if (rep.parity[a] and rep.parity[b]) else 1
    # level 0: elements indexed by position in img.basis; act[b] = dict over M
    lev_deg = [d for d, p, X in img.basis]
    lev_par = [p for d, p, X in img.basis]
    act = []
    for d, p, X in img.basis:
        cols = {}
        for (r, c), v in X.items():
            cols.setdefault(c, {})[r] = v
        act.append(cols)
    prev = (lev_deg, lev_par, act)     # g_{k-1}
    results = []
    for k in range(1, kmax + 1):
        pdeg, ppar, pact = prev
        # candidate degrees delta: deg(psi) - deg(v_a)
        bydeg = {}
        for c, d in enumerate(pdeg):
            bydeg.setdefault(d, []).append(c)
        cands = set()
        for d in bydeg:
            for a in range(N):
                cands.add(tuple(x - y for x, y in zip(d, rep.deg[a])))
        new_deg, new_par, new_act = [], [], []
        for delta in sorted(cands):
            # unknowns: (a, c) with deg(psi_c) = delta + deg(v_a)
            unk = {}
            for a in range(N):
                for c in bydeg.get(vadd(delta, rep.deg[a]), ()):
                    unk[(a, c)] = len(unk)
            if not unk:
                continue
            rows = []
            # equations for pairs (a,b), a<=b: sum_c x_{ac} psi_c(v_b)
            #   - s(a,b) sum_c x_{bc} psi_c(v_a) = 0 in g_{k-2}
            ua = {}
            for (a, c), i in unk.items():
                ua.setdefault(a, []).append((c, i))
            alist = sorted(ua)
            for ia, a in enumerate(alist):
                for b in range(N):
                    # only pairs where at least one side has unknowns
                    if b in ua and b < a:
                        continue
                    row = {}
                    for c, i in ua[a]:
                        for tgt, v in pact[c].get(b, {}).items():
                            row.setdefault(tgt, {})
                            row[tgt][i] = add(K, row[tgt].get(i, K.zero), v)
                    s = sgn(a, b)
                    # p = 2: for an odd v the condition is [phi(v), v] = 0
                    # (the squaring), not the vacuous 2[phi(v), v] = 0
                    sq = (a == b and rep.parity[a] and getattr(K, 'p', 0) == 2)
                    for c, i in (() if sq else ua.get(b, ())):
                        for tgt, v in pact[c].get(a, {}).items():
                            vv = v if s == -1 else neg(K, v)
                            row.setdefault(tgt, {})
                            row[tgt][i] = add(K, row[tgt].get(i, K.zero), vv)
                    for tgt, r in row.items():
                        r = {i: v for i, v in r.items() if not K.is_zero(v)}
                        if r:
                            rows.append(r)
            sols = nullspace_sparse(K, rows, len(unk))
            if not sols:
                continue
            inv = {i: ac for ac, i in unk.items()}
            par = None
            for s_ in sols:
                phi = {}
                for i, v in s_.items():
                    a, c = inv[i]
                    phi.setdefault(a, {})
                    phi[a][c] = add(K, phi[a].get(c, K.zero), v)
                new_deg.append(delta)
                # parity: parity(psi_c) + parity(v_a), for any (a,c) in support
                i0 = next(iter(s_))
                a0, c0 = inv[i0]
                new_par.append((ppar[c0] + rep.parity[a0]) % 2)
                new_act.append(phi)
        ev = sum(1 for p in new_par if p == 0)
        od = len(new_par) - ev
        results.append((ev, od))
        line = "  g_%d: %d|%d" % (k, ev, od)
        if verbose:
            print(line, flush=True)
        if out:
            out.write(line + "\n"); out.flush()
        if not new_deg:
            break
        prev = (new_deg, new_par, new_act)
    return results


def analyse(name, A, par, p, lam, kmax=4, identity=False, grading=False, field=None,
            **kw):
    rep = Rep(A, par, p, lam, field=field, **kw)
    img = Image(rep, extra_identity=identity, extra_grading=grading)
    print("%s  lambda=%s  sdim M=%s  sdim g0=%s" % (name, lam, rep.sdim(), img.dims()),
          flush=True)
    return prolong(rep, img, kmax=kmax)


if __name__ == "__main__":
    # calibration: sl(3) over F_3, the 3-dim module: g_1 must be 15-dim (svect(3))
    sl3 = [[2, -1], [-1, 2]]
    r = analyse("sl(3) p=3", sl3, [0, 0], 3, [0, 1], kmax=3)
    assert r[0] == (15, 0), r
    # sl(3) over Q (via p=101), the 3-dim module: g_1 = 15 (svect), g_2 = 24
    r = analyse("sl(3) p=101", sl3, [0, 0], 101, [0, 1], kmax=2)
    assert r[0] == (15, 0), r
    # gl(3) = sl(3) + identity over p=101: vect(3): g_1 = 18
    r = analyse("gl(3) p=101", sl3, [0, 0], 101, [0, 1], kmax=2, identity=True)
    assert r[0] == (18, 0), r
    # sl(3) adjoint: no prolong
    r = analyse("sl(3) ad p=101", sl3, [0, 0], 101, [1, 1], kmax=2)
    assert r[0] == (0, 0), r
    # sp(4): tautological 4-dim: prolong = h(4) : g_1 = S^3(V) = 20
    sp4 = [[2, -1], [-2, 2]]
    r = analyse("sp(4) p=101", sp4, [0, 0], 101, [0, 1], kmax=2)
    assert r[0] == (20, 0), r
    # osp(1|2) on its 1|2 module: prolong = k(1|?)... just print
    r = analyse("osp(1|2) p=101 L(1)", [[1]], [1], 101, [1], kmax=3)
    # sl(1|2) tautological 1|2 : prolong = svect(1|2)?: g_1 nonzero
    r = analyse("sl(1|2) p=101", [[0, -1], [-1, 2]], [1, 0], 101, [1, 0], kmax=2)
    print("calibration done")


def prolong_plain(K, parity, g0, kmax=3, verbose=True):
    """The Cartan prolong without using any grading: parity = parities of the
    basis of g_{-1} (N vectors); g0 = list of (operator dict {(r,c):v}, parity)."""
    N = len(parity)
    act = []
    ppar = []
    for X, px in g0:
        cols = {}
        for (r, c), v in X.items():
            cols.setdefault(c, {})[r] = v
        act.append(cols); ppar.append(px)
    res = []
    for k in range(1, kmax + 1):
        dims = [0, 0]
        new_act, new_par = [], []
        for pphi in (0, 1):
            unk = {}
            for a in range(N):
                for c in range(len(act)):
                    if (ppar[c] + parity[a]) % 2 == pphi:
                        unk[(a, c)] = len(unk)
            if not unk:
                continue
            ua = {}
            for (a, c), i in unk.items():
                ua.setdefault(a, []).append((c, i))
            rows = []
            for a in range(N):
                for b in range(a, N):
                    s = -1 if (parity[a] and parity[b]) else 1
                    sq = (a == b and parity[a] and getattr(K, 'p', 0) == 2)
                    row = {}
                    for c, i in ua.get(a, ()):
                        for t, v in act[c].get(b, {}).items():
                            row.setdefault(t, {})
                            row[t][i] = add(K, row[t].get(i, K.zero), v)
                    for c, i in (() if sq else ua.get(b, ())):
                        for t, v in act[c].get(a, {}).items():
                            vv = v if s == -1 else neg(K, v)
                            row.setdefault(t, {})
                            row[t][i] = add(K, row[t].get(i, K.zero), vv)
                    for t, r in row.items():
                        r = {i: v for i, v in r.items() if not K.is_zero(v)}
                        if r:
                            rows.append(r)
            sols = nullspace_sparse(K, rows, len(unk))
            inv = {i: ac for ac, i in unk.items()}
            for s_ in sols:
                phi = {}
                for i, v in s_.items():
                    a, c = inv[i]
                    phi.setdefault(a, {})[c] = v
                new_act.append(phi); new_par.append(pphi)
            dims[pphi] = len(sols)
        res.append(tuple(dims))
        if verbose:
            print("  g_%d: %d|%d" % (k, dims[0], dims[1]), flush=True)
        if not new_act:
            break
        act, ppar = new_act, new_par
    return res
